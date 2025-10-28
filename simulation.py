import matplotlib.pyplot as plt

from drone import Drone
from environment import river, forest, forest_edge, cliffs, climb_entries
from log import logger
from utils import (
    manhattan,
    get_effective_vision_range,
    is_in_cover,
    has_line_of_sight,
    units_spotted_by_vision,
    compute_unit_defend_position,
    compute_bridge_defend_position,
    get_bridges,
)
from config import SUPPRESSION_RECOVERY_RATE

class Simulation:
    """
    Manages the execution of a simulation, including unit behavior,
    line-of-sight updates, drone spotting, HTN planning, and visualization.
    """

    def __init__(self, friendly_units, enemy_units, visualize=False, plan_name="Unknown Plan", enable_animation=False, animation_dir="animations"):
        """
        Initialize the simulation environment.

        Args:
            friendly_units (list): List of FriendlyUnit instances.
            enemy_units (list): List of EnemyUnit instances.
            visualize (bool): Whether to enable visualization.
            plan_name (str): Identifier for the planning session.
            enable_animation (bool): Whether to enable animation recording.
            animation_dir (str): Directory to save animation files.
        """
        self.friendly_units = friendly_units
        self.friendly_units_dict = {u.name: u for u in friendly_units}
        # Keep references to the original unit lists so Gantt charts can
        # include units that are removed from active play (e.g. destroyed
        # or with zero group size).
        self.all_friendly_units = list(friendly_units)

        self.enemy_units = enemy_units
        self.enemy_units_dict = {e.name: e for e in enemy_units}
        self.all_enemy_units = list(enemy_units)
        # Track the next bridge index for delay mission assignments so that
        # enemies evenly cover available bridges.
        self.next_delay_bridge_index = 0

        self.friendly_drone = Drone("friendly", "enemy")
        self.enemy_drone = Drone("enemy", "friendly")

        self.river = river
        self.forest = forest
        self.forest_edge = forest_edge
        self.cliffs = cliffs
        self.climb_entries = climb_entries

        active_enemy = next((e for e in enemy_units if e.state.get("enemy_alive")), None)
        for u in self.friendly_units:
            u.state["enemy"] = active_enemy.state if active_enemy else {}
            u.state.update({
                "visible_enemies": [],
                "all_enemies": [e.state for e in self.enemy_units],
                "total_enemies": len(self.enemy_units),
                "scout_steps": 0
            })
            u.state["hit_this_step"] = False
            u.state["attacked_by"] = []
            u.sim = self

        for e in self.enemy_units:
            e.state["hit_this_step"] = False
            e.state["attacked_by"] = []
            e.sim = self
            e.initialize_defend_lines()

            if e.state.get("mission") == "DelayMission" and "delay_bridges" not in e.state:
                bridges = get_bridges()
                num_bridges = len(bridges)
                bridge_idx = self.next_delay_bridge_index % num_bridges
                self.next_delay_bridge_index += 1
                order = list(range(bridge_idx, num_bridges))
                e.state.update(
                    {
                        "delay_bridges": bridges,
                        "delay_bridge_order": order,
                        "delay_order_pos": 0,
                        "delay_stage": 0,
                        "current_delay_bridge": bridge_idx,
                        "delay_health_thresholds": [0.75, 0.5],
                    }
                )
                pos = compute_bridge_defend_position(self, e, bridges[bridge_idx])
                e.state["defend_position"] = pos
                e.state["position"] = pos
                e.state["picked_position"] = True
                e.state["delay_retreating"] = False

        self.step_count = 0
        self.defensive_line = 0
        self.visualize = visualize
        self.plan_name = plan_name
        self.enable_animation = enable_animation
        self.animation_dir = animation_dir
        self.animator = None

        if self.visualize or self.enable_animation:
            from plotter import SimulationPlotter
            plt.ion()
            if self.visualize:
                plt.show(block=False)
            self.plotter = SimulationPlotter(self, visualize=self.visualize)
        
        # Initialize animation system if enabled
        if self.enable_animation:
            from animation import SimulationAnimator
            self.animator = SimulationAnimator(self, self.animation_dir, self.plan_name)

        # Cache staging locations shared across friendly units so that
        # each phase uses a consistent staging point. This cache is
        # cleared whenever a target is destroyed and units replan.
        self.group_staging_cache = {}
        # Track when all assigned units have reached their staging points for a
        # given attack phase. This allows WaitForGroup to sync across units
        # even when their execute order differs each step.
        self.group_arrival_flags = {}

    def _check_defensive_lines(self):
        """Advance defensive line if any enemy falls below health thresholds."""
        thresholds = [0.75, 0.5]
        idx = self.defensive_line
        if idx >= len(thresholds):
            return
        attacked = None
        for e in self.enemy_units:
            if not e.state.get("enemy_alive", False):
                continue
            max_health = e.state.get("max_health", 0)
            health = e.state.get("health", 0)
            if max_health <= 0:
                continue
            # Only trigger line advancement when a living enemy has taken
            # damage and dropped below the configured threshold. This avoids
            # repositioning before any engagement has occurred.
            if health < max_health and health <= thresholds[idx] * max_health:
                attacked = e
                break
        if not attacked:
            return
        self.defensive_line = idx + 1
        center = attacked.state["defend_lines"][self.defensive_line]
        for enemy in self.enemy_units:
            lines = enemy.state.get("defend_lines", [])
            if len(lines) <= self.defensive_line:
                continue
            target = lines[self.defensive_line]
            cx, cy = center
            tx, ty = target
            if enemy is attacked:
                new_pos = target
            else:
                new_pos = ((tx + cx) // 2, (ty + cy) // 2)
            enemy.state["defend_position"] = new_pos
            enemy.state["picked_position"] = True
            enemy.state["force_delay"] = True
            enemy.state["current_line"] = self.defensive_line

    def update_enemy_behavior(self):
        """Update each enemy unit's plan and execute its next action."""
        friendly_units = [u for u in self.friendly_units if u.state.get("health", 0) > 0]
        for enemy in self.enemy_units:
            if enemy.state["enemy_alive"]:
                enemy.update_plan(friendly_units)
                enemy.execute_next_task()
                enemy.current_goal = enemy.get_goal_position()
                if self.visualize:
                    logger.info(f"{enemy.state['name']} position: {enemy.state['position']}")
                    logger.info(f"{enemy.state['name']}'s current goal: {enemy.current_goal}")

    def update_friendly_enemy_info(self):
        """Update each friendly unit's knowledge of the closest visible enemy."""
        active_enemies = [e for e in self.enemy_units if e.state["enemy_alive"]]
        for u in self.friendly_units:
            closest_enemy = None
            min_distance = float('inf')
            for e in active_enemies:
                distance = manhattan(u.state["position"], e.state["position"])
                has_los = has_line_of_sight(e.state["position"], u.state["position"])
                in_cover = is_in_cover(u.state["position"])
                stealth_modifier = u.state.get("stealth_modifier", 0)
                effective_vision_range = get_effective_vision_range(
                    e.state.get("vision_range", 20), stealth_modifier, in_cover, has_los)
                if distance <= effective_vision_range and has_los and distance < min_distance:
                    min_distance = distance
                    closest_enemy = e.state
            u.state["enemy"] = closest_enemy or {}
            if self.visualize:
                logger.info(f"{u.name} state['enemy']: {u.state['enemy'].get('name', 'None')}")
                visible_enemies = u.state.get("visible_enemies", [])
                logger.info(f"{u.name} at {u.state['position']} sees enemies: {visible_enemies}")

    def finalize_task_logs(self):
        """Close any open task logs for friendly units."""
        # Use the full list of friendly units so that units removed from the
        # active roster (e.g. destroyed units) still have their final task
        # segments closed out for reporting/visualization.
        for u in getattr(self, "all_friendly_units", self.friendly_units):
            current = getattr(u, "_current_task", None)
            if current is not None:
                u.task_log.append((current, u._task_start_step, self.step_count))
                u._current_task = None

    def evaluate_plan(self):
        """Calculate and return simulation score and state summary."""
        self.finalize_task_logs()
        total_friendly = sum(u.state["health"] for u in self.friendly_units)
        max_friendly = sum(u.state["max_health"] for u in self.friendly_units)
        health = sum(e.state["health"] for e in self.enemy_units if e.state["enemy_alive"])
        max_enemy = sum(e.state["max_health"] for e in self.enemy_units)
        outpost_secured = any(u.state.get("outpost_secured", False) for u in self.friendly_units)
        steps = self.step_count
        friendly_ratio = total_friendly / max_friendly if max_friendly > 0 else 0
        enemy_ratio = health / max_enemy if max_enemy > 0 else 0

        max_steps = getattr(self, "max_steps", steps) or steps
        step_ratio = steps / max_steps if max_steps > 0 else 0
        score = (
            friendly_ratio * 20
            - enemy_ratio * 20
            + (10 if outpost_secured else -10)
            - step_ratio * 10
        )

        print(
            f"\n\n\nScore: {score:.1f} \n Health: {total_friendly} \n Enemy Health: {health} \n Outpost Secured: {outpost_secured} \n Steps Taken: {steps}"
        )
        return {
            "score": score,
            "health": total_friendly,
            "enemy_health": health,
            "outpost_secured": outpost_secured,
            "steps_taken": steps
        }
    
    def print_plan_trees(self):
        """Print plan trees for all friendly units."""
        for unit in self.friendly_units:
            print(f"Plan tree for {unit.name}:")
            unit.print_plan_tree()
            print()

    def step(self):
        """
        Execute a single simulation step:
        - Update drones and visibility
        - Update enemy and friendly behavior
        - Refresh plot if enabled
        """
        self.step_count += 1
        # Avoid periodic repositioning of enemy units unless they've actually
        # taken damage. This prevents enemies from "retreating" or picking new
        # defensive positions before any engagement has occurred.
        if (self.step_count % 50 == 0 and any(e.state.get("health", 0) < e.state.get("max_health", 0) for e in self.enemy_units)):
            for enemy in self.enemy_units:
                if enemy.state.get("enemy_alive", False):
                    enemy.state["defend_position"] = compute_unit_defend_position(self, enemy)
                    enemy.state["picked_position"] = True
        if self.visualize:
            logger.info(f"--- Simulation Step {self.step_count} ---")

        for unit in self.friendly_units + self.enemy_units:
            unit.state["hit_this_step"] = False
            unit.state["attacked_by"] = []

        self.friendly_drone.update(self)
        self.enemy_drone.update(self)
        self._check_defensive_lines()

        drone_seen_by_friendlies = set(self.friendly_drone.last_known.keys())
        drone_seen_by_enemies = set(self.enemy_drone.last_known.keys())

        for friend in self.friendly_units:
            los_list = [e.name for e in units_spotted_by_vision(friend, self.enemy_units)]
            friend.state["visible_enemies"] = los_list
            merged = set(los_list) | drone_seen_by_friendlies
            friend.state["spotted_enemies"] = [
                n for n in merged
                if n in self.enemy_units_dict and self.enemy_units_dict[n].state["current_group_size"] > 0
            ]

        for enemy in self.enemy_units:
            los_list = [u.name for u in units_spotted_by_vision(enemy, self.friendly_units)]
            enemy.state["visible_enemies"] = los_list
            merged = set(los_list) | drone_seen_by_enemies
            enemy.state["spotted_enemies"] = [
                n for n in merged
                if n in self.friendly_units_dict and self.friendly_units_dict[n].state["current_group_size"] > 0
            ]

        self.update_friendly_enemy_info()
        self.update_enemy_behavior()

        for u in self.friendly_units:
            last = set(u.state.get("_last_spotted", []))
            now = set(u.state["spotted_enemies"])
            if now - last:
                logger.info(f"{u.name} spotted new foes {now - last}; forcing replan")
                u.update_plan(force_replan=True)
            elif not u.current_plan:
                logger.info(f"{u.name} has empty plan; replanning")
                u.update_plan(force_replan=True)
            elif isinstance(u.current_plan[0], tuple) and u.current_plan[0][0] == "AttackEnemy":
                tgt = u.current_plan[0][1]
                if not any(e.name == tgt and e.state.get("enemy_alive", False) for e in self.enemy_units):
                    logger.info(f"{u.name}'s target {tgt} died; replanning")
                    u.update_plan(force_replan=True)
            else:
                u.update_plan()

            u.state["_last_spotted"] = list(now)
            logger.info(f"{u.name} current plan: {u.current_plan}")
            if self.visualize and u.current_plan and isinstance(u.current_plan[0], tuple):
                logger.info(f"{u.name} targeting: {u.current_plan[0][1]}")
            u.execute_next_task()
            if self.visualize:
                logger.info(f"{u.name}'s current goal: {u.get_goal_position()}")

        for unit in self.friendly_units + self.enemy_units:
            if not unit.state.get("hit_this_step", False):
                suppression = unit.state.get("suppression_from_enemy", 0.0)
                if suppression > 0:
                    unit.state["suppression_from_enemy"] = max(0.0, suppression - SUPPRESSION_RECOVERY_RATE)

        if self.visualize:
            self.plotter.update()
        
        # Save animation frame if animation is enabled
        if self.enable_animation and self.animator:
            self.animator.save_step_image(self.step_count)

    def run(self, max_steps=500):
        """
        Run the simulation for up to max_steps or until the mission is complete.

        Returns:
            dict: Evaluation results after the simulation ends.
        """
        self.step_count = 0
        self.max_steps = max_steps
        for u in self.friendly_units:
            u.update_plan(force_replan=True)
        for e in self.enemy_units:
            e.update_plan(self.friendly_units)

        if self.visualize:
            self.plotter.update()
            plt.pause(0.5)

        for _ in range(max_steps):
            alive_friendlies = [u for u in self.friendly_units if u.state.get("health", 0) > 0]
            if not alive_friendlies:
                if self.visualize:
                    self.plotter.update()
                    logger.info("\nMission failed: All friendly units defeated.")
                if self.enable_animation and self.animator:
                    result = self.evaluate_plan()
                    self.animator.finalize_animation(result)
                    return result
                return self.evaluate_plan()
            if any(u.state.get("outpost_secured", False) for u in alive_friendlies):
                if self.visualize:
                    self.plotter.update()
                    logger.info("\nMission accomplished: Outpost secured!")
                if self.enable_animation and self.animator:
                    result = self.evaluate_plan()
                    self.animator.finalize_animation(result)
                    return result
                return self.evaluate_plan()
            self.debug_paths = []
            self.step()

            alive_friendlies = [u for u in self.friendly_units if u.state.get("current_group_size", 0) > 0]
            if len(alive_friendlies) != len(self.friendly_units):
                self.friendly_units = alive_friendlies
                self.friendly_units_dict = {u.name: u for u in self.friendly_units}
            if not self.friendly_units:
                if self.visualize:
                    self.plotter.update()
                    logger.info("\nMission failed: All friendly units defeated.")
                if self.enable_animation and self.animator:
                    result = self.evaluate_plan()
                    self.animator.finalize_animation(result)
                    return result
                return self.evaluate_plan()

            alive_enemies = [e for e in self.enemy_units if e.state.get("current_group_size", 0) > 0]
            if len(alive_enemies) != len(self.enemy_units):
                self.enemy_units = alive_enemies
                self.enemy_units_dict = {e.name: e for e in self.enemy_units}

            self._reassign_idle_units()

            if self.visualize:
                self.plotter.update()
                plt.pause(0.5)
                while self.plotter.paused:
                    plt.pause(0.2)

        if self.visualize:
            self.plotter.update()
            logger.info("\nMission incomplete after maximum steps.")
        
        # Finalize animation if enabled
        if self.enable_animation and self.animator:
            result = self.evaluate_plan()
            self.animator.finalize_animation(result)
            return result
        
        return self.evaluate_plan()

    def _reassign_idle_units(self):
        """Reassign idle friendly units if a phase loses all its attackers."""
        seq = getattr(self, "attack_sequence", [])
        if not seq:
            return

        updated = False
        new_seq = list(seq)

        for idx, (enemy_name, attack_type, assigned) in enumerate(seq):
            enemy = self.enemy_units_dict.get(enemy_name)
            if not enemy or not enemy.state.get("enemy_alive", False):
                continue

            alive_assigned = [
                name for name in assigned
                if name in self.friendly_units_dict
            ]

            if alive_assigned:
                new_seq[idx] = (enemy_name, attack_type, alive_assigned)
                continue

            idle_units = [
                u for u in self.friendly_units
                if not u.current_plan or u.current_plan == ["Hold"] or (
                    len(u.current_plan) == 1 and u.current_plan[0] == "Hold"
                )
            ]
            if not idle_units:
                continue

            logger.info(
                f"All attackers for phase {idx} vs {enemy_name} destroyed; "
                f"reassigning idle units {[u.name for u in idle_units]}"
            )

            new_seq[idx] = (
                enemy_name,
                attack_type,
                [u.name for u in idle_units],
            )

            if hasattr(self, "group_staging_cache"):
                self.group_staging_cache.pop((enemy_name, idx), None)
            if hasattr(self, "group_arrival_flags"):
                self.group_arrival_flags.pop((enemy_name, idx), None)

            for u in idle_units:
                u.update_plan(force_replan=True)

            updated = True

        if updated:
            self.attack_sequence = new_seq
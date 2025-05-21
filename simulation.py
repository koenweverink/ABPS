import matplotlib.pyplot as plt

from drone import Drone
from environment import river, forest, forest_edge, cliffs, climb_entries
from log import logger
from utils import manhattan, get_effective_vision_range, is_in_cover, has_line_of_sight, units_spotted_by_vision

class Simulation:
    def __init__(
        self,
        friendly_units,
        enemy_units,
        visualize=False,
        plan_name="Unknown Plan",
    ):
        # ---- core state setup ----
        self.friendly_units = friendly_units
        self.friendly_units_dict = {u.name: u for u in friendly_units}

        self.enemy_units = enemy_units
        self.enemy_units_dict  = {e.name: e for e in enemy_units}

        # Recon drones
        self.friendly_drone = Drone(
            side="friendly", target_side="enemy",
            n_cols=3, n_rows=2, stay_rounds=10, spot_prob=0.2
        )
        self.enemy_drone = Drone(
            side="enemy", target_side="friendly",
            n_cols=3, n_rows=2, stay_rounds=10, spot_prob=0.2
        )
        
        self.river = river
        self.forest = forest
        self.forest_edge = forest_edge
        self.cliffs = cliffs
        self.climb_entries = climb_entries

        # Initialize each friendly unit’s view state
        active_enemy = next((e for e in enemy_units if e.state.get("enemy_alive")), None)
        for u in self.friendly_units:
            u.state["enemy"]           = active_enemy.state if active_enemy else {}
            u.state["visible_enemies"] = []
            u.state["all_enemies"]     = [e.state for e in self.enemy_units]
            u.state["total_enemies"]   = len(self.enemy_units)
            u.state["scout_steps"]     = 0
            u.sim = self

        for e in self.enemy_units:
            e.sim = self

        self.step_count = 0
        self.visualize  = visualize
        self.plan_name  = plan_name

        # ---- plotting hand-off ----
        if self.visualize:
            # Hand off _all_ plotting to SimulationPlotter
            from plotter import SimulationPlotter
            plt.ion()
            plt.show(block=False)
            self.plotter = SimulationPlotter(self, visualize=True)

    def update_enemy_behavior(self):
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
        active_enemies = [e for e in self.enemy_units if e.state["enemy_alive"]]
        for u in self.friendly_units:
            if active_enemies:
                closest_enemy = None
                min_distance = float('inf')
                for e in active_enemies:
                    distance = manhattan(u.state["position"], e.state["position"])
                    has_los = has_line_of_sight(e.state["position"], u.state["position"])
                    in_cover = is_in_cover(u.state["position"])
                    stealth_modifier = u.state.get("stealth_modifier", 0)
                    effective_vision_range = get_effective_vision_range(
                        e.state.get("vision_range", 20),
                        stealth_modifier,
                        in_cover,
                        has_los
                    )
                    if distance <= effective_vision_range and has_los:
                        if distance < min_distance:
                            min_distance = distance
                            closest_enemy = e.state
                u.state["enemy"] = closest_enemy or {}
                if self.visualize:
                    logger.info(f"{u.name} state['enemy']: {u.state['enemy'].get('name', 'None')}")
            else:
                for u in self.friendly_units:
                    u.state["enemy"] = {}
                    if self.visualize:
                        logger.info(f"{u.name} state['enemy']: None (no active enemies)")
            if self.visualize:
                visible_enemies = u.state.get("visible_enemies", [])
                logger.info(f"{u.name} at {u.state['position']} sees enemies: {visible_enemies}")

    def evaluate_plan(self):
        total_friendly = sum(u.state["health"] for u in self.friendly_units)
        max_friendly = sum(u.state["max_health"] for u in self.friendly_units)
        health = sum(e.state["health"] for e in self.enemy_units if e.state["enemy_alive"])
        max_enemy = sum(e.state["max_health"] for e in self.enemy_units)
        outpost_secured = any(u.state.get("outpost_secured", False) for u in self.friendly_units)
        steps = self.step_count
        friendly_ratio = total_friendly / max_friendly if max_friendly > 0 else 0
        enemy_ratio = health / max_enemy if max_enemy > 0 else 0
        score = (friendly_ratio * 20) - (enemy_ratio * 20) + (10 if outpost_secured else -10) - 0.1 * steps
        return {
            "score": score,
            "health": total_friendly,
            "enemy_health": health,
            "outpost_secured": outpost_secured,
            "steps_taken": steps
        }

    def step(self):
        # 0) Advance the clock & log header
        self.step_count += 1
        if self.visualize:
            logger.info(f"--- Simulation Step {self.step_count} ---")

        # 2) Refresh drones
        drone_seen_friends = self.friendly_drone.update(self)  # set of enemy names
        drone_seen_enemies = self.enemy_drone.update(self)     # set of friendly names

        # 3) Build LOS‐based visibility via your helper
        #    and merge in drone reports for spotted_enemies
        self.friendly_drone.update(self)
        self.enemy_drone.update(self)
        # Now read out what they saw
        drone_seen_by_friendlies = set(self.friendly_drone.last_known.keys())
        drone_seen_by_enemies    = set(self.enemy_drone.last_known.keys())

        # Build LOS via helper + merge in drone reports
        for friend in self.friendly_units:
            los_list = [e.name for e in units_spotted_by_vision(friend, self.enemy_units)]
            friend.state["visible_enemies"] = los_list

            merged = set(los_list) | drone_seen_by_friendlies
            friend.state["spotted_enemies"] = [
                n for n in merged
                if n in self.enemy_units_dict
                and self.enemy_units_dict[n].state["current_group_size"] > 0
            ]

        for enemy in self.enemy_units:
            los_list = [u.name for u in units_spotted_by_vision(enemy, self.friendly_units)]
            enemy.state["visible_enemies"] = los_list

            merged = set(los_list) | drone_seen_by_enemies
            enemy.state["spotted_enemies"] = [
                n for n in merged
                if n in self.friendly_units_dict
                and self.friendly_units_dict[n].state["current_group_size"] > 0
            ]

        self.update_friendly_enemy_info()
        # 4) Enemy turn
        self.update_enemy_behavior()

        # 5) Friendly turn: HTN replan & execution
        for u in self.friendly_units:
            last = set(u.state.get("_last_spotted", []))
            now  = set(u.state["spotted_enemies"])
            # force‐replan on truly new sightings or dead targets
            if now - last:
                logger.info(f"{u.name} spotted new foes {now - last}; forcing replan")
                u.update_plan(force_replan=True)
            elif not u.current_plan:
                logger.info(f"{u.name} has empty plan; replanning")
                u.update_plan(force_replan=True)
            elif (
                isinstance(u.current_plan[0], tuple)
                and u.current_plan[0][0] == "AttackEnemy"
            ):
                tgt = u.current_plan[0][1]
                if not any(e.name == tgt and e.state.get("enemy_alive", False)
                        for e in self.enemy_units):
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

        # 6) Refresh the plot
        if self.visualize:
            self.plotter.update()


    def run(self, max_steps=500):
        """
        Main simulation loop:
        - Perform initial planning
        - For up to max_steps:
            * Check for mission success
            * step()
            * Prune dead enemies
            * Update visualization
        - Return the evaluation of the executed plan
        """
        # 0) Initial planning
        self.step_count = 0
        for u in self.friendly_units:
            u.update_plan(force_replan=True)
        for e in self.enemy_units:
            e.update_plan(self.friendly_units)

        if self.visualize:
            self.plotter.update()
            plt.pause(0.5)

        # 1) Main loop
        for _ in range(max_steps):
            # 1a) Success check: all surviving friendlies at outpost
            alive_friendlies = [
                u for u in self.friendly_units
                if u.state.get("health", 0) > 0
            ]
            if (alive_friendlies
                    and all(u.state["position"] == u.state.get("outpost_position")
                            for u in alive_friendlies)):
                if self.visualize:
                    self.plotter.update()
                    logger.info("\nMission accomplished: Outpost secured!")
                return self.evaluate_plan()

            # 1b) Advance one step
            self.step()

            # 1c) Prune any dead enemies so planners never target them
            alive_friendlies = [
                u for u in self.friendly_units
                if u.state.get("current_group_size", 0) > 0
            ]
            if len(alive_friendlies) != len(self.friendly_units):
                self.friendly_units = alive_friendlies
                self.friendly_units_dict = {u.name: u for u in self.friendly_units}

            alive_enemies = [
                e for e in self.enemy_units
                if e.state.get("current_group_size", 0) > 0
            ]
            if len(alive_enemies) != len(self.enemy_units):
                self.enemy_units = alive_enemies
                self.enemy_units_dict = {e.name: e for e in self.enemy_units}

            # 1d) Visualization update
            if self.visualize:
                self.plotter.update()
                plt.pause(0.5)

                while self.plotter.paused:
                    plt.pause(0.2)

        # 2) If we exhaust max_steps without success
        if self.visualize:
            self.plotter.update()
            logger.info("\nMission incomplete after maximum steps.")
        return self.evaluate_plan()
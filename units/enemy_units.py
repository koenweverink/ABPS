from htn_planner import HTNPlanner
from log import logger
from utils import (
    manhattan,
    has_line_of_sight,
    is_in_cover,
    get_effective_vision_range,
    perform_attack,
    compute_retreat_point,
    compute_unit_defend_position,
    compute_bridge_defend_position,
    get_bridges,
    astar,
)
from terrain_utils import sign
from config import GRID_WIDTH


class EnemyUnit:
    """Base class for enemy-controlled units using HTN-based planning and execution."""

    def __init__(self, name, state, domain):
        """Initialize an enemy unit with a given state and planning domain."""
        self.name = name
        self.state = state
        self.planner = HTNPlanner(domain)
        self.current_plan = []
        self.last_position = state["position"]
        self.last_health = state["health"]
        self.last_group_size = state["current_group_size"]
        # Track the last executed task so the planner can
        # conditionally allow certain actions like FaceEnemy
        self.state["last_task"] = None

    def initialize_defend_lines(self):
        """Precompute defensive line positions (far, mid, close)."""
        if self.state.get("defend_lines"):
            return
        x, y = self.state.get("position", (0, 0))
        lines = [
            (max(0, x - 30), y),
            (max(0, x - 20), y),
            (max(0, x - 10), y),
        ]
        self.state["defend_lines"] = lines
        self.state["current_line"] = 0
        self.state["defend_position"] = lines[0]
        self.state["picked_position"] = True


    def _filter_visible_friendlies(self, friendly_units):
        """Return friendlies visible to this unit."""
        pos = self.state["position"]
        base_range = self.state.get("vision_range", 20)
        visible = []
        for u in friendly_units:
            if u.state.get("health", 0) <= 0:
                continue
            u_pos = u.state["position"]
            los = has_line_of_sight(pos, u_pos)
            dist = manhattan(pos, u_pos)
            eff_range = get_effective_vision_range(
                base_range,
                u.state.get("stealth_modifier", 0),
                is_in_cover(u_pos),
                los,
            )
            if dist <= eff_range and los:
                visible.append(u)
        return visible

    def update_plan(self, friendly_units, force_replan=False, visible=None):
        """Update the current plan using the HTN planner given friendly unit positions."""
        mission = self.state.get("mission", "DefendAreaMission")
        s = self.state.copy()
        s.update({
            "sim": self.sim,
            "spotted_enemies": self.state.get("spotted_enemies", []),
            "unit": self,
            # Provide last executed task so domain logic can
            # make context-sensitive decisions (e.g. FaceEnemy)
            "last_task": self.state.get("last_task"),
        })

        # Determine visible friendlies before evaluating domain conditions so
        # that condition_can_see_friendly has access to this information.
        if visible is None:
            visible = self._filter_visible_friendlies(friendly_units)
        s["friendly_units"] = visible
        s["target_position"] = (
            min(
                visible,
                key=lambda u: manhattan(self.state["position"], u.state["position"]),
            ).state["position"]
            if visible
            else self.state["position"]
        )

        valid = [
            (cond, subs)
            for cond, subs in self.planner.domain[mission]
            if cond(s)
        ]
        # logger.info(f"{self.name} valid methods (count={len(valid)}): {valid}")
        if not valid:
            self.current_plan = [("Hold", None)]
            return

        idx = 0  # use highest priority method

        cond, subtasks = valid[idx]
        plan = subtasks(s) if callable(subtasks) else list(subtasks)
        self.current_plan = plan or [("Hold", None)]
        # logger.info(f"{self.name} updated plan: {self.current_plan}")

        self.state["_last_valid_count"] = len(valid)
        self.last_health = self.state["health"]

    def execute_next_task(self):
        """Execute the next task in the current plan."""
        self.state["is_attacking"] = False
        if self.state["health"] <= 0 or not self.state["enemy_alive"]:
            return

        # If suppression reduces accuracy to zero, fall back until recovered
        base_acc = self.state.get("accuracy", 0.0)
        suppression = self.state.get("suppression_from_enemy", 0.0)
        effective_acc = base_acc - suppression
        if effective_acc <= 0 or self.state.get("accuracy_recovering", False):
            if suppression <= 0:
                self.state["accuracy_recovering"] = False
                self.state.pop("accuracy_retreat_point", None)
            else:
                self.state["accuracy_recovering"] = True
                self._execute_accuracy_retreat()
                self.state["last_task"] = "Retreat"
            return

        # Maintain battle position bonuses every turn while stationary
        if self.state.get("in_battle_position", False):
            first = (
                self.current_plan[0]
                if self.current_plan
                else None
            )
            if not (
                first == "BattlePosition"
                or (
                    isinstance(first, tuple) and first[0] == "BattlePosition"
                )
            ):
                self._execute_battle_position()

        visible = self._filter_visible_friendlies(self.sim.friendly_units)

        # If holding position but an enemy is visible, face and attack manually
        if (
            self.state.get("in_battle_position", False)
            and self.current_plan
            and self.current_plan[0] == "Hold"
            and visible
        ):
            nearest = min(
                visible,
                key=lambda u: manhattan(self.state["position"], u.state["position"]),
            )
            dx = nearest.state["position"][0] - self.state["position"][0]
            dy = nearest.state["position"][1] - self.state["position"][1]
            if dx or dy:
                self.state["facing"] = (sign(dx), sign(dy))
            if self.can_attack(nearest):
                self.state["is_attacking"] = True
                self._resolve_attack(nearest)
            return

        if (
            visible
            and not self.state.get("delay_retreating", False)
            and not (
                self.current_plan and self.current_plan[0] in ("Retreat", "Delay")
            )
        ):
            nearest = min(
                visible,
                key=lambda u: manhattan(self.state["position"], u.state["position"]),
            )
            if self.can_attack(nearest):
                self.state["is_attacking"] = True
                self._resolve_attack(nearest)
                self.state["last_task"] = "AttackEnemy"
                return

        if visible:
            self.update_plan(
                self.sim.friendly_units, force_replan=True, visible=visible
            )

        for u in self.sim.friendly_units:
            if (
                u.state["health"] > 0
                and manhattan(self.state["position"], u.state["position"]) <= self.state["attack_range"]
                and has_line_of_sight(self.state["position"], u.state["position"])
            ):
                visible = visible + [u] if u not in visible else visible
                self.update_plan(
                    self.sim.friendly_units, force_replan=True, visible=visible
                )
                break

        if not self.current_plan:
            return

        task, arg = self.current_plan[0] if isinstance(self.current_plan[0], tuple) else (self.current_plan[0], None)

        old_pos = self.state["position"]

        if task == "Move":
            self._execute_move()
        elif task == "MoveToPosition":
            self._execute_move_to_position()
        elif task == "PickPosition":
            self._execute_pick_position()
        elif task == "FaceEnemy":
            self._execute_face_enemy(arg)
        elif task == "AttackEnemy":
            self._execute_attack()
        elif task == "Retreat":
            self._execute_retreat()
        elif task == "Delay":
            self._execute_delay()
        elif task == "BattlePosition":
            self._execute_battle_position()

        if self.state["position"] != old_pos and self.state.get("in_battle_position", False):
            self.state.update({"turns_in_battle_position": 0, "hasty_done": False, "entrenched_done": False, "in_battle_position": False})

        # Record the task we just attempted for context in future planning
        self.state["last_task"] = task

    def _execute_move(self):
        """Perform movement toward the nearest friendly unit."""
        self.state["move_credit"] += self.state["speed"]
        steps = int(self.state["move_credit"])
        self.state["move_credit"] -= steps
        # logger.info(f"{self.name} taking {steps} steps")
        for _ in range(steps):
            old_pos = self.state["position"]
            target = min(
                self.sim.friendly_units,
                key=lambda u: manhattan(old_pos, u.state["position"]),
            ).state["position"]

            full_path = astar(
                old_pos,
                target,
                enemy_units=self.sim.friendly_units,
                unit=self.state["type"],
            )

            if full_path and len(full_path) >= 2:
                next_pos = full_path[1]
                if not hasattr(self.sim, "debug_paths"):
                    self.sim.debug_paths = []
                self.sim.debug_paths.append({"unit": self, "path": full_path})
            else:
                next_pos = old_pos

            self.state["position"] = next_pos
            if manhattan(self.state["position"], target) <= self.state["attack_range"]:
                if self.current_plan:
                    self.current_plan.pop(0)
            dx = self.state["position"][0] - old_pos[0]
            dy = self.state["position"][1] - old_pos[1]
            if dx or dy:
                self.state["facing"] = (sign(dx), sign(dy))

    def _execute_attack(self):
        """Find a valid attack target and resolve the attack."""
        target_unit = None
        min_dist = float("inf")
        pos = self.state["position"]
        rng = self.state["attack_range"]
        for u in self.sim.friendly_units:
            if u.state.get("health", 0) <= 0:
                continue
            u_pos = u.state["position"]
            d = manhattan(pos, u_pos)
            if d <= rng and has_line_of_sight(pos, u_pos):
                if d < min_dist:
                    target_unit, min_dist = u, d
        if target_unit:
            self.state["is_attacking"] = True
            self._resolve_attack(target_unit)
        self.current_plan.pop(0)

    def _resolve_attack(self, target):
        """Run the actual damage model against a chosen target."""
        perform_attack(self, target)

    def _execute_retreat(self):
        """Move toward a designated fallback point."""
        if "retreat_point" not in self.state:
            self.state["retreat_point"] = compute_retreat_point(self.sim)
        target = self.state["retreat_point"]

        self.state["move_credit"] += self.state["speed"]
        steps = int(self.state["move_credit"])
        self.state["move_credit"] -= steps
        for _ in range(steps):
            old_pos = self.state["position"]
            if old_pos == target:
                break

            full_path = astar(
                old_pos,
                target,
                enemy_units=self.sim.friendly_units,
                unit=self.state["type"],
            )

            if not full_path or len(full_path) < 2:
                # Path blocked or unreachable: hold here
                self.state["defend_position"] = old_pos
                if self.current_plan:
                    self.current_plan.pop(0)
                self._execute_battle_position()
                return

            next_pos = full_path[1]
            if not hasattr(self.sim, "debug_paths"):
                self.sim.debug_paths = []
            self.sim.debug_paths.append({"unit": self, "path": full_path})

            self.state["position"] = next_pos
            dx = self.state["position"][0] - old_pos[0]
            dy = self.state["position"][1] - old_pos[1]
            if dx or dy:
                self.state["facing"] = (sign(dx), sign(dy))
        if self.state["position"] == target:
            self.state["has_retreated"] = True
            self.state["defend_position"] = target
            self.state["picked_position"] = True
            if self.current_plan:
                self.current_plan.pop(0)
            # Immediately adopt battle position and face nearest enemy
            self._execute_battle_position()
            visible = self._filter_visible_friendlies(self.sim.friendly_units)
            if visible:
                nearest = min(visible, key=lambda u: manhattan(self.state["position"], u.state["position"]))
                dx = nearest.state["position"][0] - self.state["position"][0]
                dy = nearest.state["position"][1] - self.state["position"][1]
                if dx or dy:
                    self.state["facing"] = (sign(dx), sign(dy))

    def _execute_accuracy_retreat(self):
        """Step back to regroup when accuracy is fully suppressed."""
        if "accuracy_retreat_point" not in self.state:
            self.state["accuracy_retreat_point"] = compute_retreat_point(self.sim, max_distance=20)
        target = self.state["accuracy_retreat_point"]



    def _execute_delay(self):
        """Assign the unit to guard bridges and retreat on low health."""
        self.state.pop("force_delay", None)
        self.state["delay_retreating"] = True

        health_ratio = self.state.get("health", 0) / max(1, self.state.get("max_health", 1))

        if "delay_bridge_order" not in self.state:
            bridges = get_bridges()
            num_bridges = len(bridges)
            bridge_idx = self.sim.next_delay_bridge_index % num_bridges
            self.sim.next_delay_bridge_index += 1
            order = list(range(bridge_idx, len(bridges)))
            self.state.update(
                {
                    "delay_bridges": bridges,
                    "delay_bridge_order": order,
                    "delay_order_pos": 0,
                    "delay_stage": 0,
                    "current_delay_bridge": bridge_idx,
                    "delay_health_thresholds": [0.75, 0.5],
                }
            )
            self.state["defend_position"] = compute_bridge_defend_position(
                self.sim, self, bridges[bridge_idx]
            )
            self.state["picked_position"] = True
        else:
            bridges = self.state["delay_bridges"]
            order = self.state["delay_bridge_order"]
            pos = self.state["delay_order_pos"]
            thresholds = self.state.get("delay_health_thresholds", [0.75, 0.5])
            stage = self.state.get("delay_stage", 0)
            if stage < len(thresholds) and health_ratio < thresholds[stage] and pos < len(order) - 1:
                pos += 1
                self.state["delay_stage"] = stage + 1

            self.state["delay_order_pos"] = pos
            new_bridge = order[pos]
            self.state["current_delay_bridge"] = new_bridge
            self.state["defend_position"] = compute_bridge_defend_position(
                self.sim, self, bridges[new_bridge]
            )
            self.state["picked_position"] = True

        if self.current_plan:
            self.current_plan.pop(0)

    def _execute_battle_position(self):
        """Switch to a defensive posture and apply armor bonuses over time."""
        self.state["in_battle_position"] = True
        turns = self.state.get("turns_in_battle_position", 0) + 1
        self.state["turns_in_battle_position"] = turns

        if turns == 1 and not self.state.get("hasty_done", False):
            front, flank = (2, 3) if is_in_cover(self.state["position"]) else (1, 2)
            self._apply_armor_bonus(front, flank)
            self.state["hasty_done"] = True
            # logger.info(f"{self.name} hasty position: +{front} front, +{flank} side/rear")

        elif turns == 5 and not self.state.get("entrenched_done", False):
            self._apply_armor_bonus(2, 8)
            self.state["entrenched_done"] = True
            # logger.info(f"{self.name} entrenched position: +2 front, +8 side/rear")

    def _apply_armor_bonus(self, front, flank):
        """Apply bonus armor to front, side, and rear based on posture."""
        self.state["armor_front"] = self.state["base_armor_front"] + front
        self.state["armor_side"] = self.state["base_armor_side"] + flank
        self.state["armor_rear"] = self.state["base_armor_rear"] + flank

    def _execute_pick_position(self):
        """Select a defensive position if not already chosen."""
        if "defend_position" not in self.state:
            self.state["defend_position"] = compute_unit_defend_position(self.sim, self)
        self.state["picked_position"] = True
        if self.current_plan:
            self.current_plan.pop(0)

    def _execute_move_to_position(self):
        """Move toward the preset defend_position."""
        target = self.state.get("defend_position", self.state["position"])
        self.state["move_credit"] += self.state["speed"]
        steps = int(self.state["move_credit"])
        self.state["move_credit"] -= steps
        for _ in range(steps):
            old_pos = self.state["position"]
            if old_pos == target:
                if self.current_plan:
                    self.current_plan.pop(0)
                if self.current_plan:
                    next_task = self.current_plan[0]
                    if next_task == "BattlePosition" or (
                        isinstance(next_task, tuple) and next_task[0] == "BattlePosition"
                    ):
                        self.current_plan.pop(0)
                self.state["delay_retreating"] = False
                self._execute_battle_position()
                break

            full_path = astar(
                old_pos,
                target,
                enemy_units=self.sim.friendly_units,
                unit=self.state["type"],
            )

            if not full_path or len(full_path) < 2:
                self.state["defend_position"] = old_pos
                if self.current_plan:
                    self.current_plan.pop(0)
                self.state["delay_retreating"] = False
                self._execute_battle_position()
                return

            next_pos = full_path[1]
            if not hasattr(self.sim, "debug_paths"):
                self.sim.debug_paths = []
            self.sim.debug_paths.append({"unit": self, "path": full_path})

            self.state["position"] = next_pos
            dx = self.state["position"][0] - old_pos[0]
            dy = self.state["position"][1] - old_pos[1]
            if dx or dy:
                self.state["facing"] = (sign(dx), sign(dy))
        else:
            if self.state["position"] == target:
                self.state["delay_retreating"] = False
                self._execute_battle_position()
                return

    def _execute_face_enemy(self, name):
        """Orient the unit toward a specific enemy without moving, then pop if successful."""
        # Only allow facing when the previous task was a stationary action
        # to avoid interrupting behaviors like hit-and-run.
        if self.state.get("last_task") not in ("Hold", "BattlePosition"):
            if self.current_plan and self.current_plan[0][0] == "FaceEnemy":
                self.current_plan.pop(0)
            return
        if name is None:
            visible = self._filter_visible_friendlies(self.sim.friendly_units)
            enemy = min(visible, key=lambda u: manhattan(self.state["position"], u.state["position"])) if visible else None
        else:
            enemy = next((u for u in self.sim.friendly_units if u.name == name), None)

        if enemy:
            dx = enemy.state["position"][0] - self.state["position"][0]
            dy = enemy.state["position"][1] - self.state["position"][1]
            if dx != 0:
                dx = int(dx / abs(dx))
            if dy != 0:
                dy = int(dy / abs(dy))
            self.state["facing"] = (dx, dy)

            # logger.info(f"{self.name}: Now facing {enemy.name} at {enemy.state['position']}")

            # ✅ Only pop after successful facing
            if self.current_plan and self.current_plan[0][0] == "FaceEnemy":
                self.current_plan.pop(0)
        else:
            logger.warning(f"{self.name}: No valid enemy found to face, holding plan.")


    def get_goal_position(self):
        """Determine current target location based on top-level plan."""
        if not self.current_plan:
            return self.state["position"]
        task, _ = self.current_plan[0] if isinstance(self.current_plan[0], tuple) else (self.current_plan[0], None)
        if task in ("Move", "AttackEnemy"):
            last_known = self.sim.enemy_drone.last_known
            return min(last_known.values(), key=lambda p: manhattan(self.state["position"], p)) if last_known else self.state["position"]
        elif task == "Retreat":
            return self.state.get("retreat_point", (9, 9))
        return self.state["position"]

    def needs_update(self):
        """Determine whether the unit needs to replan based on any significant state change."""
        return (
            self.state["position"] != self.last_position
            or abs(self.state["health"] - self.last_health) > 0.1
            or self.state["current_group_size"] != self.last_group_size
            or not self.state["enemy_alive"]
        )


class EnemyTank(EnemyUnit):
    """Tank unit with line-of-sight and range-based attack logic."""
    def can_attack(self, target):
        """Return True if the given target is alive, in LOS and within range."""
        alive = target.state.get("health", 0) > 0
        in_range = manhattan(self.state["position"], target.state["position"]) <= self.state["attack_range"]
        return alive and in_range and has_line_of_sight(self.state["position"], target.state["position"])


class EnemyInfantry(EnemyTank):
    """Infantry with same attack logic as tank."""
    pass


class EnemyAntiTank(EnemyTank):
    """Anti-tank unit with same attack logic as tank."""
    pass


class EnemyArtillery(EnemyUnit):
    """Artillery unit with simplified range-only attack logic."""
    def can_attack(self, target):
        """Artillery only needs range check (no LOS)."""
        return target.state.get("health", 0) > 0 and manhattan(self.state["position"], target.state["position"]) <= self.state["attack_range"]

import copy, random

from htn_planner import HTNPlanner
from log import logger
from utils import (
    manhattan, has_line_of_sight, is_in_cover, get_effective_vision_range,
    next_step, perform_attack
)
from terrain_utils import sign


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

    def _choose_branch(self, valid):
        """Selects and remembers one valid HTN method branch from the available options."""
        valid_count = len(valid)
        if valid_count > 1 and "_branch_choice" not in self.state:
            idx = random.randrange(valid_count)
            self.state["_branch_choice"] = idx
        elif "_branch_choice" in self.state:
            idx = min(self.state["_branch_choice"], valid_count - 1)
            self.state["_branch_choice"] = idx
        else:
            idx = 0
        return idx

    def _filter_visible_friendlies(self, friendly_units):
        """Filter visible enemy units based on vision range, cover, stealth, and LOS."""
        visible = []
        for u in friendly_units:
            if u.state.get("health", 0) <= 0:
                continue
            dist = manhattan(self.state["position"], u.state["position"])
            los = has_line_of_sight(self.state["position"], u.state["position"])
            stealth = u.state.get("stealth_modifier", 0)
            in_cover = is_in_cover(u.state["position"])
            eff_rng = get_effective_vision_range(self.state.get("vision_range", 20), stealth, in_cover, los)
            if dist <= eff_rng and los:
                visible.append(u)
        return visible

    def update_plan(self, friendly_units, force_replan=False):
        """Update the current plan using the HTN planner given friendly unit positions."""
        mission = "DefendAreaMission"
        s = copy.deepcopy(self.state)
        s.update({"sim": self.sim, "spotted_enemies": self.state.get("spotted_enemies", []), "unit": self})

        valid = [(cond, subs) for cond, subs in self.planner.domain[mission] if cond(s)]
        logger.info(f"{self.name} valid methods (count={len(valid)}): {valid}")
        idx = self._choose_branch(valid)
        logger.info(f"{self.name} keeps branch #{idx}")

        visible = self._filter_visible_friendlies(friendly_units)
        s["friendly_units"] = visible
        s["target_position"] = min(visible, key=lambda u: manhattan(self.state["position"], u.state["position"])).state["position"] if visible else self.state["position"]

        cond, subtasks = valid[idx]
        plan = subtasks(s) if callable(subtasks) else list(subtasks)
        self.current_plan = plan or [("Hold", None)]
        logger.info(f"{self.name} updated plan: {self.current_plan}")

        self.state["_last_valid_count"] = len(valid)
        self.last_health = self.state["health"]

    def execute_next_task(self):
        """Execute the next task in the current plan."""
        self.state["is_attacking"] = False
        if self.state["health"] <= 0 or not self.state["enemy_alive"]:
            return

        for u in self.sim.friendly_units:
            if (u.state["health"] > 0 and manhattan(self.state["position"], u.state["position"]) <= self.state["attack_range"]
                    and has_line_of_sight(self.state["position"], u.state["position"])):
                self.update_plan(self.sim.friendly_units, force_replan=True)
                break

        if not self.current_plan:
            return

        task, arg = self.current_plan[0] if isinstance(self.current_plan[0], tuple) else (self.current_plan[0], None)

        if task not in ("BattlePosition", "AttackEnemy") and self.state.get("in_battle_position", False):
            self.state.update({"turns_in_battle_position": 0, "hasty_done": False, "entrenched_done": False, "in_battle_position": False})

        if task == "Move":
            self._execute_move()
        elif task == "AttackEnemy":
            self._execute_attack()
        elif task == "Retreat":
            self._execute_retreat()
        elif task == "BattlePosition":
            self._execute_battle_position()

    def _execute_move(self):
        """Perform movement toward the nearest friendly unit."""
        self.state["move_credit"] += self.state["speed"]
        steps = int(self.state["move_credit"])
        self.state["move_credit"] -= steps
        logger.info(f"{self.name} taking {steps} steps")
        for _ in range(steps):
            old_pos = self.state["position"]
            target = min(self.sim.friendly_units, key=lambda u: manhattan(self.state["position"], u.state["position"])).state["position"]
            self.state["position"] = next_step(self.state["position"], target)
            if manhattan(self.state["position"], target) <= self.state["attack_range"]:
                self.current_plan.pop(0)
            dx, dy = self.state["position"][0] - old_pos[0], self.state["position"][1] - old_pos[1]
            if dx or dy:
                self.state["facing"] = (sign(dx), sign(dy))

    def _execute_attack(self):
        """Find a valid attack target and resolve the attack."""
        target_unit = None
        min_dist = float('inf')
        for u in self.sim.friendly_units:
            if u.state.get("health", 0) <= 0:
                continue
            d = manhattan(self.state["position"], u.state["position"])
            if d <= self.state["attack_range"] and has_line_of_sight(self.state["position"], u.state["position"]):
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
        retreat = self.state.get("retreat_point", (9, 9))
        self.state["position"] = next_step(self.state["position"], retreat)
        if self.state["position"] == retreat:
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
            logger.info(f"{self.name} hasty position: +{front} front, +{flank} side/rear")

        elif turns == 5 and not self.state.get("entrenched_done", False):
            self._apply_armor_bonus(2, 8)
            self.state["entrenched_done"] = True
            logger.info(f"{self.name} entrenched position: +2 front, +8 side/rear")

    def _apply_armor_bonus(self, front, flank):
        """Apply bonus armor to front, side, and rear based on posture."""
        self.state["armor_front"] = self.state["base_armor_front"] + front
        self.state["armor_side"] = self.state["base_armor_side"] + flank
        self.state["armor_rear"] = self.state["base_armor_rear"] + flank

    def get_goal_position(self):
        """Determine current target location based on top-level plan."""
        if not self.current_plan:
            return self.state["position"]
        task, _ = self.current_plan[0] if isinstance(self.current_plan[0], tuple) else (self.current_plan[0], None)
        if task == "Patrol":
            return self.state["patrol_points"][self.state["current_patrol_index"]]
        elif task in ("Move", "AttackEnemy"):
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
        return target.state.get("enemy_alive", False) and has_line_of_sight(self.state["position"], target.state["position"]) and manhattan(self.state["position"], target.state["position"]) <= self.state["attack_range"]


class EnemyInfantry(EnemyTank):
    """Infantry with same attack logic as tank."""
    pass


class EnemyAntiTank(EnemyTank):
    """Anti-tank unit with same attack logic as tank."""
    pass


class EnemyArtillery(EnemyUnit):
    """Artillery unit with simplified range-only attack logic."""
    def can_attack(self, target):
        return target.state.get("enemy_alive", False) and manhattan(self.state["position"], target.state["position"]) <= self.state["attack_range"]

import copy
import math
import random

from htn_planner import HTNPlanner
from utils import (
    manhattan, has_line_of_sight, under_friendly_drone_cover, next_step,
    get_num_attacks, get_penetration_probability, perform_attack,
    all_units_at_position, compute_staging_position
)
from terrain_utils import sign
from log import logger


class FriendlyUnit:
    """Base class for friendly units using HTN-based planning and reactive combat logic."""

    def __init__(self, name, state, domain, simulation=None):
        """Initialize a friendly unit with planning state and simulation context."""
        self.name = name
        self.state = state
        self.planner = HTNPlanner(domain)
        self.current_plan = []
        self.sim = simulation
        self.last_position = state["position"]
        self.last_health = state["health"]
        self.last_group_size = state["current_group_size"]

    def update_plan(self, force_replan=False):
        """Regenerate or preserve the current plan depending on context or threats."""
        mission = "SecureOutpostMission"
        if not force_replan and self.current_plan:
            first = self.current_plan[0]
            first_task = first[0] if isinstance(first, tuple) else first
            if first_task not in ("MoveToStaging", "WaitForGroup"):
                for enemy_name, enemy in self.sim.enemy_units_dict.items():
                    if self.can_attack(enemy):
                        if self.current_plan[0][:2] != ("AttackEnemy", enemy_name):
                            force_replan = True
                            logger.info(
                                f"{self.name} forcing replan due to attackable enemy {enemy_name}"
                            )
                            break

        if force_replan or not self.current_plan:
            combined = copy.deepcopy(self.state)
            combined.update({
                "sim": self.sim,
                "unit": self,
                "enemy_units_dict": self.sim.enemy_units_dict,
                "spotted_enemies": [
                    name for name in self.state.get("spotted_enemies", [])
                    if name in self.sim.enemy_units_dict and self.sim.enemy_units_dict[name].state.get("enemy_alive", False)
                ]
            })
            self.current_plan = self.planner.plan(mission, combined) or [("Hold", None)]
            logger.info(f"{self.name} replanned: {self.current_plan}")
        else:
            logger.info(f"{self.name} current plan: {self.current_plan}")
        self.last_health = self.state["health"]

    def execute_next_task(self):
        """Execute the next action in the HTN plan."""
        logger.info(f"{self.name} is facing {self.state['facing']}")
        if not self.current_plan or self.state["health"] <= 0:
            logger.info(f"{self.name} cannot execute task: plan empty or health <= 0")
            return

        task_name, task_arg = self.current_plan[0] if isinstance(self.current_plan[0], tuple) else (self.current_plan[0], None)
        logger.info(f"{self.name} executing task: {task_name}, arg: {task_arg}")

        if task_name == "Move":
            self._execute_move(task_arg)
        elif task_name == "AttackEnemy":
            self._execute_attack(task_arg)
        elif task_name == "SecureOutpostNoArg":
            self._execute_secure_outpost()
        elif task_name == "MoveToStaging":
            self._execute_move_to_staging()
        elif task_name == "WaitForGroup":
            self._execute_wait_for_group()
        elif task_name == "Hold":
            logger.info(f"{self.name} holds position at {self.state['position']}")
            self.current_plan.pop(0)

    def _execute_move(self, target_arg):
        """Advance toward goal or enemy, adjusting for proximity and LOS to target."""
        self.state["move_credit"] += self.state["speed"]
        steps = int(self.state["move_credit"])
        self.state["move_credit"] -= steps
        logger.info(f"{self.name} taking {steps} steps")

        for _ in range(steps):
            old_pos = self.state["position"]
            goal = self.get_goal_position(("Move", target_arg))

            if self._is_about_to_attack(target_arg):
                target = self._get_enemy_by_name(target_arg)
                if target and target.state["enemy_alive"]:
                    dist = manhattan(self.state["position"], target.state["position"])
                    has_los = has_line_of_sight(self.state["position"], target.state["position"])
                    if dist <= self.state["attack_range"] and has_los:
                        logger.info(f"{self.name} within attack range of {target.name}; stopping move.")
                        self.current_plan.pop(0)
                        self.update_plan(force_replan=True)
                        return

            if self.state["position"] == goal:
                logger.info(f"{self.name} reached goal {goal}; stopping move.")
                self.current_plan.pop(0)
                break

            self.state["position"] = next_step(self.state["position"], goal, self.sim.enemy_units, unit=self.state["type"])
            dx, dy = self.state["position"][0] - old_pos[0], self.state["position"][1] - old_pos[1]
            if dx or dy:
                self.state["facing"] = (sign(dx), sign(dy))
            logger.info(f"{self.name} moved to {self.state['position']}")

    def _execute_attack(self, enemy_name):
        """Attempt to attack the specified enemy unit if possible."""
        target = self._get_enemy_by_name(enemy_name)
        if not target or not target.state["enemy_alive"]:
            logger.info(f"{self.name} cannot attack; target {enemy_name} is dead or missing.")
            self.current_plan.pop(0)
            self.update_plan(force_replan=True)
            return

        distance = manhattan(self.state["position"], target.state["position"])
        los = has_line_of_sight(self.state["position"], target.state["position"])
        if distance > self.state["attack_range"] or not los:
            logger.info(f"{self.name} cannot attack {target.name}; out of range or no LOS.")
            self.current_plan.pop(0)
            self.update_plan(force_replan=True)
            return

        perform_attack(self, target)
        if not target.state["enemy_alive"]:
            self.current_plan.pop(0)
            for unit in self.sim.friendly_units:
                unit.update_plan(force_replan=True)

    def _execute_secure_outpost(self):
        """Secure the outpost if at correct position."""
        if self.state["position"] == self.state["outpost_position"]:
            self.state["outpost_secured"] = True
            logger.info(f"{self.name} secures the outpost!")
        else:
            logger.info(f"{self.name} cannot secure outpost; not at target location.")
        self.current_plan.pop(0)
        self.current_plan.append(("Hold", None))

    def _execute_move_to_staging(self):
        """Move toward the designated staging position."""
        self.state["move_credit"] += self.state["speed"]
        steps = int(self.state["move_credit"])
        self.state["move_credit"] -= steps
        logger.info(f"{self.name} moving to staging area with {steps} steps")

        if "staging_position" not in self.state:
            if self.sim.staging_position is None:
                self.sim.staging_position = compute_staging_position(self.sim)
            self.state["staging_position"] = self.sim.staging_position
        else:
            if self.sim.staging_position is None:
                self.sim.staging_position = self.state["staging_position"]
        staging = self.sim.staging_position
        for _ in range(steps):
            old_pos = self.state["position"]
            if self.state["position"] == staging:
                logger.info(f"{self.name} reached staging area {staging}")
                self.current_plan.pop(0)
                break
            self.state["position"] = next_step(self.state["position"], staging, self.sim.enemy_units, unit=self.state["type"])
            dx, dy = self.state["position"][0] - old_pos[0], self.state["position"][1] - old_pos[1]
            if dx or dy:
                self.state["facing"] = (sign(dx), sign(dy))
            logger.info(f"{self.name} moved to {self.state['position']}")

    def _execute_wait_for_group(self):
        """Hold until all friendly units reach the staging area."""
        if "staging_position" not in self.state:
            if self.sim.staging_position is None:
                self.sim.staging_position = compute_staging_position(self.sim)
            self.state["staging_position"] = self.sim.staging_position
        staging = self.sim.staging_position
        all_ready = all_units_at_position(self.sim.friendly_units, staging)
        if all_ready:
            logger.info(f"{self.name} group assembled, proceeding")
            self.current_plan.pop(0)
            self.sim.staging_position = None
        else:
            logger.info(f"{self.name} waiting at staging for group")

    def _get_enemy_by_name(self, name):
        """Return enemy unit by name from the simulation context."""
        return next((e for e in self.sim.enemy_units if e.name == name), None)

    def _is_about_to_attack(self, target_arg):
        """Check if the next task is an attack on the specified target."""
        return (
            len(self.current_plan) > 1
            and isinstance(self.current_plan[1], tuple)
            and self.current_plan[1][0] == "AttackEnemy"
            and target_arg == self.current_plan[1][1]
        )

    def get_goal_position(self, task=None):
        """Determine movement goal based on current task and drone intelligence."""
        task_name, task_arg = task if isinstance(task, tuple) else (task, None)
        if task_name == "Move" and task_arg == "outpost":
            return self.state["outpost_position"]
        if task_name == "MoveToStaging":
            if self.sim.staging_position is None:
                self.sim.staging_position = compute_staging_position(self.sim)
            self.state["staging_position"] = self.sim.staging_position
            return self.sim.staging_position
        if task_name in ("Move", "AttackEnemy") and task_arg:
            return self.sim.friendly_drone.last_known.get(task_arg, self.state["position"])
        return self.state["position"]

    def needs_update(self):
        """Determine whether the unit has changed enough to require replanning."""
        return (
            self.state["position"] != self.last_position
            or abs(self.state["health"] - self.last_health) > 0.1
            or self.state["current_group_size"] != self.last_group_size
            or self.state["health"] <= 0
        )


class FriendlyTank(FriendlyUnit):
    """Tank unit with line-of-sight and range-based attack logic."""
    def can_attack(self, target):
        return target.state.get("enemy_alive", False) and has_line_of_sight(self.state["position"], target.state["position"]) and manhattan(self.state["position"], target.state["position"]) <= self.state["attack_range"]


class FriendlyInfantry(FriendlyTank):
    """Infantry unit reusing attack logic from tank."""
    pass


class FriendlyAntiTank(FriendlyTank):
    """Anti-tank unit reusing attack logic from tank."""
    pass


class FriendlyScout(FriendlyUnit):
    """Scout unit with custom behavior (to be defined)."""
    pass


class FriendlyArtillery(FriendlyUnit):
    """Artillery unit with support-enabled attack capability."""
    def can_attack(self, target):
        if not target.state.get("enemy_alive", False):
            return False

        distance = manhattan(self.state["position"], target.state["position"])
        if distance > self.state["attack_range"]:
            return False

        if has_line_of_sight(self.state["position"], target.state["position"]):
            return True

        for unit in self.sim.friendly_units:
            if unit is not self and has_line_of_sight(unit.state["position"], target.state["position"]):
                return True

        return under_friendly_drone_cover(self.sim, target.state["position"])

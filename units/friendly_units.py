import copy
import math
import random

from htn_planner import HTNPlanner
from utils import (
    manhattan, has_line_of_sight, under_friendly_drone_cover, next_step,
    get_num_attacks, get_penetration_probability, perform_attack,
    all_units_at_position, compute_retreat_point, compute_staging_position,
    compute_flanking_position, get_all_enemy_attack_zones, astar
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
        # Track tasks for Gantt chart generation
        self.task_log = []
        self._current_task = None
        self._task_start_step = None

    def update_plan(self, force_replan=False):
        """Update the HTN plan for this friendly unit."""

        mission = self.state.get("mission", "SecureOutpostMission")

        # Check if replan is actually needed
        if not force_replan and self.current_plan:
            first = self.current_plan[0]

            # Prioritize enemies that are actively attacking this unit
            if isinstance(first, tuple) and first[0] in ("Move", "AttackEnemy") and first[1] != "outpost":
                attackers = [
                    self.sim.enemy_units_dict[n]
                    for n in self.state.get("attacked_by", [])
                    if n in self.sim.enemy_units_dict
                    and self.sim.enemy_units_dict[n].state.get("enemy_alive", False)
                    and n in self.state.get("visible_enemies", [])
                ]
                if attackers:
                    closest = min(
                        attackers,
                        key=lambda e: manhattan(self.state["position"], e.state["position"]),
                    )
                    if self.state.get("override_target") != closest.name:
                        if self.can_attack(closest):
                            new_step = ("AttackEnemy", closest.name)
                        else:
                            new_step = ("Move", closest.name)
                        # logger.info(
                        #     f"{self.name}: switching to attacker {closest.name}"
                        # )
                        self.current_plan.insert(0, new_step)
                        self.state["targeting_enemy"] = closest.name
                        self.state["override_target"] = closest.name
                        first = self.current_plan[0]
                elif self.state.get("override_target"):
                    if first[1] == self.state["override_target"]:
                        # logger.info(
                        #     f"{self.name}: resuming target sequence after attack from {self.state['override_target']}"
                        # )
                        self.current_plan.pop(0)
                        first = self.current_plan[0] if self.current_plan else None
                    self.state.pop("override_target", None)

            if isinstance(first, tuple) and first[0] == "AttackEnemy":
                enemy_name = first[1]
                enemy = self.sim.enemy_units_dict.get(enemy_name)
                if enemy and self.can_attack(enemy):
                    # Only broadcast replan if this is a new engagement
                    if self.state.get("enemy_engaged") != enemy_name:
                        # logger.info(f"{self.name}: engaging {enemy_name}, broadcasting replan to allies")
                        self.state["enemy_engaged"] = enemy_name
                        for u in self.sim.friendly_units:
                            if u is not self and u.state.get("enemy_engaged") != enemy_name:
                                u.update_plan(force_replan=True)
                    # Do not overwrite plan on every step
                    return

        # Only replan if explicitly forced or no current plan
        if force_replan or not self.current_plan or self.current_plan == ["Hold"]:
            combined = {
                "unit": self,
                "sim": self.sim,
                "spotted_enemies": list(self.state.get("spotted_enemies", [])),
                "outpost_position": self.state.get("outpost_position"),
                "position": self.state.get("position"),
                "enemy_alive": {name: enemy.state.get("enemy_alive", False) for name, enemy in self.sim.enemy_units_dict.items()},
            }

            new_plan = self.planner.plan(mission, combined)

            if new_plan:
                self.current_plan = new_plan
                # logger.info(f"{self.name} replanned: {self.current_plan}")
                
                # Update targeting_enemy to match the first attack in the new plan
                for step in new_plan:
                    if isinstance(step, tuple) and step[0] == "AttackEnemy":
                        self.state["targeting_enemy"] = step[1]
                        break

            else:
                self.current_plan = [("Hold", None)]
                # logger.warning(f"{self.name}: no valid plan, defaulting to Hold")


    def execute_next_task(self):
        """Execute the next action in the HTN plan."""
        # logger.info(f"{self.name} is facing {self.state['facing']}")
        if not self.current_plan or self.state["health"] <= 0:
            if self._current_task is not None:
                self.task_log.append((self._current_task, self._task_start_step, self.sim.step_count))
                self._current_task = None
            # logger.info(f"{self.name} cannot execute task: plan empty or health <= 0")
            return

        base_acc = self.state.get("friendly_accuracy", 0.0)
        suppression = self.state.get("suppression_from_enemy", 0.0)
        effective_acc = base_acc - suppression
        if effective_acc <= 0 or self.state.get("accuracy_recovering", False):
            if suppression <= 0:
                self.state["accuracy_recovering"] = False
                self.state.pop("accuracy_retreat_point", None)
            else:
                self.state["accuracy_recovering"] = True
                self._execute_accuracy_retreat()
            return

        task_name, task_arg = (
            self.current_plan[0]
            if isinstance(self.current_plan[0], tuple)
            else (self.current_plan[0], None)
        )
        # Build a label that includes the task's target for certain actions
        task_label = task_name
        if task_name in {"Move", "MoveToStaging", "AttackEnemy"} and task_arg is not None:
            task_label = f"{task_name}({task_arg})"

        step = self.sim.step_count
        if task_label != self._current_task:
            if self._current_task is not None:
                self.task_log.append((self._current_task, self._task_start_step, step))
            self._current_task = task_label
            self._task_start_step = step
        # # logger_info...
        # logger.warning(f"{self.name} is about to run {task_name} with staging_type={self.state.get('staging_type')}, staging={self.state.get('staging_position')}, flank={self.state.get('flank_position')}")


        if task_name == "Move":
            self._execute_move(task_arg)
        elif task_name == "AttackEnemy":
            self._execute_attack(task_arg)
        elif task_name == "SecureOutpostNoArg":
            self._execute_secure_outpost()
        elif task_name == "MoveToStaging":
            self._execute_move_to_staging(task_arg)
        elif task_name == "MoveToFlank":
            self._execute_move_to_flank(task_arg)
        elif task_name == "WaitForGroup":
            self._execute_wait_for_group()
        elif task_name == "Hold":
            # logger.info(f"{self.name} holds position at {self.state['position']}")
            self.current_plan.pop(0)

    def _execute_move(self, target_name):
        """
        Move toward the named target, which may be an enemy unit or the 'outpost'.
        """
        # 0) Resolve the target position
        if target_name == "outpost":
            target_pos = self.state.get("outpost_position")
            if target_pos is None:
                # logger.info(f"{self.name} has no outpost_position; dropping Move('outpost')")
                self.current_plan.pop(0)
                return
            enemy = None
        else:
            enemy = self.sim.enemy_units_dict.get(target_name)
            if enemy is None or not enemy.state.get("enemy_alive", False):
                # logger.info(f"{self.name} cannot Move→{target_name}: target gone; dropping step")
                self.current_plan.pop(0)
                self.update_plan(force_replan=True)
                return
            # 1) If in range & LOS for an enemy, switch to AttackEnemy
            if self.can_attack(enemy):
                # logger.info(f"{self.name} is now in range/LOS of {target_name}, switching to AttackEnemy")
                self.current_plan[0] = ("AttackEnemy", target_name)
                return
            target_pos = enemy.state["position"]

        # 2) Move‐credit and stepping
        self.state["move_credit"] += self.state["speed"]
        steps = int(self.state["move_credit"])
        self.state["move_credit"] -= steps
        # logger.info(f"{self.name} taking {steps} steps toward {target_name}")

        for _ in range(steps):
            old = self.state["position"]
            if old == target_pos:
                # logger.info(f"{self.name} reached {target_name} at {target_pos}")
                self.current_plan.pop(0)
                break

            # Compute full path for debugging and movement
            full_path = astar(
                old,
                target_pos,
                enemy_units=self.sim.enemy_units,
                unit=self.state["type"],
            )

            if full_path and len(full_path) >= 2:
                # Take the next step along the path
                next_pos = full_path[1]

                # Store path for visualization without clobbering others
                if not hasattr(self.sim, "debug_paths"):
                    self.sim.debug_paths = []
                self.sim.debug_paths.append({"unit": self, "path": full_path})
            else:
                # No valid path found; remain in place
                next_pos = old

            self.state["position"] = next_pos
            dx, dy = next_pos[0] - old[0], next_pos[1] - old[1]
            if dx or dy:
                self.state["facing"] = (sign(dx), sign(dy))
            # logger.info(f"{self.name} moved to {next_pos}")


    def _execute_accuracy_retreat(self):
        """Step back to regroup when accuracy is fully suppressed."""
        if "accuracy_retreat_point" not in self.state:
            self.state["accuracy_retreat_point"] = compute_retreat_point(
                self.sim, max_distance=20, retreating_side="friendly"
            )
        target = self.state["accuracy_retreat_point"]

        self.state["move_credit"] += self.state["speed"]
        steps = int(self.state["move_credit"])
        self.state["move_credit"] -= steps
        # logger.info(f"{self.name} falling back to {target} with {steps} steps")

        for _ in range(steps):
            old = self.state["position"]
            if old == target:
                break
            nxt = next_step(old, target, self.sim.enemy_units, unit=self.state["type"])
            if nxt == old:
                break
            self.state["position"] = nxt
            dx, dy = nxt[0] - old[0], nxt[1] - old[1]
            if dx or dy:
                self.state["facing"] = (sign(dx), sign(dy))


    def _execute_attack(self, enemy_name):
        """Attempt to attack the specified enemy unit; fall back to Move if needed."""
        target = self._get_enemy_by_name(enemy_name)
        # 1) Dead or missing?  Pop and replan globally (to pick a new target/mission).
        if not target or not target.state.get("enemy_alive", False):
            # logger.info(f"{self.name} cannot attack; target {enemy_name} is dead or missing.")
            self.current_plan.pop(0)
            self.update_plan(force_replan=True)
            return

        # 2) Out of range or no LOS?  Swap this AttackEnemy into a Move step only.
        if not self.can_attack(target):
            # logger.info(f"{self.name} cannot attack {target.name} yet; switching to Move.")
            # Replace the pending AttackEnemy with Move toward the same enemy
            self.current_plan[0] = ("Move", enemy_name)
            return

        # 3) Otherwise, we’re good—perform the shot.
        perform_attack(self, target)

        # 4) If we killed it, pop the attack and force everyone to replan
        if not target.state.get("enemy_alive", False):
            self.current_plan.pop(0)
            # Clear shared staging cache so next phase recomputes new
            # locations for all units.
            if hasattr(self.sim, "group_staging_cache"):
                self.sim.group_staging_cache.clear()
            if hasattr(self.sim, "group_arrival_flags"):
                self.sim.group_arrival_flags.clear()
            for u in self.sim.friendly_units:
                u.update_plan(force_replan=True)


    def _execute_move_to_staging(self, enemy_name=None):
        """Move step-by-step to staging, recomputing path each step to adapt to enemy changes."""
        self.state.pop("all_arrived_flag", None)  # clear sync flag
        self.state["move_credit"] += self.state["speed"]
        steps = int(self.state["move_credit"])
        self.state["move_credit"] -= steps

        # 🔍 Ensure enemy_name is updated based on current plan if not passed
        if enemy_name is None:
            for task in self.current_plan:
                if isinstance(task, tuple) and task[0] in ("MoveToStaging", "MoveToFlank", "AttackEnemy"):
                    enemy_name = task[1]
                    break

        # Load phase-specific info for this enemy
        attack_seq = getattr(self.sim, "attack_sequence", [])
        current_phase_index = None
        for idx, (e_name, a_type, units) in enumerate(attack_seq):
            if e_name == enemy_name and self.name in units:
                current_phase_index = idx
                break
        group_key = (enemy_name, current_phase_index)
        phase_map = self.state.get("phase_attack_info", {})
        info = phase_map.get(group_key)
        if info:
            self.state["staging_position"] = info.get("staging")
            self.state["flank_position"] = info.get("flank")
            self.state["staging_type"] = info.get("staging_type")

        # logger.info(f"{self.name} moving to staging area with {steps} steps")

        staging = self.state.get("staging_position")
        staging_type = self.state.get("staging_type", "front")

        if staging is None:
            # logger.warning(f"{self.name} has no staging position — skipping task")
            self.current_plan = self.current_plan[1:]  # skip this task
            return

        # logger.warning(f"{self.name} executing STAGING with goal {staging} ({staging_type})")

        for _ in range(steps):
            old_pos = self.state["position"]
            if old_pos == staging:
                # logger.info(f"{self.name} reached staging area {staging}")
                self.current_plan.pop(0)
                break

            # --- Recompute fresh path from current pos to staging each step ---
            avoid_positions = get_all_enemy_attack_zones(self.sim, enemy_name)

            timer = self.state.get("staging_recalc_timer", 0) - 1
            if timer <= 0:
                front_unit = None
                assigned_units = []
                for idx, (e_name, a_type, units) in enumerate(attack_seq):
                    if e_name == enemy_name and self.name in units:
                        assigned_units = units
                        break
                if assigned_units:
                    candidates = [
                        self.sim.friendly_units_dict.get(n)
                        for n in assigned_units
                        if n in self.sim.friendly_units_dict
                    ]
                    if candidates:
                        front_unit = max(candidates, key=lambda u: u.state.get("armor_front", 0))

                new_staging = compute_staging_position(self.sim, enemy_name, front_unit or self, self)
                if new_staging:
                    staging = new_staging
                    self.state["staging_position"] = staging
                    if info is not None:
                        info["staging"] = staging
                timer = 15

            self.state["staging_recalc_timer"] = timer

            full_path = astar(
                old_pos,
                staging,
                enemy_units=self.sim.enemy_units,
                unit=self.state["type"],
                avoid_positions=avoid_positions
            )

            if not full_path or len(full_path) < 2:
                # logger.warning(f"{self.name}: no valid path to staging from {old_pos}; holding position")
                self.current_plan = self.current_plan[1:]  # skip this task
                break

            # Next step in path
            next_pos = full_path[1]
            self.state["position"] = next_pos

            # Store current live path for visualization, preserving paths from other units
            if not hasattr(self.sim, "debug_paths"):
                self.sim.debug_paths = []
            self.sim.debug_paths.append({"unit": self, "path": full_path})

            # Update facing
            dx = next_pos[0] - old_pos[0]
            dy = next_pos[1] - old_pos[1]
            if dx or dy:
                self.state["facing"] = (sign(dx), sign(dy))

            # logger.info(f"{self.name} moved to {self.state['position']}")


    def _execute_move_to_flank(self, enemy_name=None):
        """Move step-by-step to flank, recomputing path each step to adapt to enemy changes."""
        self.state.pop("all_arrived_flag", None)  # clear sync flag
        self.state["move_credit"] += self.state["speed"]
        steps = int(self.state["move_credit"])
        self.state["move_credit"] -= steps

        # logger.info(f"{self.name} moving to flank area with {steps} steps")

        if enemy_name is None:
            for task in self.current_plan:
                if isinstance(task, tuple) and task[0] in ("MoveToFlank", "AttackEnemy", "MoveToStaging"):
                    enemy_name = task[1]
                    break

        attack_seq = getattr(self.sim, "attack_sequence", [])
        current_phase_index = None
        assigned_units = []
        for idx, (e_name, _, units) in enumerate(attack_seq):
            if e_name == enemy_name and self.name in units:
                current_phase_index = idx
                assigned_units = units
                break
        group_key = (enemy_name, current_phase_index)
        phase_map = self.state.get("phase_attack_info", {})
        info = phase_map.get(group_key)
        if info:
            self.state["flank_position"] = info.get("flank")
            self.state["staging_position"] = info.get("staging")
            self.state["staging_type"] = info.get("staging_type")
        
        flank = self.state.get("flank_position")
        staging_type = self.state.get("staging_type", "flank")

        if flank is None:
            # logger.warning(f"{self.name} has no flank position — skipping task")
            self.current_plan = self.current_plan[1:]  # skip this task
            return

        # logger.warning(f"{self.name} executing FLANK with goal {flank} ({staging_type})")

        for _ in range(steps):
            old_pos = self.state["position"]

            if old_pos == flank:
                # logger.info(f"{self.name} reached flank area {flank}")
                self.current_plan.pop(0)
                break

            # --- Recompute fresh path from current pos to flank each step ---
            avoid_positions = get_all_enemy_attack_zones(self.sim, enemy_name)

            timer = self.state.get("flank_recalc_timer", 0) - 1
            if timer <= 0:
                front_unit = None
                if assigned_units:
                    candidates = [
                        self.sim.friendly_units_dict.get(n)
                        for n in assigned_units
                        if n in self.sim.friendly_units_dict
                    ]
                    if candidates:
                        front_unit = max(candidates, key=lambda u: u.state.get("armor_front", 0))

                new_flank, new_stage = compute_flanking_position(self.sim, enemy_name, front_unit or self, self)
                if new_flank:
                    flank = new_flank
                    self.state["flank_position"] = new_flank
                    self.state["staging_position"] = new_stage
                    if info is not None:
                        info.update({"flank": new_flank, "staging": new_stage})
                timer = 15

            self.state["flank_recalc_timer"] = timer
            full_path = astar(
                old_pos,
                flank,
                enemy_units=self.sim.enemy_units,
                unit=self.state["type"],
                avoid_positions=avoid_positions
            )

            if not full_path or len(full_path) < 2:
                # logger.warning(f"{self.name}: no valid path to flank from {old_pos}; holding position")
                self.current_plan = self.current_plan[1:]  # skip this task
                break

            # Next step in path
            next_pos = full_path[1]
            self.state["position"] = next_pos

            # Store current live path for visualization, preserving paths from all units
            if not hasattr(self.sim, "debug_paths"):
                self.sim.debug_paths = []
            self.sim.debug_paths.append({"unit": self, "path": full_path})

            # Update facing
            dx = next_pos[0] - old_pos[0]
            dy = next_pos[1] - old_pos[1]
            if dx or dy:
                self.state["facing"] = (sign(dx), sign(dy))

            # logger.info(f"{self.name} moved to {self.state['position']}")


    def _execute_wait_for_group(self):
        """
        Wait at staging until all assigned units for this enemy phase have arrived.
        Proceed to next task only when entire group is ready.
        """
        # Derive enemy_name from the next task in current_plan
        enemy_name = None
        for task in self.current_plan:
            if isinstance(task, tuple) and task[0] == "AttackEnemy":
                enemy_name = task[1]
                break

        if not enemy_name:
            # logger.warning(f"{self.name}: No AttackEnemy task found in plan — cannot WaitForGroup.")
            self.current_plan = ["Hold"]
            return

        staging_type = self.state.get("staging_type")
        staging_position = self.state.get("staging_position")

        # logger.info(f"{self.name}: WaitForGroup DEBUG — enemy_name: {enemy_name}, staging: {staging_position}, type: {staging_type}")

        if not enemy_name or staging_position is None:
            # logger.warning(f"{self.name}: Missing staging info — cannot WaitForGroup.")
            self.current_plan = ["Hold"]
            return

        attack_seq = getattr(self.sim, "attack_sequence", [])

        # Identify current phase index and assigned units
        current_phase_index = None
        assigned_units = []
        for idx, (e_name, _, units) in enumerate(attack_seq):
            if e_name == enemy_name and self.name in units:
                current_phase_index = idx
                assigned_units = units
                break

        group_key = (enemy_name, current_phase_index)

        # Load phase specific info
        phase_map = self.state.get("phase_attack_info", {})
        phase_info = phase_map.get(group_key)
        if phase_info:
            self.state["staging_position"] = phase_info.get("staging")
            self.state["flank_position"] = phase_info.get("flank")
            self.state["staging_type"] = phase_info.get("staging_type")
        arrival_flags = getattr(self.sim, "group_arrival_flags", {})
        if arrival_flags.get(group_key) or self.state.get("all_arrived_flag"):
            # logger.info(f"{self.name}: WaitForGroup sync flag detected for {group_key}")
            self.current_plan.pop(0)
            return

        if current_phase_index is None or not assigned_units:
            # logger.warning(f"{self.name}: No assigned units found for {enemy_name}, skipping WaitForGroup.")
            self.current_plan.pop(0)
            return

        # logger.info(f"{self.name}: Assigned group for {enemy_name} phase {current_phase_index}: {assigned_units}")

        # If any earlier phase shares units with this phase and its target is
        # still alive, we must wait until that phase finishes before attacking.
        for idx in range(current_phase_index):
            e_name, _, earlier_units = attack_seq[idx]
            if any(u in assigned_units for u in earlier_units):
                enemy_unit = self.sim.enemy_units_dict.get(e_name)
                if enemy_unit and enemy_unit.state.get("enemy_alive", False):
                    # logger.info(
                    #     f"{self.name}: Waiting for earlier phase {idx} ({e_name}) to complete"
                    # )
                    return

        # Check if all assigned teammates are in position and ready
        all_ready = True
        for ally_name in assigned_units:
            ally = self.sim.friendly_units_dict.get(ally_name)
            if not ally or not ally.state.get("enemy_alive", True):
                continue

            ally_target_pos = (
                ally.state.get("staging_position")
                if ally.state.get("staging_type") == "front"
                else ally.state.get("flank_position")
            )

            # logger.info(f"{self.name}: Checking ally {ally_name} — pos: {ally.state.get('position')}, target: {ally_target_pos}, type: {ally.state.get('staging_type')}")

            # Check if ally is at the correct position
            if ally_target_pos is None or ally.state.get("position") != ally_target_pos:
                # logger.info(f"{self.name}: Ally {ally_name} not at target position {ally_target_pos}")
                all_ready = False
                break

            # Check if ally is busy with an earlier phase (skip self)
            if ally_name != self.name:
                for idx in range(current_phase_index):
                    e_name, _, earlier_units = attack_seq[idx]
                    if ally_name in earlier_units:
                        if (
                            ally.state.get("targeting_enemy") == e_name
                            and ally.current_plan
                            and ally.current_plan[0] != "Hold"
                        ):
                            # logger.info(
                            #     f"{self.name}: Blocking — teammate {ally_name} still busy in earlier phase {idx}"
                            # )
                            all_ready = False
                            break

        if all_ready:
            # logger.info(f"{self.name}: All group members ready at staging — proceeding to next task.")
            # Mark phase as synchronized so other units don't keep waiting
            if not hasattr(self.sim, "group_arrival_flags"):
                self.sim.group_arrival_flags = {}
            self.sim.group_arrival_flags[group_key] = True
            for ally_name in assigned_units:
                ally = self.sim.friendly_units_dict.get(ally_name)
                if ally:
                    ally.state["all_arrived_flag"] = True
            self.current_plan.pop(0)
        else:
            logger.info(f"{self.name}: Waiting at {self.state['position']} for group to finish arriving at {staging_position}")


    def _execute_secure_outpost(self):
        """Mark the outpost as secured and finish the task."""
        self.state["outpost_secured"] = True
        # logger.info(f"{self.name} has secured the outpost at {self.state['position']}")
        if self.current_plan:
            self.current_plan.pop(0)

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

    def get_goal_position(self):
        if not self.current_plan:
            return self.state["position"]

        task = self.current_plan[0]
        task_name = task if isinstance(task, str) else task[0]
        task_arg = task[1] if isinstance(task, tuple) and len(task) > 1 else None

        if task_name == "MoveToStaging":
            if "staging_position" not in self.state and task_arg:
                self.state["staging_position"] = compute_staging_position(
                    self.sim, task_arg, self, self
                )
            return self.state.get("staging_position", self.state["position"])

        elif task_name == "MoveToFlank":
            if ("flank_position" not in self.state or "staging_position" not in self.state) and task_arg:
                flank_pos, staging_pos = compute_flanking_position(
                    self.sim, task_arg, self, self
                )
                self.state["flank_position"] = flank_pos
                self.state["staging_position"] = staging_pos
            return self.state.get("flank_position", self.state["position"])

        elif task_name == "Move" and task_arg == "outpost":
            return self.state.get("outpost_position", self.state["position"])

        elif task_name in ("Move", "AttackEnemy") and task_arg:
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

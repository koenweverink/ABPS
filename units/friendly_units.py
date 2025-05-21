import copy, math, random

from htn_planner import HTNPlanner
from utils import manhattan, has_line_of_sight, under_friendly_drone_cover, next_step, get_num_attacks, get_penetration_probability
from terrain_utils import sign
from log import logger

class FriendlyUnit:
    def __init__(self, name, state, domain, simulation=None):
        self.name = name
        self.state = state
        self.planner = HTNPlanner(domain)
        self.current_plan = []
        self.last_enemy_pos = state["outpost_position"]
        self.last_health = state["health"]
        self.last_position = state["position"]
        self.last_group_size = state["current_group_size"]
        self.sim = simulation

    def update_plan(self, force_replan=False):
        mission = "SecureOutpostMission"
        # Check if we can attack any visible enemy
        attack_possible = False
        if not force_replan and self.current_plan:
            for enemy_name, enemy in self.sim.enemy_units_dict.items():
                if self.can_attack(enemy):
                    # If current plan doesn't start with AttackEnemy for this enemy, force replan
                    if not self.current_plan or self.current_plan[0][0] != "AttackEnemy" or self.current_plan[0][1] != enemy_name:
                        force_replan = True
                        logger.info(f"{self.name} forcing replan due to attackable enemy {enemy_name}")
                        break

        if force_replan or not self.current_plan:
            combined = copy.deepcopy(self.state)
            combined["sim"] = self.sim
            combined["spotted_enemies"] = [
                name for name in self.state.get("spotted_enemies", [])
                if name in self.sim.enemy_units_dict
                and self.sim.enemy_units_dict[name].state.get("enemy_alive", False)
            ]
            combined["unit"] = self
            combined["enemy_units_dict"] = self.sim.enemy_units_dict
            new_plan = self.planner.plan(mission, combined)
            if not new_plan:
                new_plan = [("Hold", None)]
            self.current_plan = new_plan
            logger.info(f"{self.name} replanned: {self.current_plan}")
        else:
            logger.info(f"{self.name} current plan: {self.current_plan}")
        self.last_health = self.state["health"]

    def execute_next_task(self):
        logger.info(f"{self.name} is facing {self.state['facing']}")
        if not self.current_plan or self.state["health"] <= 0:
            logger.info(f"{self.name} cannot execute task: plan empty or health <= 0")
            return
        task = self.current_plan[0]
        task_name = task[0] if isinstance(task, tuple) else task
        task_arg = task[1] if isinstance(task, tuple) else None
        logger.info(f"{self.name} executing task: {task}")

        if task_name == "Move":
            self.state["move_credit"] += self.state["speed"]
            steps = int(self.state["move_credit"])
            logger.info(f"{self.name} taking {steps} steps")
            self.state["move_credit"] -= steps

            for _ in range(steps):
                old_pos = self.state["position"]
                goal = self.get_goal_position(task)
                if (len(self.current_plan) > 1 and
                    isinstance(self.current_plan[1], tuple) and
                    self.current_plan[1][0] == "AttackEnemy" and
                    task_arg == self.current_plan[1][1]):
                    target_unit = None
                    for e in self.sim.enemy_units:
                        if e.state.get("name") == task_arg:
                            target_unit = e
                            break
                    if target_unit and target_unit.state["enemy_alive"]:
                        distance = manhattan(self.state["position"], target_unit.state["position"])
                        has_los = has_line_of_sight(self.state["position"], target_unit.state["position"])
                        logger.info(f"{self.name} to {task_arg}: distance={distance}, attack_range={self.state['attack_range']}, has_los={has_los}")
                        if distance <= self.state["attack_range"] and has_los:
                            logger.info(f"{self.name} within attack range of {task_arg} at distance {distance}; stopping move.")
                            self.current_plan.pop(0)
                            self.update_plan(force_replan=True)
                            return
                if self.state["position"] == goal:
                    logger.info(f"{self.name} reached goal {goal}; stopping move.")
                    self.current_plan.pop(0)
                else:
                    self.state["position"] = next_step(
                        self.state["position"],
                        goal,
                        self.sim.enemy_units,
                        unit=self.state["type"]
                    )
                    logger.info(f"{self.name} moves toward {goal}, new position: {self.state['position']}")
                new_pos = self.state["position"]
                dx, dy = new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]
                if dx or dy:
                    self.state["facing"] = (sign(dx), sign(dy))

        elif task_name == "AttackEnemy":
            target_unit = None
            for e in self.sim.enemy_units:
                if e.state.get("name") == task_arg:
                    target_unit = e
                    break
            if not target_unit or not target_unit.state["enemy_alive"]:
                logger.info(f"{self.name} cannot attack; target {task_arg} is dead or invalid.")
                self.current_plan.pop(0)
                self.update_plan(force_replan=True)
                return
            tx, ty = target_unit.state["position"]
            x, y = self.state["position"]
            distance = manhattan(self.state["position"], target_unit.state["position"])
            has_los = has_line_of_sight(self.state["position"], target_unit.state["position"])
            if distance > self.state["attack_range"] or not has_los:
                logger.info(f"{self.name} cannot attack {target_unit.name}; out of range or no LOS.")
                self.current_plan.pop(0)
                self.update_plan(force_replan=True)
                return
            dx, dy = tx - x, ty - y
            norm = math.hypot(dx, dy)
            if norm > 0:
                fx, fy = dx / norm, dy / norm
                self.state["facing"] = (fx, fy)
                logger.info(f"{self.name} is now facing ({fx:.2f}, {fy:.2f})")
            else:
                # if you somehow are exactly on top of them, keep previous facing
                fx, fy = self.state.get("facing", (0,1))

            rate_of_fire = self.state["base_rate_of_fire"] * self.state["current_group_size"]
            num_attacks = get_num_attacks(rate_of_fire)
            effective_accuracy = max(0, self.state["friendly_accuracy"] - self.state["suppression_from_enemy"])

            # Initialize counters for detailed logging
            hits = 0
            penetrations = 0
            total_damage_dealt = 0.0

            logger.info(f"{self.name} (group size: {self.state['current_group_size']}) attacks {target_unit.name} "
                       f"(group size: {target_unit.state['current_group_size']}) {num_attacks} times, "
                       f"accuracy: {effective_accuracy:.2f}")

            for _ in range(num_attacks):
                if random.random() < effective_accuracy:
                    hits += 1
                    target_unit.state["suppression_from_enemy"] += self.state["suppression"]
                    dx = tx - x
                    dy = ty - y
                    norm = math.sqrt(dx**2 + dy**2)
                    dx, dy = (dx / norm, dy / norm) if norm > 0 else (0, 0)
                    self.state["facing"] = (dx, dy)
                    attack_dir = (dx, dy)
                    target_facing = target_unit.state.get("facing", (0, 1))
                    target_norm = math.sqrt(target_facing[0]**2 + target_facing[1]**2)
                    target_fx, target_fy = (target_facing[0] / target_norm, target_facing[1] / target_norm) if target_norm > 0 else (0, 0)
                    dot_product = attack_dir[0] * target_fx + attack_dir[1] * target_fy
                    dot_product = max(min(dot_product, 1), -1)
                    angle_deg = math.degrees(math.acos(dot_product))
                    direction = "rear" if angle_deg <= 45 else "side" if angle_deg <= 135 else "front"
                    logger.info(f"{self.name} attacking {target_unit.name} from the {direction}, suppression: {self.state['suppression']}")

                    armor_val = target_unit.state[f"armor_{direction}"]
                    logger.info(f"{self.name} attacking {target_unit.name} with armor {armor_val}")
                    D = self.state["penetration"] - armor_val
                    penetration_prob = get_penetration_probability(D)
                    if random.random() < penetration_prob:
                        penetrations += 1
                        total_damage = self.state["damage"] * self.state["current_group_size"]
                        target_unit.state["health"] -= total_damage
                        target_unit.state["cumulative_damage"] += total_damage
                        total_damage_dealt += total_damage

                        units_lost = int(target_unit.state["cumulative_damage"] // target_unit.state["base_health"])
                        if units_lost > 0:
                            target_unit.state["current_group_size"] = max(0, target_unit.state["current_group_size"] - units_lost)
                            target_unit.state["cumulative_damage"] -= units_lost * target_unit.state["base_health"]
                            logger.info(f"{target_unit.name} lost {units_lost} units, new group size: {target_unit.state['current_group_size']}")

                        logger.info(f"{self.name} penetrates {target_unit.name}, D={D:.1f}, "
                                   f"penetration prob={penetration_prob:.2f}, dealt {total_damage:.1f}, "
                                   f"health now {target_unit.state['health']:.1f}")
                        if target_unit.state["health"] <= 0 or target_unit.state["current_group_size"] <= 0:
                            target_unit.state["enemy_alive"] = False
                            target_unit.state["current_group_size"] = 0
                            target_unit.state["health"] = 0
                            target_unit.state["cumulative_damage"] = 0
                            logger.info(f"{self.name} destroyed {target_unit.name}!")
                            self.current_plan.pop(0)
                            for unit in self.sim.friendly_units:
                                unit.update_plan(force_replan=True)
                            logger.info(f"{self.name} attack summary: {hits}/{num_attacks} hits, "
                                       f"{penetrations}/{hits} penetrations, total damage dealt: {total_damage_dealt:.1f}")
                            return
            logger.info(f"{self.name} attack summary: {hits}/{num_attacks} hits, "
                       f"{penetrations}/{hits if hits > 0 else 1} penetrations, total damage dealt: {total_damage_dealt:.1f}")
            if target_unit.state["enemy_alive"]:
                logger.info(f"{self.name} continues attacking {target_unit.name}, health remaining: {target_unit.state['health']:.1f}")
            else:
                self.current_plan.pop(0)

        elif task_name == "SecureOutpostNoArg":
            if self.state["position"] == self.state["outpost_position"]:
                self.state["outpost_secured"] = True
                logger.info(f"{self.name} secures the outpost!")
                self.current_plan.pop(0)
                self.current_plan.append(("Hold", None))  # Fallback to Hold
            else:
                logger.info(f"{self.name} cannot secure outpost; not at target location.")
                self.current_plan.pop(0)
                self.current_plan.append(("Hold", None))  # Fallback to Hold

        elif task_name == "Hold":
            logger.info(f"{self.name} holds position at {self.state['position']}.")
            self.current_plan.pop(0)
            return

    def get_goal_position(self, task=None):
        if not task:
            task = self.current_plan[0] if self.current_plan else ("Hold", None)
        task_name = task[0] if isinstance(task, tuple) else task
        task_arg = task[1] if isinstance(task, tuple) else None

        if task_name == "Move" and task_arg == "outpost":
            return self.state["outpost_position"]

        # 2) For Move or AttackEnemy toward a specific enemy:
        if task_name in ("Move", "AttackEnemy") and task_arg:
            # a) try the drone's memory first
            last = self.sim.friendly_drone.last_known.get(task_arg)
            if last is not None:
                return last
        return self.state["position"]

    def needs_update(self):
        return (self.state["position"] != self.last_position or
                abs(self.state["health"] - self.last_health) > 0.1 or
                self.state["current_group_size"] != self.last_group_size or
                self.state["health"] <= 0)

class FriendlyTank(FriendlyUnit):
    def __init__(self, name, state, domain, simulation=None):
        super().__init__(name, state, domain, simulation)

    def can_attack(self, target):
        if target.state.get("enemy_alive", False):
            distance = manhattan(self.state["position"], target.state["position"])
            has_los = has_line_of_sight(self.state["position"], target.state["position"])
            return distance <= self.state["attack_range"] and has_los
        return False

class FriendlyInfantry(FriendlyUnit):
    def __init__(self, name, state, domain, simulation=None):
        super().__init__(name, state, domain, simulation)

    def can_attack(self, target):
        if target.state.get("enemy_alive", False):
            distance = manhattan(self.state["position"], target.state["position"])
            has_los = has_line_of_sight(self.state["position"], target.state["position"])
            return distance <= self.state["attack_range"] and has_los
        return False

class FriendlyArtillery(FriendlyUnit):
    def __init__(self, name, state, domain, simulation=None):
        super().__init__(name, state, domain, simulation)

    def can_attack(self, target):
        if not target.state.get("enemy_alive", False):
            return False
        
        distance = manhattan(self.state["position"], target.state["position"])
        if distance > self.state["attack_range"]:
            return False
        
        has_los = has_line_of_sight(self.state["position"], target.state["position"])
        if has_los:
            logger.info(f"{self.name} has direct LOS to {target.name}")
            return True
        
        # No direct LOS, check other friendly units' LOS
        for unit in self.sim.friendly_units:
            if unit is not self and has_line_of_sight(unit.state["position"], target.state["position"]):
                logger.info(f"{self.name} can attack {target.name} due to LOS from {unit.name}")
                return True
        
        # Check drone coverage
        if under_friendly_drone_cover(self.sim, target.state["position"]):
            logger.info(f"{self.name} can attack {target.name} due to friendly drone coverage")
            return True
        
        logger.info(f"{self.name} cannot attack {target.name}: no LOS, no friendly LOS, no drone coverage")
        return False

class FriendlyScout(FriendlyUnit):
    def __init__(self, name, state, domain, simulation=None):
        super().__init__(name, state, domain, simulation)

class FriendlyAntiTank(FriendlyUnit):
    def __init__(self, name, state, domain, simulation=None):
        super().__init__(name, state, domain, simulation)

    def can_attack(self, target):
        if target.state.get("enemy_alive", False):
            distance = manhattan(self.state["position"], target.state["position"])
            has_los = has_line_of_sight(self.state["position"], target.state["position"])
            return distance <= self.state["attack_range"] and has_los
        return False

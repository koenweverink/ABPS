import copy, random, math

from htn_planner import HTNPlanner
from log import logger
from utils import manhattan, has_line_of_sight, is_in_cover, get_effective_vision_range, next_step, get_num_attacks, get_penetration_probability
from terrain_utils import sign

class EnemyUnit:
    def __init__(self, name, state, domain):
        self.name = name
        self.state = state
        self.planner = HTNPlanner(domain)
        self.current_plan = []
        self.last_position = state["position"]
        self.last_health = state["health"]
        self.last_group_size = state["current_group_size"]

    def update_plan(self, friendly_units, force_replan=False):
        mission = "DefendAreaMission"

        # 1) Build the HTN state skeleton
        s = copy.deepcopy(self.state)
        s["sim"]            = self.sim
        s["spotted_enemies"] = self.state.get("spotted_enemies", [])
        s["unit"]           = self

        # 2) Gather all valid branches
        methods     = self.planner.domain[mission]
        valid       = [(cond, subs) for cond, subs in methods if cond(s)]
        valid_count = len(valid)
        old_count   = self.state.get("_last_valid_count", 0)
        logger.info(f"{self.name} valid methods (count={valid_count}): {valid}")

        # 4) Only *lock in* a branch the first time we actually have >1 valid choices
        if valid_count > 1 and "_branch_choice" not in self.state:
            # first multi‐branch moment: pick and lock
            idx = random.randrange(valid_count)
            self.state["_branch_choice"] = idx
            logger.info(f"{self.name} picked branch #{idx}")
        elif "_branch_choice" in self.state:
            # we already locked a branch—keep using it
            idx = self.state["_branch_choice"]
            # clamp if the number of branches has shrunk
            if idx >= valid_count:
                idx = valid_count - 1
                self.state["_branch_choice"] = idx
            logger.info(f"{self.name} keeps branch #{idx}")
        else:
            # still only one choice, no lock yet
            idx = 0
            logger.info(f"{self.name} only has branch #0")

        # 5) Filter to only truly visible friendlies
        visible = []
        for u in friendly_units:
            if u.state.get("health", 0) <= 0:
                continue
            dist     = manhattan(self.state["position"], u.state["position"])
            los      = has_line_of_sight(self.state["position"], u.state["position"])
            stealth  = u.state.get("stealth_modifier", 0)
            in_cover = is_in_cover(u.state["position"])
            eff_rng  = get_effective_vision_range(
                self.state.get("vision_range", 20),
                stealth, in_cover, los
            )
            if dist <= eff_rng and los:
                visible.append(u)
        s["friendly_units"] = visible
        if visible:
            tgt = min(visible, key=lambda u: manhattan(self.state["position"], u.state["position"]))
            s["target_position"] = tgt.state["position"]
        else:
            s["target_position"] = self.state["position"]

        # 6) Materialize the chosen branch
        cond, subtasks = valid[idx]
        plan = subtasks(s) if callable(subtasks) else list(subtasks)

        # 7) Never leave the plan empty
        self.current_plan = plan or [("Hold", None)]
        logger.info(f"{self.name} updated plan: {self.current_plan}")

        # store for next tick
        self.state["_last_valid_count"] = valid_count
        self.last_health = self.state["health"]

    def execute_next_task(self):
        self.state["is_attacking"] = False
        friendly_units = self.sim.friendly_units
        for u in friendly_units:
            if (u.state["health"]>0
                and manhattan(self.state["position"], u.state["position"]) <= self.state["attack_range"]
                and has_line_of_sight(self.state["position"], u.state["position"])):
                # a friendly is now in sight+range → force a new plan
                self.update_plan(friendly_units, force_replan=True)
                break

        logger.info(f"{self.name} is facing {self.state['facing']}")
        if not self.current_plan or not self.state["enemy_alive"]:
            return
        raw = self.current_plan[0]
        if isinstance(raw, tuple):
            task, arg = raw
        else:
            task, arg = raw, None

        if task not in ("BattlePosition", "AttackEnemy") and self.state.get("in_battle_position", False):
            self.state["turns_in_battle_position"] = 0
            self.state["hasty_done"]      = False
            self.state["entrenched_done"] = False
            self.state["in_battle_position"] = False

        if task == "Move":
            self.state["move_credit"] += self.state["speed"]
            steps = int(self.state["move_credit"])
            logger.info(f"{self.name} taking {steps} steps")
            self.state["move_credit"] -= steps
            for _ in range(steps):
                old_pos = self.state["position"]
                target = min(friendly_units, key=lambda u: manhattan(self.state["position"], u.state["position"])).state["position"]
                self.state["position"] = next_step(self.state["position"], target)
                new_pos = self.state["position"]
                if manhattan(self.state["position"], target) <= self.state["attack_range"]:
                    self.current_plan.pop(0)
                dx, dy = new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]
                if dx or dy:
                    self.state["facing"] = (sign(dx), sign(dy))
        elif task == "AttackEnemy":
            self.state["is_attacking"] = True
            target_unit = None
            min_dist = float('inf')
            for u in friendly_units:
                d = manhattan(self.state["position"], u.state["position"])
                if d <= self.state["attack_range"] and has_line_of_sight(self.state["position"], u.state["position"]):
                    if d < min_dist:
                        min_dist = d
                        target_unit = u
            if target_unit:
                tx, ty = target_unit.state["position"]
                x, y = self.state["position"]
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
                effective_accuracy = max(0, self.state["accuracy"] - self.state["suppression_from_enemy"])
                
                # Initialize counters for detailed logging
                hits = 0
                penetrations = 0
                total_damage_dealt = 0.0

                logger.info(f"{self.name} (group size: {self.state['current_group_size']}) attacks {target_unit.name} "
                           f"(group size: {target_unit.state['current_group_size']}) {num_attacks} times, "
                           f"accuracy: {effective_accuracy:.2f}")

                for _ in range(num_attacks):
                    if random.random() < effective_accuracy:
                        target_unit.state["suppression_from_enemy"] += self.state["suppression"]
                        hits += 1
                        target_pos = target_unit.state["position"]
                        dx = target_pos[0] - self.state["position"][0]
                        dy = target_pos[1] - self.state["position"][1]
                        norm = math.sqrt(dx**2 + dy**2)
                        if norm > 0:
                            dx, dy = dx / norm, dy / norm
                        else:
                            dx, dy = 0, 0
                        self.state["facing"] = dx, dy

                        attack_dir = (dx, dy)
                        target_facing = target_unit.state.get("facing", (0, 1))
                        target_norm = math.sqrt(target_facing[0]**2 + target_facing[1]**2)
                        if target_norm > 0:
                            target_fx, target_fy = target_facing[0] / target_norm, target_facing[1] / target_norm
                        else:
                            target_fx, target_fy = 0, 0

                        dot_product = attack_dir[0] * target_fx + attack_dir[1] * target_fy
                        dot_product = max(min(dot_product, 1), -1)
                        angle_deg = math.degrees(math.acos(dot_product))
                        direction = "rear" if angle_deg <= 45 else "side" if angle_deg <= 135 else "front"

                        logger.info(f"{self.name} attacking {target_unit.name} from the {direction}")

                        arm_val = target_unit.state[f"armor_{direction}"]
                        D = self.state["penetration"] - arm_val
                        penetration_prob = get_penetration_probability(D)
                        if random.random() < penetration_prob:
                            penetrations += 1
                            total_damage = self.state["damage"] * self.state["current_group_size"]
                            target_unit.state["health"] -= total_damage
                            target_unit.state["cumulative_damage"] += total_damage
                            total_damage_dealt += total_damage

                            units_lost = int(target_unit.state["cumulative_damage"] // target_unit.state["base_health"])
                            if units_lost > 0:
                                old_group_size = target_unit.state["current_group_size"]
                                target_unit.state["current_group_size"] = max(0, target_unit.state["current_group_size"] - units_lost)
                                target_unit.state["cumulative_damage"] -= units_lost * target_unit.state["base_health"]
                                logger.info(f"{target_unit.name} lost {units_lost} units, new group size: {target_unit.state['current_group_size']}")

                            logger.info(f"{self.name} penetrated {target_unit.name} with D={D:.1f}, "
                                       f"penetration prob={penetration_prob:.2f}, dealt {total_damage:.1f}, "
                                       f"health now {target_unit.state['health']:.1f}")
                            if target_unit.state["health"] <= 0 or target_unit.state["current_group_size"] <= 0:
                                logger.info(f"{target_unit.name} destroyed by {self.name}")
                                target_unit.state["enemy_alive"] = False
                                target_unit.state["current_group_size"] = 0
                                target_unit.state["health"] = 0
                                target_unit.state["cumulative_damage"] = 0
                logger.info(f"{self.name} attack summary: {hits}/{num_attacks} hits, "
                           f"{penetrations}/{hits} penetrations, total damage dealt: {total_damage_dealt:.1f}")
                self.current_plan.pop(0)
            else:
                self.current_plan.pop(0)
        elif task == "Retreat":
            retreat = self.state.get("retreat_point", (9, 9))
            self.state["position"] = next_step(self.state["position"], retreat)
            if self.state["position"] == retreat:
                self.current_plan.pop(0)
        elif task == "BattlePosition":
            # mark that we’re now in a defensive stance
            self.state["in_battle_position"] = True

            # increment the counter
            turns = self.state.get("turns_in_battle_position", 0) + 1
            self.state["turns_in_battle_position"] = turns

            # on turn 1 → hasty bonus
            if turns == 1 and not self.state.get("hasty_done", False):
                # Hasty: +2 front & +3 side/rear if in cover, else +1 front & +2 side/rear
                if is_in_cover(self.state["position"]):
                    front_delta, flank_delta = 2, 3
                else:
                    front_delta, flank_delta = 1, 2

                # apply additive bonuses from base
                self.state["armor_front"] = self.state["base_armor_front"] + front_delta
                self.state["armor_side"]  = self.state["base_armor_side"]   + flank_delta
                self.state["armor_rear"]  = self.state["base_armor_rear"]   + flank_delta

                self.state["hasty_done"] = True
                logger.info(f"{self.name} hasty position: +{front_delta} front, +{flank_delta} side/rear")

            elif turns == 5 and not self.state.get("entrenched_done", False):
                # Entrenched: +5 front & +8 side/rear flat
                front_delta, flank_delta = 2, 8

                self.state["armor_front"] = self.state["base_armor_front"] + front_delta
                self.state["armor_side"]  = self.state["base_armor_side"]   + flank_delta
                self.state["armor_rear"]  = self.state["base_armor_rear"]   + flank_delta

                self.state["entrenched_done"] = True
                logger.info(f"{self.name} entrenched position: +{front_delta} front, +{flank_delta} side/rear")


    def get_goal_position(self):
        if not self.current_plan:
            return self.state["position"]

        task = self.current_plan[0]
        if isinstance(task, (list, tuple)):
            t, _ = task
        else:
            t, _ = task, None
        if t == "Patrol":
            idx = self.state["current_patrol_index"]
            return self.state["patrol_points"][idx]
        elif t in ["Move", "AttackEnemy"]:
            last_known = self.sim.enemy_drone.last_known  # maps name -> (x,y)
            if last_known:
                # pick the nearest last‐known coordinate
                return min(last_known.values(), key=lambda p: manhattan(self.state["position"], p))
        elif t == "Retreat":
            return self.state.get("retreat_point", (9, 9))
        return self.state["position"]

    def needs_update(self):
        return (self.state["position"] != self.last_position or
                abs(self.state["health"] - self.last_health) > 0.1 or
                self.state["current_group_size"] != self.last_group_size or
                not self.state["enemy_alive"])

class EnemyTank(EnemyUnit):
    def __init__(self, name, state, domain):
        super().__init__(name, state, domain)
    
    def can_attack(self, target):
        if target.state.get("enemy_alive", False):
            distance = manhattan(self.state["position"], target.state["position"])
            has_los = has_line_of_sight(self.state["position"], target.state["position"])
            return distance <= self.state["attack_range"] and has_los
        return False

class EnemyInfantry(EnemyUnit):
    def __init__(self, name, state, domain):
        super().__init__(name, state, domain)

    def can_attack(self, target):
        if target.state.get("enemy_alive", False):
            distance = manhattan(self.state["position"], target.state["position"])
            has_los = has_line_of_sight(self.state["position"], target.state["position"])
            return distance <= self.state["attack_range"] and has_los
        return False

class EnemyAntiTank(EnemyUnit):
    def __init__(self, name, state, domain):
        super().__init__(name, state, domain)
    
    def can_attack(self, target):
        if target.state.get("enemy_alive", False):
            distance = manhattan(self.state["position"], target.state["position"])
            has_los = has_line_of_sight(self.state["position"], target.state["position"])
            return distance <= self.state["attack_range"] and has_los
        return False

class EnemyArtillery(EnemyUnit):
    def __init__(self, name, state, domain):
        super().__init__(name, state, domain)

    def can_attack(self, target):
        if target.state.get("enemy_alive", False):
            distance = manhattan(self.state["position"], target.state["position"])
            return distance <= self.state["attack_range"]
        return False
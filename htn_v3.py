import random
import heapq
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import math
import logging
import time

# Write INFO+ messages to simulation.log, DEBUG goes nowhere by default
logging.basicConfig(
    filename="simulation.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
)
logger = logging.getLogger("Sim")

###############################
# Grid, Obstacles, and LOS
###############################

GRID_WIDTH = 75
GRID_HEIGHT = 50
CELL_SIZE = 100
random.seed(42)

def get_line(start, end):
    x1, y1 = start
    x2, y2 = end
    line = []
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    x, y = x1, y1
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x2:
            line.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            line.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    line.append((x, y))
    return line

river = set()
for x in range(0, GRID_WIDTH):
    for y in range(0, GRID_HEIGHT):
        if (x < 7 and y < 25):
            river.add((x, y))
        elif (7 <= x < 25 and 20 <= y < 25):
            river.add((x, y))
        elif (x >= 50 and 35 <= y < 40):
            river.add((x, y))
        for i in range(0, 5):
            for pos in get_line((25, 20+i), (49, 35+i)):
                river.add(pos)

for x in range(11, 14):
    for y in range(20, 25):
        river.discard((x, y))

for i in range(6):
    for pos in get_line((40, 21 + i), (34, 29 + i)):
        river.discard(pos)

for x in range(61, 64):
    for y in range(35, 40):
        river.discard((x, y))

def init_forest(p, width, height):
    return {
        (x,y)
        for x in range(width)
        for y in range(height)
        if random.random() < p and (x,y) not in river
    }

def count_neighbors(forest, x, y):
    n = 0
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            if dx==0 and dy==0: continue
            if (x+dx, y+dy) in forest:
                n += 1
    return n

def smooth_forest(forest, width, height, survive=4, birth=5):
    new = set()
    for x in range(width):
        for y in range(height):
            n = count_neighbors(forest, x, y)
            if ( (x,y) in forest and n >= survive ) or ( (x,y) not in forest and n >= birth ):
                new.add((x,y))
    return new

forest = init_forest(p=0.45, width=GRID_WIDTH, height=GRID_HEIGHT)
for _ in range(4):
    forest = smooth_forest(forest, GRID_WIDTH, GRID_HEIGHT)

cliff_defs = [((45,12),(55,12),(0, 1))]
cliffs = {}
for s,e,n in cliff_defs:
    for c in get_line(s,e):
        cliffs[c] = n

climb_entries = {
    (cx - nx, cy - ny): (cx, cy)
    for (cx, cy), (nx, ny) in cliffs.items()
}

def in_bounds(pos):
    return 0 <= pos[0] < GRID_WIDTH and 0 <= pos[1] < GRID_HEIGHT

def neighbors(pos):
    results = []
    for p in [(pos[0]-1,pos[1]), (pos[0]+1,pos[1]), (pos[0],pos[1]-1), (pos[0],pos[1]+1)]:
        if not in_bounds(p) or p in river:
            continue
        if pos in cliffs:
            nx, ny = cliffs[pos]
            entry = (pos[0] - nx, pos[1] - ny)
            if p != entry:
                continue
        if p in cliffs:
            if climb_entries.get(pos) != p:
                continue
        results.append(p)
    return results

forest_edge = [pos for pos in forest if any(n not in forest for n in neighbors(pos))]
forest -= set(forest_edge)

def manhattan(p, q):
    return abs(p[0]-q[0]) + abs(p[1]-q[1])

def is_in_enemy_vision(pos, enemy_units):
    for enemy in enemy_units:
        if enemy.state["enemy_alive"]:
            distance = manhattan(pos, enemy.state["position"])
            has_los = has_line_of_sight(pos, enemy.state["position"])
            vision_range = enemy.state.get("vision_range", 20)
            if distance <= vision_range and has_los:
                return True
    return False

def astar(start, goal, enemy_units=None, unit="unknown"):
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for nxt in neighbors(current):
            if unit in ["tank", "artillery", "anti-tank"]:
                if is_in_cover(nxt):
                    continue
            new_cost = cost_so_far[current] + 1
            if unit in ["scout", "infantry"]:
                if not is_in_cover(nxt):
                    new_cost += 5
            else:
                if is_in_cover(nxt):
                    continue
            if enemy_units and is_in_enemy_vision(nxt, enemy_units):
                new_cost += 3
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + manhattan(nxt, goal)
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current
    if goal not in came_from:
        return []
    path = []
    current = goal
    while current:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def next_step(start, goal, enemy_units=None, unit="unknown"):
    path = astar(start, goal, enemy_units, unit)
    return path[1] if len(path) >= 2 else start

def has_line_of_sight(start, end):
    line = get_line(start, end)
    blocks_los = all(pos not in river for pos in line[1:-1])
    if not blocks_los:
        return False
    start_in_cover = start in forest or start in forest_edge
    end_in_cover = end in forest or end in forest_edge
    start_forest_edge = start in forest_edge
    end_forest_edge = end in forest_edge
    los_clear = all(pos not in forest for pos in line[1:-1])
    if start_forest_edge and not (end in forest or end in forest_edge):
        if los_clear:
            return True
        else:
            return False
    if end_forest_edge and not (start in forest or start in forest_edge):
        if los_clear:
            return True
        else:
            return False
    if not los_clear:
        return False
    if start_in_cover or end_in_cover:
        if not (start_forest_edge or end_forest_edge):
            return False
    return True

def is_in_cover(pos):
    return pos in forest or pos in forest_edge

def get_effective_vision_range(base_vision_range, stealth_modifier, in_cover, has_los):
    if not in_cover:
        return base_vision_range
    return base_vision_range / (1 + stealth_modifier / CELL_SIZE)

###############################
# Helper Functions
###############################

def get_num_attacks(rpm):
    exact = rpm * 0.1
    n = int(exact)
    if random.random() < (exact - n):
        n += 1
    return n

def get_penetration_probability(D):
    if D <= -3:
        return 0.0
    elif -3 < D <= 0:
        return 0.33 + 0.11 * (D + 3)
    elif 0 < D <= 6:
        return 0.66 + (0.29/6) * D
    else:
        return 0.95

def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0

###############################
# Global Enemy State Creation
###############################

def create_enemy_state(index=0):
    GROUP_SIZE = 10
    BASE_HEALTH = 20
    state = {
        "name": f"EnemyTankGroup{index+1}",
        "position": (GRID_WIDTH - 1, GRID_HEIGHT - 3 if index == 0 else GRID_HEIGHT // 2),
        "facing": (0, 1),
        "enemy_alive": True,
        "health": BASE_HEALTH * GROUP_SIZE,
        "max_health": BASE_HEALTH * GROUP_SIZE,
        "base_health": BASE_HEALTH,
        "group_size": GROUP_SIZE,
        "current_group_size": GROUP_SIZE,
        "cumulative_damage": 0.0,
        "armor_front": 17,
        "armor_side": 4,
        "armor_rear": 3,
        "outpost_position": (GRID_WIDTH - 1, 0),
        "outpost_secured": False,
        "attack_range": 2400 / CELL_SIZE,
        "accuracy": 0.7,
        "penetration": 18,
        "damage": 9,
        "suppression": 0.12,
        "base_rate_of_fire": 4.9,
        "suppression_from_enemy": 0.0,
        "patrol_points": [
            [(GRID_WIDTH - 1, GRID_HEIGHT - 3), (GRID_WIDTH - 20, GRID_HEIGHT - 3)] if index == 0
            else [(GRID_WIDTH - 1, GRID_HEIGHT // 2), (GRID_WIDTH - 3, GRID_HEIGHT // 2)]
        ][0],
        "current_patrol_index": 0,
        "vision_range": 2000 / CELL_SIZE,
        "retreat_point": (GRID_WIDTH - 1, GRID_HEIGHT - 1),
        "stealth_modifier": 0
    }
    return state

###############################
# HTN Domains and Planners
###############################

secure_outpost_domain = {
    "SecureOutpostMission": [
        (lambda state: True, ["DefeatEnemies", "SecureOutpost"]),
    ],
    "DefeatEnemies": [
        (lambda state: any(e.get("enemy_alive", False) for e in state.get("all_enemies", [])),
         lambda state: [
             task
             for enemy in state.get("all_enemies", [])
             if enemy.get("enemy_alive", False)
             for task in [("Move", enemy["name"]), ("AttackEnemy", enemy["name"])]
         ]),
        (lambda state: not any(e.get("enemy_alive", False) for e in state.get("all_enemies", [])), []),
    ],
    "SecureOutpost": [
        (lambda state: state["position"] != state["outpost_position"], [("Move", "outpost")]),
        (lambda state: state["position"] == state["outpost_position"], ["SecureOutpostNoArg"]),
    ],
    "SecureOutpostNoArg": [
        (lambda state: True, ["Hold"]),
    ],
}

class HTNPlanner:
    def __init__(self, domain):
        self.domain = domain

    def plan(self, task, state):
        task_name = task[0] if isinstance(task, tuple) else task
        task_arg = task[1] if isinstance(task, tuple) else None
        if task_name not in self.domain:
            return [task]
        for condition, subtasks in self.domain[task_name]:
            if condition(state):
                task_list = subtasks(state) if callable(subtasks) else subtasks
                plan = []
                for subtask in task_list:
                    sub_plan = self.plan(subtask, state)
                    if sub_plan is None:
                        return None
                    plan.extend(sub_plan)
                return plan
        return None

###############################
# Refactored Enemy Classes
###############################

class EnemyUnit:
    def __init__(self, name, state, domain):
        self.name = name
        self.state = state
        self.planner = HTNPlanner(domain)
        self.current_plan = []
        self.last_position = state["position"]
        self.last_health = state["health"]
        self.last_group_size = state["current_group_size"]

    def update_plan(self, friendly_units):
        combined_state = self.state.copy()
        visible_units = []
        for unit in friendly_units:
            if unit.state.get("health", 0) > 0:
                distance = manhattan(self.state["position"], unit.state["position"])
                has_los = has_line_of_sight(self.state["position"], unit.state["position"])
                stealth_modifier = unit.state.get("stealth_modifier", 0)
                in_cover = is_in_cover(unit.state["position"])
                effective_vision_range = get_effective_vision_range(
                    self.state.get("vision_range", 20), stealth_modifier, in_cover, has_los)
                if distance <= effective_vision_range and has_los:
                    visible_units.append(unit)
        combined_state["friendly_units"] = visible_units
        if visible_units:
            target = min(visible_units, key=lambda u: manhattan(self.state["position"], u.state["position"]))
            combined_state["target_position"] = target.state["position"]
        else:
            combined_state["target_position"] = self.state["position"]
        mission = "DefendAreaMission"
        new_plan = self.planner.plan(mission, combined_state)
        self.current_plan = new_plan if new_plan else []
        logger.info(f"{self.state['name']} updated plan: {self.current_plan}")

    def execute_next_task(self, friendly_units):
        logger.info(f"{self.name} is facing {self.state['facing']}")
        if not self.current_plan or not self.state["enemy_alive"]:
            return
        task = self.current_plan[0]
        if task == "Patrol":
            old_pos = self.state["position"]
            idx = self.state["current_patrol_index"]
            target = self.state["patrol_points"][idx]
            self.state["position"] = next_step(self.state["position"], target)
            new_pos = self.state["position"]
            if self.state["position"] == target:
                self.state["current_patrol_index"] = (idx + 1) % len(self.state["patrol_points"])
                self.current_plan.pop(0)
            dx, dy = new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]
            if dx or dy:
                self.state["facing"] = (sign(dx), sign(dy))
        elif task == "ChaseTarget":
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
                self.state["facing"] = (sign(dx), sign(dy))

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
                            target_unit.state["suppression_from_enemy"] += self.state["suppression"]

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

    def get_goal_position(self):
        if not self.current_plan:
            return self.state["position"]
        t = self.current_plan[0]
        if t == "Patrol":
            idx = self.state["current_patrol_index"]
            return self.state["patrol_points"][idx]
        elif t in ["ChaseTarget", "AttackEnemy"]:
            if "friendly_units" in self.state and self.state["friendly_units"]:
                return min(self.state["friendly_units"], key=lambda u: manhattan(self.state["position"], u.state["position"])).state["position"]
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

###############################
# Friendly Unit Classes
###############################

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
        if force_replan or not self.current_plan:
            self.state["all_enemies"] = [e.state for e in self.sim.enemy_units]
            new_plan = self.planner.plan(mission, self.state)
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
                    logger.info(f"{self.name} to {task_arg}: distance={distance}, attack_range={self.state['friendly_attack_range']}, has_los={has_los}")
                    if distance <= self.state["friendly_attack_range"] and has_los:
                        logger.info(f"{self.name} within attack range of {task_arg} at distance {distance}; stopping move.")
                        self.current_plan.pop(0)
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
            if distance > self.state["friendly_attack_range"] or not has_los:
                logger.info(f"{self.name} cannot attack {target_unit.name}; out of range or no LOS.")
                self.current_plan.pop(0)
                self.update_plan(force_replan=True)
                return
            dx, dy = tx - x, ty - y
            self.state["facing"] = (sign(dx), sign(dy))

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
                    logger.info(f"{self.name} attacking {target_unit.name} from the {direction}")

                    armor_val = target_unit.state[f"armor_{direction}"]
                    D = self.state["penetration"] - armor_val
                    penetration_prob = get_penetration_probability(D)
                    if random.random() < penetration_prob:
                        penetrations += 1
                        total_damage = self.state["damage"] * self.state["current_group_size"]
                        target_unit.state["health"] -= total_damage
                        target_unit.state["cumulative_damage"] += total_damage
                        total_damage_dealt += total_damage
                        target_unit.state["suppression_from_enemy"] += self.state["suppression"]

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
            self.current_plan.append(("Hold", None))  # Continue holding

    def get_goal_position(self, task=None):
        if not task:
            task = self.current_plan[0] if self.current_plan else ("Hold", None)
        task_name = task[0] if isinstance(task, tuple) else task
        task_arg = task[1] if isinstance(task, tuple) else None
        if task_name == "Move":
            if task_arg == "outpost":
                return self.state["outpost_position"]
            for enemy in self.state.get("all_enemies", []):
                if enemy["name"] == task_arg and enemy.get("enemy_alive", False):
                    return enemy["position"]
        elif task_name == "AttackEnemy":
            for enemy in self.state.get("all_enemies", []):
                if enemy["name"] == task_arg and enemy.get("enemy_alive", False):
                    return enemy["position"]
        return self.state["position"]

    def needs_update(self):
        return (self.state["position"] != self.last_position or
                abs(self.state["health"] - self.last_health) > 0.1 or
                self.state["current_group_size"] != self.last_group_size or
                self.state["health"] <= 0)

class FriendlyTank(FriendlyUnit):
    def __init__(self, name, state, domain, simulation=None):
        super().__init__(name, state, domain, simulation)

class FriendlyInfantry(FriendlyUnit):
    def __init__(self, name, state, domain, simulation=None):
        super().__init__(name, state, domain, simulation)

class FriendlyArtillery(FriendlyUnit):
    def __init__(self, name, state, domain, simulation=None):
        super().__init__(name, state, domain, simulation)

class FriendlyScout(FriendlyUnit):
    def __init__(self, name, state, domain, simulation=None):
        super().__init__(name, state, domain, simulation)

class FriendlyAntiTank(FriendlyUnit):
    def __init__(self, name, state, domain, simulation=None):
        super().__init__(name, state, domain, simulation)

###############################
# TeamCommander and Simulation Classes
###############################

class TeamCommander:
    def __init__(self, friendly_units):
        self.friendly_units = friendly_units

class Simulation:
    def __init__(self, friendly_units, enemy_units, team_commander, visualize=False, plan_name="Unknown Plan"):
        self.friendly_units = friendly_units
        self.enemy_units = enemy_units
        self.team_commander = team_commander
        active_enemy = next((e for e in enemy_units if e.state["enemy_alive"]), None)
        for u in self.friendly_units:
            u.state["enemy"] = active_enemy.state if active_enemy else {}
            u.state["visible_enemies"] = []
            u.state["all_enemies"] = [e.state for e in self.enemy_units]
            u.state["total_enemies"] = len(self.enemy_units)
            u.state["scout_steps"] = 0
            u.sim = self
        self.step_count = 0
        self.visualize = visualize
        self.plan_name = plan_name

        if self.visualize:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.fig.set_size_inches(8, 8)
            self.ax.set_aspect("equal", adjustable="box")
            self.ax.set_xlim(-CELL_SIZE/2, GRID_WIDTH * CELL_SIZE - CELL_SIZE/2)
            self.ax.set_ylim(-CELL_SIZE/2, GRID_HEIGHT * CELL_SIZE - CELL_SIZE/2)
            major_step = 500
            self.ax.set_xticks(np.arange(0, (GRID_WIDTH+1)*CELL_SIZE, major_step))
            self.ax.set_yticks(np.arange(0, (GRID_HEIGHT+1)*CELL_SIZE, major_step))
            self.ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f} m"))
            self.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f} m"))
            self.ax.grid(True)

            # Initialize static map using a grid array
            self.map_grid = np.ones((GRID_HEIGHT, GRID_WIDTH, 3))  # White background
            for x in range(GRID_WIDTH):
                for y in range(GRID_HEIGHT):
                    pos = (x, y)
                    if pos in river:
                        self.map_grid[y, x] = [0, 0, 1]  # Blue
                    elif pos in forest:
                        self.map_grid[y, x] = [0, 0.5, 0]  # Green
                    elif pos in forest_edge:
                        self.map_grid[y, x] = [0.7, 1, 0.7]  # Light green
                    elif pos in cliffs:
                        self.map_grid[y, x] = [0.6, 0.3, 0]  # Brown
                    elif pos in climb_entries:
                        self.map_grid[y, x] = [0, 0.5, 0]  # Forest green

            self.ax.imshow(
                self.map_grid,
                origin='lower',
                extent=(-CELL_SIZE/2, GRID_WIDTH * CELL_SIZE - CELL_SIZE/2,
                        -CELL_SIZE/2, GRID_HEIGHT * CELL_SIZE - CELL_SIZE/2)
            )

            # Draw climb entry arrows
            self.climb_arrows = []
            for entry, cliff_cell in climb_entries.items():
                ex, ey = entry
                cx, cy = cliff_cell
                arrow = self.ax.annotate(
                    '',
                    xy=(cx*CELL_SIZE + CELL_SIZE/2, cy*CELL_SIZE + CELL_SIZE/2),
                    xytext=(ex*CELL_SIZE + CELL_SIZE/2, ey*CELL_SIZE + CELL_SIZE/2),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1),
                    zorder=3
                )
                self.climb_arrows.append(arrow)

            # Draw outpost
            if self.friendly_units:
                outpost = self.friendly_units[0].state["outpost_position"]
                self.outpost_marker, = self.ax.plot(
                    outpost[0] * CELL_SIZE + CELL_SIZE/2,
                    outpost[1] * CELL_SIZE + CELL_SIZE/2,
                    marker='*', markersize=10, color='magenta', label='Outpost', zorder=5
                )

            # Initialize unit plots
            self.enemy_markers = []
            self.enemy_arrows = []
            self.enemy_texts = []
            for enemy in self.enemy_units:
                ex, ey = enemy.state["position"]
                marker, = self.ax.plot(
                    ex * CELL_SIZE + CELL_SIZE/2,
                    ey * CELL_SIZE + CELL_SIZE/2,
                    marker='s', markersize=8, color='green', zorder=5
                )
                self.enemy_markers.append(marker)
                arrow = self.ax.quiver(
                    ex * CELL_SIZE + CELL_SIZE/2,
                    ey * CELL_SIZE + CELL_SIZE/2,
                    0, 0,
                    color='green', edgecolor='black', linewidth=0.5, width=0.008,
                    scale=1, scale_units='xy', angles='xy', zorder=4
                )
                self.enemy_arrows.append(arrow)
                text = self.ax.text(
                    ex * CELL_SIZE + CELL_SIZE/2 + 15,
                    ey * CELL_SIZE + CELL_SIZE/2 + 15,
                    f"{enemy.state['current_group_size']}/{enemy.state['health']:.0f}",
                    fontsize=6, color='black', zorder=6
                )
                self.enemy_texts.append(text)

            self.friendly_markers = []
            self.friendly_arrows = []
            self.friendly_texts = []
            for unit in self.friendly_units:
                pos = unit.state["position"]
                color = {
                    "tank": 'red',
                    "infantry": 'blue',
                    "artillery": 'orange',
                    "scout": 'cyan',
                    "anti-tank": 'purple'
                }.get(unit.state.get("type"), 'gray')
                marker, = self.ax.plot(
                    pos[0] * CELL_SIZE + CELL_SIZE/2,
                    pos[1] * CELL_SIZE + CELL_SIZE/2,
                    marker='o', markersize=8, color=color,
                    label=unit.state.get("type").capitalize(),
                    zorder=5
                )
                self.friendly_markers.append(marker)
                arrow = self.ax.quiver(
                    pos[0] * CELL_SIZE + CELL_SIZE/2,
                    pos[1] * CELL_SIZE + CELL_SIZE/2,
                    0, 0,
                    color=color, edgecolor='black', linewidth=0.5, width=0.008,
                    scale=1, scale_units='xy', angles='xy', zorder=4
                )
                self.friendly_arrows.append(arrow)
                text = self.ax.text(
                    pos[0] * CELL_SIZE + CELL_SIZE/2 + 15,
                    pos[1] * CELL_SIZE + CELL_SIZE/2 + 15,
                    f"{unit.state['current_group_size']}/{unit.state['health']:.0f}",
                    fontsize=6, color='black', zorder=6
                )
                self.friendly_texts.append(text)

            self.ax.legend()
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            plt.pause(0.1)  # Allow canvas to stabilize

    def update_enemy_behavior(self):
        friendly_units = [u for u in self.friendly_units if u.state.get("health", 0) > 0]
        for enemy in self.enemy_units:
            if enemy.state["enemy_alive"]:
                enemy.update_plan(friendly_units)
                enemy.execute_next_task(friendly_units)
                enemy.current_goal = enemy.get_goal_position()
                if self.visualize:
                    logger.info(f"{enemy.state['name']} position: {enemy.state['position']}")
                    logger.info(f"{enemy.state['name']}'s current goal: {enemy.current_goal}")

    def update_friendly_enemy_info(self):
        active_enemies = [e for e in self.enemy_units if e.state["enemy_alive"]]
        scout = next((u for u in self.friendly_units if u.state["type"] == "scout"), None)
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
                logger.info(f"{u.name} at {u.state['position']}, visible enemies: {[e['name'] for e in u.state.get('visible_enemies', [])]}")
        if scout and self.visualize:
            logger.info(f"Scout at {scout.state['position']}, visible enemies: {[e['name'] for e in scout.state.get('visible_enemies', [])]}, scout_steps: {scout.state.get('scout_steps', 0)}")

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

    def update_plot(self):
        if not self.visualize:
            return

        start_time = time.time()
        # Check if update is needed
        needs_update = any(enemy.needs_update() for enemy in self.enemy_units) or \
                       any(unit.needs_update() for unit in self.friendly_units)
        if not needs_update:
            return

        # Restore background
        self.fig.canvas.restore_region(self.background)

        # Update title
        self.ax.set_title(f"Step {self.step_count} - {self.plan_name}")

        # Update enemy units
        for i, enemy in enumerate(self.enemy_units):
            if i >= len(self.enemy_markers):  # Safety check
                break
            if enemy.needs_update():
                enemy.last_position = enemy.state["position"]
                enemy.last_health = enemy.state["health"]
                enemy.last_group_size = enemy.state["current_group_size"]
                if enemy.state["enemy_alive"]:
                    ex, ey = enemy.state["position"]
                    cx, cy = ex * CELL_SIZE + CELL_SIZE/2, ey * CELL_SIZE + CELL_SIZE/2
                    self.enemy_markers[i].set_data([cx], [cy])
                    self.enemy_texts[i].set_position((cx + 15, cy + 15))
                    self.enemy_texts[i].set_text(f"{enemy.state['current_group_size']}/{enemy.state['health']:.0f}")
                    fx, fy = enemy.state["facing"]
                    norm = (fx**2 + fy**2)**0.5
                    if norm > 0:
                        fx, fy = fx/norm, fy/norm
                    else:
                        fx, fy = 0, 0
                    arrow_length = CELL_SIZE * 1.2
                    self.enemy_arrows[i].set_UVC(fx * arrow_length, fy * arrow_length)
                    self.enemy_arrows[i].set_offsets((cx, cy))
                else:
                    self.enemy_markers[i].set_data([], [])
                    self.enemy_texts[i].set_text('')
                    self.enemy_arrows[i].set_UVC(0, 0)
                    self.enemy_arrows[i].set_offsets((0, 0))

        # Update friendly units
        for i, unit in enumerate(self.friendly_units):
            if i >= len(self.friendly_markers):  # Safety check
                break
            if unit.needs_update():
                unit.last_position = unit.state["position"]
                unit.last_health = unit.state["health"]
                unit.last_group_size = unit.state["current_group_size"]
                if unit.state["health"] > 0:
                    pos = unit.state["position"]
                    cx, cy = pos[0] * CELL_SIZE + CELL_SIZE/2, pos[1] * CELL_SIZE + CELL_SIZE/2
                    self.friendly_markers[i].set_data([cx], [cy])
                    self.friendly_texts[i].set_position((cx + 15, cy + 15))
                    self.friendly_texts[i].set_text(f"{unit.state['current_group_size']}/{unit.state['health']:.0f}")
                    fx, fy = unit.state["facing"]
                    norm = (fx**2 + fy**2)**0.5
                    if norm > 0:
                        fx, fy = fx/norm, fy/norm
                    else:
                        fx, fy = 0, 0
                    arrow_length = CELL_SIZE * 1.2
                    self.friendly_arrows[i].set_UVC(fx * arrow_length, fy * arrow_length)
                    self.friendly_arrows[i].set_offsets((cx, cy))
                else:
                    self.friendly_markers[i].set_data([], [])
                    self.friendly_texts[i].set_text('')
                    self.friendly_arrows[i].set_UVC(0, 0)
                    self.friendly_arrows[i].set_offsets((0, 0))

        # Redraw updated elements
        artists = [self.ax.title]
        artists.extend([
            marker for i, (marker, enemy) in enumerate(zip(self.enemy_markers, self.enemy_units))
            if i < len(self.enemy_markers) and enemy.needs_update()
        ])
        artists.extend([
            arrow for i, (arrow, enemy) in enumerate(zip(self.enemy_arrows, self.enemy_units))
            if i < len(self.enemy_arrows) and enemy.needs_update()
        ])
        artists.extend([
            text for i, (text, enemy) in enumerate(zip(self.enemy_texts, self.enemy_units))
            if i < len(self.enemy_texts) and enemy.needs_update()
        ])
        artists.extend([
            marker for i, (marker, unit) in enumerate(zip(self.friendly_markers, self.friendly_units))
            if i < len(self.friendly_markers) and unit.needs_update()
        ])
        artists.extend([
            arrow for i, (arrow, unit) in enumerate(zip(self.friendly_arrows, self.friendly_units))
            if i < len(self.friendly_arrows) and unit.needs_update()
        ])
        artists.extend([
            text for i, (text, unit) in enumerate(zip(self.friendly_texts, self.friendly_units))
            if i < len(self.friendly_texts) and unit.needs_update()
        ])

        for artist in artists:
            self.ax.draw_artist(artist)

        # Blit and flush
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Minimal pause to stabilize rendering

        render_time = time.time() - start_time
        logger.info(f"Plot update took {render_time:.3f} seconds")

    def step(self):
        self.step_count += 1
        if self.visualize:
            logger.info(f"--- Simulation Step {self.step_count} ---")
        self.update_friendly_enemy_info()
        self.update_enemy_behavior()
        for unit in self.friendly_units:
            # Ensure plan is not empty before checking tasks
            if not unit.current_plan:
                logger.info(f"{unit.name} has empty plan; replanning.")
                unit.update_plan(force_replan=True)
            # Check if the current plan involves attacking a dead enemy
            if (unit.current_plan and isinstance(unit.current_plan[0], tuple) and
                unit.current_plan[0][0] == "AttackEnemy" and
                not any(e.state["name"] == unit.current_plan[0][1] and e.state["enemy_alive"]
                        for e in self.enemy_units)):
                logger.info(f"{unit.name} target enemy is dead or invalid; replanning.")
                unit.update_plan(force_replan=True)
            else:
                unit.update_plan()
            logger.info(f"{unit.name} current plan: {unit.current_plan}")
            if unit.current_plan and isinstance(unit.current_plan[0], tuple) and unit.current_plan[0][1]:
                logger.info(f"{unit.name} targeting: {unit.current_plan[0][1]}")
            unit.execute_next_task()
            if self.visualize:
                logger.info(f"{unit.name}'s current goal: {unit.get_goal_position()}")
            # Re-check plan after execution to handle dead enemies
            if (unit.current_plan and isinstance(unit.current_plan[0], tuple) and
                unit.current_plan[0][0] == "AttackEnemy" and
                not any(e.state["name"] == unit.current_plan[0][1] and e.state["enemy_alive"]
                        for e in self.enemy_units)):
                unit.update_plan(force_replan=True)

    def run(self, max_steps=200):
        self.step_count = 0
        for unit in self.friendly_units:
            unit.update_plan(force_replan=True)
        for enemy in self.enemy_units:
            enemy.update_plan(self.friendly_units)
        for _ in range(max_steps):
            if any(u.state.get("outpost_secured", False) for u in self.friendly_units):
                if self.visualize:
                    self.update_plot()
                    logger.info("\nMission accomplished: Outpost secured!")
                return self.evaluate_plan()
            self.step()
            if self.visualize:
                self.update_plot()
            alive_friendlies = [u for u in self.friendly_units if u.state["health"] > 0]
            if len(alive_friendlies) < len(self.friendly_units):
                for dead in set(self.friendly_units) - set(alive_friendlies):
                    if self.visualize:
                        logger.info(f"{dead.name} has been destroyed!")
                # Update plot elements to match alive friendlies
                if self.visualize:
                    indices_to_keep = [self.friendly_units.index(u) for u in alive_friendlies if u in self.friendly_units]
                    self.friendly_markers = [self.friendly_markers[i] for i in indices_to_keep]
                    self.friendly_arrows = [self.friendly_arrows[i] for i in indices_to_keep]
                    self.friendly_texts = [self.friendly_texts[i] for i in indices_to_keep]
                self.friendly_units = alive_friendlies
                if not self.friendly_units:
                    if self.visualize:
                        self.update_plot()
                        logger.info("\nAll friendly units have been destroyed! Mission failed.")
                    return self.evaluate_plan()
        if self.visualize:
            self.update_plot()
            logger.info("\nMission incomplete after maximum steps.")
        return self.evaluate_plan()

###############################
# Main Simulation Setup - Mode 1 (Test Specific Plan)
###############################

if __name__ == "__main__":
    TANK_GROUP_SIZE = 12
    INFANTRY_GROUP_SIZE = 12
    ARTILLERY_GROUP_SIZE = 3
    SCOUT_GROUP_SIZE = 2
    ANTI_TANK_GROUP_SIZE = 2

    tank_state_template = {
        "type": "tank",
        "position": (15, 20),
        "facing": (0, 1),
        "base_health": 20,
        "health": 20 * TANK_GROUP_SIZE,
        "max_health": 20 * TANK_GROUP_SIZE,
        "group_size": TANK_GROUP_SIZE,
        "current_group_size": TANK_GROUP_SIZE,
        "cumulative_damage": 0.0,
        "armor_front": 17,
        "armor_side": 4,
        "armor_rear": 3,
        "friendly_accuracy": 0.75,
        "base_rate_of_fire": 4.9,
        "damage": 9,
        "suppression": 0.12,
        "penetration": 18,
        "friendly_attack_range": 2400 / CELL_SIZE,
        "all_enemies_spotted": False,
        "total_enemies": 2,
        "suppression_from_enemy": 0.0,
        "scout_steps": 0,
        "stealth_modifier": 0
    }
    infantry_state_template = {
        "type": "infantry",
        "position": (14, 20),
        "facing": (0, 1),
        "base_health": 1,
        "health": 1 * INFANTRY_GROUP_SIZE,
        "max_health": 1 * INFANTRY_GROUP_SIZE,
        "group_size": INFANTRY_GROUP_SIZE,
        "current_group_size": INFANTRY_GROUP_SIZE,
        "cumulative_damage": 0.0,
        "armor_front": 0,
        "armor_side": 0,
        "armor_rear": 0,
        "friendly_accuracy": 0.50,
        "base_rate_of_fire": 294,
        "damage": 0.8,
        "suppression": 0.01,
        "penetration": 1,
        "friendly_attack_range": 1200 / CELL_SIZE,
        "all_enemies_spotted": False,
        "total_enemies": 2,
        "suppression_from_enemy": 0.0,
        "scout_steps": 0,
        "stealth_modifier": 0
    }
    artillery_state_template = {
        "type": "artillery",
        "position": (14, 19),
        "facing": (0, 1),
        "base_health": 18,
        "health": 18 * ARTILLERY_GROUP_SIZE,
        "max_health": 18 * ARTILLERY_GROUP_SIZE,
        "group_size": ARTILLERY_GROUP_SIZE,
        "current_group_size": ARTILLERY_GROUP_SIZE,
        "cumulative_damage": 0.0,
        "armor_front": 2,
        "armor_side": 1,
        "armor_rear": 1,
        "friendly_accuracy": 0.85,
        "base_rate_of_fire": 8.6,
        "damage": 3.5,
        "suppression": 15,
        "penetration": 1,
        "friendly_attack_range": 4000 / CELL_SIZE,
        "all_enemies_spotted": False,
        "total_enemies": 2,
        "suppression_from_enemy": 0.0,
        "scout_steps": 0,
        "stealth_modifier": 0
    }
    scout_state_template = {
        "type": "scout",
        "position": (16, 20),
        "facing": (0, 1),
        "base_health": 18,
        "health": 18 * SCOUT_GROUP_SIZE,
        "max_health": 18 * SCOUT_GROUP_SIZE,
        "group_size": SCOUT_GROUP_SIZE,
        "current_group_size": SCOUT_GROUP_SIZE,
        "cumulative_damage": 0.0,
        "armor_front": 3,
        "armor_side": 2,
        "armor_rear": 2,
        "friendly_accuracy": 0.35,
        "base_rate_of_fire": 115,
        "damage": 2.4,
        "suppression": 0.05,
        "penetration": 5,
        "vision_range": 2600 / CELL_SIZE,
        "friendly_attack_range": 1800 / CELL_SIZE,
        "all_enemies_spotted": False,
        "candidate_positions": [],
        "current_candidate_index": 0,
        "total_enemies": 2,
        "suppression_from_enemy": 0.0,
        "scout_steps": 0,
        "stealth_modifier": 225
    }
    anti_tank_state_template = {
        "type": "anti-tank",
        "position": (15, 19),
        "facing": (0, 1),
        "base_health": 18,
        "health": 18 * ANTI_TANK_GROUP_SIZE,
        "max_health": 18 * ANTI_TANK_GROUP_SIZE,
        "group_size": ANTI_TANK_GROUP_SIZE,
        "current_group_size": ANTI_TANK_GROUP_SIZE,
        "cumulative_damage": 0.0,
        "armor_front": 2,
        "armor_side": 1,
        "armor_rear": 1,
        "friendly_accuracy": 0.90,
        "base_rate_of_fire": 6.3,
        "damage": 15,
        "suppression": 0.10,
        "penetration": 22,
        "vision_range": 2000 / CELL_SIZE,
        "friendly_attack_range": 2800 / CELL_SIZE,
        "all_enemies_spotted": False,
        "total_enemies": 2,
        "suppression_from_enemy": 0.0,
        "scout_steps": 0,
        "stealth_modifier": 50
    }

    candidate_domain = secure_outpost_domain

    enemy_state1 = create_enemy_state(index=0)
    enemy_state2 = create_enemy_state(index=1)

    enemy_domain = {"DefendAreaMission": [
            (lambda state: any(manhattan(state["position"], u.state["position"]) <= state["attack_range"] and
                                has_line_of_sight(state["position"], u.state["position"])
                                for u in state["friendly_units"]), ["AttackEnemy"]),
            (lambda state: any(manhattan(state["position"], u.state["position"]) <= state["vision_range"] and
                                has_line_of_sight(state["position"], u.state["position"])
                                for u in state["friendly_units"]), ["ChaseTarget"]),
            (lambda state: True, ["Patrol"])
        ]
    }

    enemy_unit1 = EnemyTank("EnemyTankGroup1", enemy_state1, enemy_domain)
    enemy_unit2 = EnemyTank("EnemyTankGroup2", enemy_state2, enemy_domain)
    enemy_units = [enemy_unit1, enemy_unit2]

    tank_state = tank_state_template.copy()
    infantry_state = infantry_state_template.copy()
    artillery_state = artillery_state_template.copy()
    scout_state = scout_state_template.copy()
    anti_tank_state = anti_tank_state_template.copy()

    for state in [tank_state, infantry_state, artillery_state, scout_state, anti_tank_state]:
        state["enemy"] = enemy_state1
        state["target_enemy"] = enemy_state1
        state["outpost_position"] = enemy_state1["outpost_position"]
        state["visible_enemies"] = []

    tank = FriendlyTank("FriendlyTankGroup", tank_state, candidate_domain)
    infantry = FriendlyInfantry("FriendlyInfantryGroup", infantry_state, candidate_domain)
    artillery = FriendlyArtillery("FriendlyArtilleryGroup", artillery_state, candidate_domain)
    scout = FriendlyScout("FriendlyScoutGroup", scout_state, candidate_domain)
    anti_tank = FriendlyAntiTank("FriendlyAntiTankGroup", anti_tank_state, candidate_domain)
    friendly_units = [tank, infantry, artillery, scout, anti_tank]

    commander = TeamCommander(friendly_units)
    sim = Simulation(friendly_units, enemy_units, commander, visualize=True, plan_name="Mode1_Test_Grouped_Dynamic_Partial")

    for unit in friendly_units:
        unit.sim = sim

    result = sim.run(max_steps=300)
    logger.info("\n=== Plan Evaluation ===")
    logger.info(f"Score: {result['score']:.1f}")
    logger.info(f"Total Friendly Health Remaining: {result['health']}")
    logger.info(f"Enemy Health Remaining: {result['enemy_health']}")
    logger.info(f"Outpost Secured: {result['outpost_secured']}")
    logger.info(f"Steps Taken: {result['steps_taken']}")

    plt.ioff()
    plt.show()
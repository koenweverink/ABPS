import random
import heapq
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import itertools

###############################
# Grid, Obstacles, and LOS
###############################

GRID_WIDTH = 50
GRID_HEIGHT = 50
CELL_SIZE = 100

# Create a river (vertical obstacle) with 3 bridges.
river_columns = range(round(GRID_WIDTH/3), round(GRID_WIDTH/3) + 5)  # 5 cells wide
bridge_centers = [10, 25, 40]
bridge_group_size = 2
bridge_rows = set()
for center in bridge_centers:
    bridge_rows.update(range(center - bridge_group_size // 2, center - bridge_group_size // 2 + bridge_group_size))
river = {(x, y) for x in river_columns for y in range(GRID_HEIGHT) if y not in bridge_rows}

# Create a forest area in the top right corner.
forest = {(x, y) for x in range(GRID_WIDTH - 20, GRID_WIDTH - 5) for y in range(GRID_HEIGHT - 20, GRID_HEIGHT - 5)}

# In this example the obstacles come only from the river and forest.
obstacles = set()

def in_bounds(pos):
    return 0 <= pos[0] < GRID_WIDTH and 0 <= pos[1] < GRID_HEIGHT

def neighbors(pos):
    return [p for p in [(pos[0]+1, pos[1]), (pos[0]-1, pos[1]), (pos[0], pos[1]+1), (pos[0], pos[1]-1)]
            if in_bounds(p) and p not in obstacles and p not in river]

def manhattan(p, q):
    return abs(p[0]-q[0]) + abs(p[1]-q[1])

def is_in_enemy_vision(pos, enemy_units):
    for enemy in enemy_units:
        if enemy.state["enemy_alive"]:
            distance = manhattan(pos, enemy.state["enemy_position"])
            has_los = has_line_of_sight(pos, enemy.state["enemy_position"])
            vision_range = enemy.state.get("vision_range", 20)
            if distance <= vision_range and has_los:
                return True
    return False

def astar(start, goal, enemy_units=None):
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for nxt in neighbors(current):
            # Base cost
            new_cost = cost_so_far[current] + 1
            # Penalty for non-forest cells
            if not is_in_cover(nxt):
                new_cost += 5  # High penalty for open terrain
            # Penalty for enemy vision
            if enemy_units and is_in_enemy_vision(nxt, enemy_units):
                new_cost += 10  # Avoid enemy vision cones
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

def next_step(start, goal):
    path = astar(start, goal)
    return path[1] if len(path) >= 2 else start

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

def has_line_of_sight(start, end):
    line = get_line(start, end)
    blocks_los = all(pos not in obstacles and pos not in river for pos in line[1:-1])
    if not blocks_los:
        return False
    start_in_cover = is_in_cover(start)
    end_in_cover = is_in_cover(end)
    start_forest_edge = is_forest_edge(start)
    end_forest_edge = is_forest_edge(end)

    if start_in_cover or end_in_cover:
        if not (start_forest_edge or end_forest_edge):
            return False
    los_clear = all(pos not in forest for pos in line[1:-1])
    return los_clear

def is_in_cover(pos):
    return pos in forest

def is_forest_edge(pos):
    if not is_in_cover(pos):
        return False
    x, y = pos
    neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    return any(not is_in_cover(n) for n in neighbors if in_bounds(n))

def get_effective_vision_range(base_vision_range, stealth_modifier, in_cover, has_los):
    if not in_cover:
        return base_vision_range
    return base_vision_range / (1 + stealth_modifier / CELL_SIZE)

def generate_candidate_positions(enemy_units, vision_range=26):
    candidates = []
    # Collect all patrol points
    patrol_points = set()
    for enemy in enemy_units:
        if enemy.state["enemy_alive"]:
            patrol_points.update(enemy.state.get("patrol_points", []))
    # Find positions in forest or forest edge within vision range
    for pos in forest:
        if is_in_cover(pos) or is_forest_edge(pos):
            for patrol_pos in patrol_points:
                distance = manhattan(pos, patrol_pos)
                if distance <= vision_range and has_line_of_sight(pos, patrol_pos):
                    candidates.append(pos)
                    break
    # Sort by proximity to enemies and cover status
    def score_position(pos):
        min_distance = min(manhattan(pos, p) for p in patrol_points) if patrol_points else float('inf')
        cover_score = 2 if is_forest_edge(pos) else 1 if is_in_cover(pos) else 0
        return (min_distance, -cover_score)  # Minimize distance, maximize cover
    candidates.sort(key=score_position)
    return candidates[:4]  # Limit to 4 candidates

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

def select_enemy_for_unit(enemy_units, friendly_unit):
    viable = [e for e in enemy_units if e.state["enemy_alive"]]
    if not viable:
        return None
    closest_enemy = min(viable, key=lambda e: manhattan(friendly_unit.state["position"], e.state["enemy_position"]))
    return closest_enemy.state

def enemy_in_range_with_los(state):
    return (state["enemy"].get("enemy_alive", False) and
            manhattan(state["position"], state["enemy"]["enemy_position"]) <= state["friendly_attack_range"] and
            has_line_of_sight(state["position"], state["enemy"]["enemy_position"]))

###############################
# Global Enemy State Creation
###############################

def create_enemy_state(index=0):
    if index == 0:
        state = {
            "name": "EnemyTank1",
            "enemy_position": (GRID_WIDTH - 1, GRID_HEIGHT - 3),
            "enemy_alive": True,
            "enemy_health": 20,
            "max_health": 20,
            "enemy_armor": 17,
            "outpost_position": (GRID_WIDTH - 1, 0),
            "outpost_secured": False,
            "enemy_attack_range": 2400 / CELL_SIZE,
            "enemy_accuracy": 0.7,
            "enemy_penetration": 18,
            "enemy_damage": 9,
            "enemy_suppression": 0.12,
            "rate_of_fire": 4.9,
            "suppression": {},
            "patrol_points": [(GRID_WIDTH - 1, GRID_HEIGHT - 3), (GRID_WIDTH - 20, GRID_HEIGHT - 3)],
            "current_patrol_index": 0,
            "vision_range": 2000 / CELL_SIZE,
            "retreat_point": (GRID_WIDTH - 1, GRID_HEIGHT - 1),
            "stealth_modifier": 0
        }
    else:
        state = {
            "name": "EnemyTank2",
            "enemy_position": (GRID_WIDTH - 1, 2),
            "enemy_alive": True,
            "enemy_health": 20,
            "max_health": 20,
            "enemy_armor": 17,
            "outpost_position": (GRID_WIDTH - 1, 0),
            "outpost_secured": False,
            "enemy_attack_range": 2400 / CELL_SIZE,
            "enemy_accuracy": 0.7,
            "enemy_penetration": 18,
            "enemy_damage": 9,
            "enemy_suppression": 0.12,
            "rate_of_fire": 4.9,
            "suppression": {},
            "patrol_points": [(GRID_WIDTH - 1, 2), (GRID_WIDTH - 3, 2)],
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
        (lambda state: state["role"] == "scout" and not state.get("all_enemies_spotted", False),
         ["ScoutSearch"]),
        (lambda state: state["role"] != "scout" and not state.get("all_enemies_spotted", False),
         ["Hold"]),
        (lambda state: state.get("all_enemies_spotted", False) and state.get("target_enemy", {}).get("enemy_alive", False),
         ["DefeatEnemies", "Move", "SecureOutpost"]),
        (lambda state: state.get("all_enemies_spotted", False) and not state.get("target_enemy", {}).get("enemy_alive", False) and
         state["position"] != state["outpost_position"],
         ["Move", "SecureOutpost"]),
        (lambda state: state.get("all_enemies_spotted", False) and not state.get("target_enemy", {}).get("enemy_alive", False) and
         state["position"] == state["outpost_position"],
         ["Hold"])
    ],
    "DefeatEnemies": [
        (lambda state: state.get("target_enemy", {}).get("enemy_alive", False) and 
                    enemy_in_range_with_los({**state, "enemy": state["target_enemy"]}),
         ["AttackEnemy"]),
        (lambda state: state.get("target_enemy", {}).get("enemy_alive", False) and 
                    manhattan(state["position"], state["target_enemy"]["enemy_position"]) > state["friendly_attack_range"],
         ["Move", "AttackEnemy"])
    ],
    "ScoutSearch": [
        (lambda state: len(set(e["name"] for e in state.get("visible_enemies", []))) >= state.get("total_enemies", 2),
         ["SetAllEnemiesSpotted"]),
        (lambda state: state.get("scout_steps", 0) >= 200,  # Fallback after 200 steps
         ["SetAllEnemiesSpotted"]),
        (lambda state: True, ["Move", "SetAllEnemiesSpotted"])
    ]
}

def enemy_not_in_range_friendly(state):
    return state["enemy"].get("enemy_alive", False) and manhattan(state["position"], state["enemy"]["enemy_position"]) > state["friendly_attack_range"]

def enemy_in_range_friendly(state):
    return state["enemy"].get("enemy_alive", False) and manhattan(state["position"], state["enemy"]["enemy_position"]) <= state["friendly_attack_range"]

class HTNPlanner:
    def __init__(self, domain):
        self.domain = domain
    def plan(self, task, state):
        if task not in self.domain:
            return [task]
        for condition, subtasks in self.domain[task]:
            if condition(state):
                plan = []
                for subtask in subtasks:
                    if subtask in self.domain:
                        sub_plan = self.plan(subtask, state)
                    else:
                        sub_plan = [subtask]
                    if sub_plan is None:
                        break
                    plan.extend(sub_plan)
                else:
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
    def update_plan(self, friendly_units):
        combined_state = self.state.copy()
        combined_state["friendly_units"] = friendly_units
        if friendly_units:
            target = min(friendly_units, key=lambda u: manhattan(self.state["enemy_position"], u.state["position"]))
            combined_state["target_position"] = target.state["position"]
        else:
            combined_state["target_position"] = self.state["enemy_position"]
        mission = "DefendAreaMission"
        new_plan = self.planner.plan(mission, combined_state)
        self.current_plan = new_plan if new_plan else []
        print(f"{self.name} updated plan: {self.current_plan}")
    def execute_next_task(self, friendly_units):
        if not self.current_plan or not self.state["enemy_alive"]:
            return
        task = self.current_plan[0]
        if task == "Patrol":
            idx = self.state["current_patrol_index"]
            target = self.state["patrol_points"][idx]
            self.state["enemy_position"] = next_step(self.state["enemy_position"], target)
            if self.state["enemy_position"] == target:
                self.state["current_patrol_index"] = (idx + 1) % len(self.state["patrol_points"])
                self.current_plan.pop(0)
        elif task == "ChaseTarget":
            target = min(friendly_units, key=lambda u: manhattan(self.state["enemy_position"], u.state["position"])).state["position"]
            self.state["enemy_position"] = next_step(self.state["enemy_position"], target)
            if manhattan(self.state["enemy_position"], target) <= self.state["enemy_attack_range"]:
                self.current_plan.pop(0)
        elif task == "AttackEnemy":
            target_unit = None
            min_dist = float('inf')
            for u in friendly_units:
                d = manhattan(self.state["enemy_position"], u.state["position"])
                if d <= self.state["enemy_attack_range"] and has_line_of_sight(self.state["enemy_position"], u.state["position"]):
                    if d < min_dist:
                        min_dist = d
                        target_unit = u
            if target_unit:
                num_attacks = get_num_attacks(self.state["rate_of_fire"])
                total_suppression = sum(self.state["suppression"].values())
                effective_accuracy = max(0, self.state["enemy_accuracy"] - total_suppression)
                for _ in range(num_attacks):
                    if random.random() < effective_accuracy:
                        D = self.state["enemy_penetration"] - target_unit.state.get("armor", 0)
                        if random.random() < get_penetration_probability(D):
                            target_unit.state["friendly_health"] -= self.state["enemy_damage"]
                            if target_unit.state["friendly_health"] <= 0:
                                print(f"{target_unit.name} destroyed by {self.name}!")
                        target_unit.state["suppression_from_enemy"] = target_unit.state.get("suppression_from_enemy", 0) + self.state["enemy_suppression"]
            self.current_plan.pop(0)
        elif task == "Retreat":
            retreat = self.state.get("retreat_point", (9, 9))
            self.state["enemy_position"] = next_step(self.state["enemy_position"], retreat)
            if self.state["enemy_position"] == retreat:
                self.current_plan.pop(0)
    def get_goal_position(self):
        if not self.current_plan:
            return self.state["enemy_position"]
        t = self.current_plan[0]
        if t == "Patrol":
            idx = self.state["current_patrol_index"]
            return self.state["patrol_points"][idx]
        elif t in ["ChaseTarget", "AttackEnemy"]:
            if "friendly_units" in self.state and self.state["friendly_units"]:
                return min(self.state["friendly_units"], key=lambda u: manhattan(self.state["enemy_position"], u.state["position"])).state["position"]
        elif t == "Retreat":
            return self.state.get("retreat_point", (9, 9))
        return self.state["enemy_position"]

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
        self.last_enemy_pos = state["target_enemy"]["enemy_position"] if "target_enemy" in state else state.get("outpost_position", (0, 0))
        self.last_health = state["friendly_health"]
        self.sim = simulation  # Reference to Simulation for enemy_units access

    def update_plan(self, force_replan=False):
        mission = list(self.planner.domain.keys())[0]
        # Keep AttackEnemy if in range and LOS
        if (self.state.get("all_enemies_spotted", False) and
            self.state.get("target_enemy", {}).get("enemy_alive", False) and
            manhattan(self.state["position"], self.state["target_enemy"]["enemy_position"]) <= self.state["friendly_attack_range"] and
            has_line_of_sight(self.state["position"], self.state["target_enemy"]["enemy_position"])):
            self.current_plan = ["AttackEnemy"] + self.current_plan[1:] if self.current_plan else ["AttackEnemy"]
            print(f"{self.name} already in attack position; keeping plan: {self.current_plan}")
            return
        # Replan if forced, no plan, or need to act after spotting enemies
        if (force_replan or not self.current_plan or
            (self.state.get("all_enemies_spotted", False) and self.current_plan == ["Hold"])):
            new_plan = self.planner.plan(mission, self.state)
            if not new_plan:
                new_plan = ["Move", "SecureOutpost"] if self.state["position"] != self.state["outpost_position"] else ["Hold"]
            self.current_plan = new_plan
            print(f"{self.name} replanned: {self.current_plan}")
        else:
            print(f"{self.name} current plan: {self.current_plan}")
        self.last_enemy_pos = self.state["target_enemy"]["enemy_position"] if self.state.get("target_enemy", {}) else self.state["outpost_position"]
        self.last_health = self.state["friendly_health"]

    def execute_next_task(self):
        if not self.current_plan or self.state["friendly_health"] <= 0:
            return
        task = self.current_plan[0]
        if task == "Move":
            next_task = self.current_plan[1] if len(self.current_plan) > 1 else None
            if next_task == "AttackEnemy":
                if not self.state.get("target_enemy", {}).get("enemy_alive", False):
                    print(f"{self.name} cannot move to attack; no valid enemy target. Replanning.")
                    self.current_plan.pop(0)
                    self.update_plan(force_replan=True)
                    return
                goal = self.state["target_enemy"]["enemy_position"]
                if (self.state["target_enemy"]["enemy_alive"] and
                    manhattan(self.state["position"], goal) <= self.state["friendly_attack_range"] and
                    has_line_of_sight(self.state["position"], goal)):
                    print(f"{self.name} reached enemy range and LOS; stopping move.")
                    self.current_plan.pop(0)
                else:
                    self.state["position"] = next_step(self.state["position"], goal)
                    print(f"{self.name} moves toward enemy at {goal}, new position: {self.state['position']}")
            elif next_task == "SecureOutpost":
                goal = self.state["outpost_position"]
                if self.state["position"] == goal:
                    print(f"{self.name} reached outpost; stopping move.")
                    self.current_plan.pop(0)
                else:
                    self.state["position"] = next_step(self.state["position"], goal)
                    print(f"{self.name} moves toward outpost at {goal}, new position: {self.state['position']}")
            elif next_task == "SetAllEnemiesSpotted" and self.state["role"] == "scout":
                candidate_positions = self.state.get("candidate_positions", [])
                if candidate_positions:
                    idx = self.state.get("current_candidate_index", 0)
                    candidate = candidate_positions[idx]
                    visible_enemy_names = set(e["name"] for e in self.state.get("visible_enemies", []))
                    self.state["scout_steps"] = self.state.get("scout_steps", 0) + 1
                    if len(visible_enemy_names) >= self.state.get("total_enemies", 2):
                        print(f"{self.name} has spotted all enemies: {visible_enemy_names}; setting all_enemies_spotted flag.")
                        self.state["all_enemies_spotted"] = True
                        self.current_plan.pop(0)
                    elif self.state["scout_steps"] >= 200:
                        print(f"{self.name} reached max scout steps (200); forcing all_enemies_spotted flag.")
                        self.state["all_enemies_spotted"] = True
                        self.current_plan.pop(0)
                    else:
                        # Use stealthy pathfinding
                        path = astar(self.state["position"], candidate, self.sim.enemy_units if hasattr(self, 'sim') else [])
                        next_pos = path[1] if len(path) >= 2 else self.state["position"]
                        self.state["position"] = next_pos
                        print(f"{self.name} moves stealthily toward candidate position {candidate}, new position: {self.state['position']}")
                        print(f"{self.name} visible enemies: {visible_enemy_names}, scout_steps: {self.state['scout_steps']}")
                        if self.state["position"] == candidate:
                            idx = (idx + 1) % len(candidate_positions)
                            self.state["current_candidate_index"] = idx
                else:
                    print(f"{self.name} has no candidate positions; setting all_enemies_spotted flag.")
                    self.state["all_enemies_spotted"] = True
                    self.current_plan.pop(0)
            else:
                goal = self.get_goal_position()
                if self.state["position"] == goal:
                    print(f"{self.name} reached goal {goal}; stopping move.")
                    self.current_plan.pop(0)
                else:
                    self.state["position"] = next_step(self.state["position"], goal)
                    print(f"{self.name} moves towards {goal}, new position: {self.state['position']}")
        # Rest of the method remains unchanged
        elif task == "AttackEnemy":
            if not self.state.get("target_enemy", {}).get("enemy_alive", False):
                print(f"{self.name} cannot attack; target enemy is already destroyed. Replanning.")
                self.current_plan.pop(0)
                self.update_plan(force_replan=True)
                return
            target_pos = self.state["target_enemy"]["enemy_position"]
            distance = manhattan(self.state["position"], target_pos)
            has_los = has_line_of_sight(self.state["position"], target_pos)
            if (distance <= self.state["friendly_attack_range"] and has_los):
                num_attacks = get_num_attacks(self.state["rate_of_fire"])
                total_suppression = sum(self.state.get("suppression_from_enemy", 0).values()) if isinstance(self.state.get("suppression_from_enemy"), dict) else self.state.get("suppression_from_enemy", 0)
                effective_accuracy = max(0, self.state["friendly_accuracy"] - total_suppression)
                print(f"{self.name} attacks {self.state['target_enemy']['name']} with {num_attacks} attacks, accuracy: {effective_accuracy:.2f}")
                for _ in range(num_attacks):
                    if random.random() < effective_accuracy:
                        D = self.state["penetration"] - self.state["target_enemy"]["enemy_armor"]
                        if random.random() < get_penetration_probability(D):
                            self.state["target_enemy"]["enemy_health"] -= self.state["damage"]
                            print(f"{self.name} hits {self.state['target_enemy']['name']}, dealing {self.state['damage']} damage. Enemy health: {self.state['target_enemy']['enemy_health']}")
                            if self.state["target_enemy"]["enemy_health"] <= 0:
                                self.state["target_enemy"]["enemy_alive"] = False
                                print(f"{self.name} destroyed {self.state['target_enemy']['name']}!")
                                self.current_plan.pop(0)
                                self.update_plan(force_replan=True)
                                return
                        self.state["target_enemy"]["suppression"] = self.state["target_enemy"].get("suppression", {})
                        self.state["target_enemy"]["suppression"][self.name] = self.state["suppression"]
            else:
                print(f"{self.name} cannot attack; enemy at {target_pos} is out of range (distance: {distance}, max: {self.state['friendly_attack_range']}) or no LOS (has_los: {has_los}). Replanning.")
                self.current_plan.pop(0)
                self.update_plan(force_replan=True)
        elif task == "SecureOutpost":
            if self.state["position"] == self.state["outpost_position"]:
                self.state["outpost_secured"] = True
                print(f"{self.name} secures the outpost!")
                self.current_plan.pop(0)
            else:
                print(f"{self.name} cannot secure the outpost; not at target location.")
                self.current_plan.pop(0)
        elif task == "Hold":
            print(f"{self.name} holds position at {self.state['position']}.")
            if (self.state["role"] != "scout" and
                self.state.get("all_enemies_spotted", False) and
                self.current_plan == ["Hold"]):
                print(f"{self.name} all enemies spotted; replanning.")
                self.update_plan(force_replan=True)
        elif task == "SetAllEnemiesSpotted":
            visible_enemy_names = set(e["name"] for e in self.state.get("visible_enemies", []))
            self.state["scout_steps"] = self.state.get("scout_steps", 0) + 1
            if len(visible_enemy_names) >= self.state.get("total_enemies", 2):
                print(f"{self.name} sets all_enemies_spotted flag to True. Spotted enemies: {visible_enemy_names}")
                self.state["all_enemies_spotted"] = True
            elif self.state["scout_steps"] >= 200:
                print(f"{self.name} reached max scout steps (200) in SetAllEnemiesSpotted; forcing all_enemies_spotted flag.")
                self.state["all_enemies_spotted"] = True
            else:
                print(f"{self.name} cannot set all_enemies_spotted; only spotted: {visible_enemy_names}, scout_steps: {self.state['scout_steps']}")
            self.current_plan.pop(0)

    def get_goal_position(self):
        if self.current_plan:
            if len(self.current_plan) > 1:
                if self.current_plan[1] == "AttackEnemy":
                    return self.state["target_enemy"]["enemy_position"] if self.state.get("target_enemy", {}).get("enemy_alive", False) else self.state["outpost_position"]
                elif self.current_plan[1] == "SecureOutpost":
                    return self.state["outpost_position"]
                elif self.current_plan[1] == "SetAllEnemiesSpotted" and self.state["role"] == "scout":
                    idx = self.state.get("current_candidate_index", 0)
                    return self.state["candidate_positions"][idx]
            if self.current_plan[0] == "AttackEnemy":
                return self.state["target_enemy"]["enemy_position"] if self.state.get("target_enemy", {}).get("enemy_alive", False) else self.state["outpost_position"]
        return self.state["outpost_position"]

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

###############################
# TeamCommander and Simulation Classes
###############################

class TeamCommander:
    def __init__(self, friendly_units):
        self.friendly_units = friendly_units
    def share_tank_health(self):
        tank = next((u for u in self.friendly_units if u.state["role"] == "attacker"), None)
        if tank:
            for u in self.friendly_units:
                u.state["tank_health"] = tank.state["friendly_health"]
                u.state["tank_max_health"] = tank.state["max_health"]

class Simulation:
    def __init__(self, friendly_units, enemy_units, team_commander, visualize=False, plan_name="Unknown Plan"):
        self.friendly_units = friendly_units
        self.enemy_units = enemy_units
        self.team_commander = team_commander
        active_enemy = next((e for e in enemy_units if e.state["enemy_alive"]), None)
        for u in self.friendly_units:
            u.state["enemy"] = active_enemy.state if active_enemy else {}
            u.state["visible_enemies"] = []
            u.state["total_enemies"] = len(self.enemy_units)
            u.state["scout_steps"] = 0
            # Set dynamic candidate positions for scout
            if u.state["role"] == "scout":
                u.state["candidate_positions"] = generate_candidate_positions(self.enemy_units, u.state["vision_range"])
                if not u.state["candidate_positions"]:
                    u.state["candidate_positions"] = [(40, 45), (40, 2), (25, 45), (45, 5)]  # Fallback
            u.sim = self  # Set simulation reference
        self.step_count = 0
        self.visualize = visualize
        self.plan_name = plan_name
        if self.visualize:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8,8))

    def update_enemy_behavior(self):
        friendly_units = [u for u in self.friendly_units if u.state.get("friendly_health", 0) > 0]
        for enemy in self.enemy_units:
            if enemy.state["enemy_alive"]:
                enemy.update_plan(friendly_units)
                enemy.execute_next_task(friendly_units)
                enemy.current_goal = enemy.get_goal_position()
                if self.visualize:
                    print(f"{enemy.state['name']} position: {enemy.state['enemy_position']}")
                    print(f"{enemy.state['name']}'s current goal: {enemy.current_goal}")

    def update_friendly_enemy_info(self):
        active_enemies = [e for e in self.enemy_units if e.state["enemy_alive"]]
        scout = next((u for u in self.friendly_units if u.state["role"] == "scout"), None)
        all_enemies_spotted = False
        if scout:
            new_visible = []
            for e in self.enemy_units:
                if e.state["enemy_alive"]:
                    distance = manhattan(scout.state["position"], e.state["enemy_position"])
                    has_los = has_line_of_sight(scout.state["position"], e.state["enemy_position"])
                    in_cover = is_in_cover(e.state["enemy_position"])
                    stealth_modifier = e.state.get("stealth_modifier", 0)
                    effective_vision_range = get_effective_vision_range(
                        scout.state.get("vision_range", 0),
                        stealth_modifier,
                        in_cover,
                        has_los
                    )
                    if distance <= effective_vision_range and has_los:
                        new_visible.append(e.state)
                        print(f"{scout.name} spotted {e.state['name']} at {e.state['enemy_position']} "
                            f"(distance: {distance:.2f}, effective vision range: {effective_vision_range:.2f}, "
                            f"stealth_modifier: {stealth_modifier}, in_cover: {in_cover}, has_los: {has_los}, "
                            f"is_forest_edge: {is_forest_edge(scout.state['position'])})")
                    elif self.visualize:
                        print(f"{scout.name} failed to spot {e.state['name']} at {e.state['enemy_position']} "
                            f"(distance: {distance:.2f}, effective vision range: {effective_vision_range:.2f}, "
                            f"stealth_modifier: {stealth_modifier}, in_cover: {in_cover}, has_los: {has_los}, "
                            f"is_forest_edge: {is_forest_edge(scout.state['position'])})")
            current_names = set(e["name"] for e in scout.state.get("visible_enemies", []))
            for enemy in new_visible:
                if enemy["name"] not in current_names:
                    scout.state["visible_enemies"].append(enemy)
                    current_names.add(enemy["name"])
                    print(f"{scout.name} newly spotted {enemy['name']} at {enemy['enemy_position']}")
            all_enemies_spotted = len(set(e["name"] for e in scout.state["visible_enemies"])) >= len(self.enemy_units)
            if all_enemies_spotted:
                scout.state["all_enemies_spotted"] = True
        for u in self.friendly_units:
            u.state["all_enemies_spotted"] = all_enemies_spotted
            if all_enemies_spotted and scout:
                u.state["visible_enemies"] = scout.state["visible_enemies"]
            else:
                u.state["visible_enemies"] = [] if u != scout else u.state.get("visible_enemies", [])
            if active_enemies:
                closest_enemy = None
                min_distance = float('inf')
                for e in active_enemies:
                    distance = manhattan(u.state["position"], e.state["enemy_position"])
                    has_los = has_line_of_sight(e.state["enemy_position"], u.state["position"])
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
                        print(f"{e.state['name']} spotted {u.name} at {u.state['position']} "
                            f"(distance: {distance:.2f}, effective vision range: {effective_vision_range:.2f}, "
                            f"stealth_modifier: {stealth_modifier}, in_cover: {in_cover}, has_los: {has_los}, "
                            f"is_forest_edge: {is_forest_edge(u.state['position'])})")
                    elif self.visualize:
                        print(f"{e.state['name']} failed to spot {u.name} at {u.state['position']} "
                            f"(distance: {distance:.2f}, effective vision range: {effective_vision_range:.2f}, "
                            f"stealth_modifier: {stealth_modifier}, in_cover: {in_cover}, has_los: {has_los}, "
                            f"is_forest_edge: {is_forest_edge(u.state['position'])})")
                # Update enemy state with closest spotted friendly unit or clear if none spotted
                e.state["enemy"] = closest_enemy or {}
                if self.visualize:
                    print(f"{e.state['name']} state['enemy']: {e.state['enemy'].get('name', 'None')}")
            else:
                # No active enemies, clear enemy state
                for e in self.enemy_units:
                    e.state["enemy"] = {}
                    if self.visualize:
                        print(f"{e.state['name']} state['enemy']: None (no active enemies)")
            # Update friendly unit state
            if self.visualize:
                print(f"{u.name} at {u.state['position']}, visible enemies: {[e['name'] for e in u.state.get('visible_enemies', [])]}")
        if scout and self.visualize:
            print(f"Scout at {scout.state['position']}, visible enemies: {[e['name'] for e in scout.state.get('visible_enemies', [])]}, scout_steps: {scout.state.get('scout_steps', 0)}")

    def evaluate_plan(self):
        total_friendly = sum(u.state["friendly_health"] for u in self.friendly_units)
        max_friendly = sum(u.state["max_health"] for u in self.friendly_units)
        enemy_health = sum(e.state["enemy_health"] for e in self.enemy_units if e.state["enemy_alive"])
        max_enemy = sum(e.state["max_health"] for e in self.enemy_units)
        outpost_secured = any(u.state.get("outpost_secured", False) for u in self.friendly_units)
        steps = self.step_count
        friendly_ratio = total_friendly / max_friendly if max_friendly > 0 else 0
        enemy_ratio = enemy_health / max_enemy if max_enemy > 0 else 0
        score = (friendly_ratio * 20) - (enemy_ratio * 20) + (10 if outpost_secured else -10) - 0.1 * steps
        return {
            "score": score,
            "friendly_health": total_friendly,
            "enemy_health": enemy_health,
            "outpost_secured": outpost_secured,
            "steps_taken": steps
        }

    def update_plot(self):
        major_step = 500
        self.ax.clear()
        self.fig.set_size_inches(8, 8)
        self.ax.set_xlim(-CELL_SIZE/2, GRID_WIDTH * CELL_SIZE - CELL_SIZE/2)
        self.ax.set_ylim(-CELL_SIZE/2, GRID_HEIGHT * CELL_SIZE - CELL_SIZE/2)
        self.ax.set_xticks(np.arange(0, (GRID_WIDTH+1)*CELL_SIZE, major_step))
        self.ax.set_yticks(np.arange(0, (GRID_HEIGHT+1)*CELL_SIZE, major_step))
        self.ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f} m"))
        self.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f} m"))
        self.ax.grid(True)
        self.ax.set_aspect("equal", adjustable="box")
        for r in river:
            self.ax.add_patch(plt.Rectangle((r[0]*CELL_SIZE, r[1]*CELL_SIZE), CELL_SIZE, CELL_SIZE, color='blue'))
        for f in forest:
            self.ax.add_patch(plt.Rectangle((f[0]*CELL_SIZE, f[1]*CELL_SIZE), CELL_SIZE, CELL_SIZE, color='green'))
        if self.friendly_units:
            outpost = self.friendly_units[0].state["outpost_position"]
            ox = outpost[0] * CELL_SIZE + CELL_SIZE/2
            oy = outpost[1] * CELL_SIZE + CELL_SIZE/2
            self.ax.plot(ox, oy, marker='*', markersize=12, color='magenta', label='Outpost')
        for enemy in self.enemy_units:
            if enemy.state["enemy_alive"]:
                ex, ey = enemy.state["enemy_position"]
                cx = ex * CELL_SIZE + CELL_SIZE / 2
                cy = ey * CELL_SIZE + CELL_SIZE / 2
                self.ax.plot(cx, cy, marker='s', markersize=12, color='green', label='Enemy')
                self.ax.text(cx + 40, cy + 40, enemy.state["name"], fontsize=8, color='black', zorder=5)
        for unit in self.friendly_units:
            pos = unit.state["position"]
            cx = pos[0] * CELL_SIZE + CELL_SIZE/2
            cy = pos[1] * CELL_SIZE + CELL_SIZE/2
            if unit.state.get("role") == "attacker":
                color = 'red'
            elif unit.state.get("role") == "outpost_securer":
                color = 'blue'
            elif unit.state.get("role") == "support":
                color = 'orange'
            elif unit.state.get("role") == "scout":
                color = 'cyan'
            else:
                color = 'gray'
            self.ax.plot(cx, cy, marker='o', markersize=12, color=color, zorder=5)
            self.ax.text(cx + 40, cy + 40, unit.name, fontsize=8, color='black', zorder=5)
        self.ax.set_title(f"Simulation Step {self.step_count} - {self.plan_name}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def step(self):
        self.step_count += 1
        if self.visualize:
            print(f"\n--- Simulation Step {self.step_count} ---")
        self.update_enemy_behavior()
        self.update_friendly_enemy_info()
        self.team_commander.share_tank_health()
        for unit in self.friendly_units:
            if ("AttackEnemy" in unit.current_plan and 
                not unit.state.get("target_enemy", {}).get("enemy_alive", False)):
                print(f"{unit.name} target enemy is dead or invalid; replanning.")
                unit.update_plan(force_replan=True)
            print(f"{unit.name} current plan: {unit.current_plan}, all_enemies_spotted: {unit.state.get('all_enemies_spotted', False)}")
            if "target_enemy" in unit.state and unit.state["target_enemy"]:
                print(f"{unit.name} targeting enemy: {unit.state['target_enemy']['name']} at {unit.state['target_enemy']['enemy_position']}")
            unit.execute_next_task()
            if self.visualize:
                print(f"{unit.name}'s current goal: {unit.get_goal_position()}")
        for unit in self.friendly_units:
            if not unit.state.get("target_enemy", {}).get("enemy_alive", False):
                new_target_state = select_enemy_for_unit(self.enemy_units, unit)
                if new_target_state is not None:
                    print(f"{unit.name} switching target to {new_target_state['name']}")
                    unit.state["target_enemy"] = new_target_state
                    unit.state["enemy"] = new_target_state
                    unit.update_plan(force_replan=True)
                else:
                    unit.state["target_enemy"] = {}
                    unit.state["enemy"] = {}
        if self.visualize:
            for unit in self.friendly_units:
                print(f"State for {unit.name}: {unit.state}")
            for enemy in self.enemy_units:
                print(f"State for {enemy.name}: {enemy.state}")

    def run(self, max_steps=200):
        self.step_count = 0
        self.team_commander.share_tank_health()
        for unit in self.friendly_units:
            unit.update_plan(force_replan=True)
        for enemy in self.enemy_units:
            enemy.update_plan(self.friendly_units)
        for _ in range(max_steps):
            if all(not e.state["enemy_alive"] for e in self.enemy_units) and any(u.state.get("outpost_secured", False) for u in self.friendly_units):
                if self.visualize:
                    self.update_plot()
                    plt.pause(0.05)
                    print("\nMission accomplished: Enemy destroyed and outpost secured!")
                return self.evaluate_plan()
            self.step()
            if self.visualize:
                self.update_plot()
                plt.pause(0.05)
            alive_friendlies = [u for u in self.friendly_units if u.state["friendly_health"] > 0]
            if len(alive_friendlies) < len(self.friendly_units):
                for dead in set(self.friendly_units) - set(alive_friendlies):
                    if self.visualize:
                        print(f"{dead.name} has been destroyed!")
                self.friendly_units = alive_friendlies
                if not self.friendly_units:
                    if self.visualize:
                        self.update_plot()
                        plt.pause(0.05)
                        print("\nAll friendly units have been destroyed! Mission failed.")
                    return self.evaluate_plan()
        if self.visualize:
            self.update_plot()
            plt.pause(0.05)
            print("\nMission incomplete after maximum steps.")
        return self.evaluate_plan()

###############################
# Main Simulation Setup - Mode 1 (Test Specific Plan)
###############################

if __name__ == "__main__":
    tank_state_template = {
        "position": (0, 0),
        "friendly_health": 20,
        "max_health": 20,
        "armor": 17,
        "friendly_accuracy": 0.75,
        "rate_of_fire": 4.9,
        "damage": 9,
        "suppression": 0.12,
        "penetration": 18,
        "friendly_attack_range": 2400 / CELL_SIZE,
        "role": "attacker",
        "all_enemies_spotted": False,
        "total_enemies": 2,
        "scout_steps": 0,
        "stealth_modifier": 0
    }
    infantry_state_template = {
        "position": (0, 1),
        "friendly_health": 1,
        "max_health": 1,
        "armor": 0,
        "friendly_accuracy": 0.50,
        "rate_of_fire": 294,
        "damage": 0.8,
        "suppression": 0.01,
        "penetration": 1,
        "friendly_attack_range": 1200 / CELL_SIZE,
        "role": "outpost_securer",
        "all_enemies_spotted": False,
        "total_enemies": 2,
        "scout_steps": 0,
        "stealth_modifier": 0
    }
    artillery_state_template = {
        "position": (1, 0),
        "friendly_health": 18,
        "max_health": 18,
        "armor": 2,
        "friendly_accuracy": 0.85,
        "rate_of_fire": 8.6,
        "damage": 3.5,
        "suppression": 15,
        "penetration": 1,
        "friendly_attack_range": 4000 / CELL_SIZE,
        "role": "support",
        "all_enemies_spotted": False,
        "total_enemies": 2,
        "scout_steps": 0,
        "stealth_modifier": 0
    }
    scout_state_template = {
        "position": (1, 1),
        "friendly_health": 18,
        "max_health": 18,
        "armor": 3,
        "friendly_accuracy": 0.35,
        "rate_of_fire": 115,
        "damage": 2.4,
        "suppression": 5,
        "penetration": 5,
        "vision_range": 2600 / CELL_SIZE,
        "friendly_attack_range": 1800 / CELL_SIZE,
        "role": "scout",
        "all_enemies_spotted": False,
        "candidate_positions": [],
        "current_candidate_index": 0,
        "total_enemies": 2,
        "scout_steps": 0,
        "stealth_modifier": 225
    }

    candidate_domain = secure_outpost_domain

    enemy_state1 = create_enemy_state(index=0)
    enemy_state2 = create_enemy_state(index=1)
    enemy_unit1 = EnemyTank("EnemyTank1", enemy_state1, {
        "DefendAreaMission": [
            (lambda state: any(manhattan(state["enemy_position"], u.state["position"]) <= state["enemy_attack_range"] and 
                                has_line_of_sight(state["enemy_position"], u.state["position"]) 
                                for u in state["friendly_units"]), ["AttackEnemy"]),
            (lambda state: any(manhattan(state["enemy_position"], u.state["position"]) <= state["vision_range"] and 
                                has_line_of_sight(state["enemy_position"], u.state["position"]) 
                                for u in state["friendly_units"]), ["ChaseTarget"]),
            (lambda state: True, ["Patrol"])
        ]
    })
    enemy_unit2 = EnemyTank("EnemyTank2", enemy_state2, {
        "DefendAreaMission": [
            (lambda state: any(manhattan(state["enemy_position"], u.state["position"]) <= state["enemy_attack_range"] and 
                                has_line_of_sight(state["enemy_position"], u.state["position"]) 
                                for u in state["friendly_units"]), ["AttackEnemy"]),
            (lambda state: any(manhattan(state["enemy_position"], u.state["position"]) <= state["vision_range"] and 
                                has_line_of_sight(state["enemy_position"], u.state["position"]) 
                                for u in state["friendly_units"]), ["ChaseTarget"]),
            (lambda state: True, ["Patrol"])
        ]
    })
    enemy_units = [enemy_unit1, enemy_unit2]

    tank_state = tank_state_template.copy()
    infantry_state = infantry_state_template.copy()
    artillery_state = artillery_state_template.copy()
    scout_state = scout_state_template.copy()

    for state in [tank_state, infantry_state, artillery_state, scout_state]:
        state["enemy"] = enemy_state1
        state["target_enemy"] = enemy_state1
        state["outpost_position"] = enemy_state1["outpost_position"]
        state["visible_enemies"] = []

    # Create friendly units first
    tank = FriendlyTank("FriendlyTank", tank_state, candidate_domain)
    infantry = FriendlyInfantry("FriendlyInfantry", infantry_state, candidate_domain)
    artillery = FriendlyArtillery("FriendlyArtillery", artillery_state, candidate_domain)
    scout = FriendlyScout("FriendlyScout", scout_state, candidate_domain)
    friendly_units = [tank, infantry, artillery, scout]

    # Initialize commander and simulation
    commander = TeamCommander(friendly_units)
    sim = Simulation(friendly_units, enemy_units, commander, visualize=True, plan_name="Mode1_Test")

    # Set simulation reference for units
    for unit in friendly_units:
        unit.sim = sim

    result = sim.run(max_steps=300)
    print("\n=== Plan Evaluation ===")
    print(f"Score: {result['score']:.1f}")
    print(f"Total Friendly Health Remaining: {result['friendly_health']}")
    print(f"Enemy Health Remaining: {result['enemy_health']}")
    print(f"Outpost Secured: {result['outpost_secured']}")
    print(f"Steps Taken: {result['steps_taken']}")

    plt.ioff()
    plt.show()
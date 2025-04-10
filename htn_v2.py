import random
import heapq
import matplotlib.pyplot as plt
import numpy as np

###############################
# Grid, Obstacles, and LOS (unchanged)
###############################

GRID_WIDTH = 10
GRID_HEIGHT = 10
obstacles = {(3, 3), (3, 4), (3, 5), (4, 5), (6, 6)}

def in_bounds(pos): return 0 <= pos[0] < GRID_WIDTH and 0 <= pos[1] < GRID_HEIGHT
def neighbors(pos): return [p for p in [(pos[0]+1, pos[1]), (pos[0]-1, pos[1]), (pos[0], pos[1]+1), (pos[0], pos[1]-1)] if in_bounds(p) and p not in obstacles]
def manhattan(p, q): return abs(p[0]-q[0]) + abs(p[1]-q[1])

def astar(start, goal):
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal: break
        for nxt in neighbors(current):
            new_cost = cost_so_far[current] + 1
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + manhattan(nxt, goal)
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current
    if goal not in came_from: return []
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
            if err < 0: y += sy; err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            line.append((x, y))
            err -= dx
            if err < 0: x += sx; err += dy
            y += sy
    line.append((x, y))
    return line

def has_line_of_sight(start, end):
    line = get_line(start, end)
    return all(pos not in obstacles for pos in line[1:-1])

###############################
# Helper Functions (unchanged)
###############################

def get_num_attacks(rpm):
    exact = rpm * 0.1
    n = int(exact)
    if random.random() < (exact - n): n += 1
    return n

def get_penetration_probability(D):
    if D <= -3: return 0.0
    elif -3 < D <= 0: return 0.33 + 0.11 * (D + 3)
    elif 0 < D <= 6: return 0.66 + (0.29/6) * D
    else: return 0.95

###############################
# Global Enemy State (unchanged)
###############################

def create_enemy_state():
    return {
        "enemy_position": (7, 7),
        "enemy_alive": True,
        "enemy_health": 20,
        "max_health": 20,
        "enemy_armor": 17,
        "outpost_position": (9, 0),
        "outpost_secured": False,
        "enemy_attack_range": 3,
        "enemy_accuracy": 0.7,
        "enemy_penetration": 18,
        "enemy_damage": 9,
        "enemy_suppression": 0.12,
        "rate_of_fire": 4.9,
        "suppression": {},
        "patrol_points": [(7, 7), (7, 5), (5, 5), (5, 7)],
        "current_patrol_index": 0,
        "detection_range": 5,
        "retreat_point": (9, 9)
    }

###############################
# HTN Domain and Planner
###############################

###############################
# HTN Domains for Friendly Units
###############################

def enemy_not_in_range(state):
    return any(e["alive"] and manhattan(state["position"], e["position"]) > state["friendly_attack_range"] 
               for e in state["enemies"])

def enemy_in_range(state):
    return any(e["alive"] and manhattan(state["position"], e["position"]) <= state["friendly_attack_range"] 
               for e in state["enemies"])

def enemy_dead_not_at_outpost(state):
    return (not any(e["alive"] for e in state["enemies"])) and (state["position"] != state["enemies"][0]["outpost_position"]) and (not state["enemies"][0]["outpost_secured"])

def enemy_dead_at_outpost(state):
    return (not any(e["alive"] for e in state["enemies"])) and (state["position"] == state["enemies"][0]["outpost_position"]) and (not state["enemies"][0]["outpost_secured"])

tank_plan1_domain = {
    "DestroyEnemyMission": [
        (enemy_not_in_range, ["MoveToEnemy", "AttackEnemy"]),
        (enemy_in_range, ["AttackEnemy"]),
        (lambda state: not any(e["alive"] for e in state["enemies"]), [])
    ]
}

infantry_plan1_domain = {
    "SecureOutpostMission": [
        (lambda state: state.get("tank_health", 20) / state.get("tank_max_health", 20) < 0.25 and any(e["alive"] for e in state["enemies"]) and enemy_not_in_range(state),
         ["MoveToEnemy", "AttackEnemy"]),
        (lambda state: state.get("tank_health", 20) / state.get("tank_max_health", 20) < 0.25 and enemy_in_range(state),
         ["AttackEnemy"]),
        (lambda state: state["position"] != state["enemies"][0]["outpost_position"] and not state["enemies"][0]["outpost_secured"],
         ["MoveToOutpost", "SecureOutpost"]),
        (lambda state: state["position"] == state["enemies"][0]["outpost_position"] and not state["enemies"][0]["outpost_secured"],
         ["SecureOutpost"])
    ]
}

both_plan2_domain = {
    "EngageThenSecureMission": [
        (enemy_not_in_range, ["MoveToEnemy", "AttackEnemy"]),
        (enemy_in_range, ["AttackEnemy"]),
        (enemy_dead_not_at_outpost, ["MoveToOutpost", "SecureOutpost"]),
        (enemy_dead_at_outpost, ["SecureOutpost"])
    ]
}

artillery_plan_domain = {
    "SupportMission": [
        (enemy_not_in_range, ["MoveToEnemy", "AttackEnemy"]),
        (enemy_in_range, ["AttackEnemy"]),
        (lambda state: not any(e["alive"] for e in state["enemies"]), [])
    ]
}

###############################
# HTN Domains for Enemy Units
###############################

def enemy_has_target_in_range(state):
    for unit in state["friendly_units"]:
        dist = manhattan(state["position"], unit.state["position"])
        if dist <= state["attack_range"] and has_line_of_sight(state["position"], unit.state["position"]):
            return True
    return False

def enemy_has_target_in_detection(state):
    for unit in state["friendly_units"]:
        dist = manhattan(state["position"], unit.state["position"])
        if dist <= state["detection_range"] and has_line_of_sight(state["position"], unit.state["position"]):
            return True
    return False

def enemy_low_health(state):
    return state["health"] / state["max_health"] < 0.25  # Updated from "enemy_health" to "health"

enemy_plan_domain = {
    "DefendAreaMission": [
        (enemy_low_health, ["Retreat"]),
        (enemy_has_target_in_range, ["AttackTarget"]),
        (enemy_has_target_in_detection, ["ChaseTarget"]),
        (lambda state: True, ["Patrol"])
    ]
}

class HTNPlanner:
    def __init__(self, domain): self.domain = domain
    def plan(self, task, state):
        if task not in self.domain: return [task]
        for condition, subtasks in self.domain[task]:
            if condition(state):
                plan = []
                for subtask in subtasks:
                    sub_plan = self.plan(subtask, state)
                    if sub_plan is None: break
                    plan.extend(sub_plan)
                else: return plan
        return None

###############################
# Team Commander (modified to share tank health)
###############################

class TeamCommander:
    def __init__(self, friendly_units): 
        self.friendly_units = friendly_units
    
    def assign_roles(self):
        for unit in self.friendly_units:
            if isinstance(unit, FriendlyTank):
                unit.state["role"] = "attacker"
            elif isinstance(unit, FriendlyInfantry):
                unit.state["role"] = "outpost_securer"
            elif isinstance(unit, FriendlyArtillery):
                unit.state["role"] = "support"
    
    def communicate_enemy_position(self):
        for unit in self.friendly_units:
            for enemy in unit.state["enemies"]:
                if has_line_of_sight(unit.state["position"], enemy["position"]):
                    observed = enemy["position"]
                    for other in self.friendly_units:
                        for e in other.state["enemies"]:
                            if e["position"] == enemy["position"]:
                                e["position"] = observed
                    break
    
    def share_tank_health(self):
        tank = next((unit for unit in self.friendly_units if isinstance(unit, FriendlyTank)), None)
        if tank:
            tank_health = tank.state["friendly_health"]
            tank_max_health = tank.state["max_health"]
            for unit in self.friendly_units:
                unit.state["tank_health"] = tank_health
                unit.state["tank_max_health"] = tank_max_health

###############################
# Friendly Unit Classes (unchanged)
###############################

class FriendlyUnit:
    def __init__(self, name, state, domain):
        self.name = name
        self.state = state
        self.planner = HTNPlanner(domain)
        self.current_plan = []
        self.last_enemy_pos = state["enemies"][0]["position"]
        self.last_health = state["friendly_health"]

    def update_plan(self, force_replan=False):
        mission = "DestroyEnemyMission" if isinstance(self, FriendlyTank) and self.planner.domain == tank_plan1_domain else \
                  "SecureOutpostMission" if isinstance(self, FriendlyInfantry) and self.planner.domain == infantry_plan1_domain else \
                  "EngageThenSecureMission" if self.planner.domain == both_plan2_domain else \
                  "SupportMission" if isinstance(self, FriendlyArtillery) else "UnknownMission"
        if force_replan and self.planner.domain == infantry_plan1_domain and isinstance(self, FriendlyInfantry):
            new_plan = self.planner.plan(mission, self.state)
            self.current_plan = new_plan if new_plan else []
            print(f"{self.name} replanned due to tank health < 25%: {self.current_plan}")
        elif not self.current_plan:
            new_plan = self.planner.plan(mission, self.state)
            self.current_plan = new_plan if new_plan else []
            print(f"{self.name} replanned: {self.current_plan}")
        else:
            print(f"{self.name} current plan: {self.current_plan}")
        self.last_enemy_pos = min((e["position"] for e in self.state["enemies"] if e["alive"]), 
                                 key=lambda p: manhattan(self.state["position"], p), default=self.state["enemies"][0]["position"])
        self.last_health = self.state["friendly_health"]

    def should_replan(self):
        if self.planner.domain == infantry_plan1_domain and isinstance(self, FriendlyInfantry):
            tank_health_low = self.state.get("tank_health", 20) / self.state.get("tank_max_health", 20) < 0.25
            return tank_health_low
        return False

    def attack_enemy(self):
        target_enemy = min((e for e in self.state["enemies"] if e["alive"]), 
                           key=lambda e: manhattan(self.state["position"], e["position"]), default=None)
        if target_enemy:
            enemy_pos = target_enemy["position"]
            if manhattan(self.state["position"], enemy_pos) <= self.state["friendly_attack_range"] and has_line_of_sight(self.state["position"], enemy_pos):
                num_attacks = get_num_attacks(self.state["rate_of_fire"])
                suppression_on_self = self.state.get("suppression_from_enemy", 0)
                effective_accuracy = max(0, self.state["friendly_accuracy"] - suppression_on_self)
                for _ in range(num_attacks):
                    if random.random() < effective_accuracy:
                        D = self.state["penetration"] - target_enemy.get("armor", 0)
                        if random.random() < get_penetration_probability(D):
                            target_enemy["health"] -= self.state["damage"]
                            if target_enemy["health"] <= 0: 
                                target_enemy["alive"] = False
                                print(f"{self.name} destroyed an enemy at {enemy_pos}!")
                        target_enemy["suppression"][self.name] = target_enemy["suppression"].get(self.name, 0) + self.state["suppression"]

    def execute_next_task(self):
        if self.current_plan:
            task = self.current_plan[0]
            if task == "MoveToEnemy":
                current_pos = self.state["position"]
                goal_pos = self.get_goal_position()
                next_pos = next_step(current_pos, goal_pos)
                self.state["position"] = next_pos
                if any(manhattan(next_pos, e["position"]) <= self.state["friendly_attack_range"] and 
                       has_line_of_sight(next_pos, e["position"]) for e in self.state["enemies"] if e["alive"]):
                    self.current_plan.pop(0)
            elif task == "AttackEnemy":
                self.attack_enemy()
                self.current_plan.pop(0)
            elif task == "MoveToOutpost":
                current_pos = self.state["position"]
                goal_pos = self.state["enemies"][0]["outpost_position"]
                next_pos = next_step(current_pos, goal_pos)
                self.state["position"] = next_pos
                if next_pos == goal_pos:
                    self.current_plan.pop(0)
            elif task == "SecureOutpost":
                if self.state["position"] == self.state["enemies"][0]["outpost_position"]:
                    self.state["enemies"][0]["outpost_secured"] = True
                    self.current_plan.pop(0)
                else:
                    self.current_plan.pop(0)

    def get_goal_position(self):
        if isinstance(self, FriendlyTank) and self.planner.domain == tank_plan1_domain:
            alive_enemies = [e for e in self.state["enemies"] if e["alive"]]
            if not alive_enemies:
                return self.state["position"]
            closest_enemy = min(alive_enemies, key=lambda e: manhattan(self.state["position"], e["position"]))
            if (manhattan(self.state["position"], closest_enemy["position"]) <= self.state["friendly_attack_range"] and 
                has_line_of_sight(self.state["position"], closest_enemy["position"])):
                return self.state["position"]
            return closest_enemy["position"]
        elif isinstance(self, FriendlyInfantry) and self.planner.domain == infantry_plan1_domain:
            if self.state.get("tank_health", 20) / self.state.get("tank_max_health", 20) < 0.25 and any(e["alive"] for e in self.state["enemies"]):
                alive_enemies = [e for e in self.state["enemies"] if e["alive"]]
                return min(alive_enemies, key=lambda e: manhattan(self.state["position"], e["position"]))["position"]
            return self.state["enemies"][0]["outpost_position"]
        elif isinstance(self, FriendlyArtillery):
            alive_enemies = [e for e in self.state["enemies"] if e["alive"]]
            if not alive_enemies:
                return self.state["position"]
            closest_enemy = min(alive_enemies, key=lambda e: manhattan(self.state["position"], e["position"]))
            if (manhattan(self.state["position"], closest_enemy["position"]) <= self.state["friendly_attack_range"] and 
                has_line_of_sight(self.state["position"], closest_enemy["position"])):
                return self.state["position"]
            return closest_enemy["position"]
        else:
            alive_enemies = [e for e in self.state["enemies"] if e["alive"]]
            if not alive_enemies:
                return self.state["enemies"][0]["outpost_position"]
            return min(alive_enemies, key=lambda e: manhattan(self.state["position"], e["position"]))["position"]

class FriendlyTank(FriendlyUnit): pass
class FriendlyInfantry(FriendlyUnit): pass
class FriendlyArtillery(FriendlyUnit): pass

class EnemyUnit:
    def __init__(self, name, state, domain, friendly_units):
        self.name = name
        self.state = state
        self.planner = HTNPlanner(domain)
        self.friendly_units = friendly_units
        self.current_plan = []

    def update_plan(self):
        mission = "DefendAreaMission"
        combined_state = self.state.copy()
        combined_state["friendly_units"] = self.friendly_units
        new_plan = self.planner.plan(mission, combined_state)
        self.current_plan = new_plan if new_plan else []
        print(f"{self.name} replanned: {self.current_plan}")

    def execute_next_task(self):
        if self.current_plan and self.state["alive"]:
            task = self.current_plan[0]
            if task == "Patrol":
                idx = self.state["current_patrol_index"]
                target = self.state["patrol_points"][idx]
                self.state["position"] = next_step(self.state["position"], target)
                if self.state["position"] == target:
                    self.state["current_patrol_index"] = (idx + 1) % len(self.state["patrol_points"])
                if self.state["position"] == target:
                    self.current_plan.pop(0)
            elif task == "ChaseTarget":
                target = min(self.friendly_units, key=lambda u: manhattan(self.state["position"], u.state["position"])).state["position"]
                self.state["position"] = next_step(self.state["position"], target)
                if enemy_has_target_in_range(self.state | {"friendly_units": self.friendly_units}):
                    self.current_plan.pop(0)
            elif task == "AttackTarget":
                target_unit = None
                min_dist = float('inf')
                for unit in self.friendly_units:
                    dist = manhattan(self.state["position"], unit.state["position"])
                    if dist <= self.state["attack_range"] and has_line_of_sight(self.state["position"], unit.state["position"]):
                        if dist < min_dist:
                            min_dist = dist
                            target_unit = unit
                if target_unit:
                    num_attacks = get_num_attacks(self.state["rate_of_fire"])
                    total_suppression = sum(self.state["suppression_received"].values()) if "suppression_received" in self.state else 0
                    effective_accuracy = max(0, self.state["accuracy"] - total_suppression)
                    for _ in range(num_attacks):
                        if random.random() < effective_accuracy:
                            D = self.state["penetration"] - target_unit.state.get("armor", 0)
                            if random.random() < get_penetration_probability(D):
                                target_unit.state["friendly_health"] -= self.state["damage"]
                                if target_unit.state["friendly_health"] <= 0:
                                    print(f"{target_unit.name} destroyed by {self.name}!")
                            target_unit.state["suppression_from_enemy"] = target_unit.state.get("suppression_from_enemy", 0) + self.state["suppression"]
                self.current_plan.pop(0)
            elif task == "Retreat":
                retreat_point = self.state.get("retreat_point", (9, 9))
                self.state["position"] = next_step(self.state["position"], retreat_point)
                if self.state["position"] == retreat_point:
                    self.current_plan.pop(0)

    def get_goal_position(self):
        if not self.current_plan:
            return self.state["position"]
        task = self.current_plan[0]
        if task == "Patrol":
            idx = self.state["current_patrol_index"]
            return self.state["patrol_points"][idx]
        elif task == "ChaseTarget" or task == "AttackTarget":
            return min(self.friendly_units, key=lambda u: manhattan(self.state["position"], u.state["position"])).state["position"]
        elif task == "Retreat":
            return self.state.get("retreat_point", (9, 9))
        return self.state["position"]

class EnemyTank(EnemyUnit): pass

###############################
# Enemy Attack (unchanged)
###############################

def enemy_attack(target_state, effective_accuracy):
    num_attacks = get_num_attacks(target_state["enemy"]["rate_of_fire"])
    for _ in range(num_attacks):
        if random.random() < effective_accuracy:
            D = target_state["enemy"]["enemy_penetration"] - target_state.get("armor", 0)
            if random.random() < get_penetration_probability(D):
                target_state["friendly_health"] -= target_state["enemy"]["enemy_damage"]
            target_state["suppression_from_enemy"] = target_state.get("suppression_from_enemy", 0) + target_state["enemy"]["enemy_suppression"]


###############################
# Team Commander
###############################

class TeamCommander:
    def __init__(self, friendly_units): 
        self.friendly_units = friendly_units
    
    def assign_roles(self):
        for unit in self.friendly_units:
            if isinstance(unit, FriendlyTank):
                unit.state["role"] = "attacker"
            elif isinstance(unit, FriendlyInfantry):
                unit.state["role"] = "outpost_securer"
            elif isinstance(unit, FriendlyArtillery):
                unit.state["role"] = "support"
    
    def communicate_enemy_position(self):
        for unit in self.friendly_units:
            for enemy in unit.state["enemies"]:
                if has_line_of_sight(unit.state["position"], enemy["position"]):
                    observed = enemy["position"]
                    for other in self.friendly_units:
                        for e in other.state["enemies"]:
                            if e["position"] == enemy["position"]:
                                e["position"] = observed
                    break
    
    def share_tank_health(self):
        tank = next((unit for unit in self.friendly_units if isinstance(unit, FriendlyTank)), None)
        if tank:
            tank_health = tank.state["friendly_health"]
            tank_max_health = tank.state["max_health"]
            for unit in self.friendly_units:
                unit.state["tank_health"] = tank_health
                unit.state["tank_max_health"] = tank_max_health

###############################
# Friendly Unit Classes
###############################

class FriendlyUnit:
    def __init__(self, name, state, domain):
        self.name = name
        self.state = state
        self.planner = HTNPlanner(domain)
        self.current_plan = []
        self.last_enemy_pos = state["enemies"][0]["position"]
        self.last_health = state["friendly_health"]

    def update_plan(self, force_replan=False):
        mission = "DestroyEnemyMission" if isinstance(self, FriendlyTank) and self.planner.domain == tank_plan1_domain else \
                  "SecureOutpostMission" if isinstance(self, FriendlyInfantry) and self.planner.domain == infantry_plan1_domain else \
                  "EngageThenSecureMission" if self.planner.domain == both_plan2_domain else \
                  "SupportMission" if isinstance(self, FriendlyArtillery) else "UnknownMission"
        if force_replan and self.planner.domain == infantry_plan1_domain and isinstance(self, FriendlyInfantry):
            new_plan = self.planner.plan(mission, self.state)
            self.current_plan = new_plan if new_plan else []
            print(f"{self.name} replanned due to tank health < 25%: {self.current_plan}")
        elif not self.current_plan:
            new_plan = self.planner.plan(mission, self.state)
            self.current_plan = new_plan if new_plan else []
            print(f"{self.name} replanned: {self.current_plan}")
        else:
            print(f"{self.name} current plan: {self.current_plan}")
        self.last_enemy_pos = min((e["position"] for e in self.state["enemies"] if e["alive"]), 
                                 key=lambda p: manhattan(self.state["position"], p), default=self.state["enemies"][0]["position"])
        self.last_health = self.state["friendly_health"]

    def should_replan(self):
        if self.planner.domain == infantry_plan1_domain and isinstance(self, FriendlyInfantry):
            tank_health_low = self.state.get("tank_health", 20) / self.state.get("tank_max_health", 20) < 0.25
            return tank_health_low
        return False

    def attack_enemy(self):
        target_enemy = min((e for e in self.state["enemies"] if e["alive"]), 
                           key=lambda e: manhattan(self.state["position"], e["position"]), default=None)
        if target_enemy:
            enemy_pos = target_enemy["position"]
            if manhattan(self.state["position"], enemy_pos) <= self.state["friendly_attack_range"] and has_line_of_sight(self.state["position"], enemy_pos):
                num_attacks = get_num_attacks(self.state["rate_of_fire"])
                suppression_on_self = self.state.get("suppression_from_enemy", 0)
                effective_accuracy = max(0, self.state["friendly_accuracy"] - suppression_on_self)
                for _ in range(num_attacks):
                    if random.random() < effective_accuracy:
                        D = self.state["penetration"] - target_enemy.get("armor", 0)
                        if random.random() < get_penetration_probability(D):
                            target_enemy["health"] -= self.state["damage"]
                            if target_enemy["health"] <= 0: 
                                target_enemy["alive"] = False
                                print(f"{self.name} destroyed an enemy at {enemy_pos}!")
                        # Ensure target_enemy["suppression"] is a dict and update it
                        if "suppression" not in target_enemy or not isinstance(target_enemy["suppression"], dict):
                            target_enemy["suppression"] = {}
                        target_enemy["suppression"][self.name] = target_enemy["suppression"].get(self.name, 0) + self.state["suppression"]

    def execute_next_task(self):
        if self.current_plan:
            task = self.current_plan[0]
            if task == "MoveToEnemy":
                current_pos = self.state["position"]
                goal_pos = self.get_goal_position()
                next_pos = next_step(current_pos, goal_pos)
                self.state["position"] = next_pos
                if any(manhattan(next_pos, e["position"]) <= self.state["friendly_attack_range"] and 
                       has_line_of_sight(next_pos, e["position"]) for e in self.state["enemies"] if e["alive"]):
                    self.current_plan.pop(0)
            elif task == "AttackEnemy":
                self.attack_enemy()
                self.current_plan.pop(0)
            elif task == "MoveToOutpost":
                current_pos = self.state["position"]
                goal_pos = self.state["enemies"][0]["outpost_position"]
                next_pos = next_step(current_pos, goal_pos)
                self.state["position"] = next_pos
                if next_pos == goal_pos:
                    self.current_plan.pop(0)
            elif task == "SecureOutpost":
                if self.state["position"] == self.state["enemies"][0]["outpost_position"]:
                    self.state["enemies"][0]["outpost_secured"] = True
                    self.current_plan.pop(0)
                else:
                    self.current_plan.pop(0)

    def get_goal_position(self):
        if isinstance(self, FriendlyTank) and self.planner.domain == tank_plan1_domain:
            alive_enemies = [e for e in self.state["enemies"] if e["alive"]]
            if not alive_enemies:
                return self.state["position"]
            closest_enemy = min(alive_enemies, key=lambda e: manhattan(self.state["position"], e["position"]))
            if (manhattan(self.state["position"], closest_enemy["position"]) <= self.state["friendly_attack_range"] and 
                has_line_of_sight(self.state["position"], closest_enemy["position"])):
                return self.state["position"]
            return closest_enemy["position"]
        elif isinstance(self, FriendlyInfantry) and self.planner.domain == infantry_plan1_domain:
            if self.state.get("tank_health", 20) / self.state.get("tank_max_health", 20) < 0.25 and any(e["alive"] for e in self.state["enemies"]):
                alive_enemies = [e for e in self.state["enemies"] if e["alive"]]
                return min(alive_enemies, key=lambda e: manhattan(self.state["position"], e["position"]))["position"]
            return self.state["enemies"][0]["outpost_position"]
        elif isinstance(self, FriendlyArtillery):
            alive_enemies = [e for e in self.state["enemies"] if e["alive"]]
            if not alive_enemies:
                return self.state["position"]
            closest_enemy = min(alive_enemies, key=lambda e: manhattan(self.state["position"], e["position"]))
            if (manhattan(self.state["position"], closest_enemy["position"]) <= self.state["friendly_attack_range"] and 
                has_line_of_sight(self.state["position"], closest_enemy["position"])):
                return self.state["position"]
            return closest_enemy["position"]
        else:
            alive_enemies = [e for e in self.state["enemies"] if e["alive"]]
            if not alive_enemies:
                return self.state["enemies"][0]["outpost_position"]
            return min(alive_enemies, key=lambda e: manhattan(self.state["position"], e["position"]))["position"]

class FriendlyTank(FriendlyUnit): pass
class FriendlyInfantry(FriendlyUnit): pass
class FriendlyArtillery(FriendlyUnit): pass

###############################
# Enemy Unit Classes
###############################

class EnemyUnit:
    def __init__(self, name, state, domain, friendly_units):
        self.name = name
        self.state = state
        self.planner = HTNPlanner(domain)
        self.friendly_units = friendly_units
        self.current_plan = []

    def update_plan(self):
        mission = "DefendAreaMission"
        combined_state = self.state.copy()
        combined_state["friendly_units"] = self.friendly_units
        new_plan = self.planner.plan(mission, combined_state)
        self.current_plan = new_plan if new_plan else []
        print(f"{self.name} replanned: {self.current_plan}")

    def execute_next_task(self):
        if self.current_plan and self.state["alive"]:
            task = self.current_plan[0]
            if task == "Patrol":
                idx = self.state["current_patrol_index"]
                target = self.state["patrol_points"][idx]
                self.state["position"] = next_step(self.state["position"], target)
                if self.state["position"] == target:
                    self.state["current_patrol_index"] = (idx + 1) % len(self.state["patrol_points"])
                if self.state["position"] == target:
                    self.current_plan.pop(0)
            elif task == "ChaseTarget":
                target = min(self.friendly_units, key=lambda u: manhattan(self.state["position"], u.state["position"])).state["position"]
                self.state["position"] = next_step(self.state["position"], target)
                if enemy_has_target_in_range(self.state | {"friendly_units": self.friendly_units}):
                    self.current_plan.pop(0)
            elif task == "AttackTarget":
                target_unit = None
                min_dist = float('inf')
                for unit in self.friendly_units:
                    dist = manhattan(self.state["position"], unit.state["position"])
                    if dist <= self.state["attack_range"] and has_line_of_sight(self.state["position"], unit.state["position"]):
                        if dist < min_dist:
                            min_dist = dist
                            target_unit = unit
                if target_unit:
                    num_attacks = get_num_attacks(self.state["rate_of_fire"])
                    total_suppression = sum(self.state["suppression_received"].values()) if "suppression_received" in self.state else 0
                    effective_accuracy = max(0, self.state["accuracy"] - total_suppression)
                    for _ in range(num_attacks):
                        if random.random() < effective_accuracy:
                            D = self.state["penetration"] - target_unit.state.get("armor", 0)
                            if random.random() < get_penetration_probability(D):
                                target_unit.state["friendly_health"] -= self.state["damage"]
                                if target_unit.state["friendly_health"] <= 0:
                                    print(f"{target_unit.name} destroyed by {self.name}!")
                            target_unit.state["suppression_from_enemy"] = target_unit.state.get("suppression_from_enemy", 0) + self.state["suppression"]
                self.current_plan.pop(0)
            elif task == "Retreat":
                retreat_point = self.state.get("retreat_point", (9, 9))
                self.state["position"] = next_step(self.state["position"], retreat_point)
                if self.state["position"] == retreat_point:
                    self.current_plan.pop(0)

    def get_goal_position(self):
        if not self.current_plan:
            return self.state["position"]
        task = self.current_plan[0]
        if task == "Patrol":
            idx = self.state["current_patrol_index"]
            return self.state["patrol_points"][idx]
        elif task == "ChaseTarget" or task == "AttackTarget":
            return min(self.friendly_units, key=lambda u: manhattan(self.state["position"], u.state["position"])).state["position"]
        elif task == "Retreat":
            return self.state.get("retreat_point", (9, 9))
        return self.state["position"]

class EnemyTank(EnemyUnit): pass

###############################
# Simulation Class
###############################

class Simulation:
    def __init__(self, friendly_units, enemy_units, team_commander, visualize=False, plan_name="Unknown Plan"):
        self.enemy_states = [unit.state for unit in enemy_units]  # Shared enemy states
        self.friendly_units = friendly_units
        for unit in self.friendly_units:
            unit.state["enemies"] = self.enemy_states
        self.enemy_units = enemy_units
        self.team_commander = team_commander
        self.step_count = 0
        self.visualize = visualize
        self.plan_name = plan_name
        if self.visualize:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6,6))

    def update_enemy_behavior(self):
        for enemy_unit in self.enemy_units:
            if enemy_unit.state["alive"]:
                enemy_unit.update_plan()
                enemy_unit.execute_next_task()
                if self.visualize:
                    goal = enemy_unit.get_goal_position()
                    print(f"{enemy_unit.name}'s current A* goal: {goal}")

    def evaluate_plan(self):
        total_friendly_health = sum(unit.state["friendly_health"] for unit in self.friendly_units)
        max_friendly_health = sum(unit.state["max_health"] for unit in self.friendly_units)
        total_enemy_health = sum(enemy["health"] if enemy["alive"] else 0 for enemy in self.enemy_states)
        max_enemy_health = sum(enemy["max_health"] for enemy in self.enemy_states)
        outpost_secured = self.enemy_states[0]["outpost_secured"]
        steps_taken = self.step_count
        
        score = (total_friendly_health / max_friendly_health * 20) - (total_enemy_health / max_enemy_health * 20) + \
                (10 if outpost_secured else -10) - 0.1 * steps_taken

        return {
            "score": score,
            "friendly_health": total_friendly_health,
            "enemy_health": total_enemy_health,
            "outpost_secured": outpost_secured,
            "steps_taken": steps_taken
        }

    def update_plot(self):
        self.ax.clear()
        self.ax.set_xlim(-1, GRID_WIDTH)
        self.ax.set_ylim(-1, GRID_HEIGHT)
        self.ax.set_xticks(range(GRID_WIDTH))
        self.ax.set_yticks(range(GRID_HEIGHT))
        self.ax.grid(True)
        for obs in obstacles:
            self.ax.add_patch(plt.Rectangle(obs, 1, 1, color='black'))
        outpost = self.enemy_states[0]["outpost_position"]
        self.ax.plot(outpost[0]+0.5, outpost[1]+0.5, marker='*', markersize=15, color='magenta', label='Outpost')
        for i, enemy in enumerate(self.enemy_units):
            if enemy.state["alive"]:
                enemy_pos = enemy.state["position"]
                self.ax.plot(enemy_pos[0]+0.5, enemy_pos[1]+0.5, marker='s', markersize=12, color='green', label=f'{enemy.name}')
                frac = enemy.state["health"] / enemy.state["max_health"]
                bar_width = 0.8 * frac
                self.ax.add_patch(plt.Rectangle((enemy_pos[0]+0.1, enemy_pos[1]+0.8), bar_width, 0.1, color='green'))
                self.ax.add_patch(plt.Rectangle((enemy_pos[0]+0.1, enemy_pos[1]+0.8), 0.8, 0.1, fill=False, edgecolor='black'))
        for unit in self.friendly_units:
            pos = unit.state["position"]
            color = 'red' if unit.state.get("role") == "attacker" else 'blue' if unit.state.get("role") == "outpost_securer" else 'yellow'
            markersize = 12 if unit.state.get("armor", 0) > 2 else 10 if unit.state.get("armor", 0) > 0 else 8
            self.ax.plot(pos[0]+0.5, pos[1]+0.5, marker='o', markersize=markersize, color=color)
            self.ax.text(pos[0]+0.2, pos[1]+0.2, unit.name, fontsize=9, color='black')
            max_hp = unit.state["max_health"]
            hp = unit.state["friendly_health"]
            frac = hp / max_hp if max_hp > 0 else 0
            bar_width = 0.8 * frac
            self.ax.add_patch(plt.Rectangle((pos[0]+0.1, pos[1]+0.8), bar_width, 0.1, color='green'))
            self.ax.add_patch(plt.Rectangle((pos[0]+0.1, pos[1]+0.8), 0.8, 0.1, fill=False, edgecolor='black'))
        self.ax.set_title(f"Simulation Step {self.step_count} - {self.plan_name}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run(self, max_steps=100):
        self.step_count = 0
        self.team_commander.share_tank_health()
        for unit in self.friendly_units:
            unit.update_plan(force_replan=True)
        for enemy_unit in self.enemy_units:
            enemy_unit.update_plan()
        for _ in range(max_steps):
            if not any(enemy["alive"] for enemy in self.enemy_states) and self.enemy_states[0]["outpost_secured"]:
                if self.visualize: 
                    self.update_plot()
                    plt.pause(0.5)
                    print("\nMission accomplished: All enemies destroyed and outpost secured!")
                return self.evaluate_plan()
            self.step()
            if self.visualize: 
                self.update_plot()
                plt.pause(0.5)
            for unit in self.friendly_units:
                if unit.state["friendly_health"] <= 0:
                    if self.visualize: 
                        self.update_plot()
                        plt.pause(0.5)
                        print(f"\n{unit.name} has been destroyed! Mission failed.")
                    return self.evaluate_plan()
        if self.visualize: 
            self.update_plot()
            plt.pause(0.5)
            print("\nMission incomplete after maximum steps.")
        return self.evaluate_plan()

    def step(self):
        self.step_count += 1
        if self.visualize: print(f"\n--- Simulation Step {self.step_count} ---")
        self.team_commander.assign_roles()
        self.team_commander.communicate_enemy_position()
        self.team_commander.share_tank_health()
        self.update_enemy_behavior()

        outpost_pos = self.enemy_states[0]["outpost_position"]
        friendly_at_outpost = any(unit.state["position"] == outpost_pos for unit in self.friendly_units)
        if not friendly_at_outpost and self.enemy_states[0]["outpost_secured"]:
            self.enemy_states[0]["outpost_secured"] = False
            if self.visualize: print(f"Outpost no longer secured - no friendly units present.")

        for unit in self.friendly_units:
            if unit.should_replan():
                if self.visualize: print(f"{unit.name} triggered replanning due to tank health < 25%.")
                unit.update_plan(force_replan=True)
            else:
                unit.update_plan()
            unit.execute_next_task()
            goal = unit.get_goal_position()
            if self.visualize:
                if goal:
                    print(f"{unit.name}'s current A* goal: {goal}")
                else:
                    print(f"{unit.name} has no A* goal.")
        if self.visualize:
            for unit in self.friendly_units:
                print(f"State for {unit.name}: {unit.state}")
            for enemy_unit in self.enemy_units:
                print(f"State for {enemy_unit.name}: {enemy_unit.state}")

###############################
# Main Simulation Setup
###############################

if __name__ == "__main__":
    # Friendly unit templates
    tank_state_template = {
        "position": (0, 0), "friendly_health": 20, "max_health": 20, "armor": 17, "friendly_accuracy": 0.75,
        "rate_of_fire": 4.9, "damage": 9, "suppression": 0.12, "penetration": 18, "friendly_attack_range": 3, "role": "attacker"
    }
    infantry_state_template = {
        "position": (0, 1), "friendly_health": 1, "max_health": 1, "armor": 0, "friendly_accuracy": 0.50,
        "rate_of_fire": 294, "damage": 0.8, "suppression": 0.01, "penetration": 1, "friendly_attack_range": 2, "role": "outpost_securer"
    }
    artillery_state_template = {
        "position": (1, 1), "friendly_health": 18, "max_health": 18, "armor": 2, "friendly_accuracy": 0.85,
        "rate_of_fire": 8.6, "damage": 3.5, "suppression": 15, "penetration": 15, "friendly_attack_range": 8, "role": "support"
    }

    # Enemy unit templates
    enemy_tank_template_1 = {
        "position": (7, 7), "health": 20, "max_health": 20, "armor": 17, "accuracy": 0.7,
        "rate_of_fire": 4.9, "damage": 9, "suppression": 0.12, "penetration": 18, "attack_range": 3,
        "alive": True, "outpost_position": (9, 0), "outpost_secured": False,
        "patrol_points": [(7, 7), (7, 5), (5, 5), (5, 7)], "current_patrol_index": 0,
        "detection_range": 5, "retreat_point": (9, 9), "suppression_received": {}
    }
    enemy_tank_template_2 = {
        "position": (5, 6), "health": 20, "max_health": 20, "armor": 17, "accuracy": 0.7,
        "rate_of_fire": 4.9, "damage": 9, "suppression": 0.12, "penetration": 18, "attack_range": 3,
        "alive": True, "outpost_position": (9, 0), "outpost_secured": False,
        "patrol_points": [(5, 6), (6, 6), (5, 5), (4, 6)], "current_patrol_index": 0,
        "detection_range": 5, "retreat_point": (9, 9), "suppression_received": {}
    }

    mode = input("Enter mode (1: Test Plan 1, 2: Test Plan 2, 3: Compare Plans): ")

    if mode in ["1", "2"]:
        plan_name = "Plan 1" if mode == "1" else "Plan 2"
        domain_tank = tank_plan1_domain if mode == "1" else both_plan2_domain
        domain_infantry = infantry_plan1_domain if mode == "1" else both_plan2_domain
        domain_artillery = artillery_plan_domain
        
        print(f"\nTesting {plan_name} with Visualization and Debug Output...")
        # Create enemy units first
        enemy_tank_1_state = enemy_tank_template_1.copy()
        enemy_tank_2_state = enemy_tank_template_2.copy()
        enemy_units = [
            EnemyTank("EnemyTank1", enemy_tank_1_state, enemy_plan_domain, []),
            EnemyTank("EnemyTank2", enemy_tank_2_state, enemy_plan_domain, [])
        ]
        enemy_states = [unit.state for unit in enemy_units]

        # Add enemy states to friendly unit states
        tank_state = tank_state_template.copy()
        tank_state["enemies"] = enemy_states
        infantry_state = infantry_state_template.copy()
        infantry_state["enemies"] = enemy_states
        artillery_state = artillery_state_template.copy()
        artillery_state["enemies"] = enemy_states
        
        tank = FriendlyTank("FriendlyTank", tank_state, domain_tank)
        infantry = FriendlyInfantry("FriendlyInfantry", infantry_state, domain_infantry)
        artillery = FriendlyArtillery("FriendlyArtillery", artillery_state, domain_artillery)
        friendly_units = [tank, infantry, artillery]

        # Update enemy units with friendly units reference
        for enemy_unit in enemy_units:
            enemy_unit.friendly_units = friendly_units

        sim = Simulation(friendly_units, enemy_units, TeamCommander(friendly_units), visualize=True, plan_name=plan_name)
        result = sim.run()

        print("\n=== Plan Evaluation ===")
        print(f"Score: {result['score']:.1f}")
        print(f"Total Friendly Health Remaining: {result['friendly_health']:.1f}/{sum(unit.state['max_health'] for unit in sim.friendly_units)}")
        print(f"Total Enemy Health Remaining: {result['enemy_health']:.1f}/{sum(enemy.state['max_health'] for enemy in sim.enemy_units)}")
        print(f"Outpost Secured: {result['outpost_secured']}")
        print(f"Number of Steps Taken: {result['steps_taken']}")

    elif mode == "3":
        plan1_scores = []
        plan2_scores = []
        num_runs = 10

        for i in range(num_runs):
            # Plan 1
            enemy_tank_1_state1 = enemy_tank_template_1.copy()
            enemy_tank_2_state1 = enemy_tank_template_2.copy()
            enemy_units1 = [
                EnemyTank("EnemyTank1", enemy_tank_1_state1, enemy_plan_domain, []),
                EnemyTank("EnemyTank2", enemy_tank_2_state1, enemy_plan_domain, [])
            ]
            enemy_states1 = [unit.state for unit in enemy_units1]

            tank_state1 = tank_state_template.copy()
            tank_state1["enemies"] = enemy_states1
            infantry_state1 = infantry_state_template.copy()
            infantry_state1["enemies"] = enemy_states1
            artillery_state1 = artillery_state_template.copy()
            artillery_state1["enemies"] = enemy_states1
            tank1 = FriendlyTank("FriendlyTank", tank_state1, tank_plan1_domain)
            infantry1 = FriendlyInfantry("FriendlyInfantry", infantry_state1, infantry_plan1_domain)
            artillery1 = FriendlyArtillery("FriendlyArtillery", artillery_state1, artillery_plan_domain)
            friendly_units1 = [tank1, infantry1, artillery1]
            for enemy_unit in enemy_units1:
                enemy_unit.friendly_units = friendly_units1
            sim1 = Simulation(friendly_units1, enemy_units1, TeamCommander(friendly_units1), visualize=False, plan_name="Plan 1")
            result1 = sim1.run()
            plan1_scores.append(result1)
            print(f"Plan 1, Run {i+1}: Score = {result1['score']:.1f}")

            # Plan 2
            enemy_tank_2_state1 = enemy_tank_template_1.copy()
            enemy_tank_2_state2 = enemy_tank_template_2.copy()
            enemy_units2 = [
                EnemyTank("EnemyTank1", enemy_tank_2_state1, enemy_plan_domain, []),
                EnemyTank("EnemyTank2", enemy_tank_2_state2, enemy_plan_domain, [])
            ]
            enemy_states2 = [unit.state for unit in enemy_units2]

            tank_state2 = tank_state_template.copy()
            tank_state2["enemies"] = enemy_states2
            infantry_state2 = infantry_state_template.copy()
            infantry_state2["enemies"] = enemy_states2
            artillery_state2 = artillery_state_template.copy()
            artillery_state2["enemies"] = enemy_states2
            tank2 = FriendlyTank("FriendlyTank", tank_state2, both_plan2_domain)
            infantry2 = FriendlyInfantry("FriendlyInfantry", infantry_state2, both_plan2_domain)
            artillery2 = FriendlyArtillery("FriendlyArtillery", artillery_state2, artillery_plan_domain)
            friendly_units2 = [tank2, infantry2, artillery2]
            for enemy_unit in enemy_units2:
                enemy_unit.friendly_units = friendly_units2
            sim2 = Simulation(friendly_units2, enemy_units2, TeamCommander(friendly_units2), visualize=False, plan_name="Plan 2")
            result2 = sim2.run()
            plan2_scores.append(result2)
            print(f"Plan 2, Run {i+1}: Score = {result2['score']:.1f}")

        avg_plan1_score = np.mean([run["score"] for run in plan1_scores])
        avg_plan2_score = np.mean([run["score"] for run in plan2_scores])

        print("\n=== Simulation Results ===")
        print(f"Plan 1 Average Score: {avg_plan1_score:.1f}")
        print(f"Plan 2 Average Score: {avg_plan2_score:.1f}")

        best_plan = "Plan 1" if avg_plan1_score > avg_plan2_score else "Plan 2"
        best_domain_tank = tank_plan1_domain if best_plan == "Plan 1" else both_plan2_domain
        best_domain_infantry = infantry_plan1_domain if best_plan == "Plan 1" else both_plan2_domain
        best_domain_artillery = artillery_plan_domain

        print(f"\nRunning Best Plan ({best_plan}) with Visualization...")
        enemy_tank_best_1_state = enemy_tank_template_1.copy()
        enemy_tank_best_2_state = enemy_tank_template_2.copy()
        enemy_units_best = [
            EnemyTank("EnemyTank1", enemy_tank_best_1_state, enemy_plan_domain, []),
            EnemyTank("EnemyTank2", enemy_tank_best_2_state, enemy_plan_domain, [])
        ]
        enemy_states_best = [unit.state for unit in enemy_units_best]

        tank_state_best = tank_state_template.copy()
        tank_state_best["enemies"] = enemy_states_best
        infantry_state_best = infantry_state_template.copy()
        infantry_state_best["enemies"] = enemy_states_best
        artillery_state_best = artillery_state_template.copy()
        artillery_state_best["enemies"] = enemy_states_best
        tank_best = FriendlyTank("FriendlyTank", tank_state_best, best_domain_tank)
        infantry_best = FriendlyInfantry("FriendlyInfantry", infantry_state_best, best_domain_infantry)
        artillery_best = FriendlyArtillery("FriendlyArtillery", artillery_state_best, best_domain_artillery)
        friendly_units_best = [tank_best, infantry_best, artillery_best]
        for enemy_unit in enemy_units_best:
            enemy_unit.friendly_units = friendly_units_best
        sim_best = Simulation(friendly_units_best, enemy_units_best, TeamCommander(friendly_units_best), visualize=True, plan_name=best_plan)
        result_best = sim_best.run()

        print("\n=== Best Plan Evaluation ===")
        print(f"Score: {result_best['score']:.1f}")
        print(f"Total Friendly Health Remaining: {result_best['friendly_health']:.1f}/{sum(unit.state['max_health'] for unit in sim_best.friendly_units)}")
        print(f"Total Enemy Health Remaining: {result_best['enemy_health']:.1f}/{sum(enemy.state['max_health'] for enemy in sim_best.enemy_units)}")
        print(f"Outpost Secured: {result_best['outpost_secured']}")
        print(f"Number of Steps Taken: {result_best['steps_taken']}")
        if result_best['friendly_health'] <= 0:
            print("Evaluation: Mission failed - all friendly units destroyed.")
        elif not any(enemy["alive"] for enemy in sim_best.enemy_states) and result_best["outpost_secured"]:
            print("Evaluation: Mission succeeded - all enemies destroyed and outpost secured!")
        else:
            print("Evaluation: Mission incomplete - check enemy status and outpost.")

    else:
        print("Invalid mode selected. Please enter 1, 2, or 3.")

    plt.ioff()
    plt.show()
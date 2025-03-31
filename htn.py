import heapq
import math
import copy
import random

# ----- Debug Output Helper -----
VERBOSE = True
def debug_print(message, depth=0):
    if VERBOSE:
        indent = "  " * depth
        print(f"{indent}{message}")

# ----- Helper Functions: A* Pathfinding and LOS -----
def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    def neighbors(pos):
        (x, y) = pos
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                yield (nx, ny)
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
    visited = set()
    while open_set:
        est, cost, current, path = heapq.heappop(open_set)
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)
        for nxt in neighbors(current):
            if nxt not in visited:
                new_cost = cost + 1
                heapq.heappush(open_set, (new_cost + heuristic(nxt, goal), new_cost, nxt, path + [nxt]))
    return None

def line_of_sight(grid, start, goal):
    x0, y0 = start; x1, y1 = goal
    dx = abs(x1 - x0); dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1; sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            if grid[x][y] == 1:
                return False
            err -= dy
            if err < 0:
                y += sy; err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            if grid[x][y] == 1:
                return False
            err -= dx
            if err < 0:
                x += sx; err += dy
            y += sy
    return grid[x1][y1] == 0

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# ----- Basic HTN Framework -----
class Task:
    def __init__(self, name, args=None):
        self.name = name
        self.args = args if args is not None else []
    def __repr__(self):
        return f"{self.name}({', '.join(map(str, self.args))})"

class Operator:
    def __init__(self, name, precond, effect):
        self.name = name
        self.precond = precond    # function(state, *args) -> bool
        self.effect = effect      # function(state, *args) -> new state
    def applicable(self, state, *args):
        return self.precond(state, *args)
    def apply(self, state, *args):
        return self.effect(state, *args)

class Method:
    def __init__(self, task_name, precond, subtasks_fn):
        self.task_name = task_name
        self.precond = precond     # function(state, *args) -> bool
        self.subtasks_fn = subtasks_fn  # function(state, *args) -> list of Tasks
    def applicable(self, state, *args):
        return self.precond(state, *args)
    def decompose(self, state, *args):
        return self.subtasks_fn(state, *args)

def seek_plan(state, tasks, operators, methods, depth=0):
    debug_print(f"seek_plan with tasks: {tasks}", depth)
    if not tasks:
        debug_print("No tasks left; returning []", depth)
        return []
    task = tasks[0]
    rest = tasks[1:]
    debug_print(f"Processing: {task}", depth)
    if task.name in operators:
        op = operators[task.name]
        debug_print(f"Found operator for {task.name}; checking preconditions...", depth)
        if op.applicable(state, *task.args):
            debug_print(f"Preconditions met for {task}", depth)
            new_state = op.apply(state, *task.args)
            plan_rest = seek_plan(new_state, rest, operators, methods, depth+1)
            if plan_rest is not None:
                return [task] + plan_rest
            else:
                debug_print(f"Backtracking from {task}", depth)
        else:
            debug_print(f"Preconditions failed for {task}", depth)
        return None
    if task.name in methods:
        debug_print(f"Found method for compound task {task.name}; trying methods...", depth)
        for m in methods[task.name]:
            if m.applicable(state, *task.args):
                debug_print(f"Method applicable for {task.name}; decomposing...", depth)
                subtasks = m.decompose(state, *task.args)
                debug_print(f"Decomposed {task} into {subtasks}", depth)
                new_tasks = subtasks + rest
                plan = seek_plan(state, new_tasks, operators, methods, depth+1)
                if plan is not None:
                    return plan
                else:
                    debug_print(f"Method for {task} failed; trying next method...", depth)
        return None
    debug_print(f"No operator/method for {task.name}", depth)
    return None

# ----- Domain Setup with Extended Attributes -----
initial_state = {
    "grid": [
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0]
    ],
    "units": {
        "tank": {
            "pos": (0,0),
            "health": 100,
            "fuel": 100,
            "attack_range": 3,
            "accuracy": 0.8,
            "damage": 40,
            "suppression_per_hit": 15
        },
        "infantry": {
            "pos": (0,1),
            "health": 100,
            "fuel": 100,
            "attack_range": 2,
            "accuracy": 0.6,
            "damage": 20,
            "suppression_per_hit": 10
        }
    },
    "enemy": {
        "type": "tank",
        "pos": (2,2),
        "health": 100,
        "fuel": 100,
        "attack_range": 3,
        "accuracy": 0.7,
        "damage": 35,
        "suppression_per_hit": 10
    },
    "outpost": (4,4)
}

damage_values = {
    "tank": 40,
    "infantry": 20,
    "enemy_tank": 35
}

# ----- Friendly Operators (with fuel cost and extended attributes) -----
def precond_move(state, unit_name, destination):
    unit = state["units"].get(unit_name)
    if not unit:
        return False
    if not astar(state["grid"], unit["pos"], destination):
        debug_print(f"No path from {unit['pos']} to {destination}")
        return False
    if unit["fuel"] < 10:
        debug_print(f"{unit_name} lacks sufficient fuel ({unit['fuel']}) to move.")
        return False
    return True

def effect_move(state, unit_name, destination):
    new_state = copy.deepcopy(state)
    unit = new_state["units"][unit_name]
    old_pos = unit["pos"]
    unit["pos"] = destination
    unit["fuel"] -= 10  # Fuel cost per move
    debug_print(f"{unit_name} moved from {old_pos} to {destination} (fuel left: {unit['fuel']})")
    return new_state

move_op = Operator("move", precond_move, effect_move)

def precond_attack(state, unit_name):
    e = state["enemy"]
    if e["health"] <= 0:
        return False
    if unit_name not in state["units"]:
        return False
    unit = state["units"][unit_name]
    pos = unit["pos"]
    if not line_of_sight(state["grid"], pos, e["pos"]):
        debug_print(f"{unit_name} has no line-of-sight to enemy at {e['pos']}.")
        return False
    if manhattan(pos, e["pos"]) > unit["attack_range"]:
        debug_print(f"{unit_name} is out of range (range: {unit['attack_range']}) from enemy at {e['pos']}.")
        return False
    return True

def effect_attack(state, unit_name):
    new_state = copy.deepcopy(state)
    unit = new_state["units"][unit_name]
    e = new_state["enemy"]
    hit_roll = random.random()
    debug_print(f"{unit_name} attack roll: {hit_roll:.2f} (accuracy: {unit['accuracy']})")
    if hit_roll <= unit["accuracy"]:
        e["health"] -= unit["damage"]
        debug_print(f"{unit_name} hit enemy at {state['enemy']['pos']} for {unit['damage']} damage (enemy HP: {e['health']})")
        debug_print(f"Suppression applied: {unit['suppression_per_hit']}")
    else:
        debug_print(f"{unit_name} missed the enemy.")
    return new_state

attack_op = Operator("attack", precond_attack, effect_attack)

operators = {
    "move": move_op,
    "attack": attack_op
}

# ----- Methods for Friendly HTN Tasks -----
# Updated secure_outpost method: only add tasks for conditions that are not yet met.
def precond_secure_outpost(st):
    return True

def subtasks_secure_outpost(st):
    tasks = []
    # Add neutralize_enemy task if enemy still has health.
    if st["enemy"]["health"] > 0:
        tasks.append(Task("neutralize_enemy"))
    # Add occupy_outpost if infantry is not at the outpost.
    if st["units"].get("infantry", {}).get("pos") != st["outpost"]:
        tasks.append(Task("occupy_outpost"))
    return tasks

secure_outpost_m = Method("secure_outpost", precond_secure_outpost, subtasks_secure_outpost)

def precond_neutralize_enemy(st):
    return st["enemy"]["health"] > 0

def subtasks_neutralize_enemy(st):
    if attack_op.applicable(st, "tank"):
        return [Task("attack", ["tank"])]
    else:
        return [Task("move", ["tank", st["enemy"]["pos"]]),
                Task("attack", ["tank"])]
        
neutralize_enemy_m = Method("neutralize_enemy", precond_neutralize_enemy, subtasks_neutralize_enemy)

def precond_occupy_outpost(st):
    inf = st["units"].get("infantry")
    return inf is not None and (inf["pos"] != st["outpost"])

def subtasks_occupy_outpost(st):
    return [Task("move", ["infantry", st["outpost"]])]

occupy_outpost_m = Method("occupy_outpost", precond_occupy_outpost, subtasks_occupy_outpost)

methods = {
    "secure_outpost": [secure_outpost_m],
    "neutralize_enemy": [neutralize_enemy_m],
    "occupy_outpost": [occupy_outpost_m]
}

# ----- Simulation Loop (Friendly Only) -----
state = copy.deepcopy(initial_state)
friendly_tasks = [Task("secure_outpost")]
max_steps = 50
step = 0
while step < max_steps:
    debug_print(f"\n--- Simulation Step {step} ---")
    
    f_plan = seek_plan(state, friendly_tasks, operators, methods)
    if f_plan:
        action = f_plan[0]
        if action.name in operators:
            state = operators[action.name].apply(state, *action.args)
        friendly_tasks = f_plan[1:] if f_plan[1:] else [Task("secure_outpost")]
    else:
        debug_print("No friendly plan found.")
    
    # Termination: mission complete when enemy is neutralized AND infantry is at outpost.
    if state["enemy"]["health"] <= 0 and state["units"].get("infantry", {}).get("pos") == state["outpost"]:
        debug_print("Mission complete: enemy neutralized and outpost secured.")
        break
    if not state["units"]:
        debug_print("All friendly units defeated.")
        break
    step += 1

debug_print("\nFinal State:")
print("Final State:", state)

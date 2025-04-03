import heapq
import math
import copy
import random

# ----- Debug Output Helper -----
VERBOSE = True  # Set to False to reduce verbosity
def debug_print(message, depth=0):
    if VERBOSE:
        indent = "  " * depth
        print(f"{indent}{message}")

# ----- Helper Functions: A* Pathfinding, LOS, Manhattan -----
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
    x0, y0 = start
    x1, y1 = goal
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    rows, cols = len(grid), len(grid[0])
    
    while True:
        if not (0 <= x0 < rows and 0 <= y0 < cols) or grid[x0][y0] == 1:
            return False
        if x0 == x1 and y0 == y1:
            return True
        e2 = 2 * err
        if e2 > -dy:
            if sx != 0 and 0 <= x0 + sx < rows:
                if (x0 == 1 and 0 <= y0 < cols and grid[1][y0] == 1) or \
                   (x0 == 2 and 0 <= y0 < cols and grid[1][y0] == 1):
                    return False
            err -= dy
            x0 += sx
        if e2 < dx:
            if sy != 0 and 0 <= x0 < rows:
                if (x0 == 2 and 0 <= y0 < cols and grid[1][y0] == 1) or \
                   (x0 == 0 and 0 <= y0 < cols and grid[1][y0] == 1):
                    return False
            err += dx
            y0 += sy

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def find_reposition_target(state, unit_name):
    grid = state["grid"]
    enemy_pos = state["enemy"]["pos"]
    unit = state["units"][unit_name]
    best = None
    best_dist = float('inf')
    rows = len(grid)
    cols = len(grid[0])
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0 or (i, j) == enemy_pos and state["enemy"]["health"] > 0:
                continue
            if astar(grid, unit["pos"], (i, j)) is None:
                continue
            if not line_of_sight(grid, (i, j), enemy_pos):
                continue
            d = manhattan((i, j), enemy_pos)
            if d > unit["attack_range"]:
                continue
            if d < best_dist:
                best_dist = d
                best = (i, j)
    return best

# ----- HTN Framework Classes -----
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

# ----- Enumerate Candidate Plans -----
def enumerate_plans(state, tasks, operators, methods, depth=0):
    if not tasks:
        return [[]]
    candidate_plans = []
    task = tasks[0]
    rest = tasks[1:]
    if task.name in operators:
        op = operators[task.name]
        if op.applicable(state, *task.args):
            new_state = op.apply(state, *task.args)
            for sp in enumerate_plans(new_state, rest, operators, methods, depth+1):
                candidate_plans.append([task] + sp)
    if task.name in methods:
        for m in methods[task.name]:
            if m.applicable(state, *task.args):
                subtasks = m.decompose(state, *task.args)
                new_tasks = subtasks + rest
                for sp in enumerate_plans(state, new_tasks, operators, methods, depth+1):
                    candidate_plans.append(sp)
    return candidate_plans

# ----- Domain Setup with Extended Attributes and Coordination Flag -----
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
            "fuel": 1000,
            "attack_range": 3,
            "accuracy": 0.8,
            "damage": 40,
            "suppression_per_hit": 15
        },
        "infantry": {
            "pos": (0,1),
            "health": 100,
            "fuel": 1000,
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
    "outpost": (4,4),
    "support_fired": False,
    "tank_waited": False
}

# ----- Friendly Operators (Cell-by-Cell Movement) -----
def precond_move(state, unit_name, destination):
    unit = state["units"].get(unit_name)
    if not unit:
        return False
    path = astar(state["grid"], unit["pos"], destination)
    if not path:
        debug_print(f"No path from {unit['pos']} to {destination}")
        return False
    if unit["fuel"] < 10:
        debug_print(f"{unit_name} lacks fuel ({unit['fuel']})")
        return False
    return True

def effect_move(state, unit_name, destination):
    new_state = copy.deepcopy(state)
    unit = new_state["units"][unit_name]
    current_pos = unit["pos"]
    path = astar(new_state["grid"], current_pos, destination)
    if path and len(path) > 1:
        next_cell = path[1]
    else:
        next_cell = destination
    unit["pos"] = next_cell
    unit["fuel"] -= 10
    debug_print(f"{unit_name} moved from {current_pos} to {next_cell} (fuel left: {unit['fuel']})")
    return new_state

move_op = Operator("move", precond_move, effect_move)

def precond_attack(state, unit_name):
    e = state["enemy"]
    if e["health"] <= 0 or unit_name not in state["units"]:
        return False
    unit = state["units"][unit_name]
    pos = unit["pos"]
    if not line_of_sight(state["grid"], pos, e["pos"]):
        debug_print(f"{unit_name} has no LOS to enemy at {e['pos']}.")
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
        if unit_name == "tank":
            new_state["support_fired"] = False
    else:
        debug_print(f"{unit_name} missed the enemy.")
    return new_state

attack_op = Operator("attack", precond_attack, effect_attack)

def precond_support_fire(state, unit_name):
    unit = state["units"].get(unit_name)
    if not unit:
        return False
    enemy_pos = state["enemy"]["pos"]
    if not line_of_sight(state["grid"], unit["pos"], enemy_pos):
        debug_print(f"{unit_name} has no LOS for support fire to {enemy_pos}.")
        return False
    if manhattan(unit["pos"], enemy_pos) > unit["attack_range"]:
        debug_print(f"{unit_name} is out of range (range: {unit['attack_range']}) for support fire to {enemy_pos}.")
        return False
    if state.get("support_fired", False):
        debug_print("Support fire already provided this cycle.")
        return False
    return True

def effect_support_fire(state, unit_name):
    new_state = copy.deepcopy(state)
    enemy = new_state["enemy"]
    enemy["accuracy"] = max(0, enemy["accuracy"] - 0.1)
    new_state["support_fired"] = True
    debug_print(f"{unit_name} provided support fire. Enemy accuracy reduced to {enemy['accuracy']:.2f}")
    return new_state

support_fire_op = Operator("support_fire", precond_support_fire, effect_support_fire)

# Wait operator for the tank.
def precond_wait(state, unit_name):
    if unit_name == "tank":
        if not state.get("support_fired", False):
            return True
        return False
    return True

def effect_wait(state, unit_name):
    new_state = copy.deepcopy(state)
    if unit_name == "tank":
        new_state["tank_waited"] = True
    debug_print(f"{unit_name} waited.")
    return new_state

wait_op = Operator("wait", precond_wait, effect_wait)

# Reposition operator for infantry using dynamic search.
def precond_reposition(state, unit_name):
    unit = state["units"].get(unit_name)
    if not unit or unit["fuel"] < 10:
        return False
    target = find_reposition_target(state, unit_name)
    if target is None or unit["pos"] == target:
        debug_print(f"No valid reposition target or already at target for {unit_name}.")
        return False
    return True

def effect_reposition(state, unit_name):
    new_state = copy.deepcopy(state)
    unit = new_state["units"][unit_name]
    target = find_reposition_target(new_state, unit_name)
    if target is None:
        debug_print(f"No reposition target found for {unit_name}; not repositioning.")
        return new_state
    current_pos = unit["pos"]
    path = astar(new_state["grid"], current_pos, target)
    if path and len(path) > 1:
        next_cell = path[1]  # Move one step
        unit["pos"] = next_cell
        unit["fuel"] -= 10
        new_state["tank_waited"] = False
        debug_print(f"{unit_name} repositioned from {current_pos} to {next_cell} toward {target} (fuel left: {unit['fuel']})")
    return new_state

reposition_op = Operator("reposition", precond_reposition, effect_reposition)

def find_reposition_target(state, unit_name):
    grid = state["grid"]
    enemy_pos = state["enemy"]["pos"]
    unit = state["units"][unit_name]
    best = None
    best_dist = float('inf')
    rows = len(grid)
    cols = len(grid[0])
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0:
                continue
            if astar(grid, unit["pos"], (i,j)) is None:
                continue
            if not line_of_sight(grid, (i,j), enemy_pos):
                continue
            d = manhattan((i,j), enemy_pos)
            if d < best_dist:
                best = (i,j)
                best_dist = d
    return best

reposition_op = Operator("reposition", precond_reposition, effect_reposition)

def precond_move_to_safe_position(state, unit_name):
    unit = state["units"].get(unit_name)
    if not unit or unit["fuel"] < 10:
        return False
    enemy_pos = state["enemy"]["pos"]
    grid = state["grid"]
    rows, cols = len(grid), len(grid[0])
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and (i, j) != unit["pos"] and astar(grid, unit["pos"], (i, j)) is not None:
                dist = manhattan((i, j), enemy_pos)
                if dist > state["enemy"]["attack_range"]:  # Must be > 3
                    return True
    return False

def effect_move_to_safe_position(state, unit_name):
    new_state = copy.deepcopy(state)
    unit = new_state["units"][unit_name]
    enemy_pos = new_state["enemy"]["pos"]
    grid = new_state["grid"]
    rows, cols = len(grid), len(grid[0])
    best_pos = None
    best_dist = float('inf')
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and (i, j) != unit["pos"]:
                path = astar(grid, unit["pos"], (i, j))
                if path is None:
                    continue
                dist = manhattan((i, j), enemy_pos)
                if dist > state["enemy"]["attack_range"]:
                    if dist < best_dist:  # Closest safe position
                        best_pos = (i, j)
                        best_dist = dist
    if best_pos:
        path = astar(grid, unit["pos"], best_pos)
        next_cell = path[1] if len(path) > 1 else best_pos
        unit["pos"] = next_cell
        unit["fuel"] -= 10
        debug_print(f"{unit_name} moved to safe position {next_cell} (fuel left: {unit['fuel']})")
    return new_state

move_to_safe_op = Operator("move_to_safe_position", precond_move_to_safe_position, effect_move_to_safe_position)

# ----- Operators Dictionary -----
operators = {
    "move": move_op,
    "attack": attack_op,
    "support_fire": support_fire_op,
    "wait": wait_op,
    "reposition": reposition_op,
    "move_to_safe_position": move_to_safe_op
}

# ----- Methods for Friendly HTN Tasks -----
def precond_secure_outpost(st):
    return True

def subtasks_secure_outpost_original(st):
    tasks = []
    if st["enemy"]["health"] > 0:
        tasks.append(Task("neutralize_enemy"))
    if st["units"].get("infantry", {}).get("pos") != st["outpost"]:
        tasks.append(Task("occupy_outpost"))
    return tasks

secure_outpost_original = Method("secure_outpost", precond_secure_outpost, subtasks_secure_outpost_original)

def precond_neutralize_enemy_with_support(st):
    return st["enemy"]["health"] > 0

def find_optimal_position(state, unit_name, enemy_pos):
    grid = state["grid"]
    unit = state["units"][unit_name]
    current_pos = unit["pos"]
    attack_range = unit["attack_range"]
    rows, cols = len(grid), len(grid[0])
    best_pos = None
    min_dist_to_current = float('inf')
    
    for i in range(rows):
        for j in range(cols):
            pos = (i, j)
            if grid[i][j] != 0 or pos == enemy_pos:  # Skip obstacles and enemy pos
                continue
            if not line_of_sight(grid, pos, enemy_pos):
                continue
            dist_to_enemy = manhattan(pos, enemy_pos)
            if dist_to_enemy > attack_range:
                continue
            dist_to_current = manhattan(pos, current_pos)
            path = astar(grid, current_pos, pos)
            if path and dist_to_current < min_dist_to_current:
                min_dist_to_current = dist_to_current
                best_pos = pos
    
    return best_pos

def subtasks_neutralize_enemy_with_support(st):
    tasks = []
    enemy_pos = st["enemy"]["pos"]  # (2,2)
    tank_pos = st["units"]["tank"]["pos"]
    infantry_pos = st["units"]["infantry"]["pos"]
    tank_range = st["units"]["tank"]["attack_range"]  # 3
    infantry_range = st["units"]["infantry"]["attack_range"]  # 2

    # Find optimal positions
    tank_target = find_optimal_position(st, "tank", enemy_pos)
    infantry_target = find_optimal_position(st, "infantry", enemy_pos)

    # Move tank to its target if not already in range with LOS
    if tank_target and (not line_of_sight(st["grid"], tank_pos, enemy_pos) or 
                        manhattan(tank_pos, enemy_pos) > tank_range):
        path = astar(st["grid"], tank_pos, tank_target)
        if path:
            for i in range(1, len(path)):
                next_pos = path[i]
                tasks.append(Task("move", ["tank", tank_target]))
                # Stop moving if in range and has LOS
                if (line_of_sight(st["grid"], next_pos, enemy_pos) and 
                    manhattan(next_pos, enemy_pos) <= tank_range):
                    break

    # Move infantry to its target if not already in range with LOS
    if infantry_target and (not line_of_sight(st["grid"], infantry_pos, enemy_pos) or 
                           manhattan(infantry_pos, enemy_pos) > infantry_range):
        path = astar(st["grid"], infantry_pos, infantry_target)
        if path:
            for i in range(1, len(path)):
                next_pos = path[i]
                tasks.append(Task("move", ["infantry", infantry_target]))
                if (line_of_sight(st["grid"], next_pos, enemy_pos) and 
                    manhattan(next_pos, enemy_pos) <= infantry_range):
                    break

    # Add support fire and attack if both units are in position
    tank_in_position = (line_of_sight(st["grid"], tank_pos, enemy_pos) and 
                        manhattan(tank_pos, enemy_pos) <= tank_range)
    infantry_in_position = (line_of_sight(st["grid"], infantry_pos, enemy_pos) and 
                           manhattan(infantry_pos, enemy_pos) <= infantry_range)
    
    if tank_in_position and infantry_in_position:
        tasks.append(Task("support_fire", ["infantry"]))
        enemy_hp = st["enemy"]["health"]
        tank_dmg = st["units"]["tank"]["damage"]
        attacks_needed = (enemy_hp + tank_dmg - 1) // tank_dmg
        tasks.extend([Task("attack", ["tank"]) for _ in range(attacks_needed)])

    return tasks

neutralize_enemy_with_support = Method(
    "neutralize_enemy_with_support",
    precond_neutralize_enemy_with_support,
    subtasks_neutralize_enemy_with_support
)
        
neutralize_enemy_with_support = Method("neutralize_enemy_with_support", precond_neutralize_enemy_with_support, subtasks_neutralize_enemy_with_support)

def precond_neutralize_enemy_with_support_wait(st):
    return (st["enemy"]["health"] > 0 and 
            (not line_of_sight(st["grid"], st["units"]["infantry"]["pos"], st["enemy"]["pos"]) or 
             manhattan(st["units"]["tank"]["pos"], st["enemy"]["pos"]) > st["enemy"]["attack_range"]))

def subtasks_neutralize_enemy_with_support_wait(st):
    tasks = []
    enemy_pos = st["enemy"]["pos"]  # (2,2)
    tank_pos = st["units"]["tank"]["pos"]
    infantry_pos = st["units"]["infantry"]["pos"]
    tank_range = st["units"]["tank"]["attack_range"]  # 3
    infantry_range = st["units"]["infantry"]["attack_range"]  # 2
    enemy_range = st["enemy"]["attack_range"]  # 3

    # Find optimal positions within range and with LOS
    tank_target = find_optimal_position(st, "tank", enemy_pos)
    infantry_target = find_optimal_position(st, "infantry", enemy_pos)

    # Tank: Move to a safe position (out of enemy range) if needed, then to attack position
    if (not line_of_sight(st["grid"], tank_pos, enemy_pos) or 
        manhattan(tank_pos, enemy_pos) > tank_range or 
        manhattan(tank_pos, enemy_pos) <= enemy_range):
        if manhattan(tank_pos, enemy_pos) <= enemy_range:  # If in enemy range, move to safe first
            safe_target = None
            rows, cols = len(st["grid"]), len(st["grid"][0])
            for i in range(rows):
                for j in range(cols):
                    pos = (i, j)
                    if (st["grid"][i][j] == 0 and pos != enemy_pos and 
                        manhattan(pos, enemy_pos) > enemy_range and 
                        astar(st["grid"], tank_pos, pos)):
                        safe_target = pos
                        break
                if safe_target:
                    break
            if safe_target:
                path = astar(st["grid"], tank_pos, safe_target)
                for i in range(1, len(path)):
                    next_pos = path[i]
                    tasks.append(Task("move", ["tank", safe_target]))
                    if manhattan(next_pos, enemy_pos) > enemy_range:
                        break
                tank_pos = path[-1]  # Update for next step planning

        # Move to attack position if not already in range with LOS
        if tank_target and (not line_of_sight(st["grid"], tank_pos, enemy_pos) or 
                            manhattan(tank_pos, enemy_pos) > tank_range):
            path = astar(st["grid"], tank_pos, tank_target)
            if path:
                for i in range(1, len(path)):
                    next_pos = path[i]
                    tasks.append(Task("move", ["tank", tank_target]))
                    if (line_of_sight(st["grid"], next_pos, enemy_pos) and 
                        manhattan(next_pos, enemy_pos) <= tank_range):
                        break

    # Tank waits if support hasn’t fired
    if not st["support_fired"]:
        tasks.append(Task("wait", ["tank"]))

    # Infantry: Move to attack position if not already in range with LOS
    if infantry_target and (not line_of_sight(st["grid"], infantry_pos, enemy_pos) or 
                           manhattan(infantry_pos, enemy_pos) > infantry_range):
        path = astar(st["grid"], infantry_pos, infantry_target)
        if path:
            for i in range(1, len(path)):
                next_pos = path[i]
                tasks.append(Task("move", ["infantry", infantry_target]))
                if (line_of_sight(st["grid"], next_pos, enemy_pos) and 
                    manhattan(next_pos, enemy_pos) <= infantry_range):
                    break

    # Add support fire and attack if both units are in position
    tank_in_position = (line_of_sight(st["grid"], tank_pos, enemy_pos) and 
                        manhattan(tank_pos, enemy_pos) <= tank_range)
    infantry_in_position = (line_of_sight(st["grid"], infantry_pos, enemy_pos) and 
                           manhattan(infantry_pos, enemy_pos) <= infantry_range)
    
    if tank_in_position and infantry_in_position:
        tasks.append(Task("support_fire", ["infantry"]))
        enemy_hp = st["enemy"]["health"]
        tank_dmg = st["units"]["tank"]["damage"]
        attacks_needed = (enemy_hp + tank_dmg - 1) // tank_dmg
        tasks.extend([Task("attack", ["tank"]) for _ in range(attacks_needed)])

    return tasks


neutralize_enemy_with_support_wait = Method("neutralize_enemy_with_support", precond_neutralize_enemy_with_support_wait, subtasks_neutralize_enemy_with_support_wait)


def precond_occupy_outpost(st):
    inf = st["units"].get("infantry")
    return inf is not None and (inf["pos"] != st["outpost"])

def subtasks_occupy_outpost(st):
    return [Task("move", ["infantry", st["outpost"]])]

occupy_outpost_m = Method("occupy_outpost", precond_occupy_outpost, subtasks_occupy_outpost)

def subtasks_secure_outpost_coordinated(st):
    tasks = []
    if st["enemy"]["health"] > 0:
        tasks.append(Task("neutralize_enemy_with_support"))
    if st["units"]["tank"]["pos"] != st["outpost"]:
        path = astar(st["grid"], st["units"]["tank"]["pos"], st["outpost"])
        if path:
            for i in range(1, len(path)):
                tasks.append(Task("move", ["tank", st["outpost"]]))
    if st["units"]["infantry"]["pos"] != st["outpost"]:
        path = astar(st["grid"], st["units"]["infantry"]["pos"], st["outpost"])
        if path:
            for i in range(1, len(path)):
                tasks.append(Task("move", ["infantry", st["outpost"]]))
    return tasks

secure_outpost_coordinated = Method("secure_outpost", precond_secure_outpost, subtasks_secure_outpost_coordinated)

methods = {
    "secure_outpost": [secure_outpost_original, secure_outpost_coordinated],
    "neutralize_enemy": [],
    "neutralize_enemy_with_support": [neutralize_enemy_with_support, neutralize_enemy_with_support_wait],
    "occupy_outpost": [occupy_outpost_m]
}

# ----- Main Simulation Loop (Using a Current Plan) -----
state = copy.deepcopy(initial_state)
friendly_tasks = [Task("secure_outpost")]
max_steps = 50
step = 0
executed_plan = []
current_plan = []  # Holds the candidate plan once generated

while step < max_steps:
    debug_print(f"\n--- Simulation Step {step} ---")
    debug_print(f"Tank pos: {state['units']['tank']['pos']}, Infantry pos: {state['units']['infantry']['pos']}, Enemy HP: {state['enemy']['health']}")
    
    if not current_plan:
        candidates = enumerate_plans(state, friendly_tasks, operators, methods)
        candidates = [plan for plan in candidates if plan]
        debug_print(f"Candidate Plans: {candidates}")
        if not candidates:
            debug_print("No non-empty candidate plans found; replanning with secure_outpost.")
            friendly_tasks = [Task("secure_outpost")]
            candidates = enumerate_plans(state, friendly_tasks, operators, methods)
            candidates = [plan for plan in candidates if plan]
            if not candidates:
                debug_print("Still no non-empty plans. Exiting simulation.")
                break
        for plan in candidates:
            if any(action.name == "move_to_safe_position" for action in plan) and any(action.name == "wait" for action in plan):
                current_plan = plan
                break
        else:
            current_plan = candidates[0]
        debug_print(f"Chosen candidate plan: {current_plan}")
    
    if not current_plan:
        continue
    
    action = current_plan.pop(0)
    if operators[action.name].applicable(state, *action.args):
        state = operators[action.name].apply(state, *action.args)
        debug_print(f"Executed action: {action}")
        executed_plan.append(action)
    else:
        debug_print(f"Action {action} not applicable; skipping and replanning.")
        current_plan = []
    
    if state["enemy"]["health"] <= 0 and state["units"]["tank"]["pos"] == state["outpost"] and state["units"]["infantry"]["pos"] == state["outpost"]:
        debug_print("Mission complete: enemy neutralized and outpost secured by both units.")
        break
    elif state["enemy"]["health"] <= 0 and (state["units"]["tank"]["pos"] != state["outpost"] or state["units"]["infantry"]["pos"] != state["outpost"]):
        friendly_tasks = [Task("secure_outpost")]
        current_plan = []
    elif not current_plan:
        friendly_tasks = [Task("secure_outpost")]
    
    step += 1

debug_print("\nFinal State:")
print("Final State:", state)
print("\nExecuted Plan:", executed_plan)

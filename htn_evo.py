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
    "support_fired": False
}
damage_values = {
    "tank": 40,
    "infantry": 20,
    "enemy_tank": 35
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
    if e["health"] <= 0:
        return False
    if unit_name not in state["units"]:
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
    if not line_of_sight(state["grid"], unit["pos"], state["enemy"]["pos"]):
        debug_print(f"{unit_name} has no LOS for support fire.")
        return False
    if manhattan(unit["pos"], state["enemy"]["pos"]) > 4:
        debug_print(f"{unit_name} is too far for support fire from enemy at {state['enemy']['pos']}.")
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

operators = {
    "move": move_op,
    "attack": attack_op,
    "support_fire": support_fire_op
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

def subtasks_neutralize_enemy_with_support(st):
    # Coordinated method: if infantry has LOS and support not provided, do support_fire.
    if precond_support_fire(st, "infantry"):
        return [Task("support_fire", ["infantry"])]
    if st["units"]["tank"]["pos"] != st["enemy"]["pos"]:
        return [Task("move", ["tank", st["enemy"]["pos"]])]
    return [Task("attack", ["tank"])]
        
neutralize_enemy_with_support = Method("neutralize_enemy_with_support", precond_neutralize_enemy_with_support, subtasks_neutralize_enemy_with_support)

# Alternative coordinated method: if infantry lacks LOS, move it to a fallback cell.
def precond_neutralize_enemy_with_support_alt(st):
    if st["enemy"]["health"] > 0 and not line_of_sight(st["grid"], st["units"]["infantry"]["pos"], st["enemy"]["pos"]):
        return True
    return False

def subtasks_neutralize_enemy_with_support_alt(st):
    fallback = (2,1)  # fallback cell for infantry to get LOS
    return [Task("move", ["infantry", fallback]),
            Task("support_fire", ["infantry"]),
            Task("move", ["tank", st["enemy"]["pos"]]),
            Task("attack", ["tank"])]
        
neutralize_enemy_with_support_alt = Method("neutralize_enemy_with_support", precond_neutralize_enemy_with_support_alt, subtasks_neutralize_enemy_with_support_alt)

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
    if st["units"].get("infantry", {}).get("pos") != st["outpost"]:
        tasks.append(Task("occupy_outpost"))
    return tasks

secure_outpost_coordinated = Method("secure_outpost", precond_secure_outpost, subtasks_secure_outpost_coordinated)

methods = {
    "secure_outpost": [secure_outpost_original, secure_outpost_coordinated],
    "neutralize_enemy": [],  # Not used in coordinated approach.
    "neutralize_enemy_with_support": [neutralize_enemy_with_support, neutralize_enemy_with_support_alt],
    "occupy_outpost": [occupy_outpost_m]
}

# ----- Evolutionary Search Functions -----
def simulate_plan(state, plan):
    """Simulate executing a candidate plan from the given state."""
    s = copy.deepcopy(state)
    for action in plan:
        if action.name in operators and operators[action.name].applicable(s, *action.args):
            s = operators[action.name].apply(s, *action.args)
        else:
            # If action becomes inapplicable, break.
            break
    return s

def fitness(state, plan):
    """Evaluate the fitness of a plan from the given state.
       Higher fitness for lower enemy health, with a bonus if mission achieved.
    """
    s_final = simulate_plan(state, plan)
    enemy_hp = s_final["enemy"]["health"]
    bonus = 0
    # If mission complete (enemy neutralized and infantry at outpost), add large bonus.
    if enemy_hp <= 0 and s_final["units"]["infantry"]["pos"] == s_final["outpost"]:
        bonus = 1000
    # Fitness: bonus + (100 - enemy_hp); lower enemy_hp gives higher fitness.
    return bonus + (100 - enemy_hp)

def tournament_selection(population, state, k=3):
    """Select one candidate plan from population using tournament selection."""
    selected = random.sample(population, min(k, len(population)))
    selected.sort(key=lambda plan: fitness(state, plan), reverse=True)
    return selected[0]

def crossover(plan1, plan2):
    """One-point crossover between two candidate plans (if possible)."""
    if len(plan1) < 2 or len(plan2) < 2:
        return plan1[:]  # nothing to crossover
    cp1 = random.randint(1, len(plan1)-1)
    cp2 = random.randint(1, len(plan2)-1)
    child = plan1[:cp1] + plan2[cp2:]
    return child

def mutate(plan, state):
    """Randomly mutate one action in the plan.
       For a move action, randomly shift the destination by ±1 in one dimension.
    """
    new_plan = plan[:]
    if not new_plan:
        return new_plan
    idx = random.randint(0, len(new_plan)-1)
    action = new_plan[idx]
    # Only mutate move actions for simplicity.
    if action.name == "move":
        unit_name, dest = action.args[0], action.args[1]
        # Randomly choose an offset.
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        new_dest = (dest[0] + dx, dest[1] + dy)
        # Check grid bounds (assume grid size from initial_state)
        rows = len(state["grid"])
        cols = len(state["grid"][0])
        new_dest = (max(0, min(new_dest[0], rows-1)), max(0, min(new_dest[1], cols-1)))
        new_plan[idx] = Task("move", [unit_name, new_dest])
    return new_plan

def evolve_population(state, population, generations=10, pop_size=10):
    for gen in range(generations):
        new_population = []
        # Evaluate fitness of current population.
        pop_fitness = [fitness(state, plan) for plan in population]
        # If any plan achieves mission complete, return it.
        for plan, fit in zip(population, pop_fitness):
            if fit >= 1000:
                return [plan]
        # Create new population by selection, crossover, and mutation.
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, state)
            parent2 = tournament_selection(population, state)
            child = crossover(parent1, parent2)
            child = mutate(child, state)
            new_population.append(child)
        population = new_population
    return population

# ----- Main Evolutionary Simulation Loop -----
state = copy.deepcopy(initial_state)
friendly_tasks = [Task("secure_outpost")]
max_steps = 50
step = 0
while step < max_steps:
    debug_print(f"\n--- Simulation Step {step} ---")
    debug_print(f"Tank pos: {state['units']['tank']['pos']}, Infantry pos: {state['units']['infantry']['pos']}")
    
    # Generate initial candidate plans using the HTN.
    candidates = enumerate_plans(state, friendly_tasks, operators, methods)
    debug_print(f"Initial Candidate Plans: {candidates}")
    if not candidates:
        debug_print("No candidate plans found. Exiting simulation.")
        break
    # Evolve the candidate plans for several generations.
    evolved = evolve_population(state, candidates, generations=10, pop_size=10)
    # Choose the best candidate from the evolved population.
    best_plan = max(evolved, key=lambda plan: fitness(state, plan))
    debug_print(f"Best Candidate Plan: {best_plan} with fitness {fitness(state, best_plan)}")
    
    # Execute only the first action from the best candidate plan.
    action = best_plan[0]
    state = operators[action.name].apply(state, *action.args)
    debug_print(f"Executed action: {action}")
    
    # Reset top-level task for next round.
    friendly_tasks = [Task("secure_outpost")]
    
    # Termination: mission complete when enemy is neutralized and infantry at outpost.
    if state["enemy"]["health"] <= 0 and state["units"].get("infantry", {}).get("pos") == state["outpost"]:
        debug_print("Mission complete: enemy neutralized and outpost secured.")
        break
    if not state["units"]:
        debug_print("All friendly units defeated.")
        break
    step += 1

debug_print("\nFinal State:")
print("Final State:", state)

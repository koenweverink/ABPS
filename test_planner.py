from copy import deepcopy
from heapq import heappush, heappop
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Terrain map remains unchanged
terrain_map = [
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0]
]

# Bresenham's line-of-sight algorithm (unchanged)
def line_of_sight(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    n = 1 + dx + dy
    x_inc = 1 if x2 > x1 else -1
    y_inc = 1 if y2 > y1 else -1
    error = dx - dy
    dx *= 2
    dy *= 2
    for _ in range(n):
        if (x, y) != (x1, y1) and (x, y) != (x2, y2):
            if terrain_map[x][y] == 1:
                return False
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return True

# Agent class (unchanged)
class Agent:
    def __init__(self, name, x, y, fuel, attack_strength, attack_range=1, 
                 v_max=6.0, s_eff=1.0, c_cover=1.0, n_noise=1.0, e_elev=1.0):
        self.name = name
        self.x = x
        self.y = y
        self.fuel = fuel
        self.attack_strength = attack_strength
        self.attack_range = attack_range
        self.detected_by_enemy = False
        self.detects_enemy = False
        self.v_max = v_max
        self.s_eff = s_eff
        self.c_cover = c_cover
        self.n_noise = n_noise
        self.e_elev = e_elev

# State class (unchanged)
class State:
    def __init__(self):
        self.data = {
            "agents": {
                "scouts": Agent("scouts", 0, 0, 800, 0, attack_range=5, v_max=6.0),
                "infantry": Agent("infantry", 0, 0, 800, 15, attack_range=2, v_max=4.0),
                "tanks": Agent("tanks", 0, 0, 1700, 30, attack_range=3, v_max=3.0)
            },
            "enemy": Agent("enemy", 9, 9, float('inf'), 10, attack_range=2, v_max=5.0),
            "mission_complete": False
        }

# A* Pathfinding (unchanged)
def heuristic(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def a_star(start_x, start_y, goal_x, goal_y):
    if not (0 <= start_x < len(terrain_map) and 0 <= start_y < len(terrain_map[0]) and
            0 <= goal_x < len(terrain_map) and 0 <= goal_y < len(terrain_map[0])):
        return None
    open_list = [(0, start_x, start_y, [])]
    closed = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while open_list:
        f_score, x, y, path = heappop(open_list)
        if (x, y) == (goal_x, goal_y):
            return path + [(x, y)]
        if (x, y) in closed:
            continue
        closed.add((x, y))
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < len(terrain_map) and 0 <= new_y < len(terrain_map[0]) and
                terrain_map[new_x][new_y] == 0 and (new_x, new_y) not in closed):
                g_score = len(path) + 1
                h_score = heuristic(new_x, new_y, goal_x, goal_y)
                f_score_new = g_score + h_score
                new_path = path + [(x, y)]
                heappush(open_list, (f_score_new, new_x, new_y, new_path))
    return None

# Primitive tasks (unchanged)
def move(state, agent_name, x, y):
    agent = state.data["agents"].get(agent_name, state.data["enemy"] if agent_name == "enemy" else None)
    if not agent:
        return False
    fuel_cost = 20 if agent_name == "tanks" else 10
    if (abs(agent.x - x) + abs(agent.y - y) == 1 and terrain_map[x][y] == 0 and agent.fuel >= fuel_cost):
        agent.x = x
        agent.y = y
        agent.fuel -= fuel_cost
        print(f"Success: {agent_name} moved to ({x}, {y}), fuel left: {agent.fuel}")
        return True
    print(f"Failed: {agent_name} cannot move to ({x}, {y}) (fuel: {agent.fuel}, terrain: {terrain_map[x][y]})")
    return False

def scout_area(state, agent_name):
    agent = state.data["agents"][agent_name]
    enemy = state.data["enemy"]
    distance = abs(agent.x - enemy.x) + abs(agent.y - enemy.y)
    if distance <= agent.attack_range:
        if line_of_sight(agent.x, agent.y, enemy.x, enemy.y):
            enemy.attack_strength = max(10, enemy.attack_strength - 20)
            print(f"Success: {agent_name} scouted enemy from ({agent.x}, {agent.y}), enemy strength: {enemy.attack_strength}")
            return True
        else:
            print(f"Failed: {agent_name} cannot scout enemy from ({agent.x}, {agent.y}) due to obstacles.")
            return False
    print(f"Failed: {agent_name} is too far to scout enemy from ({agent.x}, {agent.y}) (distance: {distance}, required: <= {agent.attack_range})")
    return False

def attack(state, agent_name):
    agent = state.data["agents"][agent_name]
    enemy = state.data["enemy"]
    fuel_cost = 10 if agent_name == "tanks" else 0
    distance = abs(agent.x - enemy.x) + abs(agent.y - enemy.y)
    if distance <= agent.attack_range and (agent_name != "tanks" or agent.fuel >= fuel_cost):
        if line_of_sight(agent.x, agent.y, enemy.x, enemy.y):
            enemy.attack_strength -= agent.attack_strength
            if agent_name == "tanks":
                agent.fuel -= fuel_cost
            print(f"Success: {agent_name} attacked enemy from ({agent.x}, {agent.y}), enemy strength: {enemy.attack_strength}, fuel: {agent.fuel}")
            return True
        else:
            print(f"Failed: {agent_name} cannot attack enemy from ({agent.x}, {agent.y}) because line-of-sight is blocked.")
            return False
    print(f"Failed: {agent_name} cannot attack enemy from ({agent.x}, {agent.y}) (distance: {distance}, required: <= {agent.attack_range})")
    return False

def secure_outpost(state):
    enemy = state.data["enemy"]
    enemy_x, enemy_y = enemy.x, enemy.y
    all_here = all(agent.x == enemy_x and agent.y == enemy_y for agent in state.data["agents"].values())
    if enemy.attack_strength <= 0 and all_here:
        state.data["mission_complete"] = True
        print("Success: Outpost secured")
        return True
    if enemy.attack_strength <= 0:
        print("Enemy defeated but not all units have converged to the outpost.")
        return False
    print(f"Failed: Enemy strength still {enemy.attack_strength}")
    return False

# Generate move tasks (unchanged)
def generate_move_tasks(agent_name, start_x, start_y, goal_x, goal_y):
    path = a_star(start_x, start_y, goal_x, goal_y)
    if not path or len(path) < 2:
        return []
    move_tasks = []
    for i in range(1, len(path)):
        x, y = path[i]
        move_tasks.append(("move", agent_name, x, y))
    return move_tasks

# Detection function (unchanged)
def update_detection(state):
    enemy = state.data["enemy"]
    enemy_x, enemy_y = enemy.x, enemy.y

    for agent_name, agent in state.data["agents"].items():
        dist = abs(agent.x - enemy_x) + abs(agent.y - enemy_y)
        detection_threshold = agent.v_max * agent.s_eff * agent.c_cover * agent.n_noise * agent.e_elev
        if dist <= detection_threshold and line_of_sight(agent.x, agent.y, enemy_x, enemy_y):
            agent.detects_enemy = True
        else:
            agent.detects_enemy = False

        enemy_threshold = enemy.v_max * enemy.s_eff * enemy.c_cover * enemy.n_noise * enemy.e_elev
        if dist <= enemy_threshold and line_of_sight(enemy_x, enemy_y, agent.x, agent.y):
            agent.detected_by_enemy = True
        else:
            agent.detected_by_enemy = False

# Secure outpost methods (unchanged)
def secure_outpost_methods(state):
    temp_state = deepcopy(state)
    tasks = []
    enemy = temp_state.data["enemy"]
    enemy_x, enemy_y = enemy.x, enemy.y

    def compute_target(agent):
        cur_x, cur_y = agent.x, agent.y
        while True:
            current_distance = abs(cur_x - enemy_x) + abs(cur_y - enemy_y)
            if current_distance <= agent.attack_range and current_distance != 0:
                return (cur_x, cur_y)
            dx = enemy_x - cur_x
            dy = enemy_y - cur_y
            move_options = []
            if dx != 0:
                move_options.append((cur_x + (1 if dx > 0 else -1), cur_y))
            if dy != 0:
                move_options.append((cur_x, cur_y + (1 if dy > 0 else -1)))
            if dx != 0 and dy != 0:
                move_options.append((cur_x + (1 if dx > 0 else -1), cur_y + (1 if dy > 0 else -1)))
            best_option = None
            best_dist = float('inf')
            for option in move_options:
                x_new, y_new = option
                if 0 <= x_new < len(terrain_map) and 0 <= y_new < len(terrain_map[0]) and terrain_map[x_new][y_new] == 0:
                    dist = abs(x_new - enemy_x) + abs(y_new - enemy_y)
                    if dist == 0:
                        continue
                    if dist < best_dist:
                        best_dist = dist
                        best_option = option
            if best_option is None:
                return (cur_x, cur_y)
            cur_x, cur_y = best_option

    for agent_name in ["scouts", "infantry", "tanks"]:
        agent = temp_state.data["agents"][agent_name]
        while abs(agent.x - enemy_x) + abs(agent.y - enemy_y) > state.data["agents"][agent_name].attack_range:
            target = compute_target(agent)
            move_tasks = generate_move_tasks(agent_name, agent.x, agent.y, target[0], target[1])
            if not move_tasks:
                break
            tasks.extend(move_tasks)
            for t in move_tasks:
                if t[0] == "move":
                    move(temp_state, t[1], t[2], t[3])
    tasks.append(("scout_area", "scouts"))
    tasks.append(("attack", "infantry"))
    tasks.append(("attack", "tanks"))
    for agent_name in ["scouts", "infantry", "tanks"]:
        agent = temp_state.data["agents"][agent_name]
        if (agent.x, agent.y) != (enemy_x, enemy_y):
            move_tasks = generate_move_tasks(agent_name, agent.x, agent.y, enemy_x, enemy_y)
            tasks.extend(move_tasks)
            for t in move_tasks:
                if t[0] == "move":
                    move(temp_state, t[1], t[2], t[3])
    tasks.append(("secure_outpost",))
    return [tasks]

# HTN Planner (unchanged)
def htn_planner(state, tasks):
    plan = []
    task_stack = tasks[:]
    while task_stack:
        current_task = task_stack.pop(0)
        temp_state = deepcopy(state)
        if current_task[0] == "move":
            if move(temp_state, current_task[1], current_task[2], current_task[3]):
                state.data = temp_state.data
                plan.append(current_task)
            else:
                print(f"Task {current_task} failed, no plan possible.")
                return None
        elif current_task[0] == "scout_area":
            if scout_area(temp_state, current_task[1]):
                state.data = temp_state.data
                plan.append(current_task)
            else:
                print(f"Task {current_task} failed, no plan possible.")
                return None
        elif current_task[0] == "attack":
            if attack(temp_state, current_task[1]):
                state.data = temp_state.data
                plan.append(current_task)
            else:
                print(f"Task {current_task} failed, no plan possible.")
                return None
        elif current_task[0] == "secure_outpost":
            if secure_outpost(temp_state):
                state.data = temp_state.data
                plan.append(current_task)
                break
            else:
                print(f"Task {current_task} failed, no plan possible.")
                return None
        elif current_task[0] == "secure_outpost_task":
            methods = secure_outpost_methods(state)
            if not methods:
                print(f"No valid methods for {current_task}, no plan possible.")
                return None
            task_stack = methods[0] + task_stack
    return plan if state.data["mission_complete"] else None

# Updated visualization function
def visualize_plan(original_state, plan):
    state = deepcopy(original_state)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Military Operation Simulation")
    ax.set_xticks(range(len(terrain_map[0])))
    ax.set_yticks(range(len(terrain_map)))
    ax.set_xticklabels(range(len(terrain_map[0])))
    ax.set_yticklabels(range(len(terrain_map)))
    ax.grid(True)

    terrain_img = [[0 if cell == 0 else 0.5 for cell in row] for row in terrain_map]
    ax.imshow(terrain_img, cmap="Greys", alpha=0.5)

    scouts_plot, = ax.plot([], [], 'bo', label="Scouts", markersize=12, alpha=0.8)
    infantry_plot, = ax.plot([], [], 'ro', label="Infantry", markersize=12, alpha=0.8)
    tanks_plot, = ax.plot([], [], 'yo', label="Tanks", markersize=12, alpha=0.8)
    enemy_plot, = ax.plot([], [], 'k*', label="Enemy", markersize=15)

    ax.legend(loc="upper right")
    strength_text = ax.text(0.5, 1.05, f"Enemy Strength: {state.data['enemy'].attack_strength}",
                            transform=ax.transAxes, ha="center", fontsize=10)

    detection_texts = {
        "scouts": ax.text(0, 0, "", color="magenta", fontsize=8, visible=False),
        "infantry": ax.text(0, 0, "", color="magenta", fontsize=8, visible=False),
        "tanks": ax.text(0, 0, "", color="magenta", fontsize=8, visible=False),
        "enemy": ax.text(0, 0, "", color="red", fontsize=8, visible=False)
    }

    def init():
        s = state.data["agents"]["scouts"]
        i = state.data["agents"]["infantry"]
        t = state.data["agents"]["tanks"]
        e = state.data["enemy"]
        scouts_plot.set_data([s.y], [s.x])
        infantry_plot.set_data([i.y], [i.x])
        tanks_plot.set_data([t.y], [t.x])
        enemy_plot.set_data([e.y], [e.x])
        strength_text.set_text(f"Enemy Strength: {state.data['enemy'].attack_strength}")
        for name, txt in detection_texts.items():
            txt.set_visible(False)
        return scouts_plot, infantry_plot, tanks_plot, enemy_plot, strength_text, *detection_texts.values()

    def update(frame):
        print(f"Processing frame {frame}/{len(plan)}")
        if frame < len(plan):
            task = plan[frame]
            print(f"Executing: {task}")
            if task[0] == "move":
                move(state, task[1], task[2], task[3])
            elif task[0] == "scout_area":
                scout_area(state, task[1])
            elif task[0] == "attack":
                attack(state, task[1])
            elif task[0] == "secure_outpost":
                secure_outpost(state)

        update_detection(state)

        s = state.data["agents"]["scouts"]
        i = state.data["agents"]["infantry"]
        t = state.data["agents"]["tanks"]
        e = state.data["enemy"]

        scouts_plot.set_data([s.y], [s.x])
        infantry_plot.set_data([i.y], [i.x])
        tanks_plot.set_data([t.y], [t.x])

        # Check if enemy is defeated (strength <= 0)
        if e.attack_strength <= 0:
            enemy_plot.set_data([], [])  # Remove enemy from visualization
            detection_texts["enemy"].set_visible(False)
        else:
            enemy_plot.set_data([e.y], [e.x])

        # Update detection labels
        for name, txt in detection_texts.items():
            agent = state.data["agents"].get(name, state.data["enemy"] if name == "enemy" else None)
            if name == "enemy":
                # Check if enemy is detected by any friendly agent or detects any friendly
                detected = any(a.detects_enemy for a in state.data["agents"].values()) or \
                          any(a.detected_by_enemy for a in state.data["agents"].values())
                if e.attack_strength > 0:  # Only show if enemy is alive
                    if detected:
                        txt.set_text("DETECTED")
                        # Adjust position to prevent cutoff
                        x_pos = min(max(e.y + 0.2, 0.5), len(terrain_map[0]) - 1.5)
                        y_pos = min(max(e.x - 0.2, 0.5), len(terrain_map) - 0.5)
                        txt.set_position((x_pos, y_pos))
                        txt.set_visible(True)
                    else:
                        txt.set_visible(False)
                else:
                    txt.set_visible(False)
            else:
                # Friendly agent detection
                detected = agent.detected_by_enemy or agent.detects_enemy
                if detected:
                    txt.set_text("DETECTED")
                    # Adjust position to prevent cutoff
                    x_pos = min(max(agent.y + 0.2, 0.5), len(terrain_map[0]) - 1.5)
                    y_pos = min(max(agent.x - 0.2, 0.5), len(terrain_map) - 0.5)
                    txt.set_position((x_pos, y_pos))
                    txt.set_visible(True)
                else:
                    txt.set_visible(False)

        strength_text.set_text(f"Enemy Strength: {state.data['enemy'].attack_strength}")
        return scouts_plot, infantry_plot, tanks_plot, enemy_plot, strength_text, *detection_texts.values()

    ani = FuncAnimation(fig, update, frames=len(plan), init_func=init,
                        blit=True, interval=300, repeat=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    initial_state = State()
    state_for_planning = deepcopy(initial_state)
    print("Initial state:")
    for agent_name, agent in state_for_planning.data["agents"].items():
        print(f"{agent_name} at ({agent.x}, {agent.y}), fuel: {agent.fuel}")
    print(f"Enemy at ({state_for_planning.data['enemy'].x}, {state_for_planning.data['enemy'].y}), strength: {state_for_planning.data['enemy'].attack_strength}")
    print("Terrain map:")
    for row in terrain_map:
        print(row)
    plan = htn_planner(state_for_planning, [("secure_outpost_task",)])
    if plan:
        print("\nPlan found:", plan)
        visualize_plan(initial_state, plan)
    else:
        print("\nNo plan found.")
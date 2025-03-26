# planner.py
from copy import deepcopy
from pathfinding import a_star
from tasks import move
from terrain import terrain_map

def generate_move_tasks(agent_name, start_x, start_y, goal_x, goal_y):
    path = a_star(start_x, start_y, goal_x, goal_y)
    if not path or len(path) < 2:
        return []
    move_tasks = []
    for i in range(1, len(path)):
        x, y = path[i]
        move_tasks.append(("move", agent_name, x, y))
    return move_tasks

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

    # Step 1: Move agents into attack range
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

    # Step 2: Keep attacking until the enemy is defeated
    while temp_state.data["enemy"].health > 0:
        # Scouts scout the area
        if temp_state.data["agents"]["scouts"].health > 0:
            tasks.append(("scout_area", "scouts"))
            from tasks import scout_area
            scout_area(temp_state, "scouts")
        # Infantry attacks
        if temp_state.data["agents"]["infantry"].health > 0:
            tasks.append(("attack", "infantry"))
            from tasks import attack
            attack(temp_state, "infantry")
        # Tanks attack
        if temp_state.data["agents"]["tanks"].health > 0:
            tasks.append(("attack", "tanks"))
            from tasks import attack
            attack(temp_state, "tanks")

    # Step 3: Move all surviving agents to the enemy's position
    for agent_name in ["scouts", "infantry", "tanks"]:
        agent = temp_state.data["agents"][agent_name]
        if agent.health > 0 and (agent.x, agent.y) != (enemy_x, enemy_y):
            move_tasks = generate_move_tasks(agent_name, agent.x, agent.y, enemy_x, enemy_y)
            tasks.extend(move_tasks)
            for t in move_tasks:
                if t[0] == "move":
                    move(temp_state, t[1], t[2], t[3])

    # Step 4: Secure the outpost
    tasks.append(("secure_outpost",))
    return [tasks]

def htn_planner(state, tasks):
    from tasks import move, scout_area, attack, secure_outpost
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
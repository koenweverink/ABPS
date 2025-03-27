# htn_planner.py
from state import State
from tasks import move, scout_area, attack, secure_outpost
from terrain import line_of_sight
import heapq

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def find_path(state, agent_name, target_x, target_y):
    agent = state.data["agents"][agent_name]
    start = (agent.x, agent.y)
    goal = (target_x, target_y)
    terrain_map = state.terrain_map

    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        x, y = current
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_x, next_y = x + dx, y + dy
            if 0 <= next_x < 10 and 0 <= next_y < 10 and terrain_map[next_x][next_y] == 0:
                neighbor = (next_x, next_y)
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

def scout_and_destroy(state):
    enemy = state.data["enemy"]
    enemy_x, enemy_y = enemy.x, enemy.y
    plan = []

    # Scouts: Move to a position to scout the enemy
    scouts = state.data["agents"]["scouts"]
    if not scouts.detects_enemy:
        vision_range_units = float(scouts.vision_range.replace(" m", "")) / 500
        best_pos = None
        best_path = []
        for x in range(10):
            for y in range(10):
                dist_to_enemy = abs(x - enemy_x) + abs(y - enemy_y)
                if dist_to_enemy <= vision_range_units:
                    path = find_path(state, "scouts", x, y)
                    if path and line_of_sight(x, y, enemy_x, enemy_y):
                        if not best_path or len(path) < len(best_path):
                            best_pos = (x, y)
                            best_path = path
        if best_path:
            for step_x, step_y in best_path:
                plan.append(("move", "scouts", step_x, step_y))
            plan.append(("scout_area", "scouts"))

    # After scouting, scouts can attack if the enemy is detected
    if scouts.detects_enemy:
        dist_to_enemy = abs(scouts.x - enemy_x) + abs(scouts.y - enemy_y)
        # Check if scouts are already in attack range
        if dist_to_enemy > scouts.attack_range:
            path = find_path(state, "scouts", enemy_x, enemy_y)
            for step_x, step_y in path:
                if abs(step_x - enemy_x) + abs(step_y - enemy_y) <= scouts.attack_range:
                    plan.append(("move", "scouts", step_x, step_y))
                    break
                plan.append(("move", "scouts", step_x, step_y))
        plan.append(("attack", "scouts"))

    # Infantry: Move to attack range and attack
    infantry = state.data["agents"]["infantry"]
    dist_to_enemy = abs(infantry.x - enemy_x) + abs(infantry.y - enemy_y)
    if dist_to_enemy > infantry.attack_range:
        path = find_path(state, "infantry", enemy_x, enemy_y)
        for step_x, step_y in path:
            if abs(step_x - enemy_x) + abs(step_y - enemy_y) <= infantry.attack_range:
                plan.append(("move", "infantry", step_x, step_y))
                break
            plan.append(("move", "infantry", step_x, step_y))
    plan.append(("attack", "infantry"))

    # Tanks: Move to attack range and attack
    tanks = state.data["agents"]["tanks"]
    dist_to_enemy = abs(tanks.x - enemy_x) + abs(tanks.y - enemy_y)
    if dist_to_enemy > tanks.attack_range:
        path = find_path(state, "tanks", enemy_x, enemy_y)
        for step_x, step_y in path:
            if abs(step_x - enemy_x) + abs(step_y - enemy_y) <= tanks.attack_range:
                plan.append(("move", "tanks", step_x, step_y))
                break
            plan.append(("move", "tanks", step_x, step_y))
    plan.append(("attack", "tanks"))

    # Secure the outpost
    for agent_name in ["scouts", "infantry", "tanks"]:
        agent = state.data["agents"][agent_name]
        if agent.x != enemy_x or agent.y != enemy_y:
            path = find_path(state, agent_name, enemy_x, enemy_y)
            for step_x, step_y in path:
                plan.append(("move", agent_name, step_x, step_y))
    plan.append(("secure_outpost",))

    return plan

def apply_task(state, task):
    if task[0] == "move":
        return move(state, task[1], task[2], task[3])
    elif task[0] == "scout_area":
        return scout_area(state, task[1])
    elif task[0] == "attack":
        # Repeat attack until enemy is defeated
        enemy = state.data["enemy"]
        agent_name = task[1]
        while enemy.health > 0:
            success = attack(state, agent_name)
            if not success:
                return False
            if enemy.health <= 0:
                break
        return True
    elif task[0] == "secure_outpost":
        return secure_outpost(state)
    return False

def plan_and_execute(state):
    plan = scout_and_destroy(state)
    if not plan:
        print("No plan found.")
        return False

    for task in plan:
        success = apply_task(state, task)
        if not success:
            print(f"Task {task} failed, no plan possible.")
            return False
        if state.data["mission_complete"]:
            return True
    return False
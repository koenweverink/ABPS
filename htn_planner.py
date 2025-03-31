# htn_planner.py (assumed)
from tasks import move, attack, secure_outpost
from terrain import Terrain

def apply_task(state, task):
    if task[0] == "move":
        _, agent_name, x, y = task
        return move(state, agent_name, x, y)
    elif task[0] == "attack":
        _, agent_name = task
        return attack(state, agent_name)
    elif task[0] == "secure_outpost":
        return secure_outpost(state)
    return False

def find_path(state, agent_name, target_x, target_y):
    # Simplified A* pathfinding (for demonstration)
    agent = state.data["agents"][agent_name]
    terrain = state.terrain
    start = (agent.x, agent.y)
    end = (target_x, target_y)
    open_set = [start]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: abs(start[0] - end[0]) + abs(start[1] - end[1])}
    
    while open_set:
        current = min(open_set, key=lambda pos: f_score[pos])
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1][1:]  # Reverse and exclude start
        
        open_set.remove(current)
        x, y = current
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_x, next_y = x + dx, y + dy
            if terrain.is_obstacle(next_x, next_y):
                continue
            tentative_g_score = g_score[current] + 1
            if (next_x, next_y) not in g_score or tentative_g_score < g_score[(next_x, next_y)]:
                came_from[(next_x, next_y)] = current
                g_score[(next_x, next_y)] = tentative_g_score
                f_score[(next_x, next_y)] = tentative_g_score + abs(next_x - end[0]) + abs(next_y - end[1])
                if (next_x, next_y) not in open_set:
                    open_set.append((next_x, next_y))
    return []  # No path found
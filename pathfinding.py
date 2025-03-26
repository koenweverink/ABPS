# pathfinding.py
from heapq import heappush, heappop
from terrain import terrain_map

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
import heapq, random
from environment import forest, forest_edge, river, cliffs, climb_entries
from terrain_utils import manhattan, get_line, in_bounds, neighbors
from drone import Drone
from config import CELL_SIZE

def get_effective_vision_range(base_vision_range, stealth_modifier, in_cover, has_los):
    if not in_cover:
        return base_vision_range
    return base_vision_range / (1 + stealth_modifier / CELL_SIZE)

def is_in_enemy_vision(pos, enemy_units):
    for enemy in enemy_units:
        if enemy.state["enemy_alive"]:
            distance = manhattan(pos, enemy.state["position"])
            has_los = has_line_of_sight(pos, enemy.state["position"])
            vision_range = get_effective_vision_range(
                enemy.state.get("vision_range", 20),
                enemy.state.get("stealth_modifier", 0),
                is_in_cover(pos),
                has_los
            )
            if distance <= vision_range and has_los:
                return True
    return False

def astar(start, goal, enemy_units=None, unit="unknown"):
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for nxt in neighbors(current, in_bounds, river=river, cliffs=cliffs, climb_entries=climb_entries):
            if unit in ["tank", "artillery", "anti-tank"]:
                if is_in_cover(nxt):
                    continue
            new_cost = cost_so_far[current] + 1
            if unit in ["scout", "infantry"]:
                if not is_in_cover(nxt):
                    new_cost += 5
            else:
                if is_in_cover(nxt):
                    continue
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

def next_step(start, goal, enemy_units=None, unit="unknown"):
    path = astar(start, goal, enemy_units, unit)
    return path[1] if len(path) >= 2 else start


def has_line_of_sight(start, end):
    """
    Returns False if either the observer (start) or the target (end) is in deep forest,
    or if any intervening cell is deep forest; otherwise True.
    """
    # 1) If the observer itself is in deep forest, no sight
    if start in forest and start not in forest_edge:
        return False

    # 2) If the target is in deep forest, they’re concealed
    if end in forest and end not in forest_edge:
        return False

    # 3) Now check the intervening line (you can skip endpoints since we already handled them)
    for pos in get_line(start, end):
        if pos in forest and pos not in forest_edge:
            return False

    return True

def is_in_cover(pos):
    return pos in forest or pos in forest_edge

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

def visible_spotted_enemies(state):
    """
    For the *friendly* HTN: return names of spotted *enemy* units
    (using the friendly drone’s memory, and filtering by enemy_units_dict).
    """
    sim = state["sim"]
    seen = sim.friendly_drone.last_known  # name → (x,y)
    alive = sim.enemy_units_dict          # name → EnemyUnit
    return [
        name
        for name in seen
        if name in alive and alive[name].state.get("enemy_alive", False)
    ]

def visible_spotted_friendlies(state):
    """
    For the *enemy* HTN: return names of spotted *friendly* units
    (using the enemy drone’s memory, and filtering by friendly_units_dict).
    """
    sim = state["sim"]
    seen = sim.enemy_drone.last_known
    alive = sim.friendly_units_dict
    return [
        name
        for name in seen
        if name in alive and alive[name].state.get("health", 0) > 0
    ]

def units_spotted_by_vision(unit, candidates):
    """
    Return the list of *Unit* objects in `candidates` that `unit` actually sees this tick,
    accounting for distance, LOS, stealth, and cover.
    """
    seen = []
    for u in candidates:
        if u.state.get("health",0) <= 0:
            continue
        dist  = manhattan(unit.state["position"], u.state["position"])
        los   = has_line_of_sight(unit.state["position"], u.state["position"])
        stealth = u.state.get("stealth_modifier",0)
        in_cover = is_in_cover(u.state["position"])
        eff_range = get_effective_vision_range(
            unit.state.get("vision_range",20),
            stealth, in_cover, los
        )
        if dist <= eff_range and los:
            seen.append(u)
    return seen

def names_in_drone_memory(sim, side="enemy"):
    """
    Return the sorted list of names that the sim's drone(last_known) has for `side`.
    side="enemy" → sim.friendly_drone.last_known
    side="friendly" → sim.enemy_drone.last_known
    """
    mem = (sim.friendly_drone if side=="enemy" else sim.enemy_drone).last_known
    return [ name for name, pos in mem.items() ]

def under_friendly_drone_cover(sim, target_unit):
    drone = next((u for u in sim.friendly_units
                  if isinstance(u, Drone) and u.side=="friendly"), None)
    if not drone:
        return False
    bounds = drone.areas[drone.current_area]
    return drone._in_area(target_unit.state["position"], bounds)
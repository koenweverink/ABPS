import heapq, random
from environment import forest, forest_edge, river, cliffs, climb_entries
from terrain_utils import manhattan, get_line, in_bounds, neighbors
from drone import Drone
from config import CELL_SIZE

def get_effective_vision_range(base_vision_range, stealth_modifier, in_cover, has_los):
    """Compute effective vision range considering cover and stealth."""
    if not in_cover:
        return base_vision_range
    return base_vision_range / (1 + stealth_modifier / CELL_SIZE)

def is_in_enemy_vision(pos, enemy_units):
    """Check if a position is within LOS and vision range of any living enemy unit."""
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
    """A* pathfinding that accounts for cover penalties and terrain types."""
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for nxt in neighbors(current, in_bounds, river=river, cliffs=cliffs, climb_entries=climb_entries):
            if unit in ["tank", "artillery", "anti-tank"] and is_in_cover(nxt):
                continue
            new_cost = cost_so_far[current] + 1
            if unit in ["scout", "infantry"] and not is_in_cover(nxt):
                new_cost += 5
            elif is_in_cover(nxt):
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
    """Return the next grid step along the path to a goal using A*."""
    path = astar(start, goal, enemy_units, unit)
    return path[1] if len(path) >= 2 else start

def has_line_of_sight(start, end):
    """Return True if there is unobstructed LOS between start and end positions."""
    if start in forest and start not in forest_edge:
        return False
    if end in forest and end not in forest_edge:
        return False
    for pos in get_line(start, end):
        if pos in forest and pos not in forest_edge:
            return False
    return True

def is_in_cover(pos):
    """Check if a position is considered in cover (forest or forest edge)."""
    return pos in forest or pos in forest_edge

def get_num_attacks(rpm):
    """Convert rounds per minute to a probabilistic number of attacks this tick."""
    exact = rpm * 0.1
    n = int(exact)
    if random.random() < (exact - n):
        n += 1
    return n

def get_penetration_probability(D):
    """Map the difference between penetration and armor into hit probability."""
    if D <= -3:
        return 0.0
    elif -3 < D <= 0:
        return 0.33 + 0.11 * (D + 3)
    elif 0 < D <= 6:
        return 0.66 + (0.29/6) * D
    else:
        return 0.95

def visible_spotted_enemies(state):
    """Return list of enemy names spotted by the friendly drone and still alive."""
    sim = state["sim"]
    seen = sim.friendly_drone.last_known
    alive = sim.enemy_units_dict
    return [name for name in seen if name in alive and alive[name].state.get("enemy_alive", False)]

def visible_spotted_friendlies(state):
    """Return list of friendly unit names spotted by the enemy drone and still alive."""
    sim = state["sim"]
    seen = sim.enemy_drone.last_known
    alive = sim.friendly_units_dict
    return [name for name in seen if name in alive and alive[name].state.get("health", 0) > 0]

def units_spotted_by_vision(unit, candidates):
    """Return list of visible candidate units based on LOS, stealth, and range."""
    seen = []
    for u in candidates:
        if u.state.get("health", 0) <= 0:
            continue
        dist = manhattan(unit.state["position"], u.state["position"])
        los = has_line_of_sight(unit.state["position"], u.state["position"])
        stealth = u.state.get("stealth_modifier", 0)
        in_cover = is_in_cover(u.state["position"])
        eff_range = get_effective_vision_range(unit.state.get("vision_range", 20), stealth, in_cover, los)
        if dist <= eff_range and los:
            seen.append(u)
    return seen

def names_in_drone_memory(sim, side="enemy"):
    """Return sorted list of names known to the drone for the given side."""
    mem = (sim.friendly_drone if side == "enemy" else sim.enemy_drone).last_known
    return [name for name in mem]

def under_friendly_drone_cover(sim, target_unit):
    """Check whether a target is inside the current area of any friendly drone."""
    drone = next((u for u in sim.friendly_units if isinstance(u, Drone) and u.side == "friendly"), None)
    if not drone:
        return False
    bounds = drone.areas[drone.current_area]
    return drone._in_area(target_unit.state["position"], bounds)


def all_units_at_position(units, position):
    """Return True if all given units are exactly at the specified position."""
    return all(u.state.get("position") == position for u in units if u.state.get("current_group_size", 0) > 0)


def compute_staging_position(sim, max_distance=500):
    # 1) Compute centroid of all friendly units (with group_size > 0). 
    #    If none, return (0, 0).
    friendlies = [u for u in sim.friendly_units if u.state.get("current_group_size", 0) > 0]
    if not friendlies:
        return (0, 0)

    avg_x = sum(u.state["position"][0] for u in friendlies) // len(friendlies)
    avg_y = sum(u.state["position"][1] for u in friendlies) // len(friendlies)
    start = (avg_x, avg_y)

    # 2) Gather “last known” positions of alive enemies, as seen by our drone:
    drone_mem = {
        name: pos
        for name, pos in sim.friendly_drone.last_known.items()
        if name in sim.enemy_units_dict
        and sim.enemy_units_dict[name].state.get("enemy_alive", False)
    }
    if not drone_mem:
        # No intel on any alive enemy → stay at the centroid.
        return start

    # 3) Pick the one enemy whose last-known tile is nearest to our centroid:
    closest_name = min(drone_mem, key=lambda n: manhattan(start, drone_mem[n]))
    enemy_pos   = drone_mem[closest_name]
    enemy_unit  = sim.enemy_units_dict[closest_name]
    enemy_range = enemy_unit.state.get("vision_range", 0) if enemy_unit else 0

    # 4) Run A* from “start” toward “enemy_pos” to get the exact tile-by-tile path:
    path = astar(
        start,
        enemy_pos,
        sim.enemy_units,
        unit=friendlies[0].state.get("type", "unknown")
    )
    #    path is a list of coordinates: [ start, …, enemy_pos ]

    # 5) Precompute cumulative distance from start for each index i along the path:
    travelled = [0] * len(path)
    for i in range(1, len(path)):
        travelled[i] = travelled[i-1] + manhattan(path[i-1], path[i])

    # 6) Now scan **in reverse** (from index = len(path)-1 down to 0):
    #    As soon as we find a tile that is still within max_distance AND
    #    satisfies “(dist > enemy_range)  or  (not LOS)”, we return it.
    for i in range(len(path)-1, -1, -1):
        if travelled[i] > max_distance:
            # Too far from our own start → skip this tile
            continue

        pos = path[i]
        dist_enemy = manhattan(pos, enemy_pos)
        enemy_los  = has_line_of_sight(pos, enemy_pos)

        # **Only one of these must hold true**:
        outside_enemy = (dist_enemy > enemy_range) or (not enemy_los)
        if outside_enemy:
            return pos

    # 7) If we never found a tile meeting those criteria within max_distance,
    #    fall back to the centroid.
    return start


def compute_retreat_point(sim, max_distance=500):
    """Determine a fallback position away from friendly forces."""
    from config import GRID_WIDTH, GRID_HEIGHT

    enemies = [u for u in sim.enemy_units if u.state.get("enemy_alive", False)]
    if not enemies:
        return (0, 0)

    avg_x = sum(u.state["position"][0] for u in enemies) // len(enemies)
    avg_y = sum(u.state["position"][1] for u in enemies) // len(enemies)
    start = (avg_x, avg_y)

    friendlies = [u for u in sim.friendly_units if u.state.get("health", 0) > 0]
    if not friendlies:
        return start

    fx = sum(u.state["position"][0] for u in friendlies) // len(friendlies)
    fy = sum(u.state["position"][1] for u in friendlies) // len(friendlies)
    friendly_centroid = (fx, fy)

    corners = [
        (0, 0),
        (0, GRID_HEIGHT - 1),
        (GRID_WIDTH - 1, 0),
        (GRID_WIDTH - 1, GRID_HEIGHT - 1),
    ]
    goal = max(corners, key=lambda c: manhattan(c, friendly_centroid))

    path = astar(start, goal, sim.enemy_units, unit=enemies[0].state.get("type", "unknown"))
    if not path:
        return goal

    travelled = [0] * len(path)
    for i in range(1, len(path)):
        travelled[i] = travelled[i - 1] + manhattan(path[i - 1], path[i])

    for i, pos in enumerate(path):
        if travelled[i] > max_distance:
            break
        nearest = min(friendlies, key=lambda u: manhattan(pos, u.state["position"]))
        dist = manhattan(pos, nearest.state["position"])
        los = has_line_of_sight(pos, nearest.state["position"])
        if dist > nearest.state.get("vision_range", 0) or not los:
            return pos

    return path[min(len(path) - 1, i)]


def compute_defend_position(sim, max_distance=500):
    """Choose a defensive position relative to last known friendly units."""
    enemies = [u for u in sim.enemy_units if u.state.get("enemy_alive", False)]
    if not enemies:
        return (0, 0)

    avg_x = sum(u.state["position"][0] for u in enemies) // len(enemies)
    avg_y = sum(u.state["position"][1] for u in enemies) // len(enemies)
    start = (avg_x, avg_y)

    drone_mem = {
        name: pos
        for name, pos in sim.enemy_drone.last_known.items()
        if name in sim.friendly_units_dict
        and sim.friendly_units_dict[name].state.get("health", 0) > 0
    }
    if not drone_mem:
        return start

    closest = min(drone_mem, key=lambda n: manhattan(start, drone_mem[n]))
    friendly_pos = drone_mem[closest]
    friendly_unit = sim.friendly_units_dict[closest]
    friendly_range = friendly_unit.state.get("vision_range", 0)

    path = astar(start, friendly_pos, sim.enemy_units, unit=enemies[0].state.get("type", "unknown"))
    if not path:
        return friendly_pos

    travelled = [0] * len(path)
    for i in range(1, len(path)):
        travelled[i] = travelled[i - 1] + manhattan(path[i - 1], path[i])

    for i in range(len(path) - 1, -1, -1):
        if travelled[i] > max_distance:
            continue
        pos = path[i]
        dist = manhattan(pos, friendly_pos)
        los = has_line_of_sight(pos, friendly_pos)
        if dist > friendly_range or not los:
            return pos

    return start


def perform_attack(attacker, target):
    """Resolve an attack from one unit to another, applying damage if successful."""
    import math
    import random
    from log import logger

    tx, ty = target.state["position"]
    x, y = attacker.state["position"]
    dx, dy = tx - x, ty - y
    norm = math.hypot(dx, dy)

    if norm > 0:
        fx, fy = dx / norm, dy / norm
        attacker.state["facing"] = (fx, fy)
    else:
        fx, fy = attacker.state.get("facing", (0, 1))

    base_rate_of_fire = attacker.state.get("base_rate_of_fire", 1)
    group_size = attacker.state.get("current_group_size", 1)
    rate_of_fire = base_rate_of_fire * group_size
    num_attacks = get_num_attacks(rate_of_fire)

    acc_key = "friendly_accuracy" if attacker.state.get("type", "").startswith("friendly") else "accuracy"
    effective_accuracy = max(0, attacker.state.get(acc_key, 0.5) - attacker.state.get("suppression_from_enemy", 0.0))

    hits = 0
    penetrations = 0
    total_damage_dealt = 0.0

    logger.info(f"{attacker.name} (group size: {group_size}) attacks {target.name} "
                f"(group size: {target.state['current_group_size']}) {num_attacks} times, "
                f"accuracy: {effective_accuracy:.2f}")

    for _ in range(num_attacks):
        if random.random() < effective_accuracy:
            hits += 1
            target.state["suppression_from_enemy"] += attacker.state.get("suppression", 0.0)

            attack_dir = (fx, fy)
            target_facing = target.state.get("facing", (0, 1))
            target_norm = math.hypot(*target_facing)
            target_fx, target_fy = (target_facing[0] / target_norm, target_facing[1] / target_norm) if target_norm > 0 else (0, 0)

            dot_product = attack_dir[0] * target_fx + attack_dir[1] * target_fy
            dot_product = max(min(dot_product, 1), -1)
            angle_deg = math.degrees(math.acos(dot_product))
            direction = "rear" if angle_deg <= 45 else "side" if angle_deg <= 135 else "front"

            logger.info(f"{attacker.name} attacking {target.name} from the {direction}")

            arm_val = target.state.get(f"armor_{direction}", 0)
            D = attacker.state.get("penetration", 0) - arm_val
            penetration_prob = get_penetration_probability(D)
            if random.random() < penetration_prob:
                penetrations += 1
                damage = attacker.state.get("damage", 1.0) * group_size
                target.state["health"] -= damage
                target.state["cumulative_damage"] += damage
                total_damage_dealt += damage

                base_health = target.state.get("base_health", 1.0)
                units_lost = int(target.state["cumulative_damage"] // base_health)
                if units_lost > 0:
                    target.state["current_group_size"] = max(0, target.state["current_group_size"] - units_lost)
                    target.state["cumulative_damage"] -= units_lost * base_health
                    logger.info(f"{target.name} lost {units_lost} units, new group size: {target.state['current_group_size']}")

                logger.info(f"{attacker.name} penetrated {target.name} with D={D:.1f}, "
                            f"penetration prob={penetration_prob:.2f}, dealt {damage:.1f}, "
                            f"health now {target.state['health']:.1f}")

                if target.state["health"] <= 0 or target.state["current_group_size"] <= 0:
                    logger.info(f"{target.name} destroyed by {attacker.name}")
                    target.state.update({
                        "enemy_alive": False,
                        "current_group_size": 0,
                        "health": 0,
                        "cumulative_damage": 0
                    })

    logger.info(f"{attacker.name} attack summary: {hits}/{num_attacks} hits, "
                f"{penetrations}/{hits if hits > 0 else 1} penetrations, total damage dealt: {total_damage_dealt:.1f}")

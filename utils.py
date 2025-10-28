import heapq, random, math
import numpy as np
from environment import forest, forest_edge, river, cliffs, climb_entries
from terrain_utils import manhattan, get_line, in_bounds, neighbors
from drone import Drone
from config import CELL_SIZE
from log import logger

# Cache calculated paths so that expensive A* computations are not
# performed every simulation step.  Each cached entry is keyed by the
# tuple (start_position, goal_position, unit_type) and stores the
# remaining path along with a TTL indicating how many more steps the
# cached path should be used before re-planning.  When a cached path is
# consumed, the cache is updated for the next position with the TTL
# decremented, ensuring that paths are recomputed roughly every 10
# steps.
_path_cache = {}

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

def astar(start, goal, enemy_units=None, unit="unknown", avoid_positions=None):
    """A* pathfinding that fully blocks avoid_positions (strict constraint)."""
    from heapq import heappush, heappop

    frontier = []
    heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    avoid_positions = avoid_positions or set()

    while frontier:
        _, current = heappop(frontier)

        if current == goal:
            break

        for nxt in neighbors(current, in_bounds, river=river, cliffs=cliffs, climb_entries=climb_entries):
            new_cost = cost_so_far[current] + 1
            if unit in ["scout", "infantry"] and not is_in_cover(nxt):
                new_cost += 5

            if unit in ["tank", "artillery"] and is_in_cover(nxt):
                new_cost += 5

            if unit == "anti-tank" and nxt not in forest_edge:
                new_cost += 5
            
            if nxt in avoid_positions:
                new_cost += 300

            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + manhattan(nxt, goal)
                heappush(frontier, (priority, nxt))
                came_from[nxt] = current

    if goal not in came_from:
        return []

    # Reconstruct path
    path = []
    current = goal
    while current:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


def next_step(start, goal, enemy_units=None, unit="unknown"):
    """Return the next grid step along the path to a goal using A*.

    Paths are cached so that the expensive A* search is only performed
    periodically.  Cached paths expire after roughly ten steps and will
    be recomputed when the cache misses or the goal changes.
    """

    key = (start, goal, unit)
    cached = _path_cache.pop(key, None)
    if cached:
        path, ttl = cached
        if path:
            next_pos = path[0]
            remaining = path[1:]
            ttl -= 1
            if remaining and ttl > 0:
                _path_cache[(next_pos, goal, unit)] = (remaining, ttl)
            return next_pos
        return start

    path = astar(start, goal, enemy_units, unit)
    if len(path) >= 2:
        next_pos = path[1]
        remaining = path[2:]
        ttl = 9  # recompute after ~10 steps
        if remaining and ttl > 0:
            _path_cache[(next_pos, goal, unit)] = (remaining, ttl)
        return next_pos
    return start

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


def _is_out_of_all_enemy_attack_range(pos_list, sim, target_enemy_name=None):
    """
    Check if all positions in pos_list are outside attack range & LOS of all enemies
    except the target_enemy_name (which we're allowed to approach).
    """
    for pos in pos_list:
        for enemy in sim.enemy_units:
            if not enemy.state.get("enemy_alive", False):
                continue
            if enemy.name == target_enemy_name:
                continue  # Allowed to approach this enemy

            enemy_pos = enemy.state.get("position")
            enemy_range = enemy.state.get("attack_range", 0)
            dist = manhattan(pos, enemy_pos)
            if dist <= enemy_range and has_line_of_sight(pos, enemy_pos):
                return False
    return True

def _is_path_safe_from_other_enemies(path, sim, target_enemy_name):
    """
    Check if all positions in the path are safe from LOS and range of enemies *other than* target_enemy_name.
    """
    for pos in path:
        for enemy in sim.enemy_units:
            if not enemy.state.get("enemy_alive", False):
                continue
            if enemy.name == target_enemy_name:
                continue  # Skip the currently targeted enemy

            enemy_pos = enemy.state.get("position")
            enemy_range = enemy.state.get("attack_range", 20)
            dist = manhattan(pos, enemy_pos)
            if dist <= enemy_range and has_line_of_sight(pos, enemy_pos):
                return False
    return True

def get_all_enemy_attack_zones(sim, exclude_enemy_name=None):
    """
    Compute positions that are within attack range of all alive enemies,
    excluding a given enemy if needed.
    """
    zone = set()
    for enemy in sim.enemy_units:
        if not enemy.state.get("enemy_alive", False):
            continue
        # if exclude_enemy_name and enemy.name == exclude_enemy_name:
        #     continue

        x, y = enemy.state["position"]
        enemy_range = int(enemy.state.get("attack_range", 10))

        for dx in range(-enemy_range, enemy_range + 1):
            for dy in range(-enemy_range, enemy_range + 1):
                pos = (x + dx, y + dy)
                if not in_bounds(pos):
                    continue
                # Use Euclidean distance to approximate circular zone
                if math.hypot(dx, dy) <= enemy_range:
                    zone.add(pos)

    return zone

def is_area_far_enough(area_idx1, area_idx2, min_distance=2, n_cols=3):
    """Estimate 'Manhattan' distance between drone areas on grid."""
    row1, col1 = divmod(area_idx1, n_cols)
    row2, col2 = divmod(area_idx2, n_cols)
    return abs(row1 - row2) + abs(col1 - col2) >= min_distance

def compute_staging_position(sim, enemy_name, front_unit, unit, max_distance=20):
    # # logger.info(f"[compute_staging_position] Computing staging position for {unit.state['type']} ({unit.name}) vs {enemy_name}")

    if enemy_name not in sim.friendly_drone.last_known:
        # logger.warning(f"[compute_staging_position] Enemy {enemy_name} not in drone memory")
        return None

    enemy_unit = sim.enemy_units_dict.get(enemy_name)
    if not enemy_unit or not enemy_unit.state.get("enemy_alive", False):
        # logger.warning(f"[compute_staging_position] Enemy {enemy_name} is not alive or not found")
        return None

    friendlies = [u for u in sim.friendly_units if u.state.get("current_group_size", 0) > 0]
    if not friendlies:
        # logger.warning(f"[compute_staging_position] No friendly units available for staging")
        return None

    start = unit.state["position"]
    enemy_pos = enemy_unit.state["position"]
    avoid_positions = get_all_enemy_attack_zones(sim)
    sim.debug_avoid_positions = avoid_positions

    best_pos = None
    if unit.state.get("type") == "artillery":
        best_score = float("-inf")
    else:
        best_score = float("inf")
    best_path_to_staging = []
    best_path_to_enemy = []

    search_range = int(unit.state.get("attack_range", max_distance)) if unit.state.get("type") == "artillery" else max_distance
    for dx in range(-search_range, search_range + 1, 2):
        for dy in range(-search_range, search_range + 1, 2):
            pos = (enemy_pos[0] + dx, enemy_pos[1] + dy)

            if not in_bounds(pos) or pos in river:
                continue
            if pos == enemy_pos:
                continue
            if (pos in forest or pos in forest_edge) and unit.state['type'] not in ["infantry", "anti-tank"]:
                continue
            distance_to_enemy = manhattan(pos, enemy_pos)
            if unit.state.get("type") == "artillery":
                attack_range = int(unit.state.get("attack_range", max_distance))
                if distance_to_enemy > attack_range:
                    continue
            elif distance_to_enemy > max_distance:
                continue

            extra_penalty = 0

            if pos in avoid_positions:
                for enemy in sim.enemy_units:
                    if not enemy.state.get("enemy_alive", False):
                        continue
                    enemy_pos_check = enemy.state["position"]
                    enemy_range = int(enemy.state.get("attack_range", 10))

                    if manhattan(pos, enemy_pos_check) <= enemy_range:
                        dist = manhattan(pos, enemy_pos_check)
                        if dist == 0:
                            dist = 0.1  # avoid div by zero
                        form = 1 / dist
                        penalty = 1000 * form if has_line_of_sight(pos, enemy_pos_check) else 250 * form
                        extra_penalty += penalty

                        # logger.info(
                        #     f"[penalty] Pos {pos} penalized by {enemy.name}: dist={dist:.1f}, "
                        #     f"LOS={'Yes' if has_line_of_sight(pos, enemy_pos_check) else 'No'}, "
                        #     f"penalty={penalty:.1f}"
                        # )

            # Compute paths
            path_to_staging = astar(start, pos, enemy_units=sim.enemy_units, unit=unit.state['type'], avoid_positions=avoid_positions)
            if not path_to_staging:
                continue

            path_to_enemy = astar(pos, enemy_pos, enemy_units=sim.enemy_units, unit=unit.state['type'], avoid_positions=avoid_positions)
            if not path_to_enemy:
                continue

            if unit.state.get("type") == "artillery":
                total_score = distance_to_enemy - extra_penalty - len(path_to_staging)
                if total_score > best_score:
                    best_score = total_score
                    best_pos = pos
                    best_path_to_staging = path_to_staging
                    best_path_to_enemy = path_to_enemy
            else:
                total_cost = len(path_to_staging) + len(path_to_enemy) + extra_penalty
                # logger.info(f"[compute_staging_position] Evaluating pos {pos} with total_cost {total_cost:.1f}; extra_penalty {extra_penalty:.1f}")

                if total_cost < best_score:
                    best_score = total_cost
                    best_pos = pos
                    best_path_to_staging = path_to_staging
                    best_path_to_enemy = path_to_enemy

    if best_pos:
        if not hasattr(sim, "debug_paths"):
            sim.debug_paths = []
        sim.debug_paths.append({
            "unit": unit,
            "path_to_staging": best_path_to_staging,
            "path_to_enemy": best_path_to_enemy
        })

        # logger.debug(f"[compute_staging_position] Selected {best_pos} with score {best_score:.1f}")
        return best_pos

    # logger.warning(f"[compute_staging_position] No valid pos found for {unit.state['type']} vs {enemy_name}; fallback needed")
    return None

def compute_flanking_position(sim, enemy_name, front_unit, unit, max_distance=20):
    # logger.info(f"[compute_flanking_position] Computing flanking position for {unit.state['type']} ({unit.name}) vs {enemy_name}")

    if enemy_name not in sim.friendly_drone.last_known:
        # logger.warning(f"[compute_flanking_position] Enemy {enemy_name} not in drone memory")
        return None, None

    enemy_unit = sim.enemy_units_dict.get(enemy_name)
    if not enemy_unit or not enemy_unit.state.get("enemy_alive", False):
        # logger.warning(f"[compute_flanking_position] Enemy {enemy_name} is not alive or not found")
        return None, None

    start = unit.state["position"]
    enemy_pos = enemy_unit.state["position"]
    avoid_positions = get_all_enemy_attack_zones(sim)

    # Reference vector from enemy to staging for flank angle
    staging_vector_x = front_unit.state["position"][0] - enemy_pos[0]
    staging_vector_y = front_unit.state["position"][1] - enemy_pos[1]
    norm = math.hypot(staging_vector_x, staging_vector_y) or 1.0
    ex, ey = staging_vector_x / norm, staging_vector_y / norm

    best_pos = None
    if unit.state.get("type") == "artillery":
        best_score = float("-inf")
    else:
        best_score = float("inf")
    best_path_to_flank = []
    best_path_to_enemy = []

    search_range = int(unit.state.get("attack_range", max_distance)) if unit.state.get("type") == "artillery" else max_distance
    for dx in range(-search_range, search_range + 1, 2):
        for dy in range(-search_range, search_range + 1, 2):
            pos = (enemy_pos[0] + dx, enemy_pos[1] + dy)

            if not in_bounds(pos) or pos in river or pos == enemy_pos:
                continue
            if (pos in forest or pos in forest_edge) and unit.state["type"] not in ["infantry", "anti-tank"]:
                continue
            distance_to_enemy = manhattan(pos, enemy_pos)
            if unit.state.get("type") == "artillery":
                attack_range = int(unit.state.get("attack_range", max_distance))
                if distance_to_enemy > attack_range:
                    continue
            elif distance_to_enemy > max_distance:
                continue

            # Angle check: prefer side/rear
            vx, vy = pos[0] - enemy_pos[0], pos[1] - enemy_pos[1]
            vec_norm = math.hypot(vx, vy) or 1.0
            vx, vy = vx / vec_norm, vy / vec_norm
            cos_angle = vx * ex + vy * ey
            if cos_angle > 0.7:
                continue

            # Penalize if in avoid zone
            extra_penalty = 0
            if pos in avoid_positions:
                for enemy in sim.enemy_units:
                    if not enemy.state.get("enemy_alive", False):
                        continue
                    enemy_pos_check = enemy.state["position"]
                    enemy_range = int(enemy.state.get("attack_range", 10))

                    if manhattan(pos, enemy_pos_check) <= enemy_range:
                        dist = manhattan(pos, enemy_pos_check)
                        if dist == 0:
                            dist = 0.1
                        form = 1 / dist
                        penalty = 1000 * form if has_line_of_sight(pos, enemy_pos_check) else 250 * form
                        extra_penalty += penalty

                        # logger.info(
                        #     f"[penalty] Pos {pos} penalized by {enemy.name}: dist={dist:.1f}, "
                        #     f"LOS={'Yes' if has_line_of_sight(pos, enemy_pos_check) else 'No'}, "
                        #     f"penalty={penalty:.1f}"
                        # )

            # Compute paths
            path_to_flank = astar(start, pos, enemy_units=sim.enemy_units, unit=unit.state["type"], avoid_positions=avoid_positions)
            if not path_to_flank:
                continue

            path_to_enemy = astar(pos, enemy_pos, enemy_units=sim.enemy_units, unit=unit.state['type'], avoid_positions=avoid_positions)
            if not path_to_enemy:
                continue

            if unit.state.get("type") == "artillery":
                total_score = distance_to_enemy - extra_penalty - len(path_to_flank)
                if total_score > best_score:
                    best_score = total_score
                    best_pos = pos
                    best_path_to_flank = path_to_flank
                    best_path_to_enemy = path_to_enemy
            else:
                total_cost = len(path_to_flank) + len(path_to_enemy) + extra_penalty
                # logger.info(f"[compute_flanking_position] Evaluating pos {pos} with total_cost {total_cost:.1f}; extra_penalty {extra_penalty:.1f}")

                if total_cost < best_score:
                    best_score = total_cost
                    best_pos = pos
                    best_path_to_flank = path_to_flank
                    best_path_to_enemy = path_to_enemy

    if best_pos:
        if not hasattr(sim, "debug_paths"):
            sim.debug_paths = []
        sim.debug_paths.append({
            "unit": unit,
            "path_to_flank": best_path_to_flank,
            "path_to_enemy": best_path_to_enemy
        })

        # logger.info(f"[compute_flanking_position] Selected flank pos {best_pos} with score {best_score:.1f}")
        return best_pos, front_unit.state["position"]

    # logger.warning(f"[compute_flanking_position] No valid flank pos found for {unit.state['type']} vs {enemy_name}; fallback to side")
    fallback_flank = (enemy_pos[0] + 5, enemy_pos[1])
    return fallback_flank, front_unit.state["position"]


def compute_retreat_point(sim, max_distance=500, retreating_side="enemy"):
    """Determine a fallback position for either side against the opposing force."""
    from config import GRID_WIDTH, GRID_HEIGHT

    if retreating_side == "enemy":
        retreaters = [u for u in sim.enemy_units if u.state.get("enemy_alive", False)]
        opponents = [u for u in sim.friendly_units if u.state.get("health", 0) > 0]
    else:
        retreaters = [u for u in sim.friendly_units if u.state.get("health", 0) > 0]
        opponents = [u for u in sim.enemy_units if u.state.get("enemy_alive", False)]

    if not retreaters:
        return (0, 0)
    if not opponents:
        return retreaters[0].state.get("position", (0, 0))

    # Centroid of retreating units as a starting position
    avg_x = sum(u.state["position"][0] for u in retreaters) // len(retreaters)
    avg_y = sum(u.state["position"][1] for u in retreaters) // len(retreaters)
    start = (avg_x, avg_y)

    # Opponent closest to the centroid
    closest = min(opponents, key=lambda u: manhattan(start, u.state["position"]))
    target = closest.state["position"]

    best = None
    best_score = float("-inf")

    # Search an area around the closest enemy for a covered spot out of LOS
    for dx in range(-10, 11):
        for dy in range(-10, 11):
            pos = (target[0] + dx, target[1] + dy)
            if not in_bounds(pos) or pos in river or pos in cliffs:
                continue

            path = astar(start, pos, sim.enemy_units, unit=retreaters[0].state.get("type", "unknown"))
            if not path:
                continue

            travelled = sum(manhattan(path[i - 1], path[i]) for i in range(1, len(path)))
            if travelled > max_distance:
                continue

            dist = manhattan(pos, target)
            los = has_line_of_sight(pos, target)

            score = -dist
            if is_in_cover(pos):
                score += 3
            if dist > closest.state.get("attack_range", 0) or not los:
                score += 1
            if los and dist <= closest.state.get("attack_range", 0):
                score -= 5

            if score > best_score:
                best_score = score
                best = pos

    if best:
        return best

    # Fallback to corner-based retreat if no suitable position is found
    fx = sum(u.state["position"][0] for u in opponents) // len(opponents)
    fy = sum(u.state["position"][1] for u in opponents) // len(opponents)
    opponent_centroid = (fx, fy)

    corners = [
        (0, 0),
        (0, GRID_HEIGHT - 1),
        (GRID_WIDTH - 1, 0),
        (GRID_WIDTH - 1, GRID_HEIGHT - 1),
    ]
    goal = max(corners, key=lambda c: manhattan(c, opponent_centroid))

    path = astar(start, goal, sim.enemy_units, unit=retreaters[0].state.get("type", "unknown"))
    if not path:
        return goal

    travelled = [0] * len(path)
    for i in range(1, len(path)):
        travelled[i] = travelled[i - 1] + manhattan(path[i - 1], path[i])

    for i, pos in enumerate(path):
        if travelled[i] > max_distance:
            break
        nearest = min(opponents, key=lambda u: manhattan(pos, u.state["position"]))
        dist = manhattan(pos, nearest.state["position"])
        los = has_line_of_sight(pos, nearest.state["position"])
        if dist > nearest.state.get("attack_range", 0) or not los:
            return pos

    return path[min(len(path) - 1, i)]

def compute_unit_defend_position(sim, unit, radius=50, column_range=None):
    """Select a defend position for a specific unit near the outpost.

    The radius is deliberately large so enemies end up quite spread out
    rather than clustered tightly around the outpost.

    ``column_range`` optionally restricts the search to an inclusive range of
    x-coordinates.  A heuristic penalty is also applied to positions that are
    far from the line connecting the friendly forces and the outpost so that
    defenders stay roughly between the attackers and their objective.
    """
    from environment import forest, forest_edge, river, cliffs
    from config import GRID_HEIGHT

    ox, oy = unit.state.get("outpost_position", (0, 0))
    if column_range:
        max_dist = max(abs(ox - column_range[0]), abs(ox - column_range[1]))
        radius = max(radius, max_dist + GRID_HEIGHT)

    # Determine a reference line between friendly centroid and the outpost.
    friendlies = [f for f in sim.friendly_units if f.state.get("health", 0) > 0]
    line_dx = line_dy = 0
    fx = fy = 0
    if friendlies:
        fx = sum(f.state["position"][0] for f in friendlies) / len(friendlies)
        fy = sum(f.state["position"][1] for f in friendlies) / len(friendlies)
        line_dx = ox - fx
        line_dy = oy - fy
        line_len = math.hypot(line_dx, line_dy) or 1.0

    best = unit.state["position"]
    best_score = float("-inf")

    for dx in range(-radius, radius + 1, 2):
        for dy in range(-radius, radius + 1, 2):
            pos = (ox + dx, oy + dy)
            if column_range and not (column_range[0] <= pos[0] <= column_range[1]):
                continue
            if not in_bounds(pos):
                continue
            if manhattan((ox, oy), pos) > radius:
                continue
            if pos in river or pos in cliffs or pos in forest:
                continue
            path = astar(unit.state["position"], pos, sim.enemy_units, unit.state.get("type", "unknown"))
            if len(path) < 2:
                continue
            score = -0.05 * manhattan(unit.state["position"], pos)
            score += 0.2 * manhattan((ox, oy), pos)

            # cover = is_in_cover(pos)
            if unit.state.get("type") in ("infantry", "anti-tank"):
                score += 3 if pos in forest_edge else -1
            else:
                score += 1 if pos not in forest_edge else -1

            if unit.state.get("attack_range", 0) > 2000 / CELL_SIZE:
                open_dirs = 0
                for step in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    for i in range(1, 4):
                        t = (pos[0] + step[0] * i, pos[1] + step[1] * i)
                        if not in_bounds(t) or not has_line_of_sight(pos, t):
                            break
                    else:
                        open_dirs += 1
                score += open_dirs

            # Penalize crowding near other defenders
            for e in sim.enemy_units:
                if e is unit or not e.state.get("enemy_alive", False):
                    continue
                other = e.state.get("defend_position")
                if other:
                    d = manhattan(pos, other)
                    if d < 6:
                        score -= (6 - d) * 2
                    if other[1] == pos[1]:
                        score -= 1

            # Penalize positions that are far from the friendly→outpost line
            if friendlies:
                # Perpendicular distance from candidate to the line
                dist_line = abs(line_dy * (pos[0] - fx) - line_dx * (pos[1] - fy)) / line_len
                if dist_line > 10:
                    continue
                score -= 0.005 * (dist_line ** 2)

            if score > best_score:
                best_score = score
                best = pos

    return best


def get_bridges():
    """Return lists of tiles representing the map's bridges."""
    from environment import (
        BRIDGE_X_LEFT,
        BRIDGE_X_DIAG_ANCHOR,
        BRIDGE_X_RIGHT,
        MID_Y0,
        MID_Y1,
        RIGHT_Y0,
        RIGHT_Y1,
        _shift_y,
    )

    bridges = []

    # Left bridge is a vertical gap in the center band
    left_bridge = [(BRIDGE_X_LEFT, _shift_y(y)) for y in range(MID_Y0, MID_Y1)]
    bridges.append(left_bridge)

    # Diagonal bridge lies along the diagonal connector
    diag_bridge = []
    for i in range(5):
        y0, y1 = MID_Y0 + i, RIGHT_Y0 + i
        line = [p for p in get_line((25, _shift_y(y0)), (49, _shift_y(y1))) if in_bounds(p)]
        if not line:
            continue
        idx = min(range(len(line)), key=lambda k: abs(line[k][0] - BRIDGE_X_DIAG_ANCHOR))
        diag_bridge.append(line[idx])
    bridges.append(diag_bridge)

    # Right bridge is a vertical gap in the right band
    right_bridge = [(BRIDGE_X_RIGHT, _shift_y(y)) for y in range(RIGHT_Y0, RIGHT_Y1)]
    bridges.append(right_bridge)

    return bridges


def compute_bridge_defend_position(sim, unit, bridge_tiles, radius=15):
    """Find a defensive position south of a bridge with light crowding."""
    cx = sum(x for x, _ in bridge_tiles) / len(bridge_tiles)
    cy = min(y for _, y in bridge_tiles)

    best = None
    best_score = float("-inf")

    for dx in range(-radius, radius + 1, 2):
        for dy in range(1, radius + 1, 2):
            pos = (int(cx) + dx, cy - dy)
            if not in_bounds(pos):
                continue
            if pos in river or pos in cliffs or pos in forest:
                continue
            path = astar(unit.state["position"], pos, sim.enemy_units, unit.state.get("type", "unknown"))
            if len(path) < 2:
                continue

            score = -0.1 * manhattan((int(cx), cy), pos)
            if unit.state.get("type") in ("infantry", "anti-tank"):
                score += 3 if pos in forest_edge else -1
            else:
                score += 1 if pos not in forest_edge else -1

            for e in sim.enemy_units:
                if e is unit or not e.state.get("enemy_alive", False):
                    continue
                other = e.state.get("defend_position")
                if other:
                    d = manhattan(pos, other)
                    if d < 6:
                        score -= (6 - d) * 2

            if score > best_score:
                best_score = score
                best = pos

    if not best:
        fallback = (int(cx), cy - 1)
        if (
            in_bounds(fallback)
            and fallback not in river
            and fallback not in cliffs
            and fallback not in forest
        ):
            return fallback
        return unit.state["position"]

    return best


def perform_attack(attacker, target):
    """Resolve an attack from one unit to another, applying damage if successful."""
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

    # Record that the target was attacked by this attacker this step
    target.state.setdefault("attacked_by", [])
    if attacker.name not in target.state["attacked_by"]:
        target.state["attacked_by"].append(attacker.name)

    acc_key = "friendly_accuracy" if attacker.state.get("type", "").startswith("friendly") else "accuracy"
    effective_accuracy = max(0, attacker.state.get(acc_key, 0.5) - attacker.state.get("suppression_from_enemy", 0.0))
    # Precompute values used for every attack
    attack_dir = (fx, fy)
    target_facing = target.state.get("facing", (0, 1))
    target_norm = math.hypot(*target_facing)
    target_fx, target_fy = (target_facing[0] / target_norm, target_facing[1] / target_norm) if target_norm > 0 else (0, 0)
    dot_product = attack_dir[0] * target_fx + attack_dir[1] * target_fy
    dot_product = max(min(dot_product, 1), -1)
    angle_deg = math.degrees(math.acos(dot_product))
    direction = "rear" if angle_deg <= 45 else "side" if angle_deg <= 135 else "front"
    arm_val = target.state.get(f"armor_{direction}", 0)
    D = attacker.state.get("penetration", 0) - arm_val
    penetration_prob = get_penetration_probability(D)

    hits = 0
    penetrations = 0
    total_damage_dealt = 0.0

    # Vectorized sampling to resolve multiple attacks at once
    if num_attacks > 0 and effective_accuracy > 0:
        hits = np.random.binomial(num_attacks, effective_accuracy)
    if hits > 0:
        target.state["suppression_from_enemy"] += attacker.state.get("suppression", 0.0) * hits
        target.state["hit_this_step"] = True
        if penetration_prob > 0:
            penetrations = np.random.binomial(hits, penetration_prob)
        if penetrations > 0:
            damage = attacker.state.get("damage", 1.0) * group_size
            total_damage_dealt = damage * penetrations
            target.state["health"] -= total_damage_dealt
            target.state["cumulative_damage"] += total_damage_dealt

            base_health = target.state.get("base_health", 1.0)
            units_lost = int(target.state["cumulative_damage"] // base_health)
            if units_lost > 0:
                target.state["current_group_size"] = max(0, target.state["current_group_size"] - units_lost)
                target.state["cumulative_damage"] -= units_lost * base_health

            if target.state["health"] <= 0 or target.state["current_group_size"] <= 0:
                target.state.update({
                    "enemy_alive": False,
                    "current_group_size": 0,
                    "health": 0,
                    "cumulative_damage": 0
                })

    # If a friendly unit attacks an enemy, mark the attacker so the enemy
    # can react on its next turn. This triggers the Delay branch, allowing
    # the enemy to either counterattack or fall back.
    #
    # Enemy units use the `enemy_alive` flag instead of a type prefix, so we
    # simply check for that flag to identify enemies. The previous implementation
    # looked for a type string starting with "enemy", which never matched the
    # actual templates (they use plain unit types like "infantry"). As a result,
    # enemies never recorded their attackers and could not retaliate.
    if target.state.get("enemy_alive", False):
        target.state["last_attacker"] = attacker.name

    # # logger.info(f"{attacker.name} attack summary: {hits}/{num_attacks} hits, "
    #             f"{penetrations}/{hits if hits > 0 else 1} penetrations, total damage dealt: {total_damage_dealt:.1f}")

import random
import heapq
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Circle
import numpy as np
import math
import logging
import copy

from plotter import SimulationPlotter

# Write INFO+ messages to simulation.log, DEBUG goes nowhere by default
logging.basicConfig(
    filename="simulation.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
)
logger = logging.getLogger("Sim")

###############################
# Grid, Obstacles, and LOS
###############################

GRID_WIDTH = 75
GRID_HEIGHT = 50
CELL_SIZE = 100
random.seed(42)

def get_line(start, end):
    x1, y1 = start
    x2, y2 = end
    line = []
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    x, y = x1, y1
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x2:
            line.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            line.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    line.append((x, y))
    return line

river = set()
for x in range(0, GRID_WIDTH):
    for y in range(0, GRID_HEIGHT):
        if (x < 7 and y < 25):
            river.add((x, y))
        elif (7 <= x < 25 and 20 <= y < 25):
            river.add((x, y))
        elif (x >= 50 and 35 <= y < 40):
            river.add((x, y))
        for i in range(0, 5):
            for pos in get_line((25, 20+i), (49, 35+i)):
                river.add(pos)

for x in range(11, 14):
    for y in range(20, 25):
        river.discard((x, y))

for i in range(6):
    for pos in get_line((40, 21 + i), (34, 29 + i)):
        river.discard(pos)

for x in range(61, 64):
    for y in range(35, 40):
        river.discard((x, y))

def init_forest(p, width, height):
    return {
        (x,y)
        for x in range(width)
        for y in range(height)
        if random.random() < p and (x,y) not in river
    }

def count_neighbors(forest, x, y):
    n = 0
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            if dx==0 and dy==0: continue
            if (x+dx, y+dy) in forest:
                n += 1
    return n

def smooth_forest(forest, width, height, survive=4, birth=5):
    new = set()
    for x in range(width):
        for y in range(height):
            n = count_neighbors(forest, x, y)
            if ( (x,y) in forest and n >= survive ) or ( (x,y) not in forest and n >= birth ):
                new.add((x,y))
    return new

forest = init_forest(p=0.45, width=GRID_WIDTH, height=GRID_HEIGHT)
for _ in range(4):
    forest = smooth_forest(forest, GRID_WIDTH, GRID_HEIGHT)

cliff_defs = [((36,10),(44,12),(0, 1))]
cliffs = {}
for s,e,n in cliff_defs:
    for c in get_line(s,e):
        cliffs[c] = n

climb_entries = {
    (cx - nx, cy - ny): (cx, cy)
    for (cx, cy), (nx, ny) in cliffs.items()
}

def in_bounds(pos):
    return 0 <= pos[0] < GRID_WIDTH and 0 <= pos[1] < GRID_HEIGHT

def neighbors(pos):
    results = []
    for p in [(pos[0]-1,pos[1]), (pos[0]+1,pos[1]), (pos[0],pos[1]-1), (pos[0],pos[1]+1)]:
        if not in_bounds(p) or p in river:
            continue
        if pos in cliffs:
            nx, ny = cliffs[pos]
            entry = (pos[0] - nx, pos[1] - ny)
            if p != entry:
                continue
        if p in cliffs:
            if climb_entries.get(pos) != p:
                continue
        results.append(p)
    return results

forest_edge = [pos for pos in forest if any(n not in forest for n in neighbors(pos))]
forest -= set(forest_edge)

def manhattan(p, q):
    return abs(p[0]-q[0]) + abs(p[1]-q[1])

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
        for nxt in neighbors(current):
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

###############################
# Helper Functions
###############################

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

def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0

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


###############################
# HTN Domains and Planners
###############################

secure_outpost_domain = {
    "SecureOutpostMission": [
        # Condition 1: Outpost already secured
        (lambda s: isinstance(s, dict) and any(u.state.get("outpost_secured", False)
                    for u in getattr(s.get("sim", object()), "friendly_units", [])), []),

        # Condition 2: All enemies defeated, proceed to secure outpost
        (lambda s: isinstance(s, dict) and not any(
            e.state.get("enemy_alive", False)
            for e in getattr(s.get("sim", object()), "enemy_units_dict", {}).values()
        ), [("SecureOutpost", None)]),

        # Condition 3: Spotted enemies exist, defeat them
        (lambda s: isinstance(s, dict) and any(
            name in getattr(s.get("sim", object()), "enemy_units_dict", {}) and
            s["sim"].enemy_units_dict[name].state.get("enemy_alive", False)
            for name in s.get("spotted_enemies", [])
        ), lambda s: [
            sub
            for name in sorted(
                [
                    name for name in s.get("spotted_enemies", [])
                    if name in getattr(s.get("sim", object()), "enemy_units_dict", {}) and
                    s["sim"].enemy_units_dict[name].state.get("enemy_alive", False)
                ],
                key=lambda n: manhattan(
                    s.get("unit", {}).state.get("position", (0, 0)),
                    getattr(s.get("sim", object()), "enemy_units_dict", {}).get(n, {}).state.get("position", (0, 0))
                )
            )
            for sub in [
                ("AttackEnemy", name) if s.get("unit", {}).can_attack(s["sim"].enemy_units_dict[name])
                else ("Move", name)
            ]
        ]),

        # Condition 4: Default to Hold (waiting for drone to spot enemies)
        (lambda s: isinstance(s, dict), ["Hold"]),
    ],

    "DefeatEnemies": [
        # Condition 1: Spotted enemies exist
        (
            lambda s: isinstance(s, dict) and bool(s.get("spotted_enemies", [])),
            lambda s: [
                ("AttackEnemy", name) if s.get("unit", {}).can_attack(getattr(s.get("sim", object()), "enemy_units_dict", {}).get(name))
                else ("Move", name)
                for name in s.get("spotted_enemies", [])
                if name in getattr(s.get("sim", object()), "enemy_units_dict", {}) and
                s["sim"].enemy_units_dict[name].state.get("enemy_alive", False)
            ]
        ),
        # Condition 2: Default to Hold
        (
            lambda s: isinstance(s, dict),
            ["Hold"]
        ),
    ],

    "SecureOutpost": [
        (lambda s: isinstance(s, dict) and s.get("position") != s.get("outpost_position"), [("Move", "outpost")]),
        (lambda s: isinstance(s, dict) and s.get("position") == s.get("outpost_position"), ["SecureOutpostNoArg"]),
    ],
}


class HTNPlanner:
    def __init__(self, domain):
        self.domain = domain

    def plan(self, task, state):
        # Handle task as either a string or a tuple
        task_name = task[0] if isinstance(task, tuple) else task
        task_arg = task[1] if isinstance(task, tuple) else None
        
        # If task_name is not in domain, return it as a primitive task
        if task_name not in self.domain:
            return [(task_name, task_arg)] if task_arg is not None else [task_name]
        
        # Evaluate conditions in the domain
        for condition, subtasks in self.domain[task_name]:
            if condition(state):
                task_list = subtasks(state) if callable(subtasks) else subtasks
                plan = []
                for subtask in task_list:
                    # Ensure subtask is processed correctly
                    sub_plan = self.plan(subtask, state)  # Pass the original state
                    if sub_plan is None:
                        return None
                    plan.extend(sub_plan)
                return plan
        return None

###############################
# Refactored Enemy Classes
###############################
class Drone:
    def __init__(self, side, target_side, n_cols=3, n_rows=2, stay_rounds=10, spot_prob=0.2):
        # build 6 (xmin,xmax,ymin,ymax) areas
        w, h = GRID_WIDTH / n_cols, GRID_HEIGHT / n_rows
        self.areas = [
            (j*w, (j+1)*w, i*h, (i+1)*h)
            for i in range(n_rows)
            for j in range(n_cols)
        ]
        self.stay_rounds = stay_rounds
        self.spot_prob = spot_prob
        self.rounds_in_area = 0
        self.side = side
        self.target_side = target_side
        self.current_area = 3 if self.side == "friendly" else 2
        self.route = [3, 4, 5, 2, 1, 0] if self.side == "friendly" else [2, 1, 0, 3, 4, 5]
        self.route_idx = 0

        # maps unit_name -> (last_x, last_y)
        self.last_known = {}

    def _in_area(self, pos, bounds):
        x, y = pos
        xmin, xmax, ymin, ymax = bounds
        return xmin <= x < xmax and ymin <= y < ymax
    
    def update(self, sim):
        """
        Call once per Simulation.step().
        Spots any friendly or enemy in the current area,
        then advances the area counter every stay_rounds.
        """
        logger.info(f"Drone({self.side}) update: target_side={self.target_side}")
        if self.target_side == "friendly":
            units = sim.friendly_units
        else:
            units = sim.enemy_units
        
        bounds = self.areas[self.current_area]

        # try spotting every unit in this area
        in_area = [u for u in units 
               if u.state.get("health",0)>0 
               and self._in_area(u.state["position"], bounds)]
        
        if in_area:
            # Use Poisson approximation to sample the number of spots
            # 1) compute λ = sum of individual spot-probs
            ps = [self.spot_prob for _ in in_area]  # or customize per-unit
            lam = sum(ps)
            # 2) sample how many spots this round
            k = np.random.poisson(lam)
            k = min(k, len(in_area))
            # 3) pick k units to spot (here uniform)
            spotted = np.random.choice(in_area, size=k, replace=False)
            for u in spotted:
                self.last_known[u.name] = u.state["position"]
                logger.info(f"Drone ({self.side}) spotted {u.name} at {u.state['position']}")

        # move on after enough rounds
        self.rounds_in_area += 1
        if self.rounds_in_area >= self.stay_rounds:
            self.rounds_in_area = 0
            self.route_idx = (self.route_idx + 1) % len(self.route)
            self.current_area = self.route[self.route_idx]
            logger.info(f"{self.side}'s drone moves to area #{self.current_area + 1}")


class EnemyUnit:
    def __init__(self, name, state, domain):
        self.name = name
        self.state = state
        self.planner = HTNPlanner(domain)
        self.current_plan = []
        self.last_position = state["position"]
        self.last_health = state["health"]
        self.last_group_size = state["current_group_size"]

    def update_plan(self, friendly_units, force_replan=False):
        mission = "DefendAreaMission"

        # 1) Build the HTN state skeleton
        s = copy.deepcopy(self.state)
        s["sim"]            = self.sim
        s["spotted_enemies"] = self.state.get("spotted_enemies", [])
        s["unit"]           = self

        # 2) Gather all valid branches
        methods     = self.planner.domain[mission]
        valid       = [(cond, subs) for cond, subs in methods if cond(s)]
        valid_count = len(valid)
        old_count   = self.state.get("_last_valid_count", 0)
        logger.info(f"{self.name} valid methods (count={valid_count}): {valid}")

        # 4) Only *lock in* a branch the first time we actually have >1 valid choices
        if valid_count > 1 and "_branch_choice" not in self.state:
            # first multi‐branch moment: pick and lock
            idx = random.randrange(valid_count)
            self.state["_branch_choice"] = idx
            logger.info(f"{self.name} picked branch #{idx}")
        elif "_branch_choice" in self.state:
            # we already locked a branch—keep using it
            idx = self.state["_branch_choice"]
            # clamp if the number of branches has shrunk
            if idx >= valid_count:
                idx = valid_count - 1
                self.state["_branch_choice"] = idx
            logger.info(f"{self.name} keeps branch #{idx}")
        else:
            # still only one choice, no lock yet
            idx = 0
            logger.info(f"{self.name} only has branch #0")

        # 5) Filter to only truly visible friendlies
        visible = []
        for u in friendly_units:
            if u.state.get("health", 0) <= 0:
                continue
            dist     = manhattan(self.state["position"], u.state["position"])
            los      = has_line_of_sight(self.state["position"], u.state["position"])
            stealth  = u.state.get("stealth_modifier", 0)
            in_cover = is_in_cover(u.state["position"])
            eff_rng  = get_effective_vision_range(
                self.state.get("vision_range", 20),
                stealth, in_cover, los
            )
            if dist <= eff_rng and los:
                visible.append(u)
        s["friendly_units"] = visible
        if visible:
            tgt = min(visible, key=lambda u: manhattan(self.state["position"], u.state["position"]))
            s["target_position"] = tgt.state["position"]
        else:
            s["target_position"] = self.state["position"]

        # 6) Materialize the chosen branch
        cond, subtasks = valid[idx]
        plan = subtasks(s) if callable(subtasks) else list(subtasks)

        # 7) Never leave the plan empty
        self.current_plan = plan or [("Hold", None)]
        logger.info(f"{self.name} updated plan: {self.current_plan}")

        # store for next tick
        self.state["_last_valid_count"] = valid_count
        self.last_health = self.state["health"]

    def execute_next_task(self):
        self.state["is_attacking"] = False
        friendly_units = self.sim.friendly_units
        for u in friendly_units:
            if (u.state["health"]>0
                and manhattan(self.state["position"], u.state["position"]) <= self.state["attack_range"]
                and has_line_of_sight(self.state["position"], u.state["position"])):
                # a friendly is now in sight+range → force a new plan
                self.update_plan(friendly_units, force_replan=True)
                break

        logger.info(f"{self.name} is facing {self.state['facing']}")
        if not self.current_plan or not self.state["enemy_alive"]:
            return
        raw = self.current_plan[0]
        if isinstance(raw, tuple):
            task, arg = raw
        else:
            task, arg = raw, None

        if task not in ("BattlePosition", "AttackEnemy") and self.state.get("in_battle_position", False):
            self.state["turns_in_battle_position"] = 0
            self.state["hasty_done"]      = False
            self.state["entrenched_done"] = False
            self.state["in_battle_position"] = False

        if task == "Move":
            self.state["move_credit"] += self.state["speed"]
            steps = int(self.state["move_credit"])
            logger.info(f"{self.name} taking {steps} steps")
            self.state["move_credit"] -= steps
            for _ in range(steps):
                old_pos = self.state["position"]
                target = min(friendly_units, key=lambda u: manhattan(self.state["position"], u.state["position"])).state["position"]
                self.state["position"] = next_step(self.state["position"], target)
                new_pos = self.state["position"]
                if manhattan(self.state["position"], target) <= self.state["attack_range"]:
                    self.current_plan.pop(0)
                dx, dy = new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]
                if dx or dy:
                    self.state["facing"] = (sign(dx), sign(dy))
        elif task == "AttackEnemy":
            self.state["is_attacking"] = True
            target_unit = None
            min_dist = float('inf')
            for u in friendly_units:
                d = manhattan(self.state["position"], u.state["position"])
                if d <= self.state["attack_range"] and has_line_of_sight(self.state["position"], u.state["position"]):
                    if d < min_dist:
                        min_dist = d
                        target_unit = u
            if target_unit:
                tx, ty = target_unit.state["position"]
                x, y = self.state["position"]
                dx, dy = tx - x, ty - y
                norm = math.hypot(dx, dy)
                if norm > 0:
                    fx, fy = dx / norm, dy / norm
                    self.state["facing"] = (fx, fy)
                    logger.info(f"{self.name} is now facing ({fx:.2f}, {fy:.2f})")
                else:
                    # if you somehow are exactly on top of them, keep previous facing
                    fx, fy = self.state.get("facing", (0,1))

                rate_of_fire = self.state["base_rate_of_fire"] * self.state["current_group_size"]
                num_attacks = get_num_attacks(rate_of_fire)
                effective_accuracy = max(0, self.state["accuracy"] - self.state["suppression_from_enemy"])
                
                # Initialize counters for detailed logging
                hits = 0
                penetrations = 0
                total_damage_dealt = 0.0

                logger.info(f"{self.name} (group size: {self.state['current_group_size']}) attacks {target_unit.name} "
                           f"(group size: {target_unit.state['current_group_size']}) {num_attacks} times, "
                           f"accuracy: {effective_accuracy:.2f}")

                for _ in range(num_attacks):
                    if random.random() < effective_accuracy:
                        target_unit.state["suppression_from_enemy"] += self.state["suppression"]
                        hits += 1
                        target_pos = target_unit.state["position"]
                        dx = target_pos[0] - self.state["position"][0]
                        dy = target_pos[1] - self.state["position"][1]
                        norm = math.sqrt(dx**2 + dy**2)
                        if norm > 0:
                            dx, dy = dx / norm, dy / norm
                        else:
                            dx, dy = 0, 0
                        self.state["facing"] = dx, dy

                        attack_dir = (dx, dy)
                        target_facing = target_unit.state.get("facing", (0, 1))
                        target_norm = math.sqrt(target_facing[0]**2 + target_facing[1]**2)
                        if target_norm > 0:
                            target_fx, target_fy = target_facing[0] / target_norm, target_facing[1] / target_norm
                        else:
                            target_fx, target_fy = 0, 0

                        dot_product = attack_dir[0] * target_fx + attack_dir[1] * target_fy
                        dot_product = max(min(dot_product, 1), -1)
                        angle_deg = math.degrees(math.acos(dot_product))
                        direction = "rear" if angle_deg <= 45 else "side" if angle_deg <= 135 else "front"

                        logger.info(f"{self.name} attacking {target_unit.name} from the {direction}")

                        arm_val = target_unit.state[f"armor_{direction}"]
                        D = self.state["penetration"] - arm_val
                        penetration_prob = get_penetration_probability(D)
                        if random.random() < penetration_prob:
                            penetrations += 1
                            total_damage = self.state["damage"] * self.state["current_group_size"]
                            target_unit.state["health"] -= total_damage
                            target_unit.state["cumulative_damage"] += total_damage
                            total_damage_dealt += total_damage

                            units_lost = int(target_unit.state["cumulative_damage"] // target_unit.state["base_health"])
                            if units_lost > 0:
                                old_group_size = target_unit.state["current_group_size"]
                                target_unit.state["current_group_size"] = max(0, target_unit.state["current_group_size"] - units_lost)
                                target_unit.state["cumulative_damage"] -= units_lost * target_unit.state["base_health"]
                                logger.info(f"{target_unit.name} lost {units_lost} units, new group size: {target_unit.state['current_group_size']}")

                            logger.info(f"{self.name} penetrated {target_unit.name} with D={D:.1f}, "
                                       f"penetration prob={penetration_prob:.2f}, dealt {total_damage:.1f}, "
                                       f"health now {target_unit.state['health']:.1f}")
                            if target_unit.state["health"] <= 0 or target_unit.state["current_group_size"] <= 0:
                                logger.info(f"{target_unit.name} destroyed by {self.name}")
                                target_unit.state["enemy_alive"] = False
                                target_unit.state["current_group_size"] = 0
                                target_unit.state["health"] = 0
                                target_unit.state["cumulative_damage"] = 0
                logger.info(f"{self.name} attack summary: {hits}/{num_attacks} hits, "
                           f"{penetrations}/{hits} penetrations, total damage dealt: {total_damage_dealt:.1f}")
                self.current_plan.pop(0)
            else:
                self.current_plan.pop(0)
        elif task == "Retreat":
            retreat = self.state.get("retreat_point", (9, 9))
            self.state["position"] = next_step(self.state["position"], retreat)
            if self.state["position"] == retreat:
                self.current_plan.pop(0)
        elif task == "BattlePosition":
            # mark that we’re now in a defensive stance
            self.state["in_battle_position"] = True

            # increment the counter
            turns = self.state.get("turns_in_battle_position", 0) + 1
            self.state["turns_in_battle_position"] = turns

            # on turn 1 → hasty bonus
            if turns == 1 and not self.state.get("hasty_done", False):
                # Hasty: +2 front & +3 side/rear if in cover, else +1 front & +2 side/rear
                if is_in_cover(self.state["position"]):
                    front_delta, flank_delta = 2, 3
                else:
                    front_delta, flank_delta = 1, 2

                # apply additive bonuses from base
                self.state["armor_front"] = self.state["base_armor_front"] + front_delta
                self.state["armor_side"]  = self.state["base_armor_side"]   + flank_delta
                self.state["armor_rear"]  = self.state["base_armor_rear"]   + flank_delta

                self.state["hasty_done"] = True
                logger.info(f"{self.name} hasty position: +{front_delta} front, +{flank_delta} side/rear")

            elif turns == 5 and not self.state.get("entrenched_done", False):
                # Entrenched: +5 front & +8 side/rear flat
                front_delta, flank_delta = 2, 8

                self.state["armor_front"] = self.state["base_armor_front"] + front_delta
                self.state["armor_side"]  = self.state["base_armor_side"]   + flank_delta
                self.state["armor_rear"]  = self.state["base_armor_rear"]   + flank_delta

                self.state["entrenched_done"] = True
                logger.info(f"{self.name} entrenched position: +{front_delta} front, +{flank_delta} side/rear")


    def get_goal_position(self):
        if not self.current_plan:
            return self.state["position"]

        task = self.current_plan[0]
        if isinstance(task, (list, tuple)):
            t, _ = task
        else:
            t, _ = task, None
        if t == "Patrol":
            idx = self.state["current_patrol_index"]
            return self.state["patrol_points"][idx]
        elif t in ["Move", "AttackEnemy"]:
            last_known = self.sim.enemy_drone.last_known  # maps name -> (x,y)
            if last_known:
                # pick the nearest last‐known coordinate
                return min(last_known.values(), key=lambda p: manhattan(self.state["position"], p))
        elif t == "Retreat":
            return self.state.get("retreat_point", (9, 9))
        return self.state["position"]

    def needs_update(self):
        return (self.state["position"] != self.last_position or
                abs(self.state["health"] - self.last_health) > 0.1 or
                self.state["current_group_size"] != self.last_group_size or
                not self.state["enemy_alive"])

class EnemyTank(EnemyUnit):
    def __init__(self, name, state, domain):
        super().__init__(name, state, domain)
    
    def can_attack(self, target):
        if target.state.get("enemy_alive", False):
            distance = manhattan(self.state["position"], target.state["position"])
            has_los = has_line_of_sight(self.state["position"], target.state["position"])
            return distance <= self.state["attack_range"] and has_los
        return False

class EnemyInfantry(EnemyUnit):
    def __init__(self, name, state, domain):
        super().__init__(name, state, domain)

    def can_attack(self, target):
        if target.state.get("enemy_alive", False):
            distance = manhattan(self.state["position"], target.state["position"])
            has_los = has_line_of_sight(self.state["position"], target.state["position"])
            return distance <= self.state["attack_range"] and has_los
        return False

class EnemyAntiTank(EnemyUnit):
    def __init__(self, name, state, domain):
        super().__init__(name, state, domain)
    
    def can_attack(self, target):
        if target.state.get("enemy_alive", False):
            distance = manhattan(self.state["position"], target.state["position"])
            has_los = has_line_of_sight(self.state["position"], target.state["position"])
            return distance <= self.state["attack_range"] and has_los
        return False

class EnemyArtillery(EnemyUnit):
    def __init__(self, name, state, domain):
        super().__init__(name, state, domain)

    def can_attack(self, target):
        if target.state.get("enemy_alive", False):
            distance = manhattan(self.state["position"], target.state["position"])
            return distance <= self.state["attack_range"]
        return False

###############################
# Friendly Unit Classes
###############################

class FriendlyUnit:
    def __init__(self, name, state, domain, simulation=None):
        self.name = name
        self.state = state
        self.planner = HTNPlanner(domain)
        self.current_plan = []
        self.last_enemy_pos = state["outpost_position"]
        self.last_health = state["health"]
        self.last_position = state["position"]
        self.last_group_size = state["current_group_size"]
        self.sim = simulation

    def update_plan(self, force_replan=False):
        mission = "SecureOutpostMission"
        # Check if we can attack any visible enemy
        attack_possible = False
        if not force_replan and self.current_plan:
            for enemy_name, enemy in self.sim.enemy_units_dict.items():
                if self.can_attack(enemy):
                    # If current plan doesn't start with AttackEnemy for this enemy, force replan
                    if not self.current_plan or self.current_plan[0][0] != "AttackEnemy" or self.current_plan[0][1] != enemy_name:
                        force_replan = True
                        logger.info(f"{self.name} forcing replan due to attackable enemy {enemy_name}")
                        break

        if force_replan or not self.current_plan:
            combined = copy.deepcopy(self.state)
            combined["sim"] = self.sim
            combined["spotted_enemies"] = [
                name for name in self.state.get("spotted_enemies", [])
                if name in self.sim.enemy_units_dict
                and self.sim.enemy_units_dict[name].state.get("enemy_alive", False)
            ]
            combined["unit"] = self
            combined["enemy_units_dict"] = self.sim.enemy_units_dict
            new_plan = self.planner.plan(mission, combined)
            if not new_plan:
                new_plan = [("Hold", None)]
            self.current_plan = new_plan
            logger.info(f"{self.name} replanned: {self.current_plan}")
        else:
            logger.info(f"{self.name} current plan: {self.current_plan}")
        self.last_health = self.state["health"]

    def execute_next_task(self):
        logger.info(f"{self.name} is facing {self.state['facing']}")
        if not self.current_plan or self.state["health"] <= 0:
            logger.info(f"{self.name} cannot execute task: plan empty or health <= 0")
            return
        task = self.current_plan[0]
        task_name = task[0] if isinstance(task, tuple) else task
        task_arg = task[1] if isinstance(task, tuple) else None
        logger.info(f"{self.name} executing task: {task}")

        if task_name == "Move":
            self.state["move_credit"] += self.state["speed"]
            steps = int(self.state["move_credit"])
            logger.info(f"{self.name} taking {steps} steps")
            self.state["move_credit"] -= steps

            for _ in range(steps):
                old_pos = self.state["position"]
                goal = self.get_goal_position(task)
                if (len(self.current_plan) > 1 and
                    isinstance(self.current_plan[1], tuple) and
                    self.current_plan[1][0] == "AttackEnemy" and
                    task_arg == self.current_plan[1][1]):
                    target_unit = None
                    for e in self.sim.enemy_units:
                        if e.state.get("name") == task_arg:
                            target_unit = e
                            break
                    if target_unit and target_unit.state["enemy_alive"]:
                        distance = manhattan(self.state["position"], target_unit.state["position"])
                        has_los = has_line_of_sight(self.state["position"], target_unit.state["position"])
                        logger.info(f"{self.name} to {task_arg}: distance={distance}, attack_range={self.state['attack_range']}, has_los={has_los}")
                        if distance <= self.state["attack_range"] and has_los:
                            logger.info(f"{self.name} within attack range of {task_arg} at distance {distance}; stopping move.")
                            self.current_plan.pop(0)
                            self.update_plan(force_replan=True)
                            return
                if self.state["position"] == goal:
                    logger.info(f"{self.name} reached goal {goal}; stopping move.")
                    self.current_plan.pop(0)
                else:
                    self.state["position"] = next_step(
                        self.state["position"],
                        goal,
                        self.sim.enemy_units,
                        unit=self.state["type"]
                    )
                    logger.info(f"{self.name} moves toward {goal}, new position: {self.state['position']}")
                new_pos = self.state["position"]
                dx, dy = new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]
                if dx or dy:
                    self.state["facing"] = (sign(dx), sign(dy))

        elif task_name == "AttackEnemy":
            target_unit = None
            for e in self.sim.enemy_units:
                if e.state.get("name") == task_arg:
                    target_unit = e
                    break
            if not target_unit or not target_unit.state["enemy_alive"]:
                logger.info(f"{self.name} cannot attack; target {task_arg} is dead or invalid.")
                self.current_plan.pop(0)
                self.update_plan(force_replan=True)
                return
            tx, ty = target_unit.state["position"]
            x, y = self.state["position"]
            distance = manhattan(self.state["position"], target_unit.state["position"])
            has_los = has_line_of_sight(self.state["position"], target_unit.state["position"])
            if distance > self.state["attack_range"] or not has_los:
                logger.info(f"{self.name} cannot attack {target_unit.name}; out of range or no LOS.")
                self.current_plan.pop(0)
                self.update_plan(force_replan=True)
                return
            dx, dy = tx - x, ty - y
            norm = math.hypot(dx, dy)
            if norm > 0:
                fx, fy = dx / norm, dy / norm
                self.state["facing"] = (fx, fy)
                logger.info(f"{self.name} is now facing ({fx:.2f}, {fy:.2f})")
            else:
                # if you somehow are exactly on top of them, keep previous facing
                fx, fy = self.state.get("facing", (0,1))

            rate_of_fire = self.state["base_rate_of_fire"] * self.state["current_group_size"]
            num_attacks = get_num_attacks(rate_of_fire)
            effective_accuracy = max(0, self.state["friendly_accuracy"] - self.state["suppression_from_enemy"])

            # Initialize counters for detailed logging
            hits = 0
            penetrations = 0
            total_damage_dealt = 0.0

            logger.info(f"{self.name} (group size: {self.state['current_group_size']}) attacks {target_unit.name} "
                       f"(group size: {target_unit.state['current_group_size']}) {num_attacks} times, "
                       f"accuracy: {effective_accuracy:.2f}")

            for _ in range(num_attacks):
                if random.random() < effective_accuracy:
                    hits += 1
                    target_unit.state["suppression_from_enemy"] += self.state["suppression"]
                    dx = tx - x
                    dy = ty - y
                    norm = math.sqrt(dx**2 + dy**2)
                    dx, dy = (dx / norm, dy / norm) if norm > 0 else (0, 0)
                    self.state["facing"] = (dx, dy)
                    attack_dir = (dx, dy)
                    target_facing = target_unit.state.get("facing", (0, 1))
                    target_norm = math.sqrt(target_facing[0]**2 + target_facing[1]**2)
                    target_fx, target_fy = (target_facing[0] / target_norm, target_facing[1] / target_norm) if target_norm > 0 else (0, 0)
                    dot_product = attack_dir[0] * target_fx + attack_dir[1] * target_fy
                    dot_product = max(min(dot_product, 1), -1)
                    angle_deg = math.degrees(math.acos(dot_product))
                    direction = "rear" if angle_deg <= 45 else "side" if angle_deg <= 135 else "front"
                    logger.info(f"{self.name} attacking {target_unit.name} from the {direction}, suppression: {self.state['suppression']}")

                    armor_val = target_unit.state[f"armor_{direction}"]
                    logger.info(f"{self.name} attacking {target_unit.name} with armor {armor_val}")
                    D = self.state["penetration"] - armor_val
                    penetration_prob = get_penetration_probability(D)
                    if random.random() < penetration_prob:
                        penetrations += 1
                        total_damage = self.state["damage"] * self.state["current_group_size"]
                        target_unit.state["health"] -= total_damage
                        target_unit.state["cumulative_damage"] += total_damage
                        total_damage_dealt += total_damage

                        units_lost = int(target_unit.state["cumulative_damage"] // target_unit.state["base_health"])
                        if units_lost > 0:
                            target_unit.state["current_group_size"] = max(0, target_unit.state["current_group_size"] - units_lost)
                            target_unit.state["cumulative_damage"] -= units_lost * target_unit.state["base_health"]
                            logger.info(f"{target_unit.name} lost {units_lost} units, new group size: {target_unit.state['current_group_size']}")

                        logger.info(f"{self.name} penetrates {target_unit.name}, D={D:.1f}, "
                                   f"penetration prob={penetration_prob:.2f}, dealt {total_damage:.1f}, "
                                   f"health now {target_unit.state['health']:.1f}")
                        if target_unit.state["health"] <= 0 or target_unit.state["current_group_size"] <= 0:
                            target_unit.state["enemy_alive"] = False
                            target_unit.state["current_group_size"] = 0
                            target_unit.state["health"] = 0
                            target_unit.state["cumulative_damage"] = 0
                            logger.info(f"{self.name} destroyed {target_unit.name}!")
                            self.current_plan.pop(0)
                            for unit in self.sim.friendly_units:
                                unit.update_plan(force_replan=True)
                            logger.info(f"{self.name} attack summary: {hits}/{num_attacks} hits, "
                                       f"{penetrations}/{hits} penetrations, total damage dealt: {total_damage_dealt:.1f}")
                            return
            logger.info(f"{self.name} attack summary: {hits}/{num_attacks} hits, "
                       f"{penetrations}/{hits if hits > 0 else 1} penetrations, total damage dealt: {total_damage_dealt:.1f}")
            if target_unit.state["enemy_alive"]:
                logger.info(f"{self.name} continues attacking {target_unit.name}, health remaining: {target_unit.state['health']:.1f}")
            else:
                self.current_plan.pop(0)

        elif task_name == "SecureOutpostNoArg":
            if self.state["position"] == self.state["outpost_position"]:
                self.state["outpost_secured"] = True
                logger.info(f"{self.name} secures the outpost!")
                self.current_plan.pop(0)
                self.current_plan.append(("Hold", None))  # Fallback to Hold
            else:
                logger.info(f"{self.name} cannot secure outpost; not at target location.")
                self.current_plan.pop(0)
                self.current_plan.append(("Hold", None))  # Fallback to Hold

        elif task_name == "Hold":
            logger.info(f"{self.name} holds position at {self.state['position']}.")
            self.current_plan.pop(0)
            return

    def get_goal_position(self, task=None):
        if not task:
            task = self.current_plan[0] if self.current_plan else ("Hold", None)
        task_name = task[0] if isinstance(task, tuple) else task
        task_arg = task[1] if isinstance(task, tuple) else None

        if task_name == "Move" and task_arg == "outpost":
            return self.state["outpost_position"]

        # 2) For Move or AttackEnemy toward a specific enemy:
        if task_name in ("Move", "AttackEnemy") and task_arg:
            # a) try the drone's memory first
            last = self.sim.friendly_drone.last_known.get(task_arg)
            if last is not None:
                return last
        return self.state["position"]

    def needs_update(self):
        return (self.state["position"] != self.last_position or
                abs(self.state["health"] - self.last_health) > 0.1 or
                self.state["current_group_size"] != self.last_group_size or
                self.state["health"] <= 0)

class FriendlyTank(FriendlyUnit):
    def __init__(self, name, state, domain, simulation=None):
        super().__init__(name, state, domain, simulation)

    def can_attack(self, target):
        if target.state.get("enemy_alive", False):
            distance = manhattan(self.state["position"], target.state["position"])
            has_los = has_line_of_sight(self.state["position"], target.state["position"])
            return distance <= self.state["attack_range"] and has_los
        return False

class FriendlyInfantry(FriendlyUnit):
    def __init__(self, name, state, domain, simulation=None):
        super().__init__(name, state, domain, simulation)

    def can_attack(self, target):
        if target.state.get("enemy_alive", False):
            distance = manhattan(self.state["position"], target.state["position"])
            has_los = has_line_of_sight(self.state["position"], target.state["position"])
            return distance <= self.state["attack_range"] and has_los
        return False

class FriendlyArtillery(FriendlyUnit):
    def __init__(self, name, state, domain, simulation=None):
        super().__init__(name, state, domain, simulation)

    def can_attack(self, target):
        if not target.state.get("enemy_alive", False):
            return False
        
        distance = manhattan(self.state["position"], target.state["position"])
        if distance > self.state["attack_range"]:
            return False
        
        has_los = has_line_of_sight(self.state["position"], target.state["position"])
        if has_los:
            logger.info(f"{self.name} has direct LOS to {target.name}")
            return True
        
        # No direct LOS, check other friendly units' LOS
        for unit in self.sim.friendly_units:
            if unit is not self and has_line_of_sight(unit.state["position"], target.state["position"]):
                logger.info(f"{self.name} can attack {target.name} due to LOS from {unit.name}")
                return True
        
        # Check drone coverage
        if under_friendly_drone_cover(self.sim, target.state["position"]):
            logger.info(f"{self.name} can attack {target.name} due to friendly drone coverage")
            return True
        
        logger.info(f"{self.name} cannot attack {target.name}: no LOS, no friendly LOS, no drone coverage")
        return False

class FriendlyScout(FriendlyUnit):
    def __init__(self, name, state, domain, simulation=None):
        super().__init__(name, state, domain, simulation)

class FriendlyAntiTank(FriendlyUnit):
    def __init__(self, name, state, domain, simulation=None):
        super().__init__(name, state, domain, simulation)

    def can_attack(self, target):
        if target.state.get("enemy_alive", False):
            distance = manhattan(self.state["position"], target.state["position"])
            has_los = has_line_of_sight(self.state["position"], target.state["position"])
            return distance <= self.state["attack_range"] and has_los
        return False

###############################
# TeamCommander and Simulation Classes
###############################

class TeamCommander:
    def __init__(self, friendly_units):
        self.friendly_units = friendly_units

class Simulation:
    def __init__(
        self,
        friendly_units,
        enemy_units,
        team_commander,
        visualize=False,
        plan_name="Unknown Plan",
    ):
        # ---- core state setup ----
        self.friendly_units = friendly_units
        self.friendly_units_dict = {u.name: u for u in friendly_units}

        self.enemy_units = enemy_units
        self.enemy_units_dict  = {e.name: e for e in enemy_units}

        # Recon drones
        self.friendly_drone = Drone(
            side="friendly", target_side="enemy",
            n_cols=3, n_rows=2, stay_rounds=10, spot_prob=0.2
        )
        self.enemy_drone = Drone(
            side="enemy", target_side="friendly",
            n_cols=3, n_rows=2, stay_rounds=10, spot_prob=0.2
        )

        self.team_commander = team_commander
        
        self.river = river
        self.forest = forest
        self.forest_edge = forest_edge
        self.cliffs = cliffs
        self.climb_entries = climb_entries

        # Initialize each friendly unit’s view state
        active_enemy = next((e for e in enemy_units if e.state.get("enemy_alive")), None)
        for u in self.friendly_units:
            u.state["enemy"]           = active_enemy.state if active_enemy else {}
            u.state["visible_enemies"] = []
            u.state["all_enemies"]     = [e.state for e in self.enemy_units]
            u.state["total_enemies"]   = len(self.enemy_units)
            u.state["scout_steps"]     = 0
            u.sim = self

        for e in self.enemy_units:
            e.sim = self

        self.step_count = 0
        self.visualize  = visualize
        self.plan_name  = plan_name

        # ---- plotting hand-off ----
        if self.visualize:
            # Hand off _all_ plotting to SimulationPlotter
            from plotter import SimulationPlotter
            plt.ion()
            plt.show(block=False)
            self.plotter = SimulationPlotter(self, visualize=True)

    def update_enemy_behavior(self):
        friendly_units = [u for u in self.friendly_units if u.state.get("health", 0) > 0]
        for enemy in self.enemy_units:
            if enemy.state["enemy_alive"]:
                enemy.update_plan(friendly_units)
                enemy.execute_next_task()
                enemy.current_goal = enemy.get_goal_position()
                if self.visualize:
                    logger.info(f"{enemy.state['name']} position: {enemy.state['position']}")
                    logger.info(f"{enemy.state['name']}'s current goal: {enemy.current_goal}")

    def update_friendly_enemy_info(self):
        active_enemies = [e for e in self.enemy_units if e.state["enemy_alive"]]
        for u in self.friendly_units:
            if active_enemies:
                closest_enemy = None
                min_distance = float('inf')
                for e in active_enemies:
                    distance = manhattan(u.state["position"], e.state["position"])
                    has_los = has_line_of_sight(e.state["position"], u.state["position"])
                    in_cover = is_in_cover(u.state["position"])
                    stealth_modifier = u.state.get("stealth_modifier", 0)
                    effective_vision_range = get_effective_vision_range(
                        e.state.get("vision_range", 20),
                        stealth_modifier,
                        in_cover,
                        has_los
                    )
                    if distance <= effective_vision_range and has_los:
                        if distance < min_distance:
                            min_distance = distance
                            closest_enemy = e.state
                u.state["enemy"] = closest_enemy or {}
                if self.visualize:
                    logger.info(f"{u.name} state['enemy']: {u.state['enemy'].get('name', 'None')}")
            else:
                for u in self.friendly_units:
                    u.state["enemy"] = {}
                    if self.visualize:
                        logger.info(f"{u.name} state['enemy']: None (no active enemies)")
            if self.visualize:
                visible_enemies = u.state.get("visible_enemies", [])
                logger.info(f"{u.name} at {u.state['position']} sees enemies: {visible_enemies}")

    def evaluate_plan(self):
        total_friendly = sum(u.state["health"] for u in self.friendly_units)
        max_friendly = sum(u.state["max_health"] for u in self.friendly_units)
        health = sum(e.state["health"] for e in self.enemy_units if e.state["enemy_alive"])
        max_enemy = sum(e.state["max_health"] for e in self.enemy_units)
        outpost_secured = any(u.state.get("outpost_secured", False) for u in self.friendly_units)
        steps = self.step_count
        friendly_ratio = total_friendly / max_friendly if max_friendly > 0 else 0
        enemy_ratio = health / max_enemy if max_enemy > 0 else 0
        score = (friendly_ratio * 20) - (enemy_ratio * 20) + (10 if outpost_secured else -10) - 0.1 * steps
        return {
            "score": score,
            "health": total_friendly,
            "enemy_health": health,
            "outpost_secured": outpost_secured,
            "steps_taken": steps
        }

    def step(self):
        # 0) Advance the clock & log header
        self.step_count += 1
        if self.visualize:
            logger.info(f"--- Simulation Step {self.step_count} ---")

        # 2) Refresh drones
        drone_seen_friends = self.friendly_drone.update(self)  # set of enemy names
        drone_seen_enemies = self.enemy_drone.update(self)     # set of friendly names

        # 3) Build LOS‐based visibility via your helper
        #    and merge in drone reports for spotted_enemies
        self.friendly_drone.update(self)
        self.enemy_drone.update(self)
        # Now read out what they saw
        drone_seen_by_friendlies = set(self.friendly_drone.last_known.keys())
        drone_seen_by_enemies    = set(self.enemy_drone.last_known.keys())

        # Build LOS via helper + merge in drone reports
        for friend in self.friendly_units:
            los_list = [e.name for e in units_spotted_by_vision(friend, self.enemy_units)]
            friend.state["visible_enemies"] = los_list

            merged = set(los_list) | drone_seen_by_friendlies
            friend.state["spotted_enemies"] = [
                n for n in merged
                if n in self.enemy_units_dict
                and self.enemy_units_dict[n].state["current_group_size"] > 0
            ]

        for enemy in self.enemy_units:
            los_list = [u.name for u in units_spotted_by_vision(enemy, self.friendly_units)]
            enemy.state["visible_enemies"] = los_list

            merged = set(los_list) | drone_seen_by_enemies
            enemy.state["spotted_enemies"] = [
                n for n in merged
                if n in self.friendly_units_dict
                and self.friendly_units_dict[n].state["current_group_size"] > 0
            ]

        self.update_friendly_enemy_info()
        # 4) Enemy turn
        self.update_enemy_behavior()

        # 5) Friendly turn: HTN replan & execution
        for u in self.friendly_units:
            last = set(u.state.get("_last_spotted", []))
            now  = set(u.state["spotted_enemies"])
            # force‐replan on truly new sightings or dead targets
            if now - last:
                logger.info(f"{u.name} spotted new foes {now - last}; forcing replan")
                u.update_plan(force_replan=True)
            elif not u.current_plan:
                logger.info(f"{u.name} has empty plan; replanning")
                u.update_plan(force_replan=True)
            elif (
                isinstance(u.current_plan[0], tuple)
                and u.current_plan[0][0] == "AttackEnemy"
            ):
                tgt = u.current_plan[0][1]
                if not any(e.name == tgt and e.state.get("enemy_alive", False)
                        for e in self.enemy_units):
                    logger.info(f"{u.name}'s target {tgt} died; replanning")
                    u.update_plan(force_replan=True)
            else:
                u.update_plan()

            u.state["_last_spotted"] = list(now)
            logger.info(f"{u.name} current plan: {u.current_plan}")
            if self.visualize and u.current_plan and isinstance(u.current_plan[0], tuple):
                logger.info(f"{u.name} targeting: {u.current_plan[0][1]}")
            u.execute_next_task()
            if self.visualize:
                logger.info(f"{u.name}'s current goal: {u.get_goal_position()}")

        # 6) Refresh the plot
        if self.visualize:
            self.plotter.update()


    def run(self, max_steps=500):
        """
        Main simulation loop:
        - Perform initial planning
        - For up to max_steps:
            * Check for mission success
            * step()
            * Prune dead enemies
            * Update visualization
        - Return the evaluation of the executed plan
        """
        # 0) Initial planning
        self.step_count = 0
        for u in self.friendly_units:
            u.update_plan(force_replan=True)
        for e in self.enemy_units:
            e.update_plan(self.friendly_units)

        if self.visualize:
            self.plotter.update()
            plt.pause(0.5)

        # 1) Main loop
        for _ in range(max_steps):
            # 1a) Success check: all surviving friendlies at outpost
            alive_friendlies = [
                u for u in self.friendly_units
                if u.state.get("health", 0) > 0
            ]
            if (alive_friendlies
                    and all(u.state["position"] == u.state.get("outpost_position")
                            for u in alive_friendlies)):
                if self.visualize:
                    self.plotter.update()
                    logger.info("\nMission accomplished: Outpost secured!")
                return self.evaluate_plan()

            # 1b) Advance one step
            self.step()

            # 1c) Prune any dead enemies so planners never target them
            alive_enemies = [
                e for e in self.enemy_units
                if e.state.get("current_group_size", 0) > 0
            ]
            if len(alive_enemies) != len(self.enemy_units):
                self.enemy_units = alive_enemies
                self.enemy_units_dict = {e.name: e for e in self.enemy_units}

            # 1d) Visualization update
            if self.visualize:
                self.plotter.update()
                plt.pause(0.5)

                while self.plotter.paused:
                    plt.pause(0.2)

        # 2) If we exhaust max_steps without success
        if self.visualize:
            self.plotter.update()
            logger.info("\nMission incomplete after maximum steps.")
        return self.evaluate_plan()


###############################
# Main Simulation Setup - Mode 1 (Test Specific Plan)
###############################

if __name__ == "__main__":
    TANK_GROUP_SIZE = 12
    INFANTRY_GROUP_SIZE = 12
    ARTILLERY_GROUP_SIZE = 3
    SCOUT_GROUP_SIZE = 2
    ANTI_TANK_GROUP_SIZE = 2

    ENEMY_TANK_GROUP_SIZE = 4
    ENEMY_INFANTRY_GROUP_SIZE = 4
    ENEMY_ANTI_TANK_GROUP_SIZE = 2
    ENEMY_ARTILLERY_GROUP_SIZE = 1

    tank_state_template = {
        "type": "tank",
        "position": (0, 49),
        "facing": (0, 1),
        "speed": (((75 / 3.6) * 6) / CELL_SIZE),
        "move_credit": 0,
        "base_health": 20,
        "health": 20 * TANK_GROUP_SIZE,
        "max_health": 20 * TANK_GROUP_SIZE,
        "group_size": TANK_GROUP_SIZE,
        "current_group_size": TANK_GROUP_SIZE,
        "cumulative_damage": 0.0,
        "armor_front": 17,
        "armor_side": 4,
        "armor_rear": 3,
        "friendly_accuracy": 0.75,
        "base_rate_of_fire": 4.9,
        "damage": 9,
        "suppression": 0.12,
        "penetration": 18,
        "vision_range": 2000 / CELL_SIZE,
        "attack_range": 2400 / CELL_SIZE,
        "all_enemies_spotted": False,
        "total_enemies": 2,
        "suppression_from_enemy": 0.0,
        "scout_steps": 0,
        "stealth_modifier": 0
    }
    infantry_state_template = {
        "type": "infantry",
        "position": (1, 49),
        "facing": (0, 1),
        "speed": (((65 / 3.6) * 6) / CELL_SIZE),
        "move_credit": 0,
        "base_health": 1,
        "health": 1 * INFANTRY_GROUP_SIZE,
        "max_health": 1 * INFANTRY_GROUP_SIZE,
        "group_size": INFANTRY_GROUP_SIZE,
        "current_group_size": INFANTRY_GROUP_SIZE,
        "cumulative_damage": 0.0,
        "armor_front": 0,
        "armor_side": 0,
        "armor_rear": 0,
        "friendly_accuracy": 0.50,
        "base_rate_of_fire": 294,
        "damage": 0.8,
        "suppression": 0.01,
        "penetration": 1,
        "vision_range": 2200 / CELL_SIZE,
        "attack_range": 1200 / CELL_SIZE,
        "all_enemies_spotted": False,
        "total_enemies": 2,
        "suppression_from_enemy": 0.0,
        "scout_steps": 0,
        "stealth_modifier": 0
    }
    artillery_state_template = {
        "type": "artillery",
        "position": (2, 49),
        "facing": (0, 1),
        "speed": (((65 / 3.6) * 6) / CELL_SIZE),
        "move_credit": 0,
        "base_health": 18,
        "health": 18 * ARTILLERY_GROUP_SIZE,
        "max_health": 18 * ARTILLERY_GROUP_SIZE,
        "group_size": ARTILLERY_GROUP_SIZE,
        "current_group_size": ARTILLERY_GROUP_SIZE,
        "cumulative_damage": 0.0,
        "armor_front": 2,
        "armor_side": 1,
        "armor_rear": 1,
        "friendly_accuracy": 0.85,
        "base_rate_of_fire": 8.6,
        "damage": 3.5,
        "suppression": 15,
        "penetration": 1,
        "vision_range": 1800 / CELL_SIZE,
        "attack_range": 4000 / CELL_SIZE,
        "all_enemies_spotted": False,
        "total_enemies": 2,
        "suppression_from_enemy": 0.0,
        "scout_steps": 0,
        "stealth_modifier": 0
    }
    scout_state_template = {
        "type": "scout",
        "position": (16, 20),
        "facing": (0, 1),
        "speed": (((90 / 3.6) * 6) / CELL_SIZE),
        "move_credit": 0,
        "base_health": 18,
        "health": 18 * SCOUT_GROUP_SIZE,
        "max_health": 18 * SCOUT_GROUP_SIZE,
        "group_size": SCOUT_GROUP_SIZE,
        "current_group_size": SCOUT_GROUP_SIZE,
        "cumulative_damage": 0.0,
        "armor_front": 3,
        "armor_side": 2,
        "armor_rear": 2,
        "friendly_accuracy": 0.35,
        "base_rate_of_fire": 115,
        "damage": 2.4,
        "suppression": 0.05,
        "penetration": 5,
        "vision_range": 2600 / CELL_SIZE,
        "attack_range": 1800 / CELL_SIZE,
        "all_enemies_spotted": False,
        "candidate_positions": [],
        "current_candidate_index": 0,
        "total_enemies": 2,
        "suppression_from_enemy": 0.0,
        "scout_steps": 0,
        "stealth_modifier": 225
    }
    anti_tank_state_template = {
        "type": "anti-tank",
        "position": (3, 49),
        "facing": (0, 1),
        "speed": (((65 / 3.6) * 6) / CELL_SIZE),
        "move_credit": 0,
        "base_health": 18,
        "health": 18 * ANTI_TANK_GROUP_SIZE,
        "max_health": 18 * ANTI_TANK_GROUP_SIZE,
        "group_size": ANTI_TANK_GROUP_SIZE,
        "current_group_size": ANTI_TANK_GROUP_SIZE,
        "cumulative_damage": 0.0,
        "armor_front": 2,
        "armor_side": 1,
        "armor_rear": 1,
        "friendly_accuracy": 0.90,
        "base_rate_of_fire": 6.3,
        "damage": 15,
        "suppression": 0.10,
        "penetration": 22,
        "vision_range": 2000 / CELL_SIZE,
        "attack_range": 2800 / CELL_SIZE,
        "all_enemies_spotted": False,
        "total_enemies": 2,
        "suppression_from_enemy": 0.0,
        "scout_steps": 0,
        "stealth_modifier": 50
    }
    enemy_tank_state_template = {
        "type": "enemy_tank",
        "position": (63, 15),
        "facing": (0, 1),
        "speed": (((85 / 3.6) * 6) / CELL_SIZE),
        "move_credit": 0,
        "enemy_alive": True,
        "health": 100 * ENEMY_TANK_GROUP_SIZE,
        "max_health": 100 * ENEMY_TANK_GROUP_SIZE,
        "base_health": 100,
        "group_size": ENEMY_TANK_GROUP_SIZE,
        "current_group_size": ENEMY_TANK_GROUP_SIZE,
        "cumulative_damage": 0.0,
        "armor_front": 18,
        "armor_side": 5,
        "armor_rear": 3,
        "outpost_position": (GRID_WIDTH - 1, 0),
        "outpost_secured": False,
        "attack_range": 2400 / CELL_SIZE,
        "accuracy": 0.7,
        "penetration": 18,
        "damage": 9,
        "suppression": 0.12,
        "base_rate_of_fire": 5.5,
        "suppression_from_enemy": 0.0,
        "patrol_points": [],
        "current_patrol_index": 0,
        "vision_range": 1800 / CELL_SIZE,
        "retreat_point": (GRID_WIDTH - 1, GRID_HEIGHT - 1),
        "stealth_modifier": 0
    }
    enemy_infantry_state_template = {
        "type": "infantry",
        "position": (47, 20),
        "facing": (0, 1),
        "speed": (((75 / 3.6) * 6) / CELL_SIZE),
        "move_credit": 0,
        "enemy_alive": True,
        "base_health": 1,
        "health": 1 * ENEMY_INFANTRY_GROUP_SIZE,
        "max_health": 1 * ENEMY_INFANTRY_GROUP_SIZE,
        "group_size": ENEMY_INFANTRY_GROUP_SIZE,
        "current_group_size": ENEMY_INFANTRY_GROUP_SIZE,
        "cumulative_damage": 0.0,
        "armor_front": 0,
        "armor_side": 0,
        "armor_rear": 0,
        "outpost_position": (GRID_WIDTH - 1, 0),
        "outpost_secured": False,
        "accuracy": 0.50,
        "base_rate_of_fire": 294,
        "damage": 0.8,
        "suppression": 0.01,
        "penetration": 1,
        "attack_range": 1200 / CELL_SIZE,
        "all_enemies_spotted": False,
        "total_enemies": 2,
        "suppression_from_enemy": 0.0,
        "patrol_points": [],
        "current_patrol_index": 0,
        "scout_steps": 0,
        "stealth_modifier": 0
    }
    enemy_anti_tank_state_template = {
        "type": "anti-tank",
        "position": (70, 7),
        "facing": (0, 1),
        "speed": (((75 / 3.6) * 6) / CELL_SIZE),
        "move_credit": 0,
        "enemy_alive": True,
        "base_health": 18,
        "health": 18 * ENEMY_ANTI_TANK_GROUP_SIZE,
        "max_health": 18 * ENEMY_ANTI_TANK_GROUP_SIZE,
        "group_size": ENEMY_ANTI_TANK_GROUP_SIZE,
        "current_group_size": ENEMY_ANTI_TANK_GROUP_SIZE,
        "cumulative_damage": 0.0,
        "armor_front": 2,
        "armor_side": 1,
        "armor_rear": 1,
        "outpost_position": (GRID_WIDTH - 1, 0),
        "outpost_secured": False,
        "accuracy": 0.90,
        "base_rate_of_fire": 6.3,
        "damage": 15,
        "suppression": 0.1,
        "penetration": 22,
        "attack_range": 2800 / CELL_SIZE,
        "all_enemies_spotted": False,
        "total_enemies": 2,
        "suppression_from_enemy": 0.0,
        "patrol_points": [],
        "current_patrol_index": 0,
        "scout_steps": 0,
        "stealth_modifier": 50
    }
    enemy_artillery_state_template = {
        "type": "artillery",
        "position": (73, 10),
        "facing": (0, 1),
        "speed": (((75 / 3.6) * 6) / CELL_SIZE),
        "move_credit": 0,
        "enemy_alive": True,
        "base_health": 13,
        "health": 13 * ENEMY_ANTI_TANK_GROUP_SIZE,
        "max_health": 13 * ENEMY_ANTI_TANK_GROUP_SIZE,
        "group_size": ENEMY_ANTI_TANK_GROUP_SIZE,
        "current_group_size": ENEMY_ANTI_TANK_GROUP_SIZE,
        "cumulative_damage": 0.0,
        "armor_front": 2,
        "armor_side": 1,
        "armor_rear": 1,
        "outpost_position": (GRID_WIDTH - 1, 0),
        "outpost_secured": False,
        "accuracy": 0.90,
        "base_rate_of_fire": 8.6,
        "damage": 3.5,
        "suppression": 0.15,
        "penetration": 0,
        "attack_range": 4400 / CELL_SIZE,
        "all_enemies_spotted": False,
        "total_enemies": 2,
        "suppression_from_enemy": 0.0,
        "patrol_points": [],
        "current_patrol_index": 0,
        "scout_steps": 0,
        "stealth_modifier": 0
    }

    enemy_domain = {
        "DefendAreaMission": [

            # 1) As long as there's *any* name in state["spotted_enemies"],
            #    stay in Attack mode
            (
                lambda s: bool(s["spotted_enemies"]),
                lambda s: [
                    # For each spotted enemy name, choose Move vs Attack
                    ("AttackEnemy", name)
                    if manhattan(
                        s["unit"].state["position"],
                        s["sim"].friendly_units_dict[name].state["position"]
                    ) <= s["unit"].state["attack_range"] 
                    and has_line_of_sight(
                        s["unit"].state["position"],
                        s["sim"].friendly_units_dict[name].state["position"]
                    )
                    else ("Move", name)
                    for name in s["spotted_enemies"]
                    # filter out any that have since died
                    if s["sim"].friendly_units_dict[name].state["current_group_size"] > 0
                ]
            ),

            # 2) Otherwise—no more spotted contacts—dig in
            (
                lambda s: True,
                ["BattlePosition"]
            ),
        ]
}

    enemy_state1 = copy.deepcopy(enemy_tank_state_template)
    enemy_state1["name"] = "EnemyTankGroup1"
    enemy_state1["patrol_points"] = [(GRID_WIDTH - 1, GRID_HEIGHT - 3), (GRID_WIDTH - 1, GRID_HEIGHT - 5)]
    enemy_state1["base_armor_front"] = enemy_state1["armor_front"]
    enemy_state1["base_armor_side"]  = enemy_state1["armor_side"]
    enemy_state1["base_armor_rear"]  = enemy_state1["armor_rear"]
    enemy_tank1 = EnemyTank("EnemyTankGroup1", enemy_state1, enemy_domain)

    enemy_state2 = copy.deepcopy(enemy_tank_state_template)
    enemy_state2["name"] = "EnemyTankGroup2"
    enemy_state2["position"] = (17, 5)
    enemy_state2["patrol_points"] = [(GRID_WIDTH - 1, GRID_HEIGHT - 5), (GRID_WIDTH - 1, GRID_HEIGHT - 3)]
    enemy_state2["base_armor_front"] = enemy_state2["armor_front"]
    enemy_state2["base_armor_side"]  = enemy_state2["armor_side"]
    enemy_state2["base_armor_rear"]  = enemy_state2["armor_rear"]
    enemy_tank2 = EnemyTank("EnemyTankGroup2", enemy_state2, enemy_domain)

    enemy_state3 = copy.deepcopy(enemy_infantry_state_template)
    enemy_state3["name"] = "EnemyInfantryGroup1"
    enemy_state3["base_armor_front"] = enemy_state3["armor_front"]
    enemy_state3["base_armor_side"]  = enemy_state3["armor_side"]
    enemy_state3["base_armor_rear"]  = enemy_state3["armor_rear"]
    enemy_infantry1 = EnemyInfantry("EnemyInfantryGroup1", enemy_state3, enemy_domain)

    enemy_state4 = copy.deepcopy(enemy_anti_tank_state_template)
    enemy_state4["name"] = "EnemyAntiTankGroup1"
    enemy_state4["patrol_points"] = [(GRID_WIDTH - 1, GRID_HEIGHT - 6), (GRID_WIDTH - 1, GRID_HEIGHT - 4)]
    enemy_state4["base_armor_front"] = enemy_state4["armor_front"]
    enemy_state4["base_armor_side"]  = enemy_state4["armor_side"]
    enemy_state4["base_armor_rear"]  = enemy_state4["armor_rear"]
    enemy_anti_tank1 = EnemyAntiTank("EnemyAntiTankGroup1", enemy_state4, enemy_domain)

    enemy_state5 = copy.deepcopy(enemy_artillery_state_template)
    enemy_state5["name"] = "EnemyArtilleryGroup1"
    enemy_state5["patrol_points"] = [(GRID_WIDTH - 1, GRID_HEIGHT - 7), (GRID_WIDTH - 1, GRID_HEIGHT - 4)]
    enemy_state5["base_armor_front"] = enemy_state5["armor_front"]
    enemy_state5["base_armor_side"]  = enemy_state5["armor_side"]
    enemy_state5["base_armor_rear"]  = enemy_state5["armor_rear"]
    enemy_artillery1 = EnemyArtillery("EnemyArtilleryGroup1", enemy_state5, enemy_domain)

    enemy_units = [enemy_tank1, enemy_tank2, enemy_infantry1, enemy_anti_tank1, enemy_artillery1]

    tank_state = copy.deepcopy(tank_state_template)
    infantry_state = copy.deepcopy(infantry_state_template)    
    artillery_state = copy.deepcopy(artillery_state_template)
    anti_tank_state = copy.deepcopy(anti_tank_state_template)

    for state in [tank_state, infantry_state, artillery_state, anti_tank_state]:
        state["enemy"] = enemy_tank_state_template
        state["target_enemy"] = enemy_tank_state_template
        state["outpost_position"] = enemy_tank_state_template["outpost_position"]
        state["visible_enemies"] = []

    tank = FriendlyTank("FriendlyTankGroup", tank_state, secure_outpost_domain)
    infantry = FriendlyInfantry("FriendlyInfantryGroup", infantry_state, secure_outpost_domain)
    artillery = FriendlyArtillery("FriendlyArtilleryGroup", artillery_state, secure_outpost_domain)
    # scout = FriendlyScout("FriendlyScoutGroup", scout_state, secure_outpost_domain)
    anti_tank = FriendlyAntiTank("FriendlyAntiTankGroup", anti_tank_state, secure_outpost_domain)
    friendly_units = [tank, infantry, artillery, anti_tank]

    commander = TeamCommander(friendly_units)
    sim = Simulation(friendly_units, enemy_units, commander, visualize=True, plan_name="Mode1_Test_Grouped_Dynamic_Partial")

    for unit in friendly_units:
        unit.sim = sim

    result = sim.run(max_steps=300)
    logger.info("\n=== Plan Evaluation ===")
    logger.info(f"Score: {result['score']:.1f}")
    logger.info(f"Total Friendly Health Remaining: {result['health']}")
    logger.info(f"Enemy Health Remaining: {result['enemy_health']}")
    logger.info(f"Outpost Secured: {result['outpost_secured']}")
    logger.info(f"Steps Taken: {result['steps_taken']}")

    plt.ioff()
    plt.show()
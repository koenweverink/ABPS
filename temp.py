import random
import heapq
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import itertools

###############################
# Grid, Obstacles, and LOS
###############################

GRID_WIDTH = 50
GRID_HEIGHT = 50
CELL_SIZE = 100  # each cell is 100 m

# Example: create a river obstacle in a vertical band with three bridges.
river_columns = range(round(GRID_WIDTH/3), round(GRID_WIDTH/3)+5)  # river width: 5 cells
bridge_centers = [10, 25, 40]
bridge_group_size = 2
bridge_rows = set()
for center in bridge_centers:
    bridge_rows.update(range(center - bridge_group_size // 2, center - bridge_group_size // 2 + bridge_group_size))
river = {(x, y) for x in river_columns for y in range(GRID_HEIGHT) if y not in bridge_rows}

# Forest in the top right corner.
forest = {(x, y) for x in range(GRID_WIDTH - 20, GRID_WIDTH - 5)
                for y in range(GRID_HEIGHT - 20, GRID_HEIGHT - 5)}

# Combine obstacles: here you may combine static obstacles (like a river, forest, etc.)
obstacles = river.union(forest)

def in_bounds(pos): 
    return 0 <= pos[0] < GRID_WIDTH and 0 <= pos[1] < GRID_HEIGHT

def neighbors(pos): 
    return [p for p in [(pos[0]+1, pos[1]), (pos[0]-1, pos[1]),
                         (pos[0], pos[1]+1), (pos[0], pos[1]-1)]
            if in_bounds(p) and p not in obstacles]

def manhattan(p, q): 
    return abs(p[0]-q[0]) + abs(p[1]-q[1])

def astar(start, goal):
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for nxt in neighbors(current):
            new_cost = cost_so_far[current] + 1
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

def next_step(start, goal):
    path = astar(start, goal)
    return path[1] if len(path) >= 2 else start

def get_line(start, end):
    x1, y1 = start; x2, y2 = end
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
                y += sy; err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            line.append((x, y))
            err -= dx
            if err < 0:
                x += sx; err += dy
            y += sy
    line.append((x, y))
    return line

def has_line_of_sight(start, end):
    line = get_line(start, end)
    return all(pos not in obstacles for pos in line[1:-1])

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

# Target selection: returns the closest alive enemy.
def select_enemy_for_unit(enemy_units, friendly_unit):
    viable = [e for e in enemy_units if e.state["enemy_alive"]]
    if not viable:
        return None
    return min(viable, key=lambda e: manhattan(friendly_unit.state["position"], e.state["enemy_position"]))

###############################
# Global Enemy State Creation
###############################

def create_enemy_state(index=0):
    if index == 0:
        return {
            "name": "EnemyTank1",
            "enemy_position": (GRID_WIDTH - 1, GRID_HEIGHT - 5),
            "enemy_alive": True,
            "enemy_health": 20,
            "max_health": 20,
            "enemy_armor": 17,
            "outpost_position": (GRID_WIDTH - 1, 0),
            "outpost_secured": False,
            "enemy_attack_range": 2400 / CELL_SIZE,   # 24 cells (2400 m)
            "enemy_accuracy": 0.7,
            "enemy_penetration": 18,
            "enemy_damage": 9,
            "enemy_suppression": 0.12,
            "rate_of_fire": 4.9,
            "suppression": {},
            "patrol_points": [(GRID_WIDTH - 1, GRID_HEIGHT - 5), (0, GRID_HEIGHT - 5)],
            "current_patrol_index": 0,
            "vision_range": 2000 / CELL_SIZE,  # 20 cells (example)
            "retreat_point": (GRID_WIDTH - 1, GRID_HEIGHT - 1)
        }
    else:
        return {
            "name": "EnemyTank2",
            "enemy_position": (GRID_WIDTH - 3, 2),
            "enemy_alive": True,
            "enemy_health": 20,
            "max_health": 20,
            "enemy_armor": 17,
            "outpost_position": (GRID_WIDTH - 1, 0),
            "outpost_secured": False,
            "enemy_attack_range": 2400 / CELL_SIZE,
            "enemy_accuracy": 0.7,
            "enemy_penetration": 18,
            "enemy_damage": 9,
            "enemy_suppression": 0.12,
            "rate_of_fire": 4.9,
            "suppression": {},
            "patrol_points": [(GRID_WIDTH - 3, 2), (GRID_WIDTH // 2, GRID_HEIGHT // 2)],
            "current_patrol_index": 0,
            "vision_range": 2000 / CELL_SIZE,
            "retreat_point": (GRID_WIDTH - 1, GRID_HEIGHT - 1)
        }

###############################
# HTN Domains and Planner
###############################
# This domain now includes a primitive Hold action.
secure_outpost_domain = {
    "SecureOutpostMission": [
        # If no enemy is spotted, hold position.
        (lambda state: not state.get("enemy_spotted", False),
         ["Hold"]),
        # Once an enemy is spotted:
        # If target enemy is alive, then defeat it then move to outpost and secure.
        (lambda state: state.get("enemy_spotted", False) and state["target_enemy"]["enemy_alive"],
         ["DefeatEnemies", "MoveToOutpost", "SecureOutpost"]),
        # If target enemy is dead and unit is not at outpost, move there and secure.
        (lambda state: state.get("enemy_spotted", False) and not state["target_enemy"]["enemy_alive"] and
         state["position"] != state["enemy"]["outpost_position"],
         ["MoveToOutpost", "SecureOutpost"]),
        # If target enemy is dead and unit is at outpost, hold.
        (lambda state: state.get("enemy_spotted", False) and not state["target_enemy"]["enemy_alive"] and
         state["position"] == state["enemy"]["outpost_position"],
         ["Hold"])
    ],
    "DefeatEnemies": [
        # If target enemy is in range with LOS, attack.
        (lambda state: state["target_enemy"]["enemy_alive"] and 
                    enemy_in_range_with_los({**state, "enemy": state["target_enemy"]}),
         ["AttackEnemy"]),
        # If target enemy is alive but out of range, move closer then attack.
        (lambda state: state["target_enemy"]["enemy_alive"] and 
                    manhattan(state["position"], state["target_enemy"]["enemy_position"]) > state["friendly_attack_range"],
         ["MoveToEnemy", "AttackEnemy"])
    ]
}

# We want a random-branch HTNPlanner.
class HTNPlanner:
    def __init__(self, domain):
        self.domain = domain
    def plan(self, task, state):
        if task not in self.domain:
            return [task]
        candidate_plans = []
        for condition, subtasks in self.domain[task]:
            if condition(state):
                plan = []
                for subtask in subtasks:
                    if subtask in self.domain:
                        sub_plan = self.plan(subtask, state)
                    else:
                        sub_plan = [subtask]
                    if sub_plan is None:
                        break
                    plan.extend(sub_plan)
                if plan is not None:
                    candidate_plans.append(plan)
        if candidate_plans:
            return random.choice(candidate_plans)
        return None

###############################
# Refactored Enemy Classes
###############################

class EnemyUnit:
    def __init__(self, name, state, domain):
        self.name = name
        self.state = state
        self.planner = HTNPlanner(domain)
        self.current_plan = []
    def update_plan(self, friendly_units):
        combined_state = self.state.copy()
        combined_state["friendly_units"] = friendly_units
        if friendly_units:
            target = min(friendly_units, key=lambda u: manhattan(self.state["enemy_position"], u.state["position"]))
            combined_state["target_position"] = target.state["position"]
        else:
            combined_state["target_position"] = self.state["enemy_position"]
        mission = "DefendAreaMission"  # (kept simple for enemy behavior)
        new_plan = self.planner.plan(mission, combined_state)
        self.current_plan = new_plan if new_plan else []
        print(f"{self.name} updated plan: {self.current_plan}")
    def execute_next_task(self, friendly_units):
        if not self.current_plan or not self.state["enemy_alive"]:
            return
        task = self.current_plan[0]
        if task == "Patrol":
            idx = self.state["current_patrol_index"]
            target = self.state["patrol_points"][idx]
            self.state["enemy_position"] = next_step(self.state["enemy_position"], target)
            if self.state["enemy_position"] == target:
                self.state["current_patrol_index"] = (idx + 1) % len(self.state["patrol_points"])
                self.current_plan.pop(0)
        elif task == "ChaseTarget":
            target = min(friendly_units, key=lambda u: manhattan(self.state["enemy_position"], u.state["position"])).state["position"]
            self.state["enemy_position"] = next_step(self.state["enemy_position"], target)
            if manhattan(self.state["enemy_position"], target) <= self.state["enemy_attack_range"]:
                self.current_plan.pop(0)
        elif task == "AttackTarget":
            target_unit = None
            min_dist = float('inf')
            for u in friendly_units:
                dist = manhattan(self.state["enemy_position"], u.state["position"])
                if (dist <= self.state["enemy_attack_range"] and 
                    has_line_of_sight(self.state["enemy_position"], u.state["position"])):
                    if dist < min_dist:
                        min_dist = dist
                        target_unit = u
            if target_unit:
                num_attacks = get_num_attacks(self.state["rate_of_fire"])
                total_suppression = sum(self.state["suppression"].values())
                effective_accuracy = max(0, self.state["enemy_accuracy"] - total_suppression)
                for _ in range(num_attacks):
                    if random.random() < effective_accuracy:
                        D = self.state["enemy_penetration"] - target_unit.state.get("armor", 0)
                        if random.random() < get_penetration_probability(D):
                            target_unit.state["friendly_health"] -= self.state["enemy_damage"]
                            if target_unit.state["friendly_health"] <= 0:
                                print(f"{target_unit.name} destroyed by {self.name}!")
                        target_unit.state["suppression_from_enemy"] = target_unit.state.get("suppression_from_enemy", 0) + self.state["enemy_suppression"]
            self.current_plan.pop(0)
        elif task == "Retreat":
            retreat_point = self.state.get("retreat_point", (9, 9))
            self.state["enemy_position"] = next_step(self.state["enemy_position"], retreat_point)
            if self.state["enemy_position"] == retreat_point:
                self.current_plan.pop(0)
    def get_goal_position(self):
        if not self.current_plan:
            return self.state["enemy_position"]
        task = self.current_plan[0]
        if task == "Patrol":
            idx = self.state["current_patrol_index"]
            return self.state["patrol_points"][idx]
        elif task in ["ChaseTarget", "AttackTarget"]:
            if "friendly_units" in self.state and self.state["friendly_units"]:
                return min(self.state["friendly_units"], key=lambda u: manhattan(self.state["enemy_position"], u.state["position"])).state["position"]
        elif task == "Retreat":
            return self.state.get("retreat_point", (9, 9))
        return self.state["enemy_position"]

class EnemyTank(EnemyUnit):
    def __init__(self, name, state, domain):
        super().__init__(name, state, domain)

###############################
# Friendly Unit Classes
###############################

class FriendlyUnit:
    def __init__(self, name, state, domain):
        self.name = name
        self.state = state
        self.planner = HTNPlanner(domain)
        self.current_plan = []
        self.last_enemy_pos = state.get("target_enemy", {}).get("enemy_position")
        self.last_health = state["friendly_health"]
    def attack_enemy(self):
        enemy_data = self.state["target_enemy"]
        enemy_pos = enemy_data["enemy_position"]
        if manhattan(self.state["position"], enemy_pos) <= self.state["friendly_attack_range"]:
            num_attacks = get_num_attacks(self.state["rate_of_fire"])
            suppression_on_self = self.state.get("suppression_from_enemy", 0)
            effective_accuracy = max(0, self.state["friendly_accuracy"] - suppression_on_self)
            for _ in range(num_attacks):
                if random.random() < effective_accuracy:
                    D = self.state["penetration"] - enemy_data.get("enemy_armor", 0)
                    if random.random() < get_penetration_probability(D):
                        enemy_data["enemy_health"] -= self.state["damage"]
                        print(f"{self.name} attack: Damage {self.state['damage']} applied; new enemy health: {enemy_data['enemy_health']}")
                        if enemy_data["enemy_health"] <= 0:
                            enemy_data["enemy_alive"] = False
                    enemy_data["suppression"][self.name] = enemy_data["suppression"].get(self.name, 0) + self.state["suppression"]
    def update_plan(self, force_replan=False):
        mission = list(self.planner.domain.keys())[0]
        if force_replan or not self.current_plan:
            new_plan = self.planner.plan(mission, self.state)
            if not new_plan:
                new_plan = self.planner.domain[mission][-1][1]
            self.current_plan = new_plan
            print(f"{self.name} replanned: {self.current_plan}")
        self.last_enemy_pos = self.state["target_enemy"]["enemy_position"]
        self.last_health = self.state["friendly_health"]
    def execute_next_task(self):
        if self.current_plan:
            task = self.current_plan[0]
            if task == "Hold":
                print(f"{self.name} holds position at {self.state['position']}.")
                self.current_plan.pop(0)
            elif task == "MoveToEnemy":
                current_pos = self.state["position"]
                goal_pos = self.get_goal_position()
                next_pos = next_step(current_pos, goal_pos)
                self.state["position"] = next_pos
                enemy_pos = self.state["target_enemy"]["enemy_position"]
                if (manhattan(next_pos, enemy_pos) <= self.state["friendly_attack_range"] and 
                    has_line_of_sight(self.state["position"], enemy_pos)):
                    self.current_plan.pop(0)
            elif task == "AttackEnemy":
                self.attack_enemy()
                if not self.state["target_enemy"]["enemy_alive"]:
                    self.current_plan.pop(0)
            elif task == "MoveToOutpost":
                current_pos = self.state["position"]
                goal_pos = self.state["enemy"]["outpost_position"]
                next_pos = next_step(current_pos, goal_pos)
                self.state["position"] = next_pos
                if next_pos == goal_pos:
                    self.current_plan.pop(0)
            elif task == "SecureOutpost":
                if self.state["position"] == self.state["enemy"]["outpost_position"]:
                    self.state["enemy"]["outpost_secured"] = True
                    self.current_plan.pop(0)
                else:
                    self.current_plan.pop(0)
    def get_goal_position(self):
        if self.current_plan:
            return self.state["target_enemy"]["enemy_position"]
        else:
            return self.state["enemy"]["outpost_position"]

class FriendlyTank(FriendlyUnit): 
    pass

class FriendlyInfantry(FriendlyUnit): 
    pass

class FriendlyArtillery(FriendlyUnit): 
    pass

class FriendlyScout(FriendlyUnit):
    pass

###############################
# Team Commander and Simulation
###############################

class TeamCommander:
    def __init__(self, friendly_units):
        self.friendly_units = friendly_units
    def share_tank_health(self):
        tank = next((unit for unit in self.friendly_units if isinstance(unit, FriendlyTank)), None)
        if tank:
            tank_health = tank.state["friendly_health"]
            tank_max_health = tank.state["max_health"]
            for unit in self.friendly_units:
                unit.state["tank_health"] = tank_health
                unit.state["tank_max_health"] = tank_max_health

class Simulation:
    def __init__(self, friendly_units, enemy_units, team_commander, visualize=False, plan_name="Unknown Plan"):
        self.friendly_units = friendly_units
        self.enemy_units = enemy_units
        self.team_commander = team_commander
        # Initialize each friendly unit with the enemy state from an active enemy (if any)
        active_enemy = next((e for e in enemy_units if e.state["enemy_alive"]), None)
        for unit in self.friendly_units:
            unit.state["enemy"] = active_enemy.state if active_enemy else {}
        self.step_count = 0
        self.visualize = visualize
        self.plan_name = plan_name
        if self.visualize:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8,8))
    def update_friendly_enemy_info(self):
        # Update the enemy_spotted flag based on any scout units.
        spotted = False
        for unit in self.friendly_units:
            if unit.state.get("role") == "scout":
                vision = unit.state.get("vision_range", 0)
                enemy = unit.state.get("enemy", {})
                enemy_pos = enemy.get("enemy_position", (0,0))
                if manhattan(unit.state["position"], enemy_pos) <= vision:
                    spotted = True
                    print(f"{unit.name} (scout) has spotted the enemy!")
        for unit in self.friendly_units:
            unit.state["enemy_spotted"] = spotted
    def update_enemy_behavior(self):
        for enemy in self.enemy_units:
            if enemy.state["enemy_alive"]:
                enemy.update_plan(self.friendly_units)
                enemy.execute_next_task(self.friendly_units)
                if self.visualize:
                    goal = enemy.get_goal_position()
                    print(f"{enemy.name}'s current A* goal: {goal}")
    def evaluate_plan(self):
        total_friendly_health = sum(u.state["friendly_health"] for u in self.friendly_units)
        max_friendly_health = sum(u.state["max_health"] for u in self.friendly_units)
        enemy_health = sum(e.state["enemy_health"] for e in self.enemy_units if e.state["enemy_alive"])
        max_enemy_health = sum(e.state["max_health"] for e in self.enemy_units)
        outpost_secured = any(e.state["outpost_secured"] for e in self.enemy_units)
        steps_taken = self.step_count
        friendly_ratio = total_friendly_health / max_friendly_health if max_friendly_health > 0 else 0
        enemy_ratio = enemy_health / max_enemy_health if max_enemy_health > 0 else 0
        score = (friendly_ratio * 20) - (enemy_ratio * 20) + (10 if outpost_secured else -10) - 0.1 * steps_taken
        return {"score": score, "friendly_health": total_friendly_health,
                "enemy_health": enemy_health, "outpost_secured": outpost_secured,
                "steps_taken": steps_taken}
    def update_plot(self):
        major_step = 500
        self.ax.clear()
        self.fig.set_size_inches(8,8)
        self.ax.set_xlim(-CELL_SIZE/2, GRID_WIDTH * CELL_SIZE - CELL_SIZE/2)
        self.ax.set_ylim(-CELL_SIZE/2, GRID_HEIGHT * CELL_SIZE - CELL_SIZE/2)
        self.ax.set_xticks(np.arange(0, (GRID_WIDTH+1)*CELL_SIZE, major_step))
        self.ax.set_yticks(np.arange(0, (GRID_HEIGHT+1)*CELL_SIZE, major_step))
        self.ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f} m"))
        self.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f} m"))
        self.ax.grid(True)
        self.ax.set_aspect("equal", adjustable="box")
        # Draw obstacles:
        for obs in obstacles:
            self.ax.add_patch(plt.Rectangle((obs[0]*CELL_SIZE, obs[1]*CELL_SIZE),
                                             CELL_SIZE, CELL_SIZE, color='black'))
        # Draw river (blue) and forest (green)
        for r in river:
            self.ax.add_patch(plt.Rectangle((r[0]*CELL_SIZE, r[1]*CELL_SIZE),
                                             CELL_SIZE, CELL_SIZE, color='blue'))
        for f in forest:
            self.ax.add_patch(plt.Rectangle((f[0]*CELL_SIZE, f[1]*CELL_SIZE),
                                             CELL_SIZE, CELL_SIZE, color='green'))
        # Draw outpost.
        if self.enemy_units:
            outpost = self.enemy_units[0].state["outpost_position"]
            outpost_x = outpost[0]*CELL_SIZE + CELL_SIZE/2
            outpost_y = outpost[1]*CELL_SIZE + CELL_SIZE/2
            self.ax.plot(outpost_x, outpost_y, marker='*', markersize=18, color='magenta')
        for enemy in self.enemy_units:
            if enemy.state["enemy_alive"]:
                ex, ey = enemy.state["enemy_position"]
                center_x = ex * CELL_SIZE + CELL_SIZE/2
                center_y = ey * CELL_SIZE + CELL_SIZE/2
                self.ax.plot(center_x, center_y, marker='s', markersize=14, color='green')
                frac = enemy.state["enemy_health"] / enemy.state["max_health"]
                bar_width = 100  # in meters
                bar_height = 15
                bar_x = center_x - bar_width/2
                bar_y = center_y + 40
                self.ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width*frac, bar_height, color='green', zorder=5))
                self.ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width, bar_height, fill=False, edgecolor='black', zorder=5))
        for unit in self.friendly_units:
            pos = unit.state["position"]
            center_x = pos[0]*CELL_SIZE + CELL_SIZE/2
            center_y = pos[1]*CELL_SIZE + CELL_SIZE/2
            role = unit.state.get("role", "unknown")
            color = {'attacker': 'red', 'outpost_securer': 'blue',
                     'support': 'orange', 'scout': 'purple'}.get(role, 'gray')
            markersize = 16
            self.ax.plot(center_x, center_y, marker='o', markersize=markersize, color=color, zorder=5)
            self.ax.text(center_x + 40, center_y + 40, unit.name, fontsize=10, color='black', zorder=5)
            hp = unit.state["friendly_health"]
            max_hp = unit.state["max_health"]
            frac = hp / max_hp if max_hp > 0 else 0
            bar_width = 100
            bar_height = 15
            bar_x = center_x - bar_width/2
            bar_y = center_y + 60
            self.ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width*frac, bar_height, color='green', zorder=5))
            self.ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width, bar_height, fill=False, edgecolor='black', zorder=5))
        self.ax.set_title(f"Simulation Step {self.step_count} - {self.plan_name}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    def step(self):
        self.step_count += 1
        print(f"\n--- Simulation Step {self.step_count} ---")
        self.update_friendly_enemy_info()
        self.team_commander.share_tank_health()
        self.update_enemy_behavior()
        for unit in self.friendly_units:
            target = unit.state.get("target_enemy")
            if target:
                print(f"{unit.name} target enemy: {target['name']} at {target['enemy_position']}")
            unit.execute_next_task()
            print(f"{unit.name} position: {unit.state['position']}")
        # Reassign target if needed.
        for unit in self.friendly_units:
            if not unit.state["target_enemy"]["enemy_alive"]:
                new_target = select_enemy_for_unit(self.enemy_units, unit)
                if new_target:
                    print(f"{unit.name} switching target to {new_target.state['name']}")
                    unit.state["target_enemy"] = new_target.state
                    unit.update_plan(force_replan=True)
        for unit in self.friendly_units:
            print(f"State for {unit.name}: {unit.state}")
        for enemy in self.enemy_units:
            print(f"State for {enemy.name}: {enemy.state}")
    def run(self, max_steps=100):
        self.step_count = 0
        self.team_commander.share_tank_health()
        for unit in self.friendly_units:
            unit.update_plan(force_replan=True)
        for enemy in self.enemy_units:
            enemy.update_plan(self.friendly_units)
        for _ in range(max_steps):
            # Check mission: all enemies dead and outpost secured.
            if all(not enemy.state["enemy_alive"] for enemy in self.enemy_units) and \
               any(enemy.state["outpost_secured"] for enemy in self.enemy_units):
                self.update_plot()
                print("\nMission accomplished: Enemy destroyed and outpost secured!")
                return self.evaluate_plan()
            self.step()
            self.update_plot()
            alive_friendlies = [u for u in self.friendly_units if u.state["friendly_health"] > 0]
            if len(alive_friendlies) < len(self.friendly_units):
                for dead in set(self.friendly_units) - set(alive_friendlies):
                    print(f"\n{dead.name} has been destroyed!")
                self.friendly_units = alive_friendlies
                if not self.friendly_units:
                    self.update_plot()
                    print("\nAll friendly units have been destroyed! Mission failed.")
                    return self.evaluate_plan()
        self.update_plot()
        print("\nMission incomplete after maximum steps.")
        return self.evaluate_plan()

###############################
# Main Simulation Setup (Mode 3 Only)
###############################

if __name__ == "__main__":
    # Friendly state templates with enemy_spotted flag.
    tank_state_template = {
        "position": (0, 0),
        "friendly_health": 20,
        "max_health": 20,
        "armor": 17,
        "friendly_accuracy": 0.75,
        "rate_of_fire": 4.9,
        "damage": 9,
        "suppression": 0.12,
        "penetration": 18,
        "friendly_attack_range": 2400 / CELL_SIZE,  # 24 cells (2400 m)
        "role": "attacker",
        "enemy_spotted": False
    }
    infantry_state_template = {
        "position": (0, 1),
        "friendly_health": 1,
        "max_health": 1,
        "armor": 0,
        "friendly_accuracy": 0.50,
        "rate_of_fire": 294,
        "damage": 0.8,
        "suppression": 0.01,
        "penetration": 1,
        "friendly_attack_range": 1200 / CELL_SIZE,  # 12 cells (1200 m)
        "role": "outpost_securer",
        "enemy_spotted": False
    }
    artillery_state_template = {
        "position": (1, 0),
        "friendly_health": 18,
        "max_health": 18,
        "armor": 2,
        "friendly_accuracy": 0.85,
        "rate_of_fire": 8.6,
        "damage": 3.5,
        "suppression": 15,
        "penetration": 1,
        "friendly_attack_range": 4000 / CELL_SIZE,  # 40 cells (4000 m)
        "role": "support",
        "enemy_spotted": False
    }
    scout_state_template = {
        "position": (1, 1),
        "friendly_health": 18,
        "max_health": 18,
        "armor": 3,
        "friendly_accuracy": 0.35,
        "rate_of_fire": 115,
        "damage": 2.4,
        "suppression": 5,
        "penetration": 5,
        "vision_range": 2600 / CELL_SIZE,       # 26 cells (~2600 m)
        "friendly_attack_range": 1800 / CELL_SIZE,  # 18 cells (~1800 m)
        "role": "scout",
        "enemy_spotted": False
    }
    
    # Hierarchical domain for SecureOutpostMission with Hold option.
    secure_outpost_domain = {
        "SecureOutpostMission": [
            # If no enemy is spotted, hold position.
            (lambda state: not state.get("enemy_spotted", False),
             ["Hold"]),
            # Once enemy is spotted:
            # If target enemy is alive, defeat it then move to outpost and secure.
            (lambda state: state.get("enemy_spotted", False) and state["target_enemy"]["enemy_alive"],
             ["DefeatEnemies", "MoveToOutpost", "SecureOutpost"]),
            # If target enemy is dead and unit is not at outpost, move there and secure.
            (lambda state: state.get("enemy_spotted", False) and not state["target_enemy"]["enemy_alive"] and
             state["position"] != state["enemy"]["outpost_position"],
             ["MoveToOutpost", "SecureOutpost"]),
            # If target enemy is dead and unit is at outpost, hold.
            (lambda state: state.get("enemy_spotted", False) and not state["target_enemy"]["enemy_alive"] and
             state["position"] == state["enemy"]["outpost_position"],
             ["Hold"])
        ],
        "DefeatEnemies": [
            (lambda state: state["target_enemy"]["enemy_alive"] and 
                        enemy_in_range_with_los({**state, "enemy": state["target_enemy"]}),
             ["AttackEnemy"]),
            (lambda state: state["target_enemy"]["enemy_alive"] and 
                        manhattan(state["position"], state["target_enemy"]["enemy_position"]) > state["friendly_attack_range"],
             ["MoveToEnemy", "AttackEnemy"])
        ]
    }

    # Use candidate_domain as secure_outpost_domain.
    candidate_domain = secure_outpost_domain

    # Enemy Target Assignment:
    num_friendlies = 4  # tank, infantry, artillery, scout.
    enemy_target_combinations = list(itertools.product([0, 1], repeat=num_friendlies))
    friendly_candidate_combinations = list(itertools.product([candidate_domain], repeat=num_friendlies))
    full_combinations = list(itertools.product(friendly_candidate_combinations, enemy_target_combinations))
    print("Total full candidate combinations (domain & enemy target assignments):", len(full_combinations))
    
    enemy_target_usage = {0: {"count": 0}, 1: {"count": 0}}
    combination_results = []
    runs_per_combination = 10
    
    for (friendly_combo, enemy_targets) in full_combinations:
        combo_label = ("SecureOutpostDomain", enemy_targets)
        scores = []
        for run in range(runs_per_combination):
            enemy_state1 = create_enemy_state(index=0)
            enemy_state2 = create_enemy_state(index=1)
            enemy_unit1 = EnemyTank("EnemyTank1", enemy_state1, enemy_plan_domain)
            enemy_unit2 = EnemyTank("EnemyTank2", enemy_state2, enemy_plan_domain)
            enemy_units = [enemy_unit1, enemy_unit2]
    
            tank_state = tank_state_template.copy()
            infantry_state = infantry_state_template.copy()
            artillery_state = artillery_state_template.copy()
            scout_state = scout_state_template.copy()
    
            # Assign enemy state.
            tank_state["enemy"] = enemy_state1
            infantry_state["enemy"] = enemy_state1
            artillery_state["enemy"] = enemy_state1
            scout_state["enemy"] = enemy_state1
    
            # Assign target_enemy based on enemy_targets.
            tank_state["target_enemy"] = enemy_unit1.state if enemy_targets[0] == 0 else enemy_unit2.state
            infantry_state["target_enemy"] = enemy_unit1.state if enemy_targets[1] == 0 else enemy_unit2.state
            artillery_state["target_enemy"] = enemy_unit1.state if enemy_targets[2] == 0 else enemy_unit2.state
            scout_state["target_enemy"] = enemy_unit1.state if enemy_targets[3] == 0 else enemy_unit2.state
    
            tank_unit = FriendlyTank("FriendlyTank", tank_state, candidate_domain)
            infantry_unit = FriendlyInfantry("FriendlyInfantry", infantry_state, candidate_domain)
            artillery_unit = FriendlyArtillery("FriendlyArtillery", artillery_state, candidate_domain)
            scout_unit = FriendlyScout("FriendlyScout", scout_state, candidate_domain)
            friendly_units = [tank_unit, infantry_unit, artillery_unit, scout_unit]
    
            # Override update_plan.
            for unit in friendly_units:
                def custom_update_plan(self, force_replan=False):
                    mission = list(self.planner.domain.keys())[0]
                    new_plan = self.planner.plan(mission, self.state)
                    if not new_plan:
                        new_plan = self.planner.domain[mission][-1][1]
                    self.current_plan = new_plan
                    print(f"{self.name} replanned: {self.current_plan}")
                    self.last_enemy_pos = self.state["target_enemy"]["enemy_position"]
                    self.last_health = self.state["friendly_health"]
                unit.update_plan = custom_update_plan.__get__(unit, unit.__class__)
    
            commander = TeamCommander(friendly_units)
            sim = Simulation(friendly_units, enemy_units, commander,
                             visualize=False, plan_name="Combo_" + str(combo_label))
            sim_result = sim.run(max_steps=100)
            scores.append(sim_result["score"])
        avg_score = np.mean(scores)
        combination_results.append((combo_label, avg_score))
        for targ in enemy_targets:
            enemy_target_usage[targ]["count"] += runs_per_combination
    
    best_combo = max(combination_results, key=lambda x: x[1])
    print("\n=== Best Combination ===")
    print("Best candidate (domain & enemy target assignment):", best_combo[0])
    print("Best combination average score:", best_combo[1])
    
    print("\n=== Overall Combination Results ===")
    for combo_label, avg_score in combination_results:
        print(f"  {combo_label}: {avg_score:.2f}")
    
    print("\nEnemy target usage:")
    for targ, info in enemy_target_usage.items():
        print(f"  Enemy {targ+1}: used {info['count']} times")
    
    overall_avg = np.mean([score for (_, score) in combination_results])
    print("\n=== Overall Experiment Summary ===")
    print(f"Overall average score: {overall_avg:.2f}")
    
    visualize = True
    if visualize:
        best_enemy_targets = best_combo[0][1]
        print("\nDetailed run for best combination:")
        print("  Enemy target assignment:", best_enemy_targets)
    
        enemy_state1 = create_enemy_state(index=0)
        enemy_state2 = create_enemy_state(index=1)
        enemy_unit1 = EnemyTank("EnemyTank1", enemy_state1, enemy_plan_domain)
        enemy_unit2 = EnemyTank("EnemyTank2", enemy_state2, enemy_plan_domain)
        enemy_units = [enemy_unit1, enemy_unit2]
    
        tank_state = tank_state_template.copy()
        infantry_state = infantry_state_template.copy()
        artillery_state = artillery_state_template.copy()
        scout_state = scout_state_template.copy()
        tank_state["enemy"] = enemy_state1
        infantry_state["enemy"] = enemy_state1
        artillery_state["enemy"] = enemy_state1
        scout_state["enemy"] = enemy_state1
        tank_state["target_enemy"] = enemy_state1 if best_enemy_targets[0] == 0 else enemy_state2
        infantry_state["target_enemy"] = enemy_state1 if best_enemy_targets[1] == 0 else enemy_state2
        artillery_state["target_enemy"] = enemy_state1 if best_enemy_targets[2] == 0 else enemy_state2
        scout_state["target_enemy"] = enemy_state1 if best_enemy_targets[3] == 0 else enemy_state2
    
        tank_unit = FriendlyTank("FriendlyTank", tank_state, secure_outpost_domain)
        infantry_unit = FriendlyInfantry("FriendlyInfantry", infantry_state, secure_outpost_domain)
        artillery_unit = FriendlyArtillery("FriendlyArtillery", artillery_state, secure_outpost_domain)
        scout_unit = FriendlyScout("FriendlyScout", scout_state, secure_outpost_domain)
        friendly_units = [tank_unit, infantry_unit, artillery_unit, scout_unit]
    
        commander = TeamCommander(friendly_units)
        sim = Simulation(friendly_units, enemy_units, commander,
                         visualize=True, plan_name="Best_Combination_Detailed")
        result = sim.run(max_steps=100)
        print("\nDetailed Evaluation:")
        print(f"  Score: {result['score']:.2f}")
        print(f"  Friendly Health: {result['friendly_health']:.2f}")
        print(f"  Enemy Health: {result['enemy_health']:.2f}")
        print(f"  Steps Taken: {result['steps_taken']:.2f}")
    
    plt.ioff()
    plt.show()

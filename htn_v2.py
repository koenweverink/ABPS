import random
import heapq
import matplotlib.pyplot as plt

###############################
# Grid, Obstacles, and LOS
###############################

GRID_WIDTH = 10
GRID_HEIGHT = 10

# Define obstacles as a set of (x,y) coordinates.
obstacles = {
    (3, 3), (3, 4), (3, 5),
    (4, 5),
    (6, 6)  # Moved from (5,5) to avoid conflict with patrol point.
}

def in_bounds(pos):
    x, y = pos
    return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT

def neighbors(pos):
    x, y = pos
    results = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    results = filter(in_bounds, results)
    return [p for p in results if p not in obstacles]

def manhattan(p, q):
    return abs(p[0]-q[0]) + abs(p[1]-q[1])

def astar(start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start))
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
    print(f"A* called with goal: {goal}. Computed path: {path}")
    if len(path) >= 2:
        return path[1]
    return start

def get_line(start, end):
    x1, y1 = start
    x2, y2 = end
    line = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
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

def has_line_of_sight(start, end):
    line = get_line(start, end)
    for pos in line[1:-1]:
        if pos in obstacles:
            return False
    return True

###############################
# Global Enemy State
###############################

global_enemy = {
    "enemy_position": (7, 7),
    "enemy_alive": True,
    "enemy_health": 100,
    "max_health": 100,
    "outpost_position": (9, 0),
    "outpost_secured": False,
    "enemy_attack_range": 3,
    "base_accuracy": 0.7,
    "suppression": {},
    "patrol_points": [(7, 7), (7, 5), (5, 5), (5, 7)],
    "current_patrol_index": 0,
    "detection_range": 5
}

###############################
# HTN Domain and Planner
###############################

def enemy_not_in_range(state):
    return state["enemy"]["enemy_alive"] and \
           manhattan(state["position"], state["enemy"]["enemy_position"]) > state["friendly_attack_range"] and \
           not state["enemy"]["outpost_secured"]

def enemy_in_range(state):
    return state["enemy"]["enemy_alive"] and \
           manhattan(state["position"], state["enemy"]["enemy_position"]) <= state["friendly_attack_range"] and \
           not state["enemy"]["outpost_secured"]

def enemy_dead_not_at_outpost(state):
    return (not state["enemy"]["enemy_alive"]) and \
           (state["position"] != state["enemy"]["outpost_position"]) and \
           (not state["enemy"]["outpost_secured"])

def enemy_dead_at_outpost(state):
    return (not state["enemy"]["enemy_alive"]) and \
           (state["position"] == state["enemy"]["outpost_position"]) and \
           (not state["enemy"]["outpost_secured"])

domain = {
    "SecureMission": [
         (enemy_not_in_range, ["MoveToEnemy", "AttackEnemy", "MoveToOutpost", "SecureOutpost"]),
         (enemy_in_range, ["AttackEnemy", "MoveToOutpost", "SecureOutpost"]),
         (enemy_dead_not_at_outpost, ["MoveToOutpost", "SecureOutpost"]),
         (enemy_dead_at_outpost, ["SecureOutpost"])
    ]
}

class HTNPlanner:
    def __init__(self, domain):
        self.domain = domain
    def plan(self, task, state):
        if task not in self.domain:
            return [task]
        methods = self.domain[task]
        for condition, subtasks in methods:
            if condition(state):
                plan = []
                for subtask in subtasks:
                    sub_plan = self.plan(subtask, state)
                    if sub_plan is None:
                        break
                    plan.extend(sub_plan)
                else:
                    return plan
        return None

###############################
# Friendly Unit Classes
###############################

TANK_ATTACK_DAMAGE = 50
INFANTRY_ATTACK_DAMAGE = 30
ENEMY_ATTACK_DAMAGE = 20

class FriendlyUnit:
    def __init__(self, name, state, attack_damage):
        self.name = name
        self.state = state  # Contains position, health, accuracy, attack range, etc.
        self.planner = HTNPlanner(domain)
        self.current_plan = []
        self.attack_damage = attack_damage
    def update_plan(self):
        plan = self.planner.plan("SecureMission", self.state)
        if plan:
            self.current_plan = plan
        else:
            self.current_plan = []
        print(f"{self.name} new plan: {self.current_plan}")
    def attack_enemy(self):
        enemy_pos = self.state["enemy"]["enemy_position"]
        if manhattan(self.state["position"], enemy_pos) <= self.state["friendly_attack_range"]:
            if not has_line_of_sight(self.state["position"], enemy_pos):
                print(f"{self.name} cannot attack: No line of sight to enemy!")
                return
            if random.random() < self.state["friendly_accuracy"]:
                self.state["enemy"]["enemy_health"] -= self.attack_damage
                print(f"{self.name} attacked enemy for {self.attack_damage} damage. Enemy health: {self.state['enemy']['enemy_health']}")
                if self.state["enemy"]["enemy_health"] <= 0:
                    self.state["enemy"]["enemy_alive"] = False
                    print(f"{self.name} destroyed the enemy tank!")
            else:
                print(f"{self.name} attacked but missed the enemy!")
            if self.name not in self.state["enemy"]["suppression"]:
                suppression_value = 0.15 if isinstance(self, FriendlyTank) else 0.10
                self.state["enemy"]["suppression"][self.name] = suppression_value
                print(f"{self.name} applied suppression of {suppression_value}.")
        else:
            print(f"{self.name} cannot attack: Enemy is out of range.")
    def execute_next_task(self):
        if self.current_plan:
            task = self.current_plan.pop(0)
            print(f"{self.name} executing task: {task}")
            if task == "MoveToEnemy":
                old_pos = self.state["position"]
                new_pos = next_step(old_pos, self.state["enemy"]["enemy_position"])
                self.state["position"] = new_pos
                print(f"{self.name} moved from {old_pos} towards enemy, now at {new_pos}")
            elif task == "AttackEnemy":
                self.attack_enemy()
            elif task == "MoveToOutpost":
                old_pos = self.state["position"]
                new_pos = next_step(old_pos, self.state["enemy"]["outpost_position"])
                self.state["position"] = new_pos
                print(f"{self.name} moved from {old_pos} towards outpost, now at {new_pos}")
            elif task == "SecureOutpost":
                if self.state["position"] == self.state["enemy"]["outpost_position"]:
                    self.state["enemy"]["outpost_secured"] = True
                    print(f"{self.name} secured the enemy outpost!")
                else:
                    print(f"{self.name} is not at the outpost to secure it. Re-planning required.")
                    self.current_plan.insert(0, "SecureOutpost")
            else:
                print(f"{self.name} executed unknown task: {task}")
        else:
            print(f"{self.name} has no tasks to execute.")
    def get_goal_position(self):
        # Return the goal passed to A* based on the first movement task in the plan.
        for task in self.current_plan:
            if task == "MoveToEnemy":
                return self.state["enemy"]["enemy_position"]
            elif task in ("MoveToOutpost", "SecureOutpost"):
                return self.state["enemy"]["outpost_position"]
        return None

class FriendlyTank(FriendlyUnit):
    def __init__(self, name, state):
        super().__init__(name, state, attack_damage=TANK_ATTACK_DAMAGE)

class FriendlyInfantry(FriendlyUnit):
    def __init__(self, name, state):
        super().__init__(name, state, attack_damage=INFANTRY_ATTACK_DAMAGE)

###############################
# Simulation with Dynamic Enemy Behavior, Obstacles, LOS, and Graphical Visualization
###############################

class Simulation:
    def __init__(self, friendly_units):
        self.friendly_units = friendly_units
        self.step_count = 0
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6,6))
    def update_enemy_behavior(self):
        if global_enemy["enemy_alive"]:
            target_found = False
            for unit in self.friendly_units:
                dist = manhattan(global_enemy["enemy_position"], unit.state["position"])
                if dist <= global_enemy["detection_range"] and has_line_of_sight(global_enemy["enemy_position"], unit.state["position"]):
                    print(f"Enemy detected {unit.name} at distance {dist}. Chasing...")
                    new_pos = next_step(global_enemy["enemy_position"], unit.state["position"])
                    global_enemy["enemy_position"] = new_pos
                    target_found = True
                    break
            if not target_found:
                patrol_points = global_enemy["patrol_points"]
                idx = global_enemy["current_patrol_index"]
                target = patrol_points[idx]
                new_pos = next_step(global_enemy["enemy_position"], target)
                global_enemy["enemy_position"] = new_pos
                print(f"Enemy patrols towards {target}, now at {global_enemy['enemy_position']}.")
                if global_enemy["enemy_position"] == target:
                    global_enemy["current_patrol_index"] = (idx + 1) % len(patrol_points)
    def enemy_counter_attack(self):
        total_suppression = sum(global_enemy["suppression"].values())
        effective_accuracy = max(0, global_enemy["base_accuracy"] - total_suppression)
        print(f"Effective enemy accuracy: {effective_accuracy:.2f}")
        for unit in self.friendly_units:
            print(f"{unit.name} friendly accuracy: {unit.state['friendly_accuracy']:.2f}")
        if global_enemy["enemy_alive"]:
            for unit in self.friendly_units:
                if manhattan(unit.state["position"], global_enemy["enemy_position"]) <= global_enemy["enemy_attack_range"] and \
                   has_line_of_sight(unit.state["position"], global_enemy["enemy_position"]):
                    if random.random() < effective_accuracy:
                        unit.state["friendly_health"] -= ENEMY_ATTACK_DAMAGE
                        print(f"Enemy counter-attacks {unit.name} successfully! {unit.name}'s health is now {unit.state['friendly_health']}.")
                    else:
                        print(f"Enemy counter-attack against {unit.name} missed!")
                    break
    def update_plot(self):
        self.ax.clear()
        self.ax.set_xlim(-1, GRID_WIDTH)
        self.ax.set_ylim(-1, GRID_HEIGHT)
        self.ax.set_xticks(range(GRID_WIDTH))
        self.ax.set_yticks(range(GRID_HEIGHT))
        self.ax.grid(True)
        # Plot obstacles.
        for obs in obstacles:
            self.ax.add_patch(plt.Rectangle(obs, 1, 1, color='black'))
        # Plot outpost.
        outpost = global_enemy["outpost_position"]
        self.ax.plot(outpost[0]+0.5, outpost[1]+0.5, marker='*', markersize=15, color='magenta', label='Outpost')
        # Plot enemy.
        if global_enemy["enemy_alive"]:
            enemy = global_enemy["enemy_position"]
            self.ax.plot(enemy[0]+0.5, enemy[1]+0.5, marker='s', markersize=12, color='green', label='Enemy')
            # Draw enemy health bar.
            fraction = global_enemy["enemy_health"] / global_enemy["max_health"]
            bar_width = 0.8 * fraction
            self.ax.add_patch(plt.Rectangle((enemy[0]+0.1, enemy[1]+0.8), bar_width, 0.1, color='green'))
            self.ax.add_patch(plt.Rectangle((enemy[0]+0.1, enemy[1]+0.8), 0.8, 0.1, fill=False, edgecolor='black'))
        # Plot friendly units and their health bars.
        for unit in self.friendly_units:
            pos = unit.state["position"]
            if isinstance(unit, FriendlyTank):
                self.ax.plot(pos[0]+0.5, pos[1]+0.5, marker='o', markersize=12, color='red', label='Tank')
            else:
                self.ax.plot(pos[0]+0.5, pos[1]+0.5, marker='o', markersize=8, color='blue', label='Infantry')
            self.ax.text(pos[0]+0.2, pos[1]+0.2, unit.name, fontsize=9, color='black')
            # Health bar for friendly unit.
            max_health = unit.state["max_health"]
            current_health = unit.state["friendly_health"]
            fraction = current_health / max_health
            bar_width = 0.8 * fraction
            self.ax.add_patch(plt.Rectangle((pos[0]+0.1, pos[1]+0.8), bar_width, 0.1, color='green'))
            self.ax.add_patch(plt.Rectangle((pos[0]+0.1, pos[1]+0.8), 0.8, 0.1, fill=False, edgecolor='black'))
        self.ax.set_title(f"Simulation Step {self.step_count}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    def step(self):
        self.step_count += 1
        print(f"\n--- Simulation Step {self.step_count} ---")
        self.update_enemy_behavior()
        for unit in self.friendly_units:
            unit.update_plan()
            unit.execute_next_task()
            goal = unit.get_goal_position()
            if goal:
                print(f"{unit.name}'s current A* goal: {goal}")
            else:
                print(f"{unit.name} has no A* goal.")
        self.enemy_counter_attack()
        for unit in self.friendly_units:
            print(f"State for {unit.name}: {unit.state}")
        print(f"Global enemy state: {global_enemy}")
        self.update_plot()
        plt.pause(0.5)

###############################
# Main
###############################

if __name__ == "__main__":
    global_enemy["base_accuracy"] = 0.7
    global_enemy["patrol_points"] = [(7, 7), (7, 5), (5, 5), (5, 7)]
    global_enemy["current_patrol_index"] = 0
    global_enemy["detection_range"] = 5

    tank_state = {
        "position": (0, 0),
        "friendly_health": 100,
        "max_health": 100,
        "friendly_accuracy": 0.8,
        "friendly_attack_range": 3,
        "enemy": global_enemy
    }
    infantry_state = {
        "position": (0, 1),
        "friendly_health": 80,
        "max_health": 80,
        "friendly_accuracy": 0.75,
        "friendly_attack_range": 2,
        "enemy": global_enemy
    }
    tank = FriendlyTank("FriendlyTank", tank_state)
    infantry = FriendlyInfantry("FriendlyInfantry", infantry_state)
    sim = Simulation([tank, infantry])
    max_steps = 50
    for _ in range(max_steps):
        if not global_enemy["enemy_alive"] and global_enemy["outpost_secured"]:
            print("\nMission accomplished: Enemy tank destroyed and outpost secured!")
            break
        sim.step()
        for unit in sim.friendly_units:
            if unit.state["friendly_health"] <= 0:
                print(f"\n{unit.name} has been destroyed! Mission failed.")
                exit()
    else:
        print("\nMission incomplete after maximum steps.")
    plt.ioff()
    plt.show()

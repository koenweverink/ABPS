import random
import heapq
import matplotlib.pyplot as plt
import numpy as np

###############################
# Grid, Obstacles, and LOS (unchanged)
###############################

GRID_WIDTH = 10
GRID_HEIGHT = 10
obstacles = {(3, 3), (3, 4), (3, 5), (4, 5), (6, 6)}

def in_bounds(pos): return 0 <= pos[0] < GRID_WIDTH and 0 <= pos[1] < GRID_HEIGHT
def neighbors(pos): return [p for p in [(pos[0]+1, pos[1]), (pos[0]-1, pos[1]), (pos[0], pos[1]+1), (pos[0], pos[1]-1)] if in_bounds(p) and p not in obstacles]
def manhattan(p, q): return abs(p[0]-q[0]) + abs(p[1]-q[1])

def astar(start, goal):
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal: break
        for nxt in neighbors(current):
            new_cost = cost_so_far[current] + 1
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + manhattan(nxt, goal)
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current
    if goal not in came_from: return []
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
            if err < 0: y += sy; err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            line.append((x, y))
            err -= dx
            if err < 0: x += sx; err += dy
            y += sy
    line.append((x, y))
    return line

def has_line_of_sight(start, end):
    line = get_line(start, end)
    return all(pos not in obstacles for pos in line[1:-1])

###############################
# Helper Functions (unchanged)
###############################

def get_num_attacks(rpm):
    exact = rpm * 0.1
    n = int(exact)
    if random.random() < (exact - n): n += 1
    return n

def get_penetration_probability(D):
    if D <= -3: return 0.0
    elif -3 < D <= 0: return 0.33 + 0.11 * (D + 3)
    elif 0 < D <= 6: return 0.66 + (0.29/6) * D
    else: return 0.95

###############################
# Global Enemy State (unchanged)
###############################

def create_enemy_state():
    return {
        "enemy_position": (7, 7),
        "enemy_alive": True,
        "enemy_health": 20,
        "max_health": 20,
        "enemy_armor": 17,
        "outpost_position": (9, 0),
        "outpost_secured": False,
        "enemy_attack_range": 3,
        "enemy_accuracy": 0.7,
        "enemy_penetration": 18,
        "enemy_damage": 9,
        "enemy_suppression": 0.12,
        "rate_of_fire": 4.9,
        "suppression": {},
        "patrol_points": [(7, 7), (7, 5), (5, 5), (5, 7)],
        "current_patrol_index": 0,
        "detection_range": 5
    }

###############################
# HTN Domain and Planner (unchanged)
###############################

def enemy_not_in_range(state):
    return state["enemy"]["enemy_alive"] and manhattan(state["position"], state["enemy"]["enemy_position"]) > state["friendly_attack_range"]

def enemy_in_range(state):
    return state["enemy"]["enemy_alive"] and manhattan(state["position"], state["enemy"]["enemy_position"]) <= state["friendly_attack_range"]

def enemy_dead_not_at_outpost(state):
    return (not state["enemy"]["enemy_alive"]) and (state["position"] != state["enemy"]["outpost_position"]) and (not state["enemy"]["outpost_secured"])

def enemy_dead_at_outpost(state):
    return (not state["enemy"]["enemy_alive"]) and (state["position"] == state["enemy"]["outpost_position"]) and (not state["enemy"]["outpost_secured"])

tank_plan1_domain = {
    "DestroyEnemyMission": [
        (lambda state: enemy_not_in_range(state) and manhattan(state["position"], state["enemy"]["enemy_position"]) > state["friendly_attack_range"],
         ["MoveToEnemy", "AttackEnemy"]),
        (enemy_in_range, ["AttackEnemy"]),
        (lambda state: not state["enemy"]["enemy_alive"], [])
    ]
}

infantry_plan1_domain = {
    "SecureOutpostMission": [
        (lambda state: state["position"] != state["enemy"]["outpost_position"] and not state["enemy"]["outpost_secured"],
         ["MoveToOutpost", "SecureOutpost"]),
        (lambda state: state["position"] == state["enemy"]["outpost_position"] and not state["enemy"]["outpost_secured"],
         ["SecureOutpost"])
    ]
}

both_plan2_domain = {
    "EngageThenSecureMission": [
        (enemy_not_in_range, ["MoveToEnemy", "AttackEnemy"]),
        (enemy_in_range, ["AttackEnemy"]),
        (enemy_dead_not_at_outpost, ["MoveToOutpost", "SecureOutpost"]),
        (enemy_dead_at_outpost, ["SecureOutpost"])
    ]
}

class HTNPlanner:
    def __init__(self, domain): self.domain = domain
    def plan(self, task, state):
        if task not in self.domain: return [task]
        for condition, subtasks in self.domain[task]:
            if condition(state):
                plan = []
                for subtask in subtasks:
                    sub_plan = self.plan(subtask, state)
                    if sub_plan is None: break
                    plan.extend(sub_plan)
                else: return plan
        return None

###############################
# Team Commander (unchanged)
###############################

class TeamCommander:
    def __init__(self, friendly_units): self.friendly_units = friendly_units
    def assign_roles(self):
        for unit in self.friendly_units:
            unit.state["role"] = "attacker" if isinstance(unit, FriendlyTank) else "outpost_securer"
    def communicate_enemy_position(self):
        for unit in self.friendly_units:
            if has_line_of_sight(unit.state["position"], unit.state["enemy"]["enemy_position"]):
                observed = unit.state["enemy"]["enemy_position"]
                for other in self.friendly_units:
                    other.state["enemy"]["enemy_position"] = observed
                return

###############################
# Friendly Unit Classes (unchanged)
###############################

class FriendlyUnit:
    def __init__(self, name, state, domain):
        self.name = name
        self.state = state
        self.planner = HTNPlanner(domain)
        self.current_plan = []

    def update_plan(self):
        mission = "DestroyEnemyMission" if isinstance(self, FriendlyTank) and self.planner.domain == tank_plan1_domain else \
                  "SecureOutpostMission" if isinstance(self, FriendlyInfantry) and self.planner.domain == infantry_plan1_domain else \
                  "EngageThenSecureMission"
        plan = self.planner.plan(mission, self.state)
        self.current_plan = plan if plan else []
        print(f"{self.name} new plan: {self.current_plan}")

    def attack_enemy(self):
        enemy_data = self.state["enemy"]
        enemy_pos = enemy_data["enemy_position"]
        if manhattan(self.state["position"], enemy_pos) <= self.state["friendly_attack_range"] and has_line_of_sight(self.state["position"], enemy_pos):
            num_attacks = get_num_attacks(self.state["rate_of_fire"])
            suppression_on_self = self.state.get("suppression_from_enemy", 0)
            effective_accuracy = max(0, self.state["friendly_accuracy"] - suppression_on_self)
            for _ in range(num_attacks):
                if random.random() < effective_accuracy:
                    D = self.state["penetration"] - enemy_data.get("enemy_armor", 0)
                    if random.random() < get_penetration_probability(D):
                        enemy_data["enemy_health"] -= self.state["damage"]
                        if enemy_data["enemy_health"] <= 0: enemy_data["enemy_alive"] = False
                    enemy_data["suppression"][self.name] = enemy_data["suppression"].get(self.name, 0) + self.state["suppression"]

    def hold_position(self):
        enemy_pos = self.state["enemy"]["enemy_position"]
        if manhattan(self.state["position"], enemy_pos) <= self.state["friendly_attack_range"] and has_line_of_sight(self.state["position"], enemy_pos):
            self.attack_enemy()

    def execute_next_task(self):
        if self.current_plan:
            task = self.current_plan.pop(0)
            if task == "MoveToEnemy":
                self.state["position"] = next_step(self.state["position"], self.get_goal_position())
            elif task == "AttackEnemy": self.attack_enemy()
            elif task == "MoveToOutpost":
                self.state["position"] = next_step(self.state["position"], self.state["enemy"]["outpost_position"])
            elif task == "SecureOutpost":
                if self.state["position"] == self.state["enemy"]["outpost_position"]:
                    self.state["enemy"]["outpost_secured"] = True
                else: self.current_plan.insert(0, "SecureOutpost")
            elif task == "HoldPosition": self.hold_position()

    def get_goal_position(self):
        if isinstance(self, FriendlyTank) and self.planner.domain == tank_plan1_domain:
            return self.state["enemy"]["enemy_position"] if self.state["enemy"]["enemy_alive"] else self.state["position"]
        elif isinstance(self, FriendlyInfantry) and self.planner.domain == infantry_plan1_domain:
            return self.state["enemy"]["outpost_position"]
        else:
            return self.state["enemy"]["enemy_position"] if self.state["enemy"]["enemy_alive"] else self.state["enemy"]["outpost_position"]

class FriendlyTank(FriendlyUnit): pass
class FriendlyInfantry(FriendlyUnit): pass

###############################
# Enemy Attack (unchanged)
###############################

def enemy_attack(target_state, effective_accuracy):
    num_attacks = get_num_attacks(target_state["enemy"]["rate_of_fire"])
    for _ in range(num_attacks):
        if random.random() < effective_accuracy:
            D = target_state["enemy"]["enemy_penetration"] - target_state.get("armor", 0)
            if random.random() < get_penetration_probability(D):
                target_state["friendly_health"] -= target_state["enemy"]["enemy_damage"]
            target_state["suppression_from_enemy"] = target_state.get("suppression_from_enemy", 0) + target_state["enemy"]["enemy_suppression"]

###############################
# Simulation Class (modified)
###############################

class Simulation:
    def __init__(self, friendly_units, team_commander, visualize=False, plan_name="Unknown Plan"):
        self.friendly_units = friendly_units
        self.team_commander = team_commander
        self.step_count = 0
        self.visualize = visualize
        self.plan_name = plan_name
        if self.visualize:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6,6))

    def update_enemy_behavior(self):
        enemy = self.friendly_units[0].state["enemy"]
        if enemy["enemy_alive"]:
            for unit in self.friendly_units:
                dist = manhattan(enemy["enemy_position"], unit.state["position"])
                if dist <= enemy["enemy_attack_range"] and has_line_of_sight(enemy["enemy_position"], unit.state["position"]):
                    if self.visualize: print(f"Enemy within attack range ({dist} tiles) of {unit.name} and has LOS. Holding position to attack.")
                    return
            target_found = False
            for unit in self.friendly_units:
                dist = manhattan(enemy["enemy_position"], unit.state["position"])
                if dist <= enemy["detection_range"] and has_line_of_sight(enemy["enemy_position"], unit.state["position"]):
                    if self.visualize: print(f"Enemy detected {unit.name} at distance {dist}. Chasing...")
                    enemy["enemy_position"] = next_step(enemy["enemy_position"], unit.state["position"])
                    target_found = True
                    break
            if not target_found:
                idx = enemy["current_patrol_index"]
                target = enemy["patrol_points"][idx]
                enemy["enemy_position"] = next_step(enemy["enemy_position"], target)
                if self.visualize: print(f"Enemy patrols towards {target}, now at {enemy['enemy_position']}.")
                if enemy["enemy_position"] == target:
                    enemy["current_patrol_index"] = (idx + 1) % len(enemy["patrol_points"])

    def enemy_counter_attack(self):
        enemy = self.friendly_units[0].state["enemy"]
        if enemy["enemy_alive"]:
            total_suppression = sum(enemy["suppression"].values())
            effective_accuracy = max(0, enemy["enemy_accuracy"] - total_suppression)
            if self.visualize: print(f"Effective enemy accuracy: {effective_accuracy:.2f}")
            for unit in self.friendly_units:
                if self.visualize: print(f"{unit.name} friendly accuracy: {unit.state['friendly_accuracy']:.2f} (suppression from enemy: {unit.state.get('suppression_from_enemy', 0):.2f})")
                dist = manhattan(unit.state["position"], enemy["enemy_position"])
                if dist <= enemy["enemy_attack_range"] and has_line_of_sight(unit.state["position"], enemy["enemy_position"]):
                    unit.state["name"] = unit.name
                    enemy_attack(unit.state, effective_accuracy)
                    break

    def evaluate_plan(self):
        total_friendly_health = sum(unit.state["friendly_health"] for unit in self.friendly_units)
        max_friendly_health = sum(unit.state["max_health"] for unit in self.friendly_units)
        enemy = self.friendly_units[0].state["enemy"]
        enemy_health = enemy["enemy_health"] if enemy["enemy_alive"] else 0
        max_enemy_health = enemy["max_health"]
        outpost_secured = enemy["outpost_secured"]
        steps_taken = self.step_count
        
        score = (total_friendly_health / max_friendly_health * 20) - (enemy_health / max_enemy_health * 20) + \
                (10 if outpost_secured else -10) - 0.1 * steps_taken

        return {
            "score": score,
            "friendly_health": total_friendly_health,
            "enemy_health": enemy_health,
            "outpost_secured": outpost_secured,
            "steps_taken": steps_taken
        }

    def update_plot(self):
        self.ax.clear()
        self.ax.set_xlim(-1, GRID_WIDTH)
        self.ax.set_ylim(-1, GRID_HEIGHT)
        self.ax.set_xticks(range(GRID_WIDTH))
        self.ax.set_yticks(range(GRID_HEIGHT))
        self.ax.grid(True)
        for obs in obstacles:
            self.ax.add_patch(plt.Rectangle(obs, 1, 1, color='black'))
        outpost = self.friendly_units[0].state["enemy"]["outpost_position"]
        self.ax.plot(outpost[0]+0.5, outpost[1]+0.5, marker='*', markersize=15, color='magenta', label='Outpost')
        enemy = self.friendly_units[0].state["enemy"]
        if enemy["enemy_alive"]:
            enemy_pos = enemy["enemy_position"]
            self.ax.plot(enemy_pos[0]+0.5, enemy_pos[1]+0.5, marker='s', markersize=12, color='green', label='Enemy')
            frac = enemy["enemy_health"] / enemy["max_health"]
            bar_width = 0.8 * frac
            self.ax.add_patch(plt.Rectangle((enemy_pos[0]+0.1, enemy_pos[1]+0.8), bar_width, 0.1, color='green'))
            self.ax.add_patch(plt.Rectangle((enemy_pos[0]+0.1, enemy_pos[1]+0.8), 0.8, 0.1, fill=False, edgecolor='black'))
        for unit in self.friendly_units:
            pos = unit.state["position"]
            color = 'red' if unit.state.get("role") == "attacker" else 'blue'
            markersize = 12 if unit.state.get("armor", 0) > 0 else 8
            self.ax.plot(pos[0]+0.5, pos[1]+0.5, marker='o', markersize=markersize, color=color)
            self.ax.text(pos[0]+0.2, pos[1]+0.2, unit.name, fontsize=9, color='black')
            max_hp = unit.state["max_health"]
            hp = unit.state["friendly_health"]
            frac = hp / max_hp
            bar_width = 0.8 * frac
            self.ax.add_patch(plt.Rectangle((pos[0]+0.1, pos[1]+0.8), bar_width, 0.1, color='green'))
            self.ax.add_patch(plt.Rectangle((pos[0]+0.1, pos[1]+0.8), 0.8, 0.1, fill=False, edgecolor='black'))
        self.ax.set_title(f"Simulation Step {self.step_count} - {self.plan_name}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run(self, max_steps=50):
        self.step_count = 0
        for _ in range(max_steps):
            if not self.friendly_units[0].state["enemy"]["enemy_alive"] and self.friendly_units[0].state["enemy"]["outpost_secured"]:
                if self.visualize: 
                    self.update_plot()
                    plt.pause(0.5)
                    print("\nMission accomplished: Enemy tank destroyed and outpost secured!")
                return self.evaluate_plan()
            self.step()
            if self.visualize: 
                self.update_plot()
                plt.pause(0.5)
            for unit in self.friendly_units:
                if unit.state["friendly_health"] <= 0:
                    if self.visualize: 
                        self.update_plot()
                        plt.pause(0.5)
                        print(f"\n{unit.name} has been destroyed! Mission failed.")
                    return self.evaluate_plan()
        if self.visualize: 
            self.update_plot()
            plt.pause(0.5)
            print("\nMission incomplete after maximum steps.")
        return self.evaluate_plan()

    def step(self):
        self.step_count += 1
        if self.visualize: print(f"\n--- Simulation Step {self.step_count} ---")
        self.team_commander.assign_roles()
        self.team_commander.communicate_enemy_position()
        self.update_enemy_behavior()
        for unit in self.friendly_units:
            unit.update_plan()
            unit.execute_next_task()
            goal = unit.get_goal_position()
            if self.visualize:
                if goal:
                    print(f"{unit.name}'s current A* goal: {goal}")
                else:
                    print(f"{unit.name} has no A* goal.")
        self.enemy_counter_attack()
        if self.visualize:
            for unit in self.friendly_units:
                print(f"State for {unit.name}: {unit.state}")
            print(f"Global enemy state: {self.friendly_units[0].state['enemy']}")

###############################
# Main Simulation Setup (modified)
###############################

if __name__ == "__main__":
    tank_state_template = {
        "position": (0, 0), "friendly_health": 20, "max_health": 20, "armor": 17, "friendly_accuracy": 0.75,
        "rate_of_fire": 4.9, "damage": 9, "suppression": 0.12, "penetration": 18, "friendly_attack_range": 3, "role": "attacker"
    }
    infantry_state_template = {
        "position": (0, 1), "friendly_health": 1, "max_health": 1, "armor": 0, "friendly_accuracy": 0.50,
        "rate_of_fire": 294, "damage": 0.8, "suppression": 0.01, "penetration": 1, "friendly_attack_range": 2, "role": "outpost_securer"
    }

    # User input to choose mode
    mode = input("Enter mode (1: Test Plan 1, 2: Test Plan 2, 3: Compare Plans): ")

    if mode in ["1", "2"]:
        # Test a single plan with debug output
        plan_name = "Plan 1" if mode == "1" else "Plan 2"
        domain_tank = tank_plan1_domain if mode == "1" else both_plan2_domain
        domain_infantry = infantry_plan1_domain if mode == "1" else both_plan2_domain
        
        print(f"\nTesting {plan_name} with Visualization and Debug Output...")
        enemy_state = create_enemy_state()
        tank_state = tank_state_template.copy(); tank_state["enemy"] = enemy_state
        infantry_state = infantry_state_template.copy(); infantry_state["enemy"] = enemy_state
        tank = FriendlyTank("FriendlyTank", tank_state, domain_tank)
        infantry = FriendlyInfantry("FriendlyInfantry", infantry_state, domain_infantry)
        sim = Simulation([tank, infantry], TeamCommander([tank, infantry]), visualize=True, plan_name=plan_name)
        result = sim.run()

        print("\n=== Plan Evaluation ===")
        print(f"Score: {result['score']:.1f}")
        print(f"Total Friendly Health Remaining: {result['friendly_health']:.1f}/{sum(unit.state['max_health'] for unit in sim.friendly_units)}")
        print(f"Enemy Health Remaining: {result['enemy_health']:.1f}/{enemy_state['max_health']}")
        print(f"Outpost Secured: {result['outpost_secured']}")
        print(f"Number of Steps Taken: {result['steps_taken']}")

    elif mode == "3":
        # Compare plans with multiple runs
        plan1_scores = []
        plan2_scores = []
        num_runs = 10

        for i in range(num_runs):
            # Plan 1
            enemy_state1 = create_enemy_state()
            tank_state1 = tank_state_template.copy(); tank_state1["enemy"] = enemy_state1
            infantry_state1 = infantry_state_template.copy(); infantry_state1["enemy"] = enemy_state1
            tank1 = FriendlyTank("FriendlyTank", tank_state1, tank_plan1_domain)
            infantry1 = FriendlyInfantry("FriendlyInfantry", infantry_state1, infantry_plan1_domain)
            sim1 = Simulation([tank1, infantry1], TeamCommander([tank1, infantry1]), visualize=False, plan_name="Plan 1")
            result1 = sim1.run()
            plan1_scores.append(result1)
            print(f"Plan 1, Run {i+1}: Score = {result1['score']:.1f}")

            # Plan 2
            enemy_state2 = create_enemy_state()
            tank_state2 = tank_state_template.copy(); tank_state2["enemy"] = enemy_state2
            infantry_state2 = infantry_state_template.copy(); infantry_state2["enemy"] = enemy_state2
            tank2 = FriendlyTank("FriendlyTank", tank_state2, both_plan2_domain)
            infantry2 = FriendlyInfantry("FriendlyInfantry", infantry_state2, both_plan2_domain)
            sim2 = Simulation([tank2, infantry2], TeamCommander([tank2, infantry2]), visualize=False, plan_name="Plan 2")
            result2 = sim2.run()
            plan2_scores.append(result2)
            print(f"Plan 2, Run {i+1}: Score = {result2['score']:.1f}")

        # Evaluate and pick the best plan
        avg_plan1_score = np.mean([run["score"] for run in plan1_scores])
        avg_plan2_score = np.mean([run["score"] for run in plan2_scores])

        print("\n=== Simulation Results ===")
        print(f"Plan 1 Average Score: {avg_plan1_score:.1f}")
        print(f"Plan 2 Average Score: {avg_plan2_score:.1f}")

        best_plan = "Plan 1" if avg_plan1_score > avg_plan2_score else "Plan 2"
        best_domain_tank = tank_plan1_domain if best_plan == "Plan 1" else both_plan2_domain
        best_domain_infantry = infantry_plan1_domain if best_plan == "Plan 1" else both_plan2_domain

        # Run the best plan with visuals and debug output
        print(f"\nRunning Best Plan ({best_plan}) with Visualization...")
        enemy_state_best = create_enemy_state()
        tank_state_best = tank_state_template.copy(); tank_state_best["enemy"] = enemy_state_best
        infantry_state_best = infantry_state_template.copy(); infantry_state_best["enemy"] = enemy_state_best
        tank_best = FriendlyTank("FriendlyTank", tank_state_best, best_domain_tank)
        infantry_best = FriendlyInfantry("FriendlyInfantry", infantry_state_best, best_domain_infantry)
        sim_best = Simulation([tank_best, infantry_best], TeamCommander([tank_best, infantry_best]), visualize=True, plan_name=best_plan)
        result_best = sim_best.run()

        print("\n=== Best Plan Evaluation ===")
        print(f"Score: {result_best['score']:.1f}")
        print(f"Total Friendly Health Remaining: {result_best['friendly_health']:.1f}/{sum(unit.state['max_health'] for unit in sim_best.friendly_units)}")
        print(f"Enemy Health Remaining: {result_best['enemy_health']:.1f}/{enemy_state_best['max_health']}")
        print(f"Outpost Secured: {result_best['outpost_secured']}")
        print(f"Number of Steps Taken: {result_best['steps_taken']}")
        if result_best['friendly_health'] <= 0:
            print("Evaluation: Mission failed - all friendly units destroyed.")
        elif not enemy_state_best["enemy_alive"] and enemy_state_best["outpost_secured"]:
            print("Evaluation: Mission succeeded - enemy destroyed and outpost secured!")
        else:
            print("Evaluation: Mission incomplete - check enemy status and outpost.")

    else:
        print("Invalid mode selected. Please enter 1, 2, or 3.")

    plt.ioff()
    plt.show()
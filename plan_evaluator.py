# plan_evaluator.py (updated)
import copy
import random
import math
from state import State
from htn_planner import apply_task, find_path

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = copy.deepcopy(state)
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0.0

    def is_fully_expanded(self):
        possible_actions = []
        for agent_name in ["infantry", "tanks"]:
            agent = self.state.data["agents"][agent_name]
            if agent.health > 0 and self.state.data["enemy"].health > 0:
                possible_actions.append(("attack", agent_name))
        return len(self.children) == len(possible_actions)

    def best_child(self, exploration_weight=1.0):
        if not self.children:
            return None
        best_child = None
        best_uct = float('-inf')
        for child in self.children:
            if child.visits == 0:
                uct = float('inf')
            else:
                exploitation = child.total_reward / child.visits
                exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                uct = exploitation + exploration
            if uct > best_uct:
                best_uct = uct
                best_child = child
        return best_child

    def expand(self):
        possible_actions = []
        for agent_name in ["infantry", "tanks"]:
            agent = self.state.data["agents"][agent_name]
            if agent.health > 0 and self.state.data["enemy"].health > 0:
                possible_actions.append(("attack", agent_name))

        tried_actions = [child.action for child in self.children]
        untried_actions = [action for action in possible_actions if action not in tried_actions]

        if not untried_actions:
            return None

        action = random.choice(untried_actions)
        new_state = copy.deepcopy(self.state)
        success = apply_task(new_state, action)
        if not success:
            return None

        child = MCTSNode(new_state, parent=self, action=action)
        self.children.append(child)
        return child

def simulate(state):
    temp_state = copy.deepcopy(state)
    initial_health = {agent_name: agent.health for agent_name, agent in temp_state.data["agents"].items()}
    turns = 0

    while temp_state.data["enemy"].health > 0:
        possible_actions = []
        for agent_name in ["infantry", "tanks"]:
            agent = temp_state.data["agents"][agent_name]
            if agent.health > 0:
                dist_to_enemy = abs(agent.x - temp_state.data["enemy"].x) + abs(agent.y - temp_state.data["enemy"].y)
                if dist_to_enemy <= agent.attack_range and temp_state.has_line_of_sight(agent.x, agent.y, temp_state.data["enemy"].x, temp_state.data["enemy"].y):
                    possible_actions.append(("attack", agent_name))

        if not possible_actions:
            return float('inf')

        # Heuristic: Prioritize tanks for attacks, use infantry only if safe
        infantry = temp_state.data["agents"]["infantry"]
        tanks = temp_state.data["agents"]["tanks"]
        enemy = temp_state.data["enemy"]
        dist_infantry_to_enemy = abs(infantry.x - enemy.x) + abs(infantry.y - enemy.y)
        enemy_can_attack_infantry = dist_infantry_to_enemy <= enemy.attack_range

        if (tanks.health > 30 and ("attack", "tanks") in possible_actions):
            action = ("attack", "tanks")
        elif (temp_state.data["enemy"].suppression > 2.0 and infantry.health > 30 and not enemy.can_attack_infantry and ("attack", "infantry") in possible_actions):
            action = ("attack", "infantry")
        else:
            return float('inf')

        success = apply_task(temp_state, action)
        turns += 1
        if not success:
            return float('inf')
        if turns > 100:
            return float('inf')

    fuel_used = sum(state.data["agents"][agent_name].fuel - temp_state.data["agents"][agent_name].fuel for agent_name in temp_state.data["agents"])
    health_lost = sum(max(0, initial_health[agent_name] - temp_state.data["agents"][agent_name].health) for agent_name in temp_state.data["agents"])
    cost = 1.0 * fuel_used + 50 * health_lost + 10 * turns
    return cost

def mcts_search(initial_state, iterations=1000):
    root = MCTSNode(initial_state)

    for _ in range(iterations):
        node = root
        while node is not None and node.is_fully_expanded() and node.children:
            node = node.best_child()
            if node is None:
                break

        if node is None:
            continue

        if node.state.data["enemy"].health <= 0:
            cost = simulate(node.state)
            current = node
            while current is not None:
                current.visits += 1
                current.total_reward += -cost
                current = current.parent
            continue

        child = node.expand()
        if child is None:
            cost = float('inf')
            current = node
            while current is not None:
                current.visits += 1
                current.total_reward += -cost
                current = current.parent
            continue
        else:
            node = child

        cost = simulate(node.state)
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += -cost
            current = current.parent

    attack_sequence = []
    current = root
    temp_state = copy.deepcopy(initial_state)
    while temp_state.data["enemy"].health > 0 and current.children:
        best_child = max(current.children, key=lambda c: c.visits if c.visits > 0 else 0, default=None)
        if best_child is None:
            return attack_sequence
        attack_sequence.append(best_child.action)
        apply_task(temp_state, best_child.action)
        current = best_child

    return attack_sequence

def generate_plans(initial_state):
    plans = []
    state = copy.deepcopy(initial_state)
    plan1 = []

    enemy_x, enemy_y = state.get_enemy_position()

    # Move tanks to a position at distance 3 from the enemy
    tanks = state.data["agents"]["tanks"]
    target_positions = [
        (enemy_x - 3, enemy_y), (enemy_x + 3, enemy_y),  # Horizontal positions
        (enemy_x, enemy_y - 3), (enemy_x, enemy_y + 3),  # Vertical positions
        (enemy_x - 2, enemy_y - 1), (enemy_x - 2, enemy_y + 1),  # Diagonal positions
        (enemy_x + 2, enemy_y - 1), (enemy_x + 2, enemy_y + 1),
        (enemy_x - 1, enemy_y - 2), (enemy_x + 1, enemy_y - 2),
        (enemy_x - 1, enemy_y + 2), (enemy_x + 1, enemy_y + 2)
    ]
    valid_target = None
    for target_x, target_y in target_positions:
        if (0 <= target_x < state.terrain.width and 0 <= target_y < state.terrain.height and
                not state.terrain.is_obstacle(target_x, target_y)):
            dist_to_enemy = abs(target_x - enemy_x) + abs(target_y - enemy_y)
            if dist_to_enemy == tanks.attack_range and state.has_line_of_sight(target_x, target_y, enemy_x, enemy_y):
                valid_target = (target_x, target_y)
                print(f"Selected target for tanks: ({target_x}, {target_y}), distance to enemy: {dist_to_enemy}")
                break

    if valid_target:
        target_x, target_y = valid_target
        path = find_path(state, "tanks", target_x, target_y)
        for step_x, step_y in path:
            dist_to_enemy = abs(step_x - enemy_x) + abs(step_y - enemy_y)
            if dist_to_enemy <= tanks.attack_range and state.has_line_of_sight(step_x, step_y, enemy_x, enemy_y):
                plan1.append(("move", "tanks", step_x, step_y))
                apply_task(state, ("move", "tanks", step_x, step_y))
                break
            plan1.append(("move", "tanks", step_x, step_y))
            apply_task(state, ("move", "tanks", step_x, step_y))
    else:
        print("No valid target position found for tanks at distance 3. Falling back to closest position within range.")
        dist_to_enemy = abs(tanks.x - enemy_x) + abs(tanks.y - enemy_y)
        if dist_to_enemy > tanks.attack_range:
            path = find_path(state, "tanks", enemy_x, enemy_y)
            for step_x, step_y in path:
                dist_to_enemy = abs(step_x - enemy_x) + abs(step_y - enemy_y)
                if dist_to_enemy <= tanks.attack_range and state.has_line_of_sight(step_x, step_y, enemy_x, enemy_y):
                    plan1.append(("move", "tanks", step_x, step_y))
                    apply_task(state, ("move", "tanks", step_x, step_y))
                    break
                plan1.append(("move", "tanks", step_x, step_y))
                apply_task(state, ("move", "tanks", step_x, step_y))

    # Move infantry to a safer position (distance 4 from enemy to avoid being attacked)
    infantry = state.data["agents"]["infantry"]
    enemy = state.data["enemy"]
    safe_distance = enemy.attack_range + 1  # Stay just outside enemy attack range
    target_positions = [
        (enemy_x - 4, enemy_y), (enemy_x + 4, enemy_y),
        (enemy_x, enemy_y - 4), (enemy_x, enemy_y + 4),
        (enemy_x - 3, enemy_y - 1), (enemy_x - 3, enemy_y + 1),
        (enemy_x + 3, enemy_y - 1), (enemy_x + 3, enemy_y + 1),
        (enemy_x - 1, enemy_y - 3), (enemy_x + 1, enemy_y - 3),
        (enemy_x - 1, enemy_y + 3), (enemy_x + 1, enemy_y + 3)
    ]
    valid_target = None
    for target_x, target_y in target_positions:
        if (0 <= target_x < state.terrain.width and 0 <= target_y < state.terrain.height and
                not state.terrain.is_obstacle(target_x, target_y)):
            dist_to_enemy = abs(target_x - enemy_x) + abs(target_y - enemy_y)
            if dist_to_enemy >= safe_distance:
                valid_target = (target_x, target_y)
                print(f"Selected target for infantry: ({target_x}, {target_y}), distance to enemy: {dist_to_enemy}")
                break

    if valid_target:
        target_x, target_y = valid_target
        path = find_path(state, "infantry", target_x, target_y)
        for step_x, step_y in path:
            dist_to_enemy = abs(step_x - enemy_x) + abs(step_y - enemy_y)
            if dist_to_enemy < safe_distance:
                break  # Stop moving if we get too close to the enemy
            plan1.append(("move", "infantry", step_x, step_y))
            apply_task(state, ("move", "infantry", step_x, step_y))
    else:
        print("No valid target position found for infantry at distance 4. Keeping infantry at current position.")

    # Estimate the number of attack actions needed to defeat the enemy
    enemy_health = state.data["enemy"].health
    tanks_accuracy = state.data["agents"]["tanks"].accuracy / 100.0
    tanks_damage = state.data["agents"]["tanks"].damage
    max_shots_per_attack = 10
    expected_hits_per_attack = max_shots_per_attack * tanks_accuracy
    expected_damage_per_attack = expected_hits_per_attack * tanks_damage
    estimated_attacks_needed = int(math.ceil(enemy_health / expected_damage_per_attack)) + 2

    print(f"Estimated attacks needed to defeat enemy: {estimated_attacks_needed} (enemy health: {enemy_health}, expected damage per attack: {expected_damage_per_attack})")

    # Keep attacking until the enemy is defeated
    attack_count = 0
    while state.data["enemy"].health > 0 and attack_count < estimated_attacks_needed:
        attack_sequence = mcts_search(state, iterations=1000)
        if not attack_sequence:
            print("MCTS failed to find a valid attack sequence")
            return plans

        for action in attack_sequence:
            plan1.append(action)
            success = apply_task(state, action)
            if not success:
                print(f"Plan failed: {action[1]} was destroyed during attack")
                return plans
            attack_count += 1
            if state.data["enemy"].health <= 0:
                break

    while state.data["enemy"].health > 0:
        plan1.append(("attack", "tanks"))
        success = apply_task(state, ("attack", "tanks"))
        if not success:
            print("Plan failed: tanks were destroyed during additional attack")
            return plans
        attack_count += 1

    print(f"Total attack actions in plan: {attack_count}")

    # Check if both agents are alive before securing the outpost
    for agent_name in ["infantry", "tanks"]:
        agent = state.data["agents"][agent_name]
        if agent.health <= 0:
            print(f"Plan failed: {agent_name} was destroyed, cannot secure outpost")
            return plans

    # Move infantry to the outpost only after the enemy is defeated
    infantry = state.data["agents"]["infantry"]
    if infantry.x != enemy_x or infantry.y != enemy_y:
        path = find_path(state, "infantry", enemy_x, enemy_y)
        for step_x, step_y in path:
            plan1.append(("move", "infantry", step_x, step_y))
            apply_task(state, ("move", "infantry", step_x, step_y))

    # Move tanks to the outpost
    tanks = state.data["agents"]["tanks"]
    if tanks.x != enemy_x or tanks.y != enemy_y:
        path = find_path(state, "tanks", enemy_x, enemy_y)
        for step_x, step_y in path:
            plan1.append(("move", "tanks", step_x, step_y))
            apply_task(state, ("move", "tanks", step_x, step_y))

    # Secure the outpost
    plan1.append(("secure_outpost",))
    apply_task(state, ("secure_outpost",))

    plans.append(("Plan 1: MCTS-optimized coordinated attack", plan1))
    return plans


def simulate_plan(state):
    temp_state = copy.deepcopy(state)
    initial_health = {agent_name: agent.health for agent_name, agent in temp_state.data["agents"].items()}
    turns = 0

    while temp_state.data["enemy"].health > 0:
        possible_actions = []
        for agent_name in ["infantry", "tanks"]:
            agent = temp_state.data["agents"][agent_name]
            if agent.health > 0:
                dist_to_enemy = abs(agent.x - temp_state.data["enemy"].x) + abs(agent.y - temp_state.data["enemy"].y)
                if dist_to_enemy <= agent.attack_range and temp_state.has_line_of_sight(agent.x, agent.y, temp_state.data["enemy"].x, temp_state.data["enemy"].y):
                    possible_actions.append(("attack", agent_name))

        if not possible_actions:
            return float('inf')

        # Heuristic: Use infantry to build suppression early, then switch to tanks
        infantry = temp_state.data["agents"]["infantry"]
        tanks = temp_state.data["agents"]["tanks"]
        enemy = temp_state.data["enemy"]
        dist_infantry_to_enemy = abs(infantry.x - enemy.x) + abs(infantry.y - enemy.y)
        enemy_can_attack_infantry = dist_infantry_to_enemy <= enemy.attack_range

        if (temp_state.data["enemy"].suppression < 1.0 and infantry.health > 30 and not enemy.can_attack_infantry and ("attack", "infantry") in possible_actions):
            action = ("attack", "infantry")
        elif tanks.health > 30 and ("attack", "tanks") in possible_actions:
            action = ("attack", "tanks")
        else:
            return float('inf')

        success = apply_task(temp_state, action)
        turns += 1
        if not success:
            return float('inf')
        if turns > 100:
            return float('inf')

    fuel_used = sum(state.data["agents"][agent_name].fuel - temp_state.data["agents"][agent_name].fuel for agent_name in temp_state.data["agents"])
    health_lost = sum(max(0, initial_health[agent_name] - temp_state.data["agents"][agent_name].health) for agent_name in temp_state.data["agents"])
    cost = 1.0 * fuel_used + 50 * health_lost + 10 * turns
    return cost


def cost_function(result):
    if not result["success"]:
        return float('inf')
    return 1.0 * result["fuel_used"] + 50 * result["health_lost"] + 10 * result["turns"]

def evaluate_plans(initial_state):
    plans = generate_plans(initial_state)
    results = []

    if not plans:
        print("Error: No plans could be generated.")
        return None

    for plan_name, plan in plans:
        print(f"\nSimulating {plan_name}...")
        result = simulate_plan(initial_state, plan)
        cost = cost_function(result)
        results.append((plan_name, plan, result, cost))
        print(f"Result: Success={result['success']}, Fuel Used={result['fuel_used']}, Health Lost={result['health_lost']}, Turns={result['turns']}, Cost={cost}")

    best_plan = min(results, key=lambda x: x[3])
    print(f"\nBest Plan: {best_plan[0]} with cost {best_plan[3]}")
    return best_plan[1]
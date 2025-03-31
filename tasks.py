# tasks.py (updated)
import random

def move(state, agent_name, x, y):
    agent = state.data["agents"][agent_name]
    terrain = state.terrain
    if terrain.is_obstacle(x, y):
        return False
    fuel_cost = abs(agent.x - x) + abs(agent.y - y)
    if agent.fuel < fuel_cost * 10:
        return False
    agent.x = x
    agent.y = y
    agent.fuel -= fuel_cost * 10
    print(f"Success: {agent_name} moved to ({x}, {y}), fuel left: {agent.fuel}")
    return True

def attack(state, agent_name):
    agent = state.data["agents"][agent_name]
    enemy = state.data["enemy"]
    if enemy.health <= 0:  # Check if enemy is already defeated
        return True  # No need to attack further
    if agent.health <= 0:
        return False
    dist = abs(agent.x - enemy.x) + abs(agent.y - enemy.y)
    if dist > agent.attack_range:
        return False
    if not state.has_line_of_sight(agent.x, agent.y, enemy.x, enemy.y):
        return False
    shots = random.randint(1, 10)
    hits = sum(1 for _ in range(shots) if random.random() < agent.accuracy / 100)
    damage = hits * agent.damage
    enemy.health -= damage
    enemy.suppression += hits * agent.suppression_per_hit
    enemy_accuracy = max(0, enemy.accuracy - (enemy.suppression * 10))
    print(f"Success: {agent_name} fired {shots} shots at enemy from ({agent.x}, {agent.y}) with {agent.weapon}, {hits} hits, enemy health: {enemy.health}, enemy suppression: {enemy.suppression:.2f}, fuel: {agent.fuel}")
    if enemy.health <= 0:
        print(f"Enemy defeated by {agent_name}!")
        return True
    enemy_hits = sum(1 for _ in range(shots) if random.random() < enemy_accuracy / 100)
    agent.health -= enemy_hits * enemy.damage
    if enemy_hits > 0:
        print(f"Enemy hit {agent_name} from ({enemy.x}, {enemy.y}) with {enemy.weapon}, {enemy_hits} hits, {agent_name} health: {agent.health}")
    else:
        print(f"Enemy missed {agent_name} from ({enemy.x}, {enemy.y}) with {enemy.weapon} (accuracy: {enemy_accuracy:.2f}%)")
    if agent.health <= 0:
        print(f"{agent_name} destroyed!")
        return False
    return True

def secure_outpost(state):
    enemy = state.data["enemy"]
    if enemy.health > 0:
        return False
    for agent_name, agent in state.data["agents"].items():
        if agent.health <= 0:
            return False
        if agent.x != enemy.x or agent.y != enemy.y:
            return False
    state.data["mission_complete"] = True
    print("Success: Outpost secured")
    return True
# tasks.py
import random
import math
from terrain import terrain_map, line_of_sight

def move(state, agent_name, x, y):
    agent = state.data["agents"].get(agent_name, state.data["enemy"] if agent_name == "enemy" else None)
    if not agent:
        return False
    fuel_cost = 20 if agent_name == "tanks" else 10
    if (abs(agent.x - x) + abs(agent.y - y) == 1 and terrain_map[x][y] == 0 and agent.fuel >= fuel_cost):
        agent.x = x
        agent.y = y
        agent.fuel -= fuel_cost
        agent.moved_last_turn = True
        print(f"Success: {agent_name} moved to ({x}, {y}), fuel left: {agent.fuel}")
        return True
    print(f"Failed: {agent_name} cannot move to ({x}, {y}) (fuel: {agent.fuel}, terrain: {terrain_map[x][y]})")
    return False

def calculate_accuracy(weapon, distance, optimal_range, suppression, moved_last_turn):
    base_accuracy = float(weapon.get("accuracy", "0%").replace("%", "")) / 100
    weapon_range_meters = float(weapon.get("range", "0 meters").replace(" meters", ""))
    weapon_range_units = weapon_range_meters / 500
    optimal_range_units = optimal_range * weapon_range_units
    delta_range = abs(distance - optimal_range_units) / 2
    movement_penalty = 0.1 if moved_last_turn else 0.0
    final_accuracy = base_accuracy * math.exp(-delta_range) * (1 - suppression) * (1 - movement_penalty)
    return max(0, min(1, final_accuracy))

def calculate_penetration_probability(ap, armor):
    if isinstance(armor, str) and "/" in armor:
        armor = float(armor.split("/")[0].strip())
    else:
        armor = float(armor)
    ap = float(ap)
    d = ap - armor
    if d <= -3:
        return 0.0
    elif -3 < d <= 0:
        return 0.33 + (0.33 / 3) * (d + 3)
    else:
        return 0.95

def calculate_rounds_fired(weapon):
    rate_of_fire = float(weapon.get("rate_of_fire", "0 rpm").replace(" rpm", ""))
    shots_per_turn = int(rate_of_fire * (6 / 60))
    return max(1, shots_per_turn)

def scout_area(state, agent_name):
    agent = state.data["agents"][agent_name]
    enemy = state.data["enemy"]
    distance = abs(agent.x - enemy.x) + abs(agent.y - enemy.y)
    # Use vision range for scouting instead of attack range
    vision_range_meters = float(agent.vision_range.replace(" m", "")) if agent.vision_range else 0
    vision_range_units = vision_range_meters / 500  # Convert to grid units
    if distance <= vision_range_units:
        if line_of_sight(agent.x, agent.y, enemy.x, enemy.y):
            agent.detects_enemy = True
            print(f"Success: {agent_name} scouted enemy from ({agent.x}, {agent.y}), enemy detected at ({enemy.x}, {enemy.y})")
            return True
        else:
            print(f"Failed: {agent_name} cannot scout enemy from ({agent.x}, {agent.y}) due to obstacles.")
            return False
    print(f"Failed: {agent_name} is too far to scout enemy from ({agent.x}, {agent.y}) (distance: {distance}, required: <= {vision_range_units})")
    return False

def attack(state, agent_name):
    agent = state.data["agents"][agent_name]
    enemy = state.data["enemy"]
    fuel_cost = 10 if agent_name == "tanks" else 0
    distance = abs(agent.x - enemy.x) + abs(agent.y - enemy.y)
    if distance <= agent.attack_range and (agent_name != "tanks" or agent.fuel >= fuel_cost):
        if line_of_sight(agent.x, agent.y, enemy.x, enemy.y):
            weapon = agent.main_weapon
            if agent_name == "infantry" and "secondary_weapon" in agent.__dict__ and agent.secondary_weapon:
                weapon = agent.secondary_weapon  # Use M72 LAW for infantry
            rounds_fired = calculate_rounds_fired(weapon)
            accuracy = calculate_accuracy(weapon, distance, optimal_range=0.5, suppression=agent.suppression, moved_last_turn=agent.moved_last_turn)
            hits = 0
            total_damage = 0
            suppression_per_hit = weapon.get("suppression", 0) / 100
            penetration_prob = calculate_penetration_probability(weapon.get("penetration", 0), enemy.front_armor)
            base_damage = weapon.get("damage", 0)
            
            for _ in range(rounds_fired):
                if random.random() < accuracy:
                    hits += 1
                    damage = base_damage
                    if random.random() > penetration_prob:
                        damage *= 0.1
                    total_damage += damage
                    enemy.suppression = min(1, enemy.suppression + suppression_per_hit)
            
            if hits == 0:
                print(f"Failed: {agent_name} missed all {rounds_fired} shots at enemy from ({agent.x}, {agent.y}) with {weapon.get('name', 'unknown weapon')} (accuracy: {accuracy:.2%})")
                return False
            
            enemy.health -= total_damage
            enemy.health = max(0, enemy.health)
            if agent_name == "tanks":
                agent.fuel -= fuel_cost
            print(f"Success: {agent_name} fired {rounds_fired} shots at enemy from ({agent.x}, {agent.y}) with {weapon.get('name', 'unknown weapon')}, {hits} hits, enemy health: {enemy.health}, enemy suppression: {enemy.suppression:.2f}, fuel: {agent.fuel}")
            return True
        else:
            print(f"Failed: {agent_name} cannot attack enemy from ({agent.x}, {agent.y}) because line-of-sight is blocked.")
            return False
    print(f"Failed: {agent_name} cannot attack enemy from ({agent.x}, {agent.y}) (distance: {distance}, required: <= {agent.attack_range})")
    return False

def enemy_attack(state, target_name):
    enemy = state.data["enemy"]
    target = state.data["agents"][target_name]
    distance = abs(enemy.x - target.x) + abs(enemy.y - target.y)
    if distance <= enemy.attack_range:
        if line_of_sight(enemy.x, enemy.y, target.x, target.y):
            weapon = enemy.main_weapon
            rounds_fired = calculate_rounds_fired(weapon)
            accuracy = calculate_accuracy(weapon, distance, optimal_range=0.5, suppression=enemy.suppression, moved_last_turn=enemy.moved_last_turn)
            hits = 0
            total_damage = 0
            suppression_per_hit = weapon.get("suppression", 0) / 100
            target_armor = getattr(target, 'front_armor', 0)
            penetration_prob = calculate_penetration_probability(weapon.get("penetration", 0), target_armor)
            base_damage = weapon.get("damage", 0)
            
            for _ in range(rounds_fired):
                if random.random() < accuracy:
                    hits += 1
                    damage = base_damage
                    if random.random() > penetration_prob:
                        damage *= 0.1
                    total_damage += damage
                    target.suppression = min(1, target.suppression + suppression_per_hit)
            
            if hits == 0:
                print(f"Failed: Enemy missed all {rounds_fired} shots at {target_name} from ({enemy.x}, {enemy.y}) with {weapon.get('name', 'unknown weapon')} (accuracy: {accuracy:.2%})")
                return False
            
            target.health -= total_damage
            target.health = max(0, target.health)
            print(f"Enemy fired {rounds_fired} shots at {target_name} from ({enemy.x}, {enemy.y}) with {weapon.get('name', 'unknown weapon')}, {hits} hits, {target_name} health: {target.health}, {target_name} suppression: {target.suppression:.2f}")
            return True
        else:
            print(f"Enemy cannot attack {target_name} from ({enemy.x}, {enemy.y}) because line-of-sight is blocked.")
            return False
    print(f"Enemy cannot attack {target_name} from ({enemy.x}, {enemy.y}) (distance: {distance}, required: <= {enemy.attack_range})")
    return False

def secure_outpost(state):
    enemy = state.data["enemy"]
    enemy_x, enemy_y = enemy.x, enemy.y
    all_here = all(agent.x == enemy_x and agent.y == enemy_y for agent in state.data["agents"].values() if agent.health > 0)
    if enemy.health <= 0 and all_here:
        state.data["mission_complete"] = True
        print("Success: Outpost secured")
        return True
    if enemy.health <= 0:
        print("Enemy defeated but not all units have converged to the outpost.")
        return False
    print(f"Failed: Enemy health still {enemy.health}")
    return False

def update_detection(state):
    enemy = state.data["enemy"]
    enemy_x, enemy_y = enemy.x, enemy.y

    for agent_name, agent in state.data["agents"].items():
        dist = abs(agent.x - enemy_x) + abs(agent.y - enemy_y)
        # Use vision range for detection instead of v_max-based threshold
        vision_range_meters = float(agent.vision_range.replace(" m", "")) if agent.vision_range else 0
        vision_range_units = vision_range_meters / 500
        if dist <= vision_range_units and line_of_sight(agent.x, agent.y, enemy_x, enemy_y):
            agent.detects_enemy = True
        else:
            agent.detects_enemy = False

        enemy_vision_range_meters = float(enemy.vision_range.replace(" m", "")) if enemy.vision_range else 0
        enemy_vision_range_units = enemy_vision_range_meters / 500
        if dist <= enemy_vision_range_units and line_of_sight(enemy_x, enemy_y, agent.x, agent.y):
            agent.detected_by_enemy = True
            if dist <= enemy.attack_range and enemy.health > 0 and agent.health > 0:
                enemy_attack(state, agent_name)
        else:
            agent.detected_by_enemy = False

        agent.suppression = max(0, agent.suppression - 0.1)
        agent.moved_last_turn = False

    enemy.suppression = max(0, enemy.suppression - 0.1)
    enemy.moved_last_turn = False
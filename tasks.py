# tasks.py
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
        print(f"Success: {agent_name} moved to ({x}, {y}), fuel left: {agent.fuel}")
        return True
    print(f"Failed: {agent_name} cannot move to ({x}, {y}) (fuel: {agent.fuel}, terrain: {terrain_map[x][y]})")
    return False

def scout_area(state, agent_name):
    agent = state.data["agents"][agent_name]
    enemy = state.data["enemy"]
    distance = abs(agent.x - enemy.x) + abs(agent.y - enemy.y)
    if distance <= agent.attack_range:
        if line_of_sight(agent.x, agent.y, enemy.x, enemy.y):
            enemy.health = max(0, enemy.health - 20)
            print(f"Success: {agent_name} scouted enemy from ({agent.x}, {agent.y}), enemy health: {enemy.health}")
            return True
        else:
            print(f"Failed: {agent_name} cannot scout enemy from ({agent.x}, {agent.y}) due to obstacles.")
            return False
    print(f"Failed: {agent_name} is too far to scout enemy from ({agent.x}, {agent.y}) (distance: {distance}, required: <= {agent.attack_range})")
    return False

def attack(state, agent_name):
    agent = state.data["agents"][agent_name]
    enemy = state.data["enemy"]
    fuel_cost = 10 if agent_name == "tanks" else 0
    distance = abs(agent.x - enemy.x) + abs(agent.y - enemy.y)
    if distance <= agent.attack_range and (agent_name != "tanks" or agent.fuel >= fuel_cost):
        if line_of_sight(agent.x, agent.y, enemy.x, enemy.y):
            enemy.health -= agent.attack_strength
            enemy.health = max(0, enemy.health)
            if agent_name == "tanks":
                agent.fuel -= fuel_cost
            print(f"Success: {agent_name} attacked enemy from ({agent.x}, {agent.y}), enemy health: {enemy.health}, fuel: {agent.fuel}")
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
            target.health -= enemy.attack_strength
            target.health = max(0, target.health)
            print(f"Enemy attacked {target_name} from ({enemy.x}, {enemy.y}), {target_name} health: {target.health}")
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
    enemy_x, enemy_y = enemy.x, enemy.y  # Fixed the typo here

    for agent_name, agent in state.data["agents"].items():
        dist = abs(agent.x - enemy_x) + abs(agent.y - enemy_y)
        detection_threshold = agent.v_max * agent.s_eff * agent.c_cover * agent.n_noise * agent.e_elev
        if dist <= detection_threshold and line_of_sight(agent.x, agent.y, enemy_x, enemy_y):
            agent.detects_enemy = True
        else:
            agent.detects_enemy = False

        enemy_threshold = enemy.v_max * enemy.s_eff * enemy.c_cover * enemy.n_noise * enemy.e_elev
        if dist <= enemy_threshold and line_of_sight(enemy_x, enemy_y, agent.x, agent.y):
            agent.detected_by_enemy = True
            if dist <= enemy.attack_range and enemy.health > 0 and agent.health > 0:
                enemy_attack(state, agent_name)
        else:
            agent.detected_by_enemy = False
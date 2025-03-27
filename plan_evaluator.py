# plan_evaluator.py
import copy
from state import State
from htn_planner import apply_task, find_path
from terrain import line_of_sight

def generate_plans(state):
    enemy = state.data["enemy"]
    enemy_x, enemy_y = enemy.x, enemy.y
    plans = []

    # Plan 1: Scouts scout, then all units attack together
    plan1 = []
    scouts = state.data["agents"]["scouts"]
    if not scouts.detects_enemy:
        vision_range_units = float(scouts.vision_range.replace(" m", "")) / 500
        best_pos = None
        best_path = []
        for x in range(10):
            for y in range(10):
                dist_to_enemy = abs(x - enemy_x) + abs(y - enemy_y)
                if dist_to_enemy <= vision_range_units:
                    path = find_path(state, "scouts", x, y)
                    if path and line_of_sight(x, y, enemy_x, enemy_y):
                        if not best_path or len(path) < len(best_path):
                            best_pos = (x, y)
                            best_path = path
        if best_path:
            for step_x, step_y in best_path:
                plan1.append(("move", "scouts", step_x, step_y))
            plan1.append(("scout_area", "scouts"))

    for agent_name in ["scouts", "infantry", "tanks"]:
        agent = state.data["agents"][agent_name]
        dist_to_enemy = abs(agent.x - enemy_x) + abs(agent.y - enemy_y)
        if dist_to_enemy > agent.attack_range:
            path = find_path(state, agent_name, enemy_x, enemy_y)
            for step_x, step_y in path:
                if abs(step_x - enemy_x) + abs(step_y - enemy_y) <= agent.attack_range:
                    plan1.append(("move", agent_name, step_x, step_y))
                    break
                plan1.append(("move", agent_name, step_x, step_y))
        plan1.append(("attack", agent_name))

    for agent_name in ["scouts", "infantry", "tanks"]:
        agent = state.data["agents"][agent_name]
        if agent.x != enemy_x or agent.y != enemy_y:
            path = find_path(state, agent_name, enemy_x, enemy_y)
            for step_x, step_y in path:
                plan1.append(("move", agent_name, step_x, step_y))
    plan1.append(("secure_outpost",))
    plans.append(("Plan 1: Scouts scout, all attack together", plan1))

    # Plan 2: Scouts scout, infantry attacks first, then others join
    plan2 = []
    if not scouts.detects_enemy:
        vision_range_units = float(scouts.vision_range.replace(" m", "")) / 500
        best_pos = None
        best_path = []
        for x in range(10):
            for y in range(10):
                dist_to_enemy = abs(x - enemy_x) + abs(y - enemy_y)
                if dist_to_enemy <= vision_range_units:
                    path = find_path(state, "scouts", x, y)
                    if path and line_of_sight(x, y, enemy_x, enemy_y):
                        if not best_path or len(path) < len(best_path):
                            best_pos = (x, y)
                            best_path = path
        if best_path:
            for step_x, step_y in best_path:
                plan2.append(("move", "scouts", step_x, step_y))
            plan2.append(("scout_area", "scouts"))

    infantry = state.data["agents"]["infantry"]
    dist_to_enemy = abs(infantry.x - enemy_x) + abs(infantry.y - enemy_y)
    if dist_to_enemy > infantry.attack_range:
        path = find_path(state, "infantry", enemy_x, enemy_y)
        for step_x, step_y in path:
            if abs(step_x - enemy_x) + abs(step_y - enemy_y) <= infantry.attack_range:
                plan2.append(("move", "infantry", step_x, step_y))
                break
            plan2.append(("move", "infantry", step_x, step_y))
    plan2.append(("attack", "infantry"))

    for agent_name in ["scouts", "tanks"]:
        agent = state.data["agents"][agent_name]
        dist_to_enemy = abs(agent.x - enemy_x) + abs(agent.y - enemy_y)
        if dist_to_enemy > agent.attack_range:
            path = find_path(state, agent_name, enemy_x, enemy_y)
            for step_x, step_y in path:
                if abs(step_x - enemy_x) + abs(step_y - enemy_y) <= agent.attack_range:
                    plan2.append(("move", agent_name, step_x, step_y))
                    break
                plan2.append(("move", agent_name, step_x, step_y))
        plan2.append(("attack", agent_name))

    for agent_name in ["scouts", "infantry", "tanks"]:
        agent = state.data["agents"][agent_name]
        if agent.x != enemy_x or agent.y != enemy_y:
            path = find_path(state, agent_name, enemy_x, enemy_y)
            for step_x, step_y in path:
                plan2.append(("move", agent_name, step_x, step_y))
    plan2.append(("secure_outpost",))
    plans.append(("Plan 2: Scouts scout, infantry attacks first", plan2))

    # Plan 3: Scouts scout, tanks attack first, then others join
    plan3 = []
    if not scouts.detects_enemy:
        vision_range_units = float(scouts.vision_range.replace(" m", "")) / 500
        best_pos = None
        best_path = []
        for x in range(10):
            for y in range(10):
                dist_to_enemy = abs(x - enemy_x) + abs(y - enemy_y)
                if dist_to_enemy <= vision_range_units:
                    path = find_path(state, "scouts", x, y)
                    if path and line_of_sight(x, y, enemy_x, enemy_y):
                        if not best_path or len(path) < len(best_path):
                            best_pos = (x, y)
                            best_path = path
        if best_path:
            for step_x, step_y in best_path:
                plan3.append(("move", "scouts", step_x, step_y))
            plan3.append(("scout_area", "scouts"))

    tanks = state.data["agents"]["tanks"]
    dist_to_enemy = abs(tanks.x - enemy_x) + abs(tanks.y - enemy_y)
    if dist_to_enemy > tanks.attack_range:
        path = find_path(state, "tanks", enemy_x, enemy_y)
        for step_x, step_y in path:
            if abs(step_x - enemy_x) + abs(step_y - enemy_y) <= tanks.attack_range:
                plan3.append(("move", "tanks", step_x, step_y))
                break
            plan3.append(("move", "tanks", step_x, step_y))
    plan3.append(("attack", "tanks"))

    for agent_name in ["scouts", "infantry"]:
        agent = state.data["agents"][agent_name]
        dist_to_enemy = abs(agent.x - enemy_x) + abs(agent.y - enemy_y)
        if dist_to_enemy > agent.attack_range:
            path = find_path(state, agent_name, enemy_x, enemy_y)
            for step_x, step_y in path:
                if abs(step_x - enemy_x) + abs(step_y - enemy_y) <= agent.attack_range:
                    plan3.append(("move", agent_name, step_x, step_y))
                    break
                plan3.append(("move", agent_name, step_x, step_y))
        plan3.append(("attack", agent_name))

    for agent_name in ["scouts", "infantry", "tanks"]:
        agent = state.data["agents"][agent_name]
        if agent.x != enemy_x or agent.y != enemy_y:
            path = find_path(state, agent_name, enemy_x, enemy_y)
            for step_x, step_y in path:
                plan3.append(("move", agent_name, step_x, step_y))
    plan3.append(("secure_outpost",))
    plans.append(("Plan 3: Scouts scout, tanks attack first", plan3))

    # Plan 4: All units move to enemy position and attack without scouting
    plan4 = []
    for agent_name in ["scouts", "infantry", "tanks"]:
        agent = state.data["agents"][agent_name]
        dist_to_enemy = abs(agent.x - enemy_x) + abs(agent.y - enemy_y)
        if dist_to_enemy > agent.attack_range:
            path = find_path(state, agent_name, enemy_x, enemy_y)
            for step_x, step_y in path:
                if abs(step_x - enemy_x) + abs(step_y - enemy_y) <= agent.attack_range:
                    plan4.append(("move", agent_name, step_x, step_y))
                    break
                plan4.append(("move", agent_name, step_x, step_y))
        plan4.append(("attack", agent_name))

    for agent_name in ["scouts", "infantry", "tanks"]:
        agent = state.data["agents"][agent_name]
        if agent.x != enemy_x or agent.y != enemy_y:
            path = find_path(state, agent_name, enemy_x, enemy_y)
            for step_x, step_y in path:
                plan4.append(("move", agent_name, step_x, step_y))
    plan4.append(("secure_outpost",))
    plans.append(("Plan 4: All move to enemy and attack without scouting", plan4))

    return plans

def simulate_plan(initial_state, plan):
    state = copy.deepcopy(initial_state)
    initial_fuel = {agent_name: agent.fuel for agent_name, agent in state.data["agents"].items()}
    initial_health = {agent_name: agent.health for agent_name, agent in state.data["agents"].items()}
    turns = 0

    for task in plan:
        success = apply_task(state, task)
        turns += 1
        if not success:
            return {
                "success": False,
                "fuel_used": sum(initial_fuel[agent_name] - state.data["agents"][agent_name].fuel for agent_name in state.data["agents"]),
                "health_lost": sum(initial_health[agent_name] - state.data["agents"][agent_name].health for agent_name in state.data["agents"]),
                "turns": turns
            }
        if state.data["mission_complete"]:
            return {
                "success": True,
                "fuel_used": sum(initial_fuel[agent_name] - state.data["agents"][agent_name].fuel for agent_name in state.data["agents"]),
                "health_lost": sum(initial_health[agent_name] - state.data["agents"][agent_name].health for agent_name in state.data["agents"]),
                "turns": turns
            }
    return {
        "success": False,
        "fuel_used": sum(initial_fuel[agent_name] - state.data["agents"][agent_name].fuel for agent_name in state.data["agents"]),
        "health_lost": sum(initial_health[agent_name] - state.data["agents"][agent_name].health for agent_name in state.data["agents"]),
        "turns": turns
    }

def cost_function(result):
    if not result["success"]:
        return float('inf')
    return 0.1 * result["fuel_used"] + 10 * result["health_lost"] + 50 * result["turns"]

def evaluate_plans(initial_state):
    plans = generate_plans(initial_state)
    results = []

    for plan_name, plan in plans:
        print(f"\nSimulating {plan_name}...")
        result = simulate_plan(initial_state, plan)
        cost = cost_function(result)
        results.append((plan_name, plan, result, cost))
        print(f"Result: Success={result['success']}, Fuel Used={result['fuel_used']}, Health Lost={result['health_lost']}, Turns={result['turns']}, Cost={cost}")

    best_plan = min(results, key=lambda x: x[3])
    print(f"\nBest Plan: {best_plan[0]} with cost {best_plan[3]}")
    return best_plan[1]
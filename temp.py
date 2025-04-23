import copy
from collections import deque

# Simplified domain with explicit progression
secure_outpost_domain = {
    "SecureOutpostMission": [
        (lambda state: not state.get("enemy_spotted", False),
         ["ScoutSearch"]),
        (lambda state: state.get("enemy_spotted", False) and state.get("enemy_alive", False),
         ["DefeatEnemy"]),
        (lambda state: state.get("enemy_spotted", False) and not state.get("enemy_alive", False) and
         state["distance_to_outpost"] > 0,
         ["Move"]),
        (lambda state: state.get("enemy_spotted", False) and not state.get("enemy_alive", False) and
         state["distance_to_outpost"] == 0 and not state.get("outpost_secured", False),
         ["SecureOutpost"]),
        (lambda state: state.get("outpost_secured", False),
         ["Hold"])
    ],
    "DefeatEnemy": [
        (lambda state: state.get("enemy_alive", False) and state["distance_to_enemy"] == 0,
         ["AttackEnemy"]),
        (lambda state: state.get("enemy_alive", False) and state["distance_to_enemy"] > 0,
         ["MoveToEnemy"])
    ],
    "ScoutSearch": [
        (lambda state: not state.get("enemy_spotted", False),
         ["SpotEnemy"])
    ]
}

# Simulate a single task and update state
def simulate_task(task, state):
    new_state = copy.deepcopy(state)
    print(f"Simulating task: {task}, Current state: {new_state}")
    if task == "Move":
        if new_state["distance_to_outpost"] > 0:
            new_state["distance_to_outpost"] -= 1
            new_state["health"] = max(0, new_state["health"] - 5)
        new_state["steps"] += 1
    elif task == "MoveToEnemy":
        if new_state["distance_to_enemy"] > 0:
            new_state["distance_to_enemy"] -= 1
            new_state["health"] = max(0, new_state["health"] - 5)
        new_state["steps"] += 1
    elif task == "AttackEnemy":
        if new_state["ammo"] > 0 and new_state["enemy_alive"]:
            new_state["ammo"] -= 1
            new_state["enemy_alive"] = False
        new_state["steps"] += 1
    elif task == "SpotEnemy":
        new_state["enemy_spotted"] = True
        new_state["steps"] += 1
    elif task == "SecureOutpost":
        if new_state["distance_to_outpost"] == 0 and not new_state["enemy_alive"]:
            new_state["outpost_secured"] = True
        new_state["steps"] += 1
    elif task == "Hold":
        new_state["steps"] += 1
    print(f"After {task}: {new_state}")
    return new_state

# Score a final state
def score_state(state):
    score = 0
    if state["outpost_secured"]:
        score += 200
    if not state["enemy_alive"]:
        score += 100
    if state["enemy_spotted"]:
        score += 10
    score -= state["steps"] * 1
    score -= (100 - state["health"]) * 2
    score -= (5 - state["ammo"]) * 10
    if state["health"] == 0:
        score -= 100
    if not state["outpost_secured"]:
        score -= 100
    return score

# Generate all possible plans
def generate_plans(initial_state):
    plans = []
    queue = deque([([], initial_state, "SecureOutpostMission")])
    iteration_count = 0
    max_iterations = 10000
    
    while queue and iteration_count < max_iterations:
        iteration_count += 1
        plan, state, task = queue.popleft()
        print(f"Iteration {iteration_count}: Exploring plan: {plan}, Task: {task}, State: {state}")
        print(f"Queue size: {len(queue)}")
        
        if state["health"] == 0 or state.get("outpost_secured", False):
            if plan and state.get("outpost_secured", False):
                plans.append(plan)
                print(f"Saved plan: {plan}")
            continue
        
        for condition, subtasks in secure_outpost_domain.get(task, []):
            if condition(state):
                print(f"Condition matched for task {task}, adding subtasks: {subtasks}")
                new_plan = plan + subtasks
                new_state = copy.deepcopy(state)
                
                for subtask in subtasks:
                    new_state = simulate_task(subtask, new_state)
                    if new_state["health"] == 0 or new_state.get("outpost_secured", False):
                        break
                
                queue.append((new_plan, new_state, "SecureOutpostMission"))
                if task in secure_outpost_domain:
                    for subtask in subtasks:
                        if subtask in secure_outpost_domain:
                            queue.append((new_plan, new_state, subtask))
                
                if new_state.get("outpost_secured", False) or not new_state.get("enemy_alive", True):
                    plans.append(new_plan)
                    print(f"Saved plan: {new_plan}")
        
        if len(plans) > 100:
            break
    
    print(f"Generated {len(plans)} plans")
    return plans

# Simulate a plan and return final state
def simulate_plan(plan, initial_state):
    state = copy.deepcopy(initial_state)
    for task in plan:
        state = simulate_task(task, state)
        if state["health"] == 0 or state.get("outpost_secured", False):
            break
    return state

# Main function
def main():
    initial_state = {
        "type": "scout",
        "health": 100,
        "ammo": 5,
        "distance_to_outpost": 10,
        "distance_to_enemy": 5,
        "enemy_spotted": False,
        "enemy_alive": True,
        "outpost_secured": False,
        "steps": 0
    }
    
    plans = generate_plans(initial_state)
    if not plans:
        print("No plans generated. Check domain conditions or initial state.")
        return
    
    results = []
    seen_outcomes = set()
    
    for plan in plans:
        final_state = simulate_plan(plan, initial_state)
        score = score_state(final_state)
        outcome_sig = (final_state["health"], final_state["ammo"], 
                       final_state["outpost_secured"], final_state["enemy_alive"], 
                       final_state["steps"])
        if outcome_sig not in seen_outcomes:
            seen_outcomes.add(outcome_sig)
            results.append((plan, final_state, score))
    
    if not results:
        print("No valid plans after simulation. Check simulation logic.")
        return
    
    results.sort(key=lambda x: x[2], reverse=True)
    print("Top 3 Plans:")
    for i, (plan, final_state, score) in enumerate(results[:3], 1):
        print(f"\nPlan {i}: Score = {score}")
        print(f"  Tasks: {plan}")
        print(f"  Final State: health={final_state['health']}, ammo={final_state['ammo']}, "
              f"outpost_secured={final_state['outpost_secured']}, enemy_alive={final_state['enemy_alive']}, "
              f"steps={final_state['steps']}")

if __name__ == "__main__":
    main()
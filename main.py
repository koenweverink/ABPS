# main.py
from copy import deepcopy
from state import State
from planner import htn_planner
from visualization import visualize_plan
from terrain import terrain_map

if __name__ == "__main__":
    initial_state = State()
    state_for_planning = deepcopy(initial_state)
    print("Initial state:")
    for agent_name, agent in state_for_planning.data["agents"].items():
        print(f"{agent_name} at ({agent.x}, {agent.y}), fuel: {agent.fuel}")
    print(f"Enemy at ({state_for_planning.data['enemy'].x}, {state_for_planning.data['enemy'].y}), strength: {state_for_planning.data['enemy'].attack_strength}")
    print("Terrain map:")
    for row in terrain_map:
        print(row)
    plan = htn_planner(state_for_planning, [("secure_outpost_task",)])
    if plan:
        print("\nPlan found:", plan)
        visualize_plan(initial_state, plan)
    else:
        print("\nNo plan found.")
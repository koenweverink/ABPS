# main.py
from state import State
from htn_planner import apply_task
from plan_evaluator import evaluate_plans

def print_state(state):
    print("Initial state:")
    for agent_name, agent in state.data["agents"].items():
        print(f"{agent_name} at ({agent.x}, {agent.y}), fuel: {agent.fuel}, health: {agent.health}")
    print(f"Enemy at ({state.data['enemy'].x}, {state.data['enemy'].y}), damage: {state.data['enemy'].main_weapon.get('damage', 0)}, health: {state.data['enemy'].health}")
    print("Terrain map:")
    for row in state.terrain_map:
        print(row)

def main():
    state = State()
    state.terrain_map = [
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0]
    ]
    print_state(state)

    # Evaluate multiple plans and select the best one
    best_plan = evaluate_plans(state)

    # Execute the best plan
    print("\nExecuting best plan...")
    for task in best_plan:
        success = apply_task(state, task)
        if not success:
            print(f"Task {task} failed, execution stopped.")
            break
        if state.data["mission_complete"]:
            print("Mission completed successfully!")
            break

if __name__ == "__main__":
    main()
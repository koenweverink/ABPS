# main.py (fixed)
from state import State, initialize_state, Terrain
from plan_evaluator import evaluate_plans
from htn_planner import apply_task

def main():
    # Initialize the state using the provided initialize_state() function
    state = initialize_state()  # This creates a State object with terrain, agents, and enemy
    
    print("Initial state:")
    print(f"infantry at ({state.data['agents']['infantry'].x}, {state.data['agents']['infantry'].y}), fuel: {state.data['agents']['infantry'].fuel}, health: {state.data['agents']['infantry'].health}")
    print(f"tanks at ({state.data['agents']['tanks'].x}, {state.data['agents']['tanks'].y}), fuel: {state.data['agents']['tanks'].fuel}, health: {state.data['agents']['tanks'].health}")
    print(f"Enemy at ({state.data['enemy'].x}, {state.data['enemy'].y}), damage: {state.data['enemy'].damage}, health: {state.data['enemy'].health}")
    print("Terrain map:")
    for y in range(state.terrain.height):
        row = []
        for x in range(state.terrain.width):
            row.append(1 if state.terrain.is_obstacle(x, y) else 0)
        print(row)

    best_plan = evaluate_plans(state)
    if best_plan is None:
        print("No valid plan found.")
        return

    print("\nExecuting best plan...")
    for task in best_plan:
        success = apply_task(state, task)
        if not success:
            print(f"Mission failed: Task {task} failed.")
            break
    if state.data["mission_complete"]:
        print("Mission completed successfully!")

if __name__ == "__main__":
    main()
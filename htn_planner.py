class HTNPlanner:
    """
    Hierarchical Task Network (HTN) planner that generates a plan by recursively
    evaluating conditions and decomposing tasks based on a defined domain.
    """

    def __init__(self, domain):
        """
        Initialize the planner with a task domain.

        Args:
            domain (dict): Dictionary mapping task names to lists of (condition, subtasks) pairs.
        """
        self.domain = domain

    def plan(self, task, state):
        """
        Generate a plan from the given task and current world state.

        Args:
            task (str or tuple): Task to decompose, optionally with an argument.
            state (dict): Current world state to evaluate conditions.

        Returns:
            list: A list of primitive tasks if a valid plan is found, otherwise None.
        """
        task_name = task[0] if isinstance(task, tuple) else task
        task_arg = task[1] if isinstance(task, tuple) else None

        # If task is not decomposable, treat it as a primitive
        if task_name not in self.domain:
            return [(task_name, task_arg)] if task_arg is not None else [task_name]

        # Decompose based on matching conditions
        for condition, subtasks in self.domain[task_name]:
            if condition(state):
                task_list = subtasks(state) if callable(subtasks) else subtasks
                plan = []
                for subtask in task_list:
                    sub_plan = self.plan(subtask, state)
                    if sub_plan is None:
                        return None
                    plan.extend(sub_plan)
                return plan

        return None

class HTNPlanner:
    def __init__(self, domain):
        self.domain = domain

    def plan(self, task, state):
        # Handle task as either a string or a tuple
        task_name = task[0] if isinstance(task, tuple) else task
        task_arg = task[1] if isinstance(task, tuple) else None
        
        # If task_name is not in domain, return it as a primitive task
        if task_name not in self.domain:
            return [(task_name, task_arg)] if task_arg is not None else [task_name]
        
        # Evaluate conditions in the domain
        for condition, subtasks in self.domain[task_name]:
            if condition(state):
                task_list = subtasks(state) if callable(subtasks) else subtasks
                plan = []
                for subtask in task_list:
                    # Ensure subtask is processed correctly
                    sub_plan = self.plan(subtask, state)  # Pass the original state
                    if sub_plan is None:
                        return None
                    plan.extend(sub_plan)
                return plan
        return None

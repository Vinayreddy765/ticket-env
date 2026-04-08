def grade(trajectory):
    """
    Easy Task:
    Assign tickets to correct agents based on matching category.
    """

    correct = sum(1 for step in trajectory if step.get("reward", 0) > 0)
    total = len(trajectory)

    return correct / total if total > 0 else 0.0
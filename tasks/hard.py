def grade(trajectory):
    """
    Hard Task:
    Complete all assignments correctly; penalize mistakes.
    """

    if not trajectory:
        return 0.0

    perfect = all(step.get("reward", 0) > 0 for step in trajectory)
    return 1.0 if perfect else 0.5
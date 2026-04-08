def grade(trajectory):
    """
    Medium Task:
    Handle multiple ticket assignments while maintaining accuracy.
    """

    score = 0
    for step in trajectory:
        if step.get("reward", 0) > 0:
            score += 1

    return min(score / 3, 1.0)
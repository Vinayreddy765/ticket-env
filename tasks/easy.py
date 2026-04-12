def grade(trajectory):
    if not trajectory:
        return 0.0

    total_reward = sum(step.get("reward", 0) for step in trajectory)
    max_possible = sum(max(step.get("reward", 0), 0) for step in trajectory)

    return round(min(total_reward / max_possible, 0.99), 3) if max_possible > 0 else 0.0



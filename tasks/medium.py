def medium(trajectory):
    if not trajectory:
        return 0.0

    MAX_TICKETS = 3
    MAX_REWARD_PER_TICKET = 3.0
    COMPLETION_BONUS = 1.0
    MAX_POSSIBLE = (MAX_TICKETS * MAX_REWARD_PER_TICKET) + COMPLETION_BONUS  # 10.0

    total_reward = sum(step.get("reward", 0) for step in trajectory)

    return round(max(0.0, min(total_reward / MAX_POSSIBLE, 0.99)), 3)

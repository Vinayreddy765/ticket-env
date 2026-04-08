import asyncio
import os
from openai import OpenAI

from openenv.core.env_client import EnvClient

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "baseline")
API_KEY = os.getenv("HF_TOKEN", "dummy")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}")


def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error}")


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}")


async def main():
    env = EnvClient(base_url=API_BASE_URL)

    rewards = []
    steps = 0

    log_start("easy", "ticket_env", MODEL_NAME)

    await env.reset()

    actions = [
        {"ticket_id": 1, "agent_id": 1},
        {"ticket_id": 2, "agent_id": 2},
    ]

    for i, action in enumerate(actions, 1):
        result = await env.step(action)

        reward = result["reward"]
        done = result["done"]

        rewards.append(reward)
        steps = i

        log_step(i, "assign", reward, done, "null")

        if done:
            break

    score = sum(rewards) / len(rewards)
    success = score > 0.5

    log_end(success, steps, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())
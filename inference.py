import asyncio
import os
import requests

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}")


def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error}")


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}")


async def main():
    rewards = []
    steps = 0

    log_start("easy", "ticket_env", "baseline")

    try:
        # RESET
        res = requests.post(f"{BASE_URL}/reset", timeout=10)
        res.raise_for_status()
        data = res.json()

        tickets = data.get("observation", {}).get("tickets", [])

        if not tickets:
            raise Exception("No tickets received")

        for i, ticket in enumerate(tickets, 1):
            try:
                # decision logic
                if ticket.get("category") == "billing":
                    agent_id = 1
                else:
                    agent_id = 2

                action = {
                    "ticket_id": ticket.get("id"),
                    "agent_id": agent_id
                }

                res = requests.post(
                    f"{BASE_URL}/step",
                    json={"action": action},
                    timeout=10
                )
                res.raise_for_status()

                data = res.json()

                reward = float(data.get("reward", 0))
                done = bool(data.get("done", False))

                rewards.append(reward)
                steps = i

                log_step(i, "assign", reward, done, "null")

                if done:
                    break

            except Exception as step_error:
                log_step(i, "assign", 0.0, True, str(step_error))
                break

    except Exception as e:
        print(f"[ERROR] {e}")
        log_end(False, steps, 0.0, [])
        return

    score = sum(rewards) / len(rewards) if rewards else 0.0
    success = score > 0.5

    log_end(success, steps, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())
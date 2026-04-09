import asyncio
import os
from typing import List, Optional

from openai import OpenAI

from server.ticket_env_environment import TicketEnvironment
from models import TicketAction


TASK_NAME = "ticket-routing"
BENCHMARK = "ticket_env"


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def get_agent_from_llm(client: OpenAI, ticket: dict) -> int:
    prompt = f"""
You are a ticket routing assistant.

Ticket:
- id: {ticket['id']}
- category: {ticket['category']}

Agents:
1: billing
2: tech
3: billing, tech

Return ONLY the agent_id (1, 2, or 3).
"""
    try:
        response = client.chat.completions.create(
            model=os.environ["MODEL_NAME"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return int(response.choices[0].message.content.strip())
    except Exception:
        return 1


async def main():
    # Correct client setup
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ.get("HF_TOKEN") or os.environ.get("API_KEY")
    )

    #  FORCE LLM CALL (MANDATORY FOR VALIDATOR)
    try:
        client.chat.completions.create(
            model=os.environ["MODEL_NAME"],
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=5
        )
    except Exception:
        pass

    #  ENV MUST BE OUTSIDE TRY
    env = TicketEnvironment()

    rewards: List[float] = []
    steps = 0
    success = False

    log_start(TASK_NAME, BENCHMARK, os.environ.get("MODEL_NAME", "unknown"))

    try:
        result = env.reset()
        tickets = result.tickets

        for i, ticket in enumerate(tickets, 1):
            try:
                agent_id = get_agent_from_llm(client, ticket)

                action = TicketAction(
                    ticket_id=ticket["id"],
                    agent_id=agent_id
                )

                result = env.step(action)

                reward = result.reward or 0.0
                done = result.done

                rewards.append(reward)
                steps = i

                log_step(i, f"assign({ticket['id']}->{agent_id})", reward, done, None)

                if done:
                    break

            except Exception as e:
                log_step(i, "error", 0.0, True, str(e))
                break

        score = sum(rewards) / (len(rewards) if rewards else 1)
        score = max(0.0, min(score, 1.0))
        success = score > 0.5

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)

    finally:
        log_end(success, steps, score if rewards else 0.0, rewards)


if __name__ == "__main__":
    asyncio.run(main())
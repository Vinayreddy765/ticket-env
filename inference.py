import asyncio
import os
import requests
from typing import List, Optional

# ── ENV VARIABLES (STRICT) 
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("API_KEY", "test-key")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# ── YOUR HF SPACE (PUBLIC URL) 
ENV_BASE_URL = os.environ.get(
    "ENV_BASE_URL",
    "https://vinay3111-ticket-env.hf.space"
)

TASK_NAME = "ticket-routing"
BENCHMARK = "ticket_env"


# ── LOGGING  
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ── RAW LLM CALL (CRITICAL — PROXY DETECTION) 
def llm_call(ticket: dict) -> int:
    url = f"{API_BASE_URL}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = f"""You are a ticket routing assistant.

Ticket:
- id: {ticket['id']}
- category: {ticket['category']}

Agents:
1: billing
2: tech
3: billing and tech

Return ONLY a single digit: 1, 2, or 3.
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 5,
    }

    res = requests.post(url, headers=headers, json=payload, timeout=20)
    res.raise_for_status()

    data = res.json()
    text = data["choices"][0]["message"]["content"].strip()

    agent_id = int(text[0])
    if agent_id not in [1, 2, 3]:
        raise ValueError(f"Invalid agent_id from LLM: {agent_id}")

    return agent_id


# ── MAIN 
async def main():
    rewards: List[float] = []
    steps = 0
    success = False

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        #  FORCE ONE LLM CALL (MANDATORY)
        _ = llm_call({"id": 0, "category": "billing"})

        # ── RESET ENV (HF SPACE) 
        res = requests.post(f"{ENV_BASE_URL}/reset", timeout=30)
        res.raise_for_status()
        data = res.json()

        tickets = data.get("observation", {}).get("tickets", [])

        for i, ticket in enumerate(tickets, 1):
            # LLM DECISION (NO FALLBACK)
            agent_id = llm_call(ticket)

            action = {
                "ticket_id": ticket["id"],
                "agent_id": agent_id
            }

            # ── STEP ENV 
            res = requests.post(
                f"{ENV_BASE_URL}/step",
                json={"action": action},
                timeout=30
            )
            res.raise_for_status()

            data = res.json()

            reward = float(data.get("reward", 0))
            done = bool(data.get("done", False))

            rewards.append(reward)
            steps = i

            log_step(i, f"assign({ticket['id']}->{agent_id})", reward, done, None)

            if done:
                break

        score = sum(rewards) / (len(rewards) if rewards else 1)
        success = score > 0.5

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)

    finally:
        log_end(success, steps, score if rewards else 0.0, rewards)


if __name__ == "__main__":
    asyncio.run(main())
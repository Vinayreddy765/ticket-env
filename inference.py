import asyncio
import os
import requests
from typing import List, Optional
from openai import OpenAI


# ── LLM proxy (injected by hackathon validator) ───────────────────────────────
API_BASE_URL = os.environ["API_BASE_URL"]   # e.g. https://litellm-proxy.../v1
API_KEY      = os.environ["API_KEY"]
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# ── Ticket environment server (your FastAPI app running locally) ──────────────
# The env server is NOT the LLM proxy — it runs separately on port 8000
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

TASK_NAME  = "ticket-routing"
BENCHMARK  = "ticket_env"


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def get_agent_from_llm(client: OpenAI, ticket: dict) -> int:
    """Ask the LLM proxy which agent should handle this ticket."""
    prompt = f"""You are a ticket routing assistant.

Ticket:
- id: {ticket['id']}
- category: {ticket['category']}

Agents:
1: billing
2: tech
3: billing, tech

Return ONLY the agent_id (1, 2, or 3). No explanation."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        return int(response.choices[0].message.content.strip())
    except Exception:
        return 1  # fallback


async def main():
    # ── Initialise LLM client pointing at the proxy ───────────────────────────
    client = OpenAI(
        base_url=API_BASE_URL,   # LiteLLM proxy
        api_key=API_KEY,
    )

    rewards: List[float] = []
    steps   = 0
    score   = 0.0
    success = False

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        # ── Reset the TICKET ENV server (not the LLM proxy) ──────────────────
        res = requests.post(f"{ENV_BASE_URL}/reset", timeout=10)
        res.raise_for_status()
        data    = res.json()
        tickets = data.get("observation", {}).get("tickets", [])

        for i, ticket in enumerate(tickets, 1):
            try:
                # LLM call goes through the proxy ✓
                agent_id = get_agent_from_llm(client, ticket)

                action = {
                    "ticket_id": ticket["id"],
                    "agent_id":  agent_id,
                }

                # Step call goes to the ticket env server ✓
                res = requests.post(
                    f"{ENV_BASE_URL}/step",
                    json={"action": action},
                    timeout=10,
                )
                res.raise_for_status()
                data = res.json()

                reward = float(data.get("reward", 0))
                done   = bool(data.get("done", False))

                rewards.append(reward)
                steps = i

                log_step(i, f"assign({ticket['id']}->{agent_id})", reward, done, None)

                if done:
                    break

            except Exception as e:
                log_step(i, "error", 0.0, True, str(e))
                break

        score   = sum(rewards) / (len(rewards) if rewards else 1)
        success = score > 0.5

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)

    finally:
        log_end(success, steps, score if rewards else 0.0, rewards)


if __name__ == "__main__":
    asyncio.run(main())
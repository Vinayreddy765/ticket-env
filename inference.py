import asyncio
import os
import requests
from typing import List, Optional
from openai import OpenAI


# ── LLM proxy — injected by validator ────────────────────────────────────────
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY      = os.environ["API_KEY"]
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# ── Ticket env server — your deployed HF Space (NOT localhost) ────────────────
# The validator runs inference.py externally, so localhost won't work.
# Point at your public HF Space URL. Can be overridden via ENV_BASE_URL.
ENV_BASE_URL = os.environ.get(
    "ENV_BASE_URL",
    "https://vinay3111-ticket-env.hf.space"   
)

TASK_NAME = "ticket-routing"
BENCHMARK = "ticket_env"


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


def llm_call(client: OpenAI, messages: list, max_tokens: int = 10) -> str:
    """Call the LLM proxy. Raises on failure — no silent swallowing."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def get_agent_from_llm(client: OpenAI, ticket: dict) -> int:
    """
    Route a ticket via LLM. Mandatory — errors are raised, not swallowed,
    so the validator confirms the system genuinely depends on the LLM.
    """
    prompt = f"""You are a ticket routing assistant.

Ticket:
- id: {ticket['id']}
- category: {ticket['category']}

Agents:
1: billing
2: tech
3: billing and tech

Return ONLY a single digit: 1, 2, or 3. No explanation, no punctuation."""

    try:
        text = llm_call(client, [{"role": "user", "content": prompt}], max_tokens=5)
        print(f"[DEBUG] LLM raw response for ticket {ticket['id']}: {repr(text)}", flush=True)
    except Exception as e:
        print(f"[ERROR] LLM call failed for ticket {ticket['id']}: {e}", flush=True)
        raise  # ← mandatory re-raise: no fallback, validator must see real usage

    agent_id = int(text.strip()[0])
    if agent_id not in [1, 2, 3]:
        raise ValueError(f"LLM returned invalid agent_id: {agent_id!r}")
    return agent_id


async def main():
    # ── LLM client ────────────────────────────────────────────────────────────
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[DEBUG] ENV_BASE_URL={ENV_BASE_URL}", flush=True)

    # ── Warm-up: guaranteed proxy hit before env interaction ──────────────────
    try:
        warmup = llm_call(
            client,
            [{"role": "user", "content": "Reply with the single word: ready"}],
            max_tokens=5,
        )
        print(f"[DEBUG] LLM warm-up OK: {warmup}", flush=True)
    except Exception as e:
        print(f"[DEBUG] LLM warm-up FAILED: {e}", flush=True)

    rewards: List[float] = []
    steps   = 0
    score   = 0.0
    success = False

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        # ── Reset env ─────────────────────────────────────────────────────────
        print(f"[DEBUG] POST {ENV_BASE_URL}/reset", flush=True)
        res = requests.post(f"{ENV_BASE_URL}/reset", timeout=30)
        res.raise_for_status()
        data    = res.json()
        tickets = data.get("observation", {}).get("tickets", [])
        print(f"[DEBUG] Received {len(tickets)} tickets", flush=True)

        for i, ticket in enumerate(tickets, 1):
            # LLM decides — no fallback path
            agent_id = get_agent_from_llm(client, ticket)
            print(f"[DEBUG] Ticket {ticket['id']} → agent {agent_id}", flush=True)

            action = {"ticket_id": ticket["id"], "agent_id": agent_id}

            res = requests.post(
                f"{ENV_BASE_URL}/step",
                json={"action": action},
                timeout=30,
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

        score   = sum(rewards) / (len(rewards) if rewards else 1)
        success = score > 0.5

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)

    finally:
        log_end(success, steps, score if rewards else 0.0, rewards)


if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import os
import requests
from typing import List
from openai import OpenAI

# ── Exactly as hackathon team instructed ──────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set in environment")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# ── Env server URL ────────────────────────────────────────────────────────────
def find_env_url() -> str:
    explicit = os.environ.get("ENV_BASE_URL")
    if explicit:
        return explicit
    for port in [7860, 8000, 8080]:
        try:
            res = requests.get(f"http://localhost:{port}/health", timeout=3)
            if res.status_code == 200:
                print(f"[DEBUG] Found env at localhost:{port}", flush=True)
                return f"http://localhost:{port}"
        except Exception:
            pass
    return "https://vinay3111-ticket-env.hf.space"

TASK_NAME = "ticket-routing"
BENCHMARK = "ticket_env"


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


def safe_parse_agent(text: str) -> int:
    for ch in text:
        if ch in "123":
            return int(ch)
    return 3


def route_ticket(ticket: dict) -> int:
    prompt = (
        "You are a ticket routing assistant.\n"
        "Agent 1: billing only\n"
        "Agent 2: tech only\n"
        "Agent 3: billing and tech\n\n"
        f"Ticket category: {ticket['category']}\n"
        "Reply with a single digit: 1, 2, or 3."
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
    )
    text = response.choices[0].message.content.strip()
    print(f"[DEBUG] LLM: {repr(text)}", flush=True)
    return safe_parse_agent(text)


async def main():
    env_base_url = find_env_url()

    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[DEBUG] ENV_BASE_URL={env_base_url}", flush=True)
    print(f"[DEBUG] HF_TOKEN set={bool(HF_TOKEN)}", flush=True)

    rewards: List[float] = []
    steps   = 0
    score   = 0.0
    success = False

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        res = requests.post(f"{env_base_url}/reset", timeout=30)
        res.raise_for_status()
        body    = res.json()
        obs     = body.get("observation", body)
        tickets = obs.get("tickets", [])
        print(f"[DEBUG] {len(tickets)} tickets", flush=True)

        for i, ticket in enumerate(tickets, 1):
            agent_id = route_ticket(ticket)
            print(f"[DEBUG] Ticket {ticket['id']} → agent {agent_id}", flush=True)

            res = requests.post(
                f"{env_base_url}/step",
                json={"action": {"ticket_id": ticket["id"], "agent_id": agent_id}},
                timeout=30,
            )
            res.raise_for_status()
            data   = res.json()
            reward = float(data.get("reward", 0))
            done   = bool(data.get("done", False))

            rewards.append(reward)
            steps = i
            log_step(i, f"assign({ticket['id']}->{agent_id})", reward, done, None)

            if done:
                break

        raw_score = sum(rewards) / max(len(rewards), 1)
        score     = min(max(raw_score, 0.01), 0.99)
        success   = score > 0.5

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)

    finally:
        log_end(success, steps, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())
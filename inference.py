import asyncio
import os
import httpx
from typing import List


# ── LLM proxy — injected by validator
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY      = os.environ["API_KEY"]
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASK_NAME = "ticket-routing"
BENCHMARK = "ticket_env"
MAX_RETRIES = 3


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


async def find_env_url(client: httpx.AsyncClient) -> str:
    explicit = os.environ.get("ENV_BASE_URL")
    if explicit:
        return explicit

    for port in [7860, 8000, 8080]:
        try:
            res = await client.get(f"http://localhost:{port}/health", timeout=3)
            if res.status_code == 200:
                print(f"[DEBUG] Found env server at localhost:{port}", flush=True)
                return f"http://localhost:{port}"
        except Exception:
            pass

    print("[DEBUG] Falling back to HF Space URL", flush=True)
    return "https://vinay3111-ticket-env.hf.space"


def safe_parse_agent(text: str, available_agent_ids: list) -> int:
    for ch in text:
        if ch.isdigit() and int(ch) in available_agent_ids:
            return int(ch)
    fallback = available_agent_ids[-1]
    print(f"[WARN] Could not parse agent from {repr(text)}, falling back to agent {fallback}", flush=True)
    return fallback


# ✅ Raw HTTP call — no SDK
async def route_ticket(http: httpx.AsyncClient, ticket: dict, agents: list) -> int:
    url = API_BASE_URL + "/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    agent_descriptions = "\n".join(
        f"Agent {a['id']}: {', '.join(a['skills'])}" for a in agents
    )

    prompt = (
        "You are a ticket routing assistant.\n"
        f"{agent_descriptions}\n\n"
        f"Ticket category: {ticket['category']}, priority: {ticket['priority']}\n"
        "Reply with a single agent id."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 5,
        "temperature": 0,
    }

    res = await http.post(url, headers=headers, json=payload, timeout=20)
    res.raise_for_status()

    data = res.json()
    text = data["choices"][0]["message"]["content"].strip()

    print(f"[DEBUG] RAW LLM response: {text}", flush=True)

    return safe_parse_agent(text, [a["id"] for a in agents])


async def post_with_retry(client: httpx.AsyncClient, url: str, json: dict) -> dict:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = await client.post(url, json=json, timeout=30)
            res.raise_for_status()
            return res.json()
        except Exception as e:
            last_error = e
            print(f"[WARN] Attempt {attempt}/{MAX_RETRIES} failed: {e}", flush=True)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError(f"All {MAX_RETRIES} attempts failed: {last_error}")


async def main():
    rewards: List[float] = []
    steps   = 0
    score   = 0.0
    success = False

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    async with httpx.AsyncClient() as http:
        env_base_url = await find_env_url(http)

        print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
        print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
        print(f"[DEBUG] ENV_BASE_URL={env_base_url}", flush=True)

        try:
            # Reset env
            res = await http.post(f"{env_base_url}/reset", timeout=30)
            res.raise_for_status()
            body    = res.json()
            obs     = body.get("observation", body)
            tickets = obs.get("tickets", [])
            agents  = obs.get("agents", [])
            print(f"[DEBUG] {len(tickets)} tickets, {len(agents)} agents", flush=True)

            for i, ticket in enumerate(tickets, 1):
                # ✅ Pass http directly — no SDK
                agent_id = await route_ticket(http, ticket, agents)
                print(f"[DEBUG] Ticket {ticket['id']} (priority={ticket['priority']}) → agent {agent_id}", flush=True)

                data   = await post_with_retry(
                    http,
                    f"{env_base_url}/step",
                    {"action": {"ticket_id": ticket["id"], "agent_id": agent_id}},
                )
                reward = float(data.get("reward", 0))
                done   = bool(data.get("done", False))

                agents = data.get("observation", {}).get("agents", agents)

                rewards.append(reward)
                steps = i
                log_step(i, f"assign({ticket['id']}->{agent_id})", reward, done, None)

                if done:
                    break

            total_reward = sum(rewards)
            max_possible = sum(r for r in rewards if r > 0) or 1.0
            raw_score    = total_reward / max_possible
            score        = round(min(max(raw_score, 0.01), 0.99), 3)
            success      = score > 0.5

        except Exception as e:
            print(f"[ERROR] {e}", flush=True)

        finally:
            log_end(success, steps, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())
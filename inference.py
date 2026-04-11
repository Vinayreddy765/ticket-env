import asyncio
import os
import requests
from typing import List, Optional, Dict
from openai import OpenAI


# ── Credentials ───────────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "https://vinay3111-ticket-env.hf.space"

TASK_NAME = "ticket-routing"
BENCHMARK = "ticket_env"


# ── Logging ───────────────────────────────────────────────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


# ── Safe parser — handles "Agent 2", "The answer is 2", "2", etc. ─────────────
def safe_parse_agent(text: str) -> int:
    for ch in text:
        if ch in "123":
            return int(ch)
    return 3  # safe fallback


# ── Rule-based fast routing (no LLM needed for simple cases) ──────────────────
def rule_based_route(ticket: dict) -> Optional[int]:
    category = ticket.get("category", "").lower()
    if category == "billing":
        return 1
    if category == "tech":
        return 2
    return None  # unknown → send to LLM


# ── Batch LLM routing — one API call for all tickets ─────────────────────────
def route_tickets_batch(client: OpenAI, tickets: list) -> Dict[int, int]:
    ticket_str = "\n".join(
        [f"{t['id']}: category={t['category']} priority={t.get('priority', 'normal')}"
         for t in tickets]
    )

    prompt = (
        "You are an intelligent ticket routing system.\n\n"
        "Routing rules:\n"
        "  Agent 1 → handles billing issues only\n"
        "  Agent 2 → handles tech issues only\n"
        "  Agent 3 → handles both billing and tech\n\n"
        "Think carefully about each ticket category, then assign the best agent.\n"
        "High priority tickets should prefer Agent 3 for best coverage.\n\n"
        f"Tickets to route:\n{ticket_str}\n\n"
        "Return ONLY lines in this exact format (one per ticket):\n"
        "ticket_id:agent_id\n"
        "Example:\n"
        "1:2\n2:1\n3:3\n"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )
        text = response.choices[0].message.content.strip()
        print(f"[DEBUG] Batch LLM response:\n{text}", flush=True)

        mapping = {}
        for line in text.split("\n"):
            line = line.strip()
            if ":" in line:
                parts = line.split(":")
                try:
                    tid = int(parts[0].strip())
                    aid = safe_parse_agent(parts[1].strip())
                    mapping[tid] = aid
                except ValueError:
                    continue
        return mapping

    except Exception as e:
        print(f"[DEBUG] Batch LLM failed: {e}", flush=True)
        return {}


# ── Per-ticket LLM routing with retry (fallback for batch failures) ───────────
def route_ticket_single(client: OpenAI, ticket: dict, retries: int = 2) -> int:
    prompt = (
        "You are a smart ticket router.\n"
        "Rules:\n"
        "  billing → agent 1\n"
        "  tech → agent 2\n"
        "  billing and tech → agent 3\n\n"
        f"Ticket category: {ticket['category']}\n"
        f"Priority: {ticket.get('priority', 'normal')}\n\n"
        "Output ONLY a single digit: 1, 2, or 3."
    )

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            text = response.choices[0].message.content.strip()
            return safe_parse_agent(text)
        except Exception as e:
            print(f"[DEBUG] Single LLM attempt {attempt + 1} failed: {e}", flush=True)

    return 3  # final fallback


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[DEBUG] ENV_BASE_URL={ENV_BASE_URL}", flush=True)

    rewards: List[float] = []
    steps = 0
    score = 0.0
    success = False

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        # ── Reset env ─────────────────────────────────────────────────────────
        print(f"[DEBUG] POST {ENV_BASE_URL}/reset", flush=True)
        res = requests.post(f"{ENV_BASE_URL}/reset", timeout=30)
        res.raise_for_status()
        body    = res.json()
        obs     = body.get("observation", body)
        tickets = obs.get("tickets", [])
        print(f"[DEBUG] {len(tickets)} tickets received", flush=True)

        # ── Batch route all tickets in ONE LLM call ───────────────────────────
        print("[DEBUG] Calling batch LLM routing...", flush=True)
        batch_mapping = route_tickets_batch(client, tickets)
        print(f"[DEBUG] Batch mapping: {batch_mapping}", flush=True)

        # ── Execute each ticket ───────────────────────────────────────────────
        for i, ticket in enumerate(tickets, 1):
            tid      = ticket["id"]
            category = ticket["category"]
            priority = ticket.get("priority", "normal")

            # Priority 1: rule-based (instant, no LLM cost)
            agent_id = rule_based_route(ticket)
            reason   = "rule-based"

            # Priority 2: batch LLM result
            if agent_id is None and tid in batch_mapping:
                agent_id = batch_mapping[tid]
                reason   = "batch-llm"

            # Priority 3: single LLM call with retry
            if agent_id is None:
                agent_id = route_ticket_single(client, ticket)
                reason   = "single-llm"

            print(f"[DEBUG] Ticket {tid} (cat={category} pri={priority}) → agent {agent_id} via {reason}", flush=True)

            res = requests.post(
                f"{ENV_BASE_URL}/step",
                json={"action": {"ticket_id": tid, "agent_id": agent_id}},
                timeout=30,
            )
            res.raise_for_status()
            data   = res.json()
            reward = float(data.get("reward", 0))
            done   = bool(data.get("done", False))

            rewards.append(reward)
            steps = i

            log_step(i, f"assign({tid}->{agent_id})[{reason}]", reward, done, None)

            if done:
                break

        score   = min(max(sum(rewards) / max(len(rewards), 1), 0.0), 1.0)
        success = score > 0.5

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)

    finally:
        log_end(success, steps, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())
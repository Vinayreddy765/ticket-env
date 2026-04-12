---
title: Ticket Env Environment Server
colorFrom: gray
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---
#  Ticket Env — OpenEnv Environment

A reinforcement learning environment where an AI agent routes customer support tickets to the most suitable human agents based on skills and priority.

Built on the [OpenEnv](https://github.com/meta-pytorch/openenv) framework by Meta.

---

## Overview

The environment simulates a real-world helpdesk routing system. At each step, the agent receives a list of open tickets and available agents, and must assign each ticket to the right agent based on skill match and priority.

---

## Project Structure
ticket_env/
├── init.py                        # Package exports
├── client.py                          # TicketEnv WebSocket client
├── models.py                          # Action and Observation models
├── inference.py                       # LLM agent inference script
├── openenv.yaml                       # OpenEnv manifest
├── pyproject.toml                     # Project metadata and dependencies
├── uv.lock                            # Locked dependencies
└── server/
├── init.py                    # Server module exports
├── app.py                         # FastAPI application
├── ticket_env_environment.py      # Core environment logic
└── Dockerfile                     # Container image definition

---

## Environment Design

### Action
```json
{ "ticket_id": 1, "agent_id": 2 }
```

### Observation
```json
{
  "tickets": [{ "id": 1, "category": "billing", "priority": 3 }],
  "agents":  [{ "id": 1, "skills": ["billing"] }],
  "done": false,
  "reward": 0.0
}
```

### Reward Function

| Outcome | Reward |
|---|---|
| Correct skill match | `+1.0 × priority` |
| Skill mismatch | `-0.5` |
| Invalid ticket or agent | `-1.0` |
| Last ticket correctly assigned | extra `+1.0` bonus |

---

## Tasks

| Task | Goal | Grader |
|---|---|---|
| Easy | Assign tickets by matching category | `tasks/easy.py` |
| Medium | Sequential assignments, partial credit | `tasks/medium.py` |
| Hard | Full correct trajectory required | `tasks/hard.py` |

Scores are between `0.0` and `0.99`.

---

## Baseline Performance

| Task | Score |
|---|---|
| Easy | ~0.99 |
| Medium | ~0.80 |
| Hard | ~0.50+ |

---

## Quick Start

### 1. Install dependencies
```bash
pip install openenv-core
```

### 2. Run the server locally
```bash
uv run server
# or
uvicorn server.app:app --reload --port 8000
```

### 3. Use the client
```python
from ticket_env import TicketEnv, TicketAction

with TicketEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(result.observation.tickets)

    result = env.step(TicketAction(ticket_id=1, agent_id=1))
    print(result.reward)
```

### 4. Run the LLM inference agent
```bash
python inference.py
```

---

## Docker

### Build
```bash
docker build -t ticket_env-env:latest -f server/Dockerfile .
```

### Run
```bash
docker run -p 8000:8000 ticket_env-env:latest
```

### Use via Docker client
```python
from ticket_env import TicketEnv, TicketAction

env = TicketEnv.from_docker_image("ticket_env-env:latest")
try:
    result = env.reset()
    result = env.step(TicketAction(ticket_id=1, agent_id=2))
    print(result.reward)
finally:
    env.close()
```

---

## Deploy to Hugging Face Spaces

```bash
# Push to your namespace (uses name from openenv.yaml)
openenv push

# Push to a specific repo
openenv push --repo-id my-org/ticket-env

# Push as private
openenv push --private
```

After deployment your space exposes:

| Endpoint | Purpose |
|---|---|
| `/web` | Interactive UI |
| `/docs` | OpenAPI / Swagger |
| `/health` | Health check |
| `/ws` | WebSocket for persistent sessions |

Live deployment: [https://vinay3111-ticket-env.hf.space](https://vinay3111-ticket-env.hf.space)

---

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit a ticket assignment |
| `GET` | `/state` | Get current episode state |
| `GET` | `/schema` | Get action/observation schemas |
| `WS` | `/ws` | Persistent WebSocket session |

---

## Future Improvements

- Dynamic ticket arrival during an episode
- Agent workload constraints
- Multi-agent coordination
- Configurable ticket and agent pools

---

## License

BSD-style license — see `LICENSE` in the repository root.  
Copyright (c) Meta Platforms, Inc. and affiliates.
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

# Ticket Env Environment

A real-world ticket routing environment where an AI agent assigns customer support tickets to the most suitable agents based on skills and categories.

## Key Features

- Real-world task: customer support ticket routing
- Structured RL environment with step/reset/state API
- Multi-task evaluation (easy → medium → hard)
- Deterministic grading system (0.0 – 1.0)
- Deployable and reproducible via OpenEnv

## Quick Start

The simplest way to use the Ticket Env environment is through the `TicketEnv` class:

```python
from ticket_env import TicketAction, TicketEnv

try:
    # Create environment from Docker image
    ticket_envenv = TicketEnv.from_docker_image("ticket_env-env:latest")

    # Reset
    result = ticket_envenv.reset()
    print(f"Reset: {result.observation.echoed_message}")

    # Send multiple messages
    messages = ["Hello, World!", "Testing echo", "Final message"]

    for msg in messages:
        result = ticket_envenv.step(TicketAction(message=msg))
        print(f"Sent: '{msg}'")
        print(f"  → Echoed: '{result.observation.echoed_message}'")
        print(f"  → Length: {result.observation.message_length}")
        print(f"  → Reward: {result.reward}")

finally:
    # Always clean up
    ticket_envenv.close()
```

That's it! The `TicketEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t ticket_env-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action
**TicketAction**: Contains a single field
- `message` (str) - The message to echo back

### Observation
**TicketObservation**: Contains the echo response and metadata
- `echoed_message` (str) - The message echoed back
- `message_length` (int) - Length of the message
- `reward` (float) - Reward based on message length (length × 0.1)
- `done` (bool) - Always False for echo environment
- `metadata` (dict) - Additional info like step count

### Reward
The reward is calculated as: `message_length × 0.1`
- "Hi" → reward: 0.2
- "Hello, World!" → reward: 1.3
- Empty message → reward: 0.0

## Advanced Usage

### Connecting to an Existing Server

If you already have a Ticket Env environment server running, you can connect directly:

```python
from ticket_env import TicketEnv

# Connect to existing server
ticket_envenv = TicketEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = ticket_envenv.reset()
result = ticket_envenv.step(TicketAction(message="Hello!"))
```

Note: When connecting to an existing server, `ticket_envenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from ticket_env import TicketAction, TicketEnv

# Connect with context manager (auto-connects and closes)
with TicketEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(TicketAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    TicketEnvironment,  # Pass class, not instance
    TicketAction,
    TicketObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from ticket_env import TicketAction, TicketEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with TicketEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(TicketAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/ticket_env_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
ticket_env/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # TicketEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── ticket_env_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
#  Ticket Routing OpenEnv Environment

## Overview

This project implements a **real-world customer support ticket routing system** using the OpenEnv framework.

The environment simulates how support tickets are assigned to agents based on their expertise — a common task in customer service platforms.

---

##  Objective

Train and evaluate AI agents to:

* Assign tickets to the most suitable agents
* Maximize correct routing
* Minimize incorrect assignments

---

##  Environment Design

### 🔹 Action Space

```json
{
  "ticket_id": int,
  "agent_id": int
}
```

### 🔹 Observation Space

```json
{
  "tickets": [
    {"id": int, "category": str, "priority": int}
    ],
  "agents": [
    {"id": int, "skills": [str]}
  ]
}
```

### 🔹 Reward Function

*  Correct assignment **+1.0**
*  Incorrect assignment  **-0.5**
*  Invalid action  **-1.0**

Provides **dense reward signals** to guide learning.

---

##  Tasks

### Easy

* Small number of tickets
* Direct mapping between ticket category and agent skill
* Goal: basic correct assignment

### Medium

* Multiple tickets
* Requires sequential decision making
* Goal: maintain accuracy across steps

### Hard

* Full completion required
* Penalizes incorrect assignments
* Goal: optimize entire trajectory

---

## Grading

Each task includes a **deterministic grader**:

* Scores between **0.0 – 1.0**
* Based on correctness of assignments
* Evaluates full trajectory

---

##  Setup & Run

```bash
pip install openenv-core
uv run server
```

---

##  Deployment

Deployed on Hugging Face Spaces:

👉 https://vinay3111-ticket-env.hf.space

---

## Inference

Run:

```bash
python inference.py
```

---

##  Motivation

Ticket routing is a real-world operational problem in:

* Customer support systems
* Helpdesk automation
* IT service management

This environment provides a structured way to train AI agents for such tasks.

---

## Baseline Performance

* Easy: ~1.0
* Medium: ~0.8
* Hard: ~0.5+

---

##  Future Improvements

* Add dynamic ticket arrival
* Introduce agent workload constraints
* Multi-agent coordination

---

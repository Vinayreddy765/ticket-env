# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Ticket Env Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - GET /health: Health check
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    # HF / Docker (flat structure)
    from models import TicketAction, TicketObservation
    from server.ticket_env_environment import TicketEnvironment
except ImportError:
    # Local development
    from ..models import TicketAction, TicketObservation
    from .ticket_env_environment import TicketEnvironment

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Create the wrapper app
app = FastAPI()

@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})

@app.get("/")
def root():
    return JSONResponse({"status": "ok", "env": "ticket_env"})

@app.get("/web")
def web():
    return JSONResponse({"status": "ok", "env": "ticket_env"})

# Create and mount the OpenEnv app
env_app = create_app(
    TicketEnvironment,
    TicketAction,
    TicketObservation,
    env_name="ticket_env",
    max_concurrent_envs=1,
)

app.mount("/", env_app)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Ticket Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TicketAction, TicketObservation
except ImportError:
    from models import TicketAction, TicketObservation


class TicketEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = TicketEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Ticket Env environment ready!"
        >>>
        >>> obs = env.step(TicketAction(message="Hello"))
    """
    tickets = []
    agents = []

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
         self._state = State(episode_id=str(uuid4()), step_count=0)
         self._reset_count = 0

    # ✅ ADD THIS
         self.tickets = []
         self.agents = []
  
    def reset(self) -> TicketObservation:
      self._state = State(episode_id=str(uuid4()), step_count=0)

      TicketEnvironment.tickets = [
        {"id": 1, "category": "billing", "priority": 1},
        {"id": 2, "category": "tech", "priority": 2},
    ]

      TicketEnvironment.agents = [
        {"id": 1, "skills": ["billing"]},
        {"id": 2, "skills": ["tech"]},
    ]

      return TicketObservation(
        tickets=TicketEnvironment.tickets,
        agents=TicketEnvironment.agents,
        done=False,
        reward=0.0
    )
    def step(self, action: TicketAction) -> TicketObservation:
      self._state.step_count += 1

      ticket_id = action.ticket_id
      agent_id = action.agent_id

      ticket = next((t for t in TicketEnvironment.tickets if t["id"] == ticket_id), None)
      agent = next((a for a in TicketEnvironment.agents if a["id"] == agent_id), None)

      if ticket is None or agent is None:
        return TicketObservation(
            tickets=TicketEnvironment.tickets,
            agents=TicketEnvironment.agents,
            done=True,
            reward=-1.0
        )

      reward = 1.0 if ticket["category"] in agent["skills"] else -0.5

      TicketEnvironment.tickets = [
        t for t in TicketEnvironment.tickets if t["id"] != ticket_id
    ]

      done = len(TicketEnvironment.tickets) == 0

      return TicketObservation(
        tickets=TicketEnvironment.tickets,
        agents=TicketEnvironment.agents,
        done=done,
        reward=reward
    )
    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

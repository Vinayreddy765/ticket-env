# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Ticket Env Environment Implementation.

A simple test environment that assigns tickets to agents based on skills and priority.
"""

from uuid import uuid4
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TicketAction, TicketObservation
except ImportError:
    from models import TicketAction, TicketObservation


DEFAULT_TICKETS = [
    {"id": 1, "category": "billing", "priority": 3},
    {"id": 2, "category": "tech", "priority": 2},
    {"id": 3, "category": "billing", "priority": 1},
]

DEFAULT_AGENTS = [
    {"id": 1, "skills": ["billing"]},
    {"id": 2, "skills": ["tech"]},
    {"id": 3, "skills": ["billing", "tech"]},
]


class TicketEnvironment(Environment):
    """
    An environment that assigns support tickets to agents based on skill match and priority.

    Each instance maintains its own isolated state, supporting concurrent sessions.

    Example:
        >>> env = TicketEnvironment()
        >>> obs = env.reset()
        >>> obs = env.step(TicketAction(ticket_id=1, agent_id=1))
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        initial_tickets: Optional[list] = None,
        initial_agents: Optional[list] = None,
    ):
        # FIX 1: Proper indentation — all init code is inside __init__
        # FIX 2: Instance-level state — tickets/agents are per-instance, not class-level
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

        # FIX 4: Configurable data — accept custom tickets/agents, fall back to defaults
        self._initial_tickets = initial_tickets or DEFAULT_TICKETS
        self._initial_agents = initial_agents or DEFAULT_AGENTS

        self.tickets = []
        self.agents = []

    def reset(self) -> TicketObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        # FIX 2: Deep copy so each episode gets a fresh independent list
        self.tickets = [t.copy() for t in self._initial_tickets]
        self.agents = [a.copy() for a in self._initial_agents]

        return TicketObservation(
            tickets=self.tickets,
            agents=self.agents,
            done=False,
            reward=0.0,
        )

    def step(self, action: TicketAction) -> TicketObservation:
        self._state.step_count += 1

        ticket_id = action.ticket_id
        agent_id = action.agent_id

        # FIX 2: Use instance variables instead of class variables
        ticket = next((t for t in self.tickets if t["id"] == ticket_id), None)
        agent = next((a for a in self.agents if a["id"] == agent_id), None)

        if ticket is None or agent is None:
            return TicketObservation(
                tickets=self.tickets,
                agents=self.agents,
                done=True,
                reward=-1.0,
            )

        if ticket["category"] in agent["skills"]:
            reward = 1.0 * ticket["priority"]
        else:
            reward = -0.5

        # Remove the assigned ticket
        self.tickets = [t for t in self.tickets if t["id"] != ticket_id]

        # FIX 3: Remove the agent after assignment so they can't be reused
        self.agents = [a for a in self.agents if a["id"] != agent_id]

        done = len(self.tickets) == 0

        # Completion bonus when last ticket is correctly assigned
        if done and reward > 0:
            reward += 1.0

        return TicketObservation(
            tickets=self.tickets,
            agents=self.agents,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
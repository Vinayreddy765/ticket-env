from enum import Enum
from typing import List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class Category(str, Enum):
    billing = "billing"
    tech = "tech"


class Ticket(BaseModel):
    id: int
    category: Category
    priority: int = Field(..., ge=1, le=3, description="Priority level (1=low, 3=high)")


class Agent(BaseModel):
    id: int
    skills: List[Category] = Field(..., description="List of categories this agent can handle")


class TicketAction(Action):
    """Action: assign a ticket to an agent"""
    ticket_id: int = Field(..., description="ID of the ticket")
    agent_id: int = Field(..., description="ID of the agent")


class TicketObservation(Observation):
    """Observation: current tickets and agents"""
    tickets: List[Ticket] = Field(default_factory=list)
    agents: List[Agent] = Field(default_factory=list)
    done: bool = False
    reward: float = 0.0
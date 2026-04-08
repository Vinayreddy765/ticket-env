from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field
from typing import List, Optional


class Ticket(BaseModel):
    id: int
    category: str
    priority: int


class Agent(BaseModel):
    id: int
    skills: List[str]


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
    metadata: Optional[dict] = None
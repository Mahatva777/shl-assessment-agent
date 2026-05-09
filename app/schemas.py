"""
Pydantic request / response models for the SHL Assessment Agent API.

These schemas match the SHL spec exactly:
    POST /chat
    Request  → ChatRequest  (list of ChatMessage objects)
    Response → ChatResponse (reply + recommendations + end flag)
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chat messages
# ---------------------------------------------------------------------------

class Role(str, Enum):
    """Roles in a multi-turn chat conversation."""

    user = "user"
    assistant = "assistant"
    system = "system"


class ChatMessage(BaseModel):
    """A single message in the conversation history."""

    role: Role = Field(..., description="Who sent this message.")
    content: str = Field(..., description="Message text.")


# ---------------------------------------------------------------------------
# Chat request
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Incoming request body for ``POST /chat``."""

    messages: List[ChatMessage] = Field(
        ...,
        min_length=1,
        description="Conversation history (at least one user message).",
    )


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

class Recommendation(BaseModel):
    """One SHL assessment recommendation returned to the user."""

    product_name: str = Field(..., description="Display name of the assessment.")
    url: str = Field(..., description="Link to the SHL product page.")
    duration_minutes: Optional[int] = Field(
        None,
        description="Estimated completion time in minutes (null if unknown).",
    )
    remote_supported: bool = Field(
        ...,
        description="Whether the assessment can be administered remotely.",
    )
    adaptive: Optional[bool] = Field(
        None,
        description="Whether the assessment adapts to the candidate.",
    )
    description: str = Field(
        "",
        description="Short description of the assessment.",
    )


# ---------------------------------------------------------------------------
# Chat response
# ---------------------------------------------------------------------------

class ChatResponse(BaseModel):
    """Response body returned by ``POST /chat``."""

    reply: str = Field(
        ...,
        description="The agent's natural-language reply.",
    )
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description="SHL assessment recommendations (may be empty when the agent is still gathering info).",
    )
    end_of_conversation: bool = Field(
        False,
        description="True when the agent considers the conversation complete.",
    )

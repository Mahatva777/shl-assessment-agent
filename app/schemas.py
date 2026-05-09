# app/schemas.py
from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class Role(str, Enum):
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    role: Role = Field(..., description="Who sent this message.")
    content: str = Field(..., description="Message text.")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(
        ...,
        min_length=1,
        description="Full conversation history (at least one message).",
    )


class Recommendation(BaseModel):
    name: str = Field(..., description="Assessment name from the SHL catalog.")
    url: str = Field(..., description="SHL catalog URL for this assessment.")
    test_type: str = Field(..., description="Test type code: K, A, P, B, or S.")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Agent's natural-language reply.")
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = Field(False)
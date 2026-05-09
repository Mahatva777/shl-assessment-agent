"""
SHL Assessment Agent – FastAPI entry-point.

Run locally with:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SHL Assessment Recommendation Agent",
    version="0.1.0",
    description=(
        "An AI-powered conversational agent that recommends SHL assessments "
        "based on job descriptions, hiring requirements, and natural-language queries."
    ),
)

# CORS – allow all origins during development; tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    """Liveness / readiness probe."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """Multi-turn chat endpoint.

    Accepts the conversation history and returns the agent's reply
    together with any SHL assessment recommendations.

    This is a **placeholder** – the full agent pipeline (state extraction →
    retrieval → scoring → LLM reply) will be wired in later.
    """
    # TODO: wire in agent.run(request.messages)
    last_user_msg = next(
        (m.content for m in reversed(request.messages) if m.role == "user"),
        "",
    )

    return ChatResponse(
        reply=(
            f"Thanks for your query: \"{last_user_msg}\". "
            "I'm still being set up — full recommendations coming soon!"
        ),
        recommendations=[],
        end_of_conversation=False,
    )

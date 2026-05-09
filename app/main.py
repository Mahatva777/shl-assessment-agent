"""
app/main.py

FastAPI application entry point.

Endpoints:
  GET  /health  -> {"status": "ok"}
  POST /chat    -> ChatResponse
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.schemas import ChatRequest, ChatResponse
from app.agent import agent

app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return agent(request.messages)


# ---------------------------------------------------------------------------
# Global error handler — keeps the service alive on unexpected failures and
# always returns a schema-compliant JSON body instead of a 500 HTML page.
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    body = ChatResponse(
        reply=(
            "I encountered an unexpected error. "
            "Please try again or rephrase your request."
        ),
        recommendations=[],
        end_of_conversation=False,
    )
    return JSONResponse(status_code=200, content=body.model_dump())
# app/main.py
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent import agent
from app.schemas import ChatRequest, ChatResponse

app = FastAPI(
    title="SHL Assessment Recommendation Agent",
    version="1.0.0",
    description=(
        "Conversational agent that recommends SHL Individual Test Solutions "
        "based on job descriptions and hiring requirements."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return agent(request.messages)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    body = ChatResponse(
        reply="I encountered an unexpected error. Please try again.",
        recommendations=[],
        end_of_conversation=False,
    )
    return JSONResponse(status_code=200, content=body.model_dump())
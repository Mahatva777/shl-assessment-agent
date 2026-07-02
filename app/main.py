# app/main.py
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static paths for pre-computed assets
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_EMBEDDINGS_PATH = str(_REPO_ROOT / "data" / "catalog_embeddings.npy")
_METADATA_PATH = str(_REPO_ROOT / "data" / "catalog_metadata.json")
_CATALOG_PATH = str(_REPO_ROOT / "data" / "shl_product_catalog.json")


# ---------------------------------------------------------------------------
# Lifespan – all startup I/O happens here, ONCE, before any request
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    """
    FastAPI lifespan hook.

    Loads all heavy resources at boot so that every request handler finds
    them already in memory.  Nothing is loaded lazily.

    Resources loaded
    ----------------
    1. Catalog items (CatalogItem list) – stored in app.agent._catalog
    2. Pre-computed embeddings + metadata – stored in app.retrieval module globals
    3. SentenceTransformer model         – stored in app.retrieval._model
    """
    t_boot = time.monotonic()
    logger.info("=== SHL Agent: lifespan startup begin ===")

    # ── 1. Load pre-computed embeddings + metadata ───────────────────────────
    from app.retrieval import load_catalog_embeddings, load_sentence_model

    load_catalog_embeddings(_EMBEDDINGS_PATH, _METADATA_PATH)

    # ── 2. Load the SentenceTransformer (query encoding only) ────────────────
    load_sentence_model()

    # ── 3. Load the catalog into agent._catalog ──────────────────────────────
    from app.agent import _bootstrap_catalog

    _bootstrap_catalog(_CATALOG_PATH)

    # ── 4. Report startup time ───────────────────────────────────────────────
    elapsed_ms = (time.monotonic() - t_boot) * 1000
    logger.info(
        "=== SHL Agent: startup complete in %.0f ms "
        "(model download excluded) ===",
        elapsed_ms,
    )
    print(
        f"[SHL Agent] Startup complete in {elapsed_ms:.0f} ms "
        f"(model download excluded).",
        flush=True,
    )

    yield  # ── application is now serving requests ──────────────────────────

    # Teardown (nothing to do – OS reclaims memory)
    logger.info("=== SHL Agent: lifespan shutdown ===")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SHL Assessment Recommendation Agent",
    version="1.0.0",
    description=(
        "Conversational agent that recommends SHL Individual Test Solutions "
        "based on job descriptions and hiring requirements."
    ),
    lifespan=lifespan,
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
    """Returns immediately – no I/O, no model calls."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        from app.agent import agent  # imported here to avoid circular at module level
        return agent(request.messages)
    except Exception:
        logger.exception("Unhandled error in /chat")
        return ChatResponse(
            reply="I encountered an error. Please try again.",
            recommendations=[],
            end_of_conversation=False,
            messages=request.messages,
        )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    body = ChatResponse(
        reply="I encountered an unexpected error. Please try again.",
        recommendations=[],
        end_of_conversation=False,
    )
    return JSONResponse(status_code=200, content=body.model_dump())
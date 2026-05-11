# app/agent.py
from __future__ import annotations
import logging
import threading
from typing import Optional
from app.catalog_loader import CatalogItem, load_catalog
from app.config import get_settings
from app.guards import (
    build_recommendations_from_names,
    build_recommendations_from_scores,
    find_catalog_item,
    sanitise_reply,
    validate_chat_response,
)
from app.llm_client import call_llm
from app.policy import get_forced_mode, should_end_conversation
from app.retrieval import build_embeddings, semantic_search
from app.schemas import ChatMessage, ChatResponse
from app.state_extraction import extract_state_from_messages

logger = logging.getLogger(__name__)

_catalog: Optional[list[CatalogItem]] = None
_catalog_lock = threading.Lock()


def _get_catalog() -> list[CatalogItem]:
    global _catalog
    if _catalog is None:
        with _catalog_lock:
            if _catalog is None:
                settings = get_settings()
                try:
                    cat = load_catalog(settings.catalog_path)
                    build_embeddings(cat)
                    _catalog = cat
                    logger.info("Catalog loaded: %d items.", len(cat))
                except Exception:
                    logger.exception("Failed to load catalog.")
                    _catalog = []
    return _catalog


def _safe(reply: str, recs=None, end=False) -> ChatResponse:
    return ChatResponse(reply=reply, recommendations=recs or [], end_of_conversation=end)


def agent(messages: list[ChatMessage]) -> ChatResponse:
    """
    Stateless agent. Never raises. Always returns a schema-valid ChatResponse.

    Architecture:
    1. Input validation
    2. State extraction (lightweight retrieval support)
    3. Deterministic safety/compare/refine gate (policy.get_forced_mode)
    4. Retrieval (semantic search + scoring)
    5. LLM call — owns mode, reply, chosen_names
    6. Grounding guard — maps chosen_names to real catalog items only
    7. Response assembly
    """

    # ── 1. Input validation ──────────────────────────────────────────────
    if not messages:
        return _safe("Hello! I'm the SHL Assessment Recommender. What role are you hiring for?")

    user_messages = [m for m in messages if m.role.value == "user"]
    if not user_messages:
        return _safe("Please describe the role you're hiring for so I can recommend SHL assessments.")

    if messages[-1].role.value != "user":
        return _safe("I'm ready to help. What role are you assessing for, or would you like to refine the recommendations?")

    last_user: ChatMessage = messages[-1]

    # ── 2. Catalog + state ───────────────────────────────────────────────
    catalog = _get_catalog()
    state = extract_state_from_messages(messages)

    # ── 3. Deterministic safety gate ─────────────────────────────────────
    forced_mode = get_forced_mode(state, last_user)

    # For forced refuse — no LLM call needed, skip retrieval.
    if forced_mode == "refuse":
        # Soft refuse for legal questions mid-conversation (C7 pattern)
        import re
        is_legal_mid = (
            len(user_messages) > 1
            and re.search(r"\b(hipaa|eeoc|gdpr|legal|complian|law|require)\b", last_user.content, re.IGNORECASE)
        )
        if is_legal_mid:
            return _safe(
                "I can't provide legal or compliance advice — please consult your legal team. "
                "I'm happy to continue recommending SHL assessments for this role."
            )
        return _safe(
            "I can only assist with SHL assessment recommendations. "
            "I'm unable to help with legal questions, general hiring advice, or non-SHL assessments. "
            "Please describe the role you're assessing and I'll suggest the right tests."
        )

    # ── 4. Retrieval ─────────────────────────────────────────────────────
    query = state.build_query_string() or last_user.content
    scored: list[tuple[float, CatalogItem]] = []
    candidates: list[CatalogItem] = []
    if catalog:
        try:
            scored = semantic_search(query, state, catalog, top_k=20)
            candidates = [item for _, item in scored]
        except Exception:
            logger.exception("Retrieval failed.")

    # For compare: locate named targets specifically
    if forced_mode == "compare":
        targets = state.compare_targets or []
        search_q = " ".join(targets) if targets else last_user.content
        try:
            pool = semantic_search(search_q, state, catalog, top_k=30)
            pool_items = [item for _, item in pool]
        except Exception:
            pool_items = candidates

        named: list[CatalogItem] = []
        seen_ids: set[str] = set()
        for target in targets:
            found = find_catalog_item(target, pool_items) or find_catalog_item(target, catalog)
            if found and found.id not in seen_ids:
                named.append(found)
                seen_ids.add(found.id)
        candidates = named if named else pool_items[:6]

    # ── 5. LLM call ───────────────────────────────────────────────────────
    result = call_llm(
        messages=messages,
        state=state,
        candidates=candidates,
        forced_mode=forced_mode,  # None means LLM decides freely
    )

    llm_mode: str = result.get("mode", "consult")
    reply: str = sanitise_reply(result.get("reply") or "Here are the SHL assessments I recommend.")
    chosen_names: list[str] = result.get("chosen_names") or []

    # ── 6. Grounding: map chosen_names to real catalog items only ─────────
    recs = []
    if llm_mode in ("recommend", "refine") and chosen_names:
        recs = build_recommendations_from_names(chosen_names, candidates, top_n=10)
        # Fallback: if LLM chose nothing valid, use top scored items
        if not recs and scored:
            recs = build_recommendations_from_scores(scored, top_n=5)
    elif llm_mode in ("recommend", "refine") and not chosen_names and scored:
        # LLM said recommend but gave no names — use retrieval fallback
        recs = build_recommendations_from_scores(scored, top_n=5)

    # Modes that must never return recommendations
    if llm_mode in ("clarify", "consult", "compare", "refuse"):
        recs = []

    # ── 7. End-of-conversation ────────────────────────────────────────────
    end = should_end_conversation(llm_mode, state)

    return validate_chat_response(ChatResponse(reply=reply, recommendations=recs, end_of_conversation=end))
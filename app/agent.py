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
from app.policy import decide_mode, pick_clarification_question, should_end_conversation
from app.retrieval import build_embeddings, semantic_search
from app.schemas import ChatMessage, ChatResponse
from app.state_extraction import extract_state_from_messages

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level catalog singleton (loaded once, never mutated after init)
# ---------------------------------------------------------------------------

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
                    logger.info("Catalog loaded and embedded: %d items.", len(cat))
                except Exception:
                    logger.exception("Failed to load/embed catalog; using empty catalog.")
                    _catalog = []
    return _catalog


# ---------------------------------------------------------------------------
# Safe response helpers
# ---------------------------------------------------------------------------

def _clarify(reply: str) -> ChatResponse:
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)


def _refuse(reply: str) -> ChatResponse:
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def agent(messages: list[ChatMessage]) -> ChatResponse:
    """
    Stateless agent: takes the full conversation history, returns the next reply.
    Never raises; always returns a schema-valid ChatResponse.
    """

    # ── 1. Input validation ──────────────────────────────────────────────
    if not messages:
        return _clarify(
            "Hello! I'm the SHL Assessment Recommender. "
            "What role are you hiring for? I'll suggest the right assessments."
        )

    user_messages = [m for m in messages if m.role.value == "user"]
    if not user_messages:
        return _clarify(
            "Please describe the role you're hiring for so I can recommend SHL assessments."
        )

    if messages[-1].role.value != "user":
        return _clarify(
            "I'm ready to help. What role are you assessing for, "
            "or would you like to refine the current recommendations?"
        )

    last_user: ChatMessage = messages[-1]

    # ── 2. Catalog ───────────────────────────────────────────────────────
    catalog = _get_catalog()

    # ── 3. State extraction ──────────────────────────────────────────────
    state = extract_state_from_messages(messages)

    # ── 4. Policy ────────────────────────────────────────────────────────
    mode = decide_mode(state, last_user)

    # ── 5. Mode dispatch ─────────────────────────────────────────────────

    if mode == "refuse":
        return _refuse(
            "I can only assist with SHL assessment recommendations. "
            "I'm not able to help with legal questions, general hiring advice, "
            "or non-SHL assessments. "
            "Please describe a role you're hiring for and I'll suggest the right tests."
        )

    if mode == "clarify":
        return _clarify(pick_clarification_question(state))

    if mode == "compare":
        targets = state.compare_targets or []
        search_query = " ".join(targets) if targets else last_user.content
        pool = semantic_search(search_query, state, catalog, top_k=20)
        pool_items = [item for _, item in pool]

        named: list[CatalogItem] = []
        seen_ids: set[str] = set()
        for target in targets:
            found = find_catalog_item(target, pool_items)
            if found and found.id not in seen_ids:
                named.append(found)
                seen_ids.add(found.id)

        compare_candidates = named if named else pool_items[:5]

        result = call_llm(mode="compare", messages=messages, state=state, candidates=compare_candidates)
        reply = sanitise_reply(result.get("reply") or "Here is a comparison of those assessments.")
        end = should_end_conversation(mode, state)
        return validate_chat_response(
            ChatResponse(reply=reply, recommendations=[], end_of_conversation=end)
        )

    # ── recommend / refine ───────────────────────────────────────────────
    query = state.build_query_string() or last_user.content
    scored = semantic_search(query, state, catalog, top_k=20)
    candidates = [item for _, item in scored]

    if not candidates:
        return _clarify(
            "I couldn't find assessments matching those constraints. "
            "Could you broaden the requirements — for example, relax the duration, "
            "language, or seniority level?"
        )

    result = call_llm(mode=mode, messages=messages, state=state, candidates=candidates)
    reply = sanitise_reply(result.get("reply") or "Here are the SHL assessments I recommend.")
    chosen_names: list[str] = result.get("chosen_names") or []

    recs = build_recommendations_from_names(chosen_names, candidates, top_n=10)
    if not recs:
        recs = build_recommendations_from_scores(scored, top_n=5)

    end = should_end_conversation(mode, state)
    return validate_chat_response(
        ChatResponse(reply=reply, recommendations=recs, end_of_conversation=end)
    )
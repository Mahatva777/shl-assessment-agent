# app/agent.py
from __future__ import annotations

import logging
import re
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


def _bootstrap_catalog(catalog_path: str) -> None:
    """
    Eagerly load the catalog from disk into the module-level _catalog global.

    Called ONCE from the FastAPI lifespan hook (app/main.py) before any
    request is served.  build_embeddings() is NOT called here because
    embeddings are pre-loaded from data/catalog_embeddings.npy by
    retrieval.load_catalog_embeddings().
    """
    global _catalog
    with _catalog_lock:
        if _catalog is not None:
            logger.debug("_bootstrap_catalog: catalog already loaded, skipping.")
            return
        try:
            cat = load_catalog(catalog_path)
            # build_embeddings is now a no-op shim – embeddings live in retrieval._emb_matrix
            build_embeddings(cat)
            _catalog = cat
            logger.info("Catalog bootstrapped: %d items.", len(cat))
        except Exception:
            logger.exception("Failed to bootstrap catalog; falling back to empty.")
            _catalog = []


def _get_catalog() -> list[CatalogItem]:
    """
    Return the module-level catalog.

    If _bootstrap_catalog() was already called from the lifespan hook this
    returns immediately.  The lazy fallback path is kept so that tests that
    import agent directly (without starting the full app) still work.
    """
    global _catalog
    if _catalog is None:
        with _catalog_lock:
            if _catalog is None:
                settings = get_settings()
                try:
                    cat = load_catalog(settings.catalog_path)
                    build_embeddings(cat)
                    _catalog = cat
                    logger.info("Catalog loaded (lazy fallback): %d items.", len(cat))
                except Exception:
                    logger.exception("Failed to load catalog; using empty.")
                    _catalog = []
    return _catalog


def _append_assistant(messages: list[ChatMessage], reply: str) -> list[dict]:
    serialised = [{"role": m.role.value, "content": m.content} for m in messages]
    serialised.append({"role": "assistant", "content": reply})
    return serialised  # type: ignore[return-value]


def _safe(
    reply: str,
    messages: list[ChatMessage] | None = None,
    recs=None,
    end: bool = False,
) -> ChatResponse:
    updated = _append_assistant(messages, reply) if messages else []
    return ChatResponse(
        reply=reply,
        recommendations=recs or [],
        end_of_conversation=end,
        messages=updated,
    )


def agent(messages: list[ChatMessage]) -> ChatResponse:
    """
    Stateless agent. Full conversation history arrives on every call.
    Never raises. Always returns a schema-valid ChatResponse.

    Change from original
    --------------------
    Step 3: passes ``messages`` to ``get_forced_mode`` so that the policy
    layer can count prior non-recommending turns and deterministically force
    ``"recommend"`` before the LLM is called.  This is the primary fix for
    the over-clarification bug.  Everything else is unchanged.
    """

    # ── 1. Input validation ──────────────────────────────────────────────
    if not messages:
        return _safe(
            "Hello! I'm the SHL Assessment Recommender. "
            "What role are you hiring for? I'll suggest the right assessments.",
            messages=[],
        )

    user_messages = [m for m in messages if m.role.value == "user"]
    if not user_messages:
        return _safe(
            "Please describe the role you're hiring for so I can recommend SHL assessments.",
            messages=messages,
        )

    if messages[-1].role.value != "user":
        return _safe(
            "I'm ready to help. What role are you assessing for, "
            "or would you like to refine the current recommendations?",
            messages=messages,
        )

    last_user: ChatMessage = messages[-1]

    # ── 2. Catalog + state ───────────────────────────────────────────────
    catalog = _get_catalog()
    state = extract_state_from_messages(messages)

    # ── 3. Deterministic safety gate ─────────────────────────────────────
    # CHANGED: pass `messages` so get_forced_mode can count non-recommending
    # turns and force "recommend" when the conversation has spent >= 1 turn
    # without a shortlist and state.has_enough_info() is True.
    forced_mode = get_forced_mode(state, last_user, messages)

    if forced_mode == "refuse":
        is_legal_mid = (
            len(user_messages) > 1
            and re.search(
                r"\b(legal\s+(?:require|complian|advi|risk)|labor\s+law"
                r"|employment\s+law|discrimination\s+law"
                r"|(?:hipaa|gdpr|eeoc|ada)\s+(?:law|regulat|legal|obligat|require)"
                r"|(?:legally|law)\s+(?:required?|mandated?))",
                last_user.content, re.IGNORECASE,
            )
        )
        if is_legal_mid:
            reply = (
                "I can't provide legal or compliance advice — please consult your legal team. "
                "I'm happy to continue recommending SHL assessments for this role."
            )
        else:
            reply = (
                "I can only assist with SHL assessment recommendations. "
                "I'm unable to help with legal questions, general hiring advice, or non-SHL assessments. "
                "Please describe the role you're assessing and I'll suggest the right tests."
            )
        return _safe(reply, messages=messages)

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

    # ── 5. Closing short-circuit ──────────────────────────────────────────
    if state.user_confirmed_final and forced_mode not in ("refuse", "compare"):
        closing_reply = "You're all set — good luck with the hiring process!"
        closing_recs = build_recommendations_from_scores(scored, top_n=10) if scored else []
        updated_messages = _append_assistant(messages, closing_reply)
        return validate_chat_response(
            ChatResponse(
                reply=closing_reply,
                recommendations=closing_recs,
                end_of_conversation=True,
                messages=updated_messages,
            )
        )

    # ── 6. Single LLM call ────────────────────────────────────────────────
    result = call_llm(
        messages=messages,
        state=state,
        candidates=candidates,
        forced_mode=forced_mode,
    )

    llm_mode: str = result.get("mode", "consult")
    reply: str = sanitise_reply(result.get("reply") or "Here are the SHL assessments I recommend.")
    chosen_names: list[str] = result.get("chosen_names") or []

    # ── 7. Grounding — only real catalog items may appear in output ───────
    recs = []
    if llm_mode in ("recommend", "refine"):
        if chosen_names:
            recs = build_recommendations_from_names(chosen_names, candidates, top_n=10)
        if not recs and scored:
            recs = build_recommendations_from_scores(scored, top_n=5)

    if llm_mode in ("clarify", "consult", "compare", "refuse"):
        recs = []

    # ── 8. End-of-conversation ────────────────────────────────────────────
    end = should_end_conversation(llm_mode, state)

    # ── 9. Build updated history for the client ───────────────────────────
    updated_messages = _append_assistant(messages, reply)

    return validate_chat_response(
        ChatResponse(
            reply=reply,
            recommendations=recs,
            end_of_conversation=end,
            messages=updated_messages,
        )
    )
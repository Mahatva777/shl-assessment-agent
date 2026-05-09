"""
Top-level agent orchestrator for the SHL Assessment Agent.
Core agent logic. Stateless: operates only on the provided message history.

Responsibilities:
    - Receive the conversation history from the ``/chat`` endpoint.
    - Orchestrate the pipeline: state extraction → retrieval → scoring →
      policy → LLM reply generation.
    - Return a ``ChatResponse`` to the caller.

"""

from __future__ import annotations

from app.schemas import ChatMessage, ChatResponse, Recommendation, CatalogItem
from app.state_extraction import extract_state_from_messages
from app.policy import decide_mode, pick_clarification_question, should_end_conversation
from app.retrieval import semantic_search
from app.llm_client import call_llm

# ---------------------------------------------------------------------------
# Test-type derivation
# ---------------------------------------------------------------------------

_KEY_PRIORITY: list[str] = [
    "Knowledge & Skills",
    "Ability & Aptitude",
    "Personality & Behavior",
    "Biodata & Situational Judgment",
    "Simulations",
]

_KEY_TO_CODE: dict[str, str] = {
    "Knowledge & Skills": "K",
    "Ability & Aptitude": "A",
    "Personality & Behavior": "P",
    "Biodata & Situational Judgment": "B",
    "Simulations": "S",
}


def derive_test_type_from_keys(keys: list[str]) -> str:
    """Return single-letter test type using deterministic priority K>A>P>B>S."""
    for key in _KEY_PRIORITY:
        if key in keys:
            return _KEY_TO_CODE[key]
    return "A"  # fallback


# ---------------------------------------------------------------------------
# Safe response helpers
# ---------------------------------------------------------------------------

def _clarify_response(reply: str) -> ChatResponse:
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)


def _refuse_response(reply: str) -> ChatResponse:
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)


def _build_recommendations(items: list[CatalogItem]) -> list[Recommendation]:
    return [
        Recommendation(
            name=item.name,
            url=item.url,
            test_type=derive_test_type_from_keys(item.keys),
        )
        for item in items[:10]  # hard cap at 10
    ]


# ---------------------------------------------------------------------------
# Main agent entry point
# ---------------------------------------------------------------------------

def agent(messages: list[ChatMessage]) -> ChatResponse:
    """
    Process a full stateless conversation history and return the next reply.
    This function is the only public interface; it must never raise.
    """

    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    if not messages:
        return _clarify_response(
            "Hello! I'm the SHL Assessment Recommender. "
            "Please describe the role you're hiring for and I'll suggest relevant assessments."
        )

    user_messages = [m for m in messages if m.role == "user"]
    if not user_messages:
        return _clarify_response(
            "Please describe the role you're hiring for so I can recommend appropriate SHL assessments."
        )

    if messages[-1].role != "user":
        return _clarify_response(
            "I'm ready to help. What role are you hiring for, or would you like to refine the current recommendations?"
        )

    last_user: str = messages[-1].content.strip()

    # ------------------------------------------------------------------
    # 2. Extract state and decide mode
    # ------------------------------------------------------------------
    state = extract_state_from_messages(messages)
    mode: str = decide_mode(state, last_user)

    # ------------------------------------------------------------------
    # 3. Mode dispatch
    # ------------------------------------------------------------------

    # --- REFUSE ---
    if mode == "refuse":
        return _refuse_response(
            "I can only help with SHL assessment recommendations. "
            "I'm not able to assist with general hiring advice, legal questions, "
            "or topics outside the SHL catalog. "
            "Please describe the role you're assessing and I'll be happy to help."
        )

    # --- CLARIFY ---
    if mode == "clarify":
        question = pick_clarification_question(state)
        return _clarify_response(question)

    # --- COMPARE ---
    if mode == "compare":
        targets: list[str] = state.compare_targets or []

        # Retrieve a pool of candidates broad enough to find all named items.
        query = " ".join(targets) if targets else last_user
        pool = semantic_search(query, top_k=20)

        # Prefer exact name matches; fall back to top pool items.
        pool_by_name: dict[str, CatalogItem] = {item.name.lower(): item for item in pool}
        candidates: list[CatalogItem] = []
        for t in targets:
            match = pool_by_name.get(t.lower())
            if match and match not in candidates:
                candidates.append(match)
        if not candidates:
            candidates = pool[:5]

        result = call_llm(mode="compare", messages=messages, state=state, candidates=candidates)
        reply = result.get("reply") or "Here is a comparison of those assessments based on the SHL catalog."
        end = should_end_conversation(mode, state)
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=end)

    # --- RECOMMEND / REFINE ---
    query = state.build_query_string()
    candidates = semantic_search(query, top_k=20)

    if not candidates:
        return _clarify_response(
            "I couldn't find assessments matching those exact constraints. "
            "Could you broaden the criteria — for example, by relaxing the language, "
            "seniority level, or duration requirements?"
        )

    result = call_llm(mode=mode, messages=messages, state=state, candidates=candidates)
    reply = result.get("reply") or "Here are the assessments I recommend based on your requirements."
    chosen_names: list[str] = result.get("chosen_names") or []

    # Map chosen names back to real catalog items only — no invented entries.
    name_to_item: dict[str, CatalogItem] = {item.name: item for item in candidates}
    chosen_items: list[CatalogItem] = [
        name_to_item[n] for n in chosen_names if n in name_to_item
    ]

    # Fallback: if LLM returned nothing usable, use top scored candidates.
    if not chosen_items:
        chosen_items = candidates[:5]

    recommendations = _build_recommendations(chosen_items)
    end = should_end_conversation(mode, state)
    return ChatResponse(reply=reply, recommendations=recommendations, end_of_conversation=end)

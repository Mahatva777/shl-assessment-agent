# app/policy.py
from __future__ import annotations

import re

from app.catalog_loader import CatalogItem
from app.schemas import ChatMessage
from app.state_extraction import ConversationState

# ---------------------------------------------------------------------------
# Refusal patterns
# ---------------------------------------------------------------------------

_REFUSE_RE = re.compile(
    r"\b(legal\s+(?:require|complian|advi|risk|liability)"
    r"|labor\s+law|employment\s+law|discrimination\s+law"
    r"|gdpr|hipaa|eeoc|ada\s+complian"
    r"|how\s+(?:to|should\s+(?:i|we))\s+(?:hire|recruit|interview|fire|terminate)"
    r"|hiring\s+(?:strategy|process|best\s+practice|tip|guide)"
    r"|salary\s+negotiat|compensation\s+structure|offer\s+letter"
    r"|background\s+check\s+(?:legal|law))\b",
    re.IGNORECASE,
)

_INJECTION_RE = re.compile(
    r"(ignore\s+(?:all\s+)?(?:prior|previous|above)\s+(?:instruction|prompt|rule)"
    r"|forget\s+(?:your|all)\s+(?:instruction|rule|prompt)"
    r"|you\s+are\s+now\s+(?:a|an)\s+(?!shl)"
    r"|disregard\s+(?:your|all|the)\s+(?:system|instruction)"
    r"|act\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?(?!an\s+shl))",
    re.IGNORECASE,
)

_NON_SHL_RE = re.compile(
    r"\b(disc\s+assessment|myers[- ]?briggs|mbti|gallup\s+strengths"
    r"|strengthsfinder|hogan\s+assessment|caliper|predictive\s+index"
    r"|wonderlic|kolbe|enneagram)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Compare detection
# ---------------------------------------------------------------------------

_COMPARE_RE = re.compile(
    r"\b(differ(?:ence|ent|s)?(?:\s+between)?|compar(?:e|ing|ison)"
    r"|versus|vs\.?|how\s+(?:does|do|is|are)\s+\S+\s+(?:differ|compare))\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Refine detection
# ---------------------------------------------------------------------------

_REFINE_RE = re.compile(
    r"\b(actually|also\s+add|add\s+(?:personality|cognitive|sjt|simulation)"
    r"|remove|drop|instead|replace|swap|switch|exclude|don'?t\s+include"
    r"|without|no\s+(?:personality|cognitive|opq|verify|sjt)"
    r"|narrow\s+down|more\s+(?:focused|specific)|fewer|shorten"
    r"|can\s+you\s+also|what\s+about\s+adding|include\s+(?:also|too)"
    r"|change\s+(?:the|my)|adjust|modify|update\s+(?:the|my))\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Retrieval confidence thresholds (Task 4)
# ---------------------------------------------------------------------------

# If top candidate score exceeds this, retrieval is considered strong.
_STRONG_RETRIEVAL_THRESHOLD = 0.62

# If recommendation_confidence() >= this, skip clarification.
_CONFIDENCE_RECOMMEND_THRESHOLD = 0.45


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decide_mode(
    state: ConversationState,
    last_user: ChatMessage,
    top_candidates: list[tuple[float, CatalogItem]] | None = None,
) -> str:
    """Determine the agent's operating mode for this turn.

    Priority: refuse > compare > refine > [retrieval-aware] clarify > recommend.

    `top_candidates` is optional scored retrieval output used to make the
    clarify/recommend decision retrieval-aware (Task 4).
    """
    text: str = last_user.content

    # ── Refuse ────────────────────────────────────────────────────────
    if (
        state.off_topic
        or _REFUSE_RE.search(text)
        or _INJECTION_RE.search(text)
        or _NON_SHL_RE.search(text)
    ):
        return "refuse"

    # ── Compare ───────────────────────────────────────────────────────
    if _COMPARE_RE.search(text) and len(state.compare_targets) >= 2:
        return "compare"

    # ── Refine ────────────────────────────────────────────────────────
    if _REFINE_RE.search(text) and state.has_enough_info():
        return "refine"

    # ── Clarify vs Recommend (retrieval-aware) ────────────────────────
    if _should_recommend(state, top_candidates):
        return "recommend"

    return "clarify"


def _should_recommend(
    state: ConversationState,
    top_candidates: list[tuple[float, CatalogItem]] | None,
) -> bool:
    """Return True when we have enough signal to recommend without clarifying.

    Decision layers (Task 1 + Task 4):
    1. If state confidence is high enough, recommend.
    2. If retrieval is strong (top score well above baseline), recommend.
    3. Otherwise defer to clarify.
    """
    # Classic sufficient-info gate (covers most cases)
    if state.has_enough_info():
        return True

    # Soft confidence gate — tolerate partial ambiguity (Task 1)
    conf = state.recommendation_confidence()
    if conf >= _CONFIDENCE_RECOMMEND_THRESHOLD:
        return True

    # Retrieval gate — if candidates strongly converge, don't clarify (Task 4)
    if top_candidates and len(top_candidates) >= 3:
        top_score = top_candidates[0][0]
        if top_score >= _STRONG_RETRIEVAL_THRESHOLD:
            return True

    return False


def should_end_conversation(mode: str, state: ConversationState) -> bool:
    if mode in ("clarify", "refine", "compare"):
        return False
    if mode == "refuse":
        return state.user_confirmed_final
    return state.user_confirmed_final


def pick_clarification_question(state: ConversationState) -> str:
    """Return the single highest-information-gain clarification question (Task 5).

    Adapts to already-known context; avoids generic or repeated prompts.
    """
    # ── No role or domain at all ───────────────────────────────────────
    if not state.role_title and len(state.domain_keywords) < 2:
        return (
            "Could you tell me what role or job family you're assessing for? "
            "For example, 'software engineer', 'sales representative', "
            "or paste a short job description."
        )

    is_leadership = _is_leadership_context(state)
    is_graduate = _is_graduate_context(state)

    # ── Known leadership context but use-case unclear ──────────────────
    if is_leadership and not state.use_case:
        return (
            "For these senior leadership roles, is the primary goal early "
            "screening against leadership norms, or in-depth benchmarking "
            "for final selection?"
        )

    # ── Known graduate context but screening vs finalist unclear ──────
    if is_graduate and not state.use_case:
        return (
            "Is this for high-volume graduate filtering at the initial "
            "screening stage, or for deeper finalist evaluation later on?"
        )

    # ── Role known, seniority still missing ───────────────────────────
    if not state.seniority_text:
        role_hint = f" for the {state.role_title}" if state.role_title else ""
        return (
            f"What experience level are you targeting{role_hint}? "
            "For example: graduate/entry-level, mid-level, manager, or director."
        )

    # ── Role + seniority known, but no test preference at all ─────────
    has_test_pref = (
        state.wants_personality or state.wants_cognitive
        or state.wants_sjt or state.wants_simulation
    )
    if not has_test_pref and not state.use_case:
        seniority = state.seniority_text or "this level"
        return (
            f"For {seniority} hires, would you like to focus on cognitive "
            "ability tests, a personality questionnaire, or a broader battery "
            "combining both?"
        )

    # ── Language not specified and might matter ────────────────────────
    if not state.language_required:
        return (
            "Does the assessment need to be available in a specific language, "
            "or is English sufficient?"
        )

    # ── Generic fallback (should rarely be reached) ───────────────────
    return (
        "Is there anything else to consider — for example, a maximum "
        "test duration or a requirement for remote administration?"
    )


# ---------------------------------------------------------------------------
# Contextual helpers
# ---------------------------------------------------------------------------

def _is_leadership_context(state: ConversationState) -> bool:
    leadership_seniority = {"executive", "director", "cxo", "ceo", "cfo", "cto"}
    if state.seniority_text and any(
        s in state.seniority_text.lower() for s in leadership_seniority
    ):
        return True
    if state.use_case in ("leadership", "benchmarking"):
        return True
    leadership_kw = {"leadership", "executive", "cxo", "director", "vp", "svp", "evp"}
    return bool(leadership_kw.intersection(set(state.domain_keywords)))


def _is_graduate_context(state: ConversationState) -> bool:
    if state.seniority_text and state.seniority_text.lower() in (
        "graduate", "intern", "trainee", "fresh", "fresher"
    ):
        return True
    if state.use_case == "graduate_hiring":
        return True
    grad_kw = {"graduate", "intern", "trainee", "campus", "fresher", "entry"}
    return bool(grad_kw.intersection(set(state.domain_keywords)))
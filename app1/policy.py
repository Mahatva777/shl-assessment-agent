# app/policy.py
from __future__ import annotations

import re

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
# Public API
# ---------------------------------------------------------------------------

def decide_mode(state: ConversationState, last_user: ChatMessage) -> str:
    """
    Determine the agent's operating mode for this turn.
    Priority: refuse > compare > refine > clarify > recommend.
    """
    text: str = last_user.content

    if state.off_topic or _REFUSE_RE.search(text) or _INJECTION_RE.search(text) or _NON_SHL_RE.search(text):
        return "refuse"

    if _COMPARE_RE.search(text) and len(state.compare_targets) >= 2:
        return "compare"

    if _REFINE_RE.search(text) and state.has_enough_info():
        return "refine"

    if not state.has_enough_info():
        return "clarify"

    return "recommend"


def should_end_conversation(mode: str, state: ConversationState) -> bool:
    """
    Return True when the conversation should be marked complete.
    Conservative default is False.
    """
    if mode in ("clarify", "refine", "compare"):
        return False
    if mode == "refuse":
        return state.user_confirmed_final
    # recommend
    return state.user_confirmed_final


def pick_clarification_question(state: ConversationState) -> str:
    """Return the single most useful clarification question given current state."""
    if not state.role_title and len(state.domain_keywords) < 2:
        return (
            "Could you tell me what role or job family you're hiring for? "
            "For example, 'software engineer', 'sales representative', "
            "or paste a short job description."
        )

    if not state.seniority_text:
        return (
            "What seniority level is this role? For example: "
            "graduate/entry-level, mid-level, manager, or executive."
        )

    has_test_pref = (
        state.wants_personality
        or state.wants_cognitive
        or state.wants_sjt
        or state.wants_simulation
    )
    if not has_test_pref:
        return (
            "What type of assessment are you looking for? For example: "
            "cognitive/aptitude tests, personality questionnaires, "
            "situational judgment tests, or job simulations."
        )

    if not state.language_required:
        return (
            "Does the assessment need to be in a specific language, "
            "or is English fine?"
        )

    return (
        "Is there anything else to consider, such as a maximum test duration "
        "or a requirement for remote administration?"
    )
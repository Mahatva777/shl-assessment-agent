# app/policy.py
"""
Lightweight conversation policy.

Deterministic code handles ONLY:
  - refusal (safety, legal, adversarial, non-SHL)
  - compare detection
  - explicit refinement detection

The LLM owns clarify / consult / recommend / refine decisions.
"""
from __future__ import annotations
import re
from app.schemas import ChatMessage
from app.state_extraction import ConversationState

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
_COMPARE_RE = re.compile(
    r"\b(differ(?:ence|ent|s)?(?:\s+between)?|compar(?:e|ing|ison)"
    r"|versus|vs\.?|how\s+(?:does|do|is|are)\s+\S+\s+(?:differ|compare))\b",
    re.IGNORECASE,
)
_REFINE_RE = re.compile(
    r"\b(actually|also\s+add|add\s+(?:personality|cognitive|sjt|simulation)"
    r"|remove|drop|instead|replace|swap|switch|exclude|don'?t\s+include"
    r"|without|no\s+(?:personality|cognitive|opq|verify|sjt)"
    r"|narrow\s+down|more\s+(?:focused|specific)|fewer|shorten"
    r"|can\s+you\s+also|what\s+about\s+adding|include\s+(?:also|too)"
    r"|change\s+(?:the|my)|adjust|modify)\b",
    re.IGNORECASE,
)


def get_forced_mode(state: ConversationState, last_user: ChatMessage) -> str | None:
    """Return forced mode if deterministic rules apply; None means LLM decides."""
    text = last_user.content
    if state.off_topic or _REFUSE_RE.search(text) or _INJECTION_RE.search(text) or _NON_SHL_RE.search(text):
        return "refuse"
    if _COMPARE_RE.search(text) and len(state.compare_targets) >= 2:
        return "compare"
    if _REFINE_RE.search(text) and state.has_enough_info():
        return "refine"
    return None


def decide_mode(
    state: ConversationState,
    last_user: ChatMessage,
    top_candidates: list | None = None,
) -> str:
    """Backward-compatible wrapper used by tests and legacy callers."""
    forced = get_forced_mode(state, last_user)
    if forced:
        return forced
    if state.has_enough_info():
        return "recommend"
    return "clarify"


def should_end_conversation(mode: str, state: ConversationState) -> bool:
    if mode in ("clarify", "consult", "compare"):
        return False
    if mode == "refuse":
        return state.user_confirmed_final
    return state.user_confirmed_final


def pick_clarification_question(state: ConversationState) -> str:
    """Fallback only — in normal operation the LLM writes the clarification."""
    if not state.role_title and len(state.domain_keywords) < 2:
        return "Could you describe the role you're assessing for? A job title or short JD works well."
    if not state.seniority_text:
        return "What seniority level is this role — graduate, mid-level, manager, or executive?"
    return "What type of assessment is the priority — cognitive, personality, situational judgment, or simulation?"
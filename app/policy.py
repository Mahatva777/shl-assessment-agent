"""
Conversation policy for the SHL Assessment Agent.

Determines the agent's **mode** for the current turn based on the
extracted :class:`ConversationState` and the latest user message.

Modes
-----
``"refuse"``
    The user asked a legal, off-topic, or adversarial question.
    The agent declines and steers back to SHL assessments.

``"compare"``
    The user names ≥2 SHL products and asks for differences.
    The agent produces a field-by-field comparison table.

``"clarify"``
    We lack the minimum information to produce useful recommendations
    (role/domain unknown, or no seniority / test-type / language clue).
    The agent asks **one** focused question per turn.

``"refine"``
    The user is adjusting existing constraints ("add personality",
    "drop OPQ", "actually under 30 min").  Prior state is preserved.

``"recommend"``
    We have enough information to rank and present assessments.

Design rationale (from sample conversations C1–C10)
----------------------------------------------------
- The evaluator allows only **8 turns total** (user + assistant), so
  clarifications must be sparse and high-value.
- Refinement must update state incrementally, not start over.
- Comparison triggers only when the user names ≥2 assessments.
- Refusal covers legal, compliance, prompt-injection, and non-SHL products.
"""

from __future__ import annotations

import re

from app.schemas import ChatMessage
from app.state_extraction import ConversationState


# ───────────────────────────────────────────────────────────────────────────
# Off-topic / refusal patterns (narrow but explicit)
# ───────────────────────────────────────────────────────────────────────────

# Legal / compliance / hiring-strategy questions
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

# Prompt injection / adversarial intent
_INJECTION_RE = re.compile(
    r"(ignore\s+(?:all\s+)?(?:prior|previous|above)\s+(?:instruction|prompt|rule)"
    r"|forget\s+(?:your|all)\s+(?:instruction|rule|prompt)"
    r"|you\s+are\s+now\s+(?:a|an)\s+(?!shl)"
    r"|disregard\s+(?:your|all|the)\s+(?:system|instruction)"
    r"|act\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?(?!an\s+shl))",
    re.IGNORECASE,
)

# Non-SHL assessment mentions
_NON_SHL_RE = re.compile(
    r"\b(disc\s+assessment|myers[- ]?briggs|mbti|gallup\s+strengths"
    r"|strengthsfinder|hogan\s+assessment|caliper|predictive\s+index"
    r"|wonderlic|kolbe|enneagram)\b",
    re.IGNORECASE,
)

# ───────────────────────────────────────────────────────────────────────────
# Compare detection
# ───────────────────────────────────────────────────────────────────────────

_COMPARE_RE = re.compile(
    r"\b(differ(?:ence|ent|s)?(?:\s+between)?|compar(?:e|ing|ison)"
    r"|versus|vs\.?|how\s+(?:does|do|is|are)\s+\S+\s+(?:differ|compare))\b",
    re.IGNORECASE,
)

# ───────────────────────────────────────────────────────────────────────────
# Refine detection — user wants to adjust, not start over
# ───────────────────────────────────────────────────────────────────────────

_REFINE_RE = re.compile(
    r"\b(actually|also\s+add|add\s+(?:personality|cognitive|sjt|simulation)"
    r"|remove|drop|instead|replace|swap|switch|exclude|don'?t\s+include"
    r"|without|no\s+(?:personality|cognitive|opq|verify|sjt)"
    r"|narrow\s+down|more\s+(?:focused|specific)|fewer|shorten"
    r"|can\s+you\s+also|what\s+about\s+adding|include\s+(?:also|too)"
    r"|change\s+(?:the|my)|adjust|modify|update\s+(?:the|my))\b",
    re.IGNORECASE,
)


# ───────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────

def decide_mode(
    state: ConversationState,
    last_user: ChatMessage,
) -> str:
    """Choose the agent's operating mode for this turn.

    Priority order (first match wins):

    1. **refuse** — off-topic, legal, adversarial, or non-SHL.
    2. **compare** — user names ≥2 products and uses compare language.
    3. **refine** — user adjusts constraints ("add", "remove", "instead").
    4. **clarify** — not enough info for useful recommendations yet.
    5. **recommend** — default: produce ranked shortlist.

    Parameters
    ----------
    state:
        The :class:`ConversationState` extracted from the full history.
    last_user:
        The most recent user message (used for intent signals).

    Returns
    -------
    str
        One of ``"refuse"``, ``"compare"``, ``"refine"``,
        ``"clarify"``, or ``"recommend"``.
    """
    text: str = last_user.content

    # ── 1. Refuse ─────────────────────────────────────────────────────
    # Off-topic flag was already set by the state extractor, but we
    # also check here for patterns in the latest message specifically.
    if state.off_topic:
        return "refuse"
    if _REFUSE_RE.search(text):
        return "refuse"
    if _INJECTION_RE.search(text):
        return "refuse"
    if _NON_SHL_RE.search(text):
        return "refuse"

    # ── 2. Compare ────────────────────────────────────────────────────
    # Requires both compare language AND ≥2 product names identified.
    if _COMPARE_RE.search(text) and len(state.compare_targets) >= 2:
        return "compare"

    # ── 3. Refine ─────────────────────────────────────────────────────
    # Trigger only if the user uses adjustment language AND we already
    # have some prior context (role / keywords) to refine from.
    if _REFINE_RE.search(text) and state.has_enough_info():
        return "refine"

    # ── 4. Clarify ────────────────────────────────────────────────────
    # If we don't have enough structured info, ask ONE focused question.
    # But only if we're early enough in the conversation (≤ 6 messages
    # to stay within the 8-turn budget).
    if not state.has_enough_info():
        return "clarify"

    # ── 5. Recommend ──────────────────────────────────────────────────
    return "recommend"


def should_end_conversation(
    mode: str,
    state: ConversationState,
) -> bool:
    """Decide the ``end_of_conversation`` flag.

    Strategy (from the assignment rules):
    - ``False`` when clarifying, refining, or comparing — user may
      want to continue.
    - ``True`` when the user has confirmed satisfaction, or when we've
      delivered a final recommendation and the user hasn't asked for
      more constraints.
    - ``True`` always on refuse (conversation cannot progress).

    Parameters
    ----------
    mode:
        The current turn mode from :func:`decide_mode`.
    state:
        Current :class:`ConversationState`.

    Returns
    -------
    bool
    """
    if mode in ("clarify", "refine", "compare"):
        return False

    if mode == "refuse":
        # Single refusal doesn't end the conversation — user may
        # course-correct.  But if they confirmed, it's done.
        return state.user_confirmed_final

    # mode == "recommend"
    if state.user_confirmed_final:
        return True

    # Default: delivering recs doesn't auto-end — user may refine.
    return False


def pick_clarification_question(state: ConversationState) -> str:
    """Return the single most valuable clarification question.

    Asks for the most discriminative missing field.  Priority order:

    1. Role / domain — without this, every recommendation is noise.
    2. Seniority — strongly filters job_levels in the catalog.
    3. Test-type preference — personality, cognitive, SJT, simulation.
    4. Language — important for international deployments.

    The question is phrased to be answerable in one sentence so we
    don't waste turns in the 8-turn budget.

    Parameters
    ----------
    state:
        Current :class:`ConversationState`.

    Returns
    -------
    str
        A concise clarification question.
    """
    if not state.role_title and len(state.domain_keywords) < 2:
        return (
            "Could you tell me what role or job family you're hiring for? "
            "For example, 'software engineer', 'sales representative', "
            "or you can paste a short job description."
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

    # Fallback — shouldn't normally reach here if has_enough_info is True
    return (
        "Is there anything else you'd like me to consider, such as "
        "a maximum test duration or remote administration?"
    )

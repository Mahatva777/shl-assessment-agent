"""
State extraction for the SHL Assessment Agent.

Scans the **full** conversation history (not just the last message) and
fills a structured :class:`ConversationState` using keyword / regex
heuristics.  The state drives the policy layer, which decides whether to
clarify, recommend, refine, compare, or refuse.

Design notes (derived from sample conversations C1-C10):
    - C1/C3/C9: Clarify seniority or language before recommending.
    - C4/C8/C9/C10: Refine when user adds/removes constraints.
    - C3/C5/C6: Compare when user names ≥2 products or asks for differences.
    - C7: Refuse legal / general hiring-strategy questions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.schemas import ChatMessage

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Conversation state
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class ConversationState:
    """Structured snapshot of everything we know from the conversation.

    Fields are progressively filled as the agent gathers information
    across multiple chat turns.  Any field that is still unknown is
    ``None`` (scalars), ``False`` (booleans), or empty (lists).
    """

    # ── Role / domain ──────────────────────────────────────────────────
    role_title: Optional[str] = None
    """Free-text job title, e.g. ``"Python Backend Engineer"``."""

    domain_keywords: list[str] = field(default_factory=list)
    """Domain / skill keywords, e.g. ``["python", "backend", "REST"]``."""

    # ── Seniority ──────────────────────────────────────────────────────
    seniority_text: Optional[str] = None
    """Raw seniority phrase, e.g. ``"junior"``, ``"mid-level"``."""

    # ── Language ───────────────────────────────────────────────────────
    language_required: Optional[str] = None
    """Required assessment language, e.g. ``"English"``, ``"Spanish"``."""

    # ── Test-type preference booleans ──────────────────────────────────
    wants_personality: bool = False
    """User explicitly wants personality / behavior assessments."""

    wants_cognitive: bool = False
    """User explicitly wants cognitive / aptitude / reasoning tests."""

    wants_sjt: bool = False
    """User explicitly wants Situational Judgment Tests (SJT)."""

    wants_simulation: bool = False
    """User explicitly wants simulation-based assessments."""

    # ── Remote / proctoring ────────────────────────────────────────────
    wants_remote: bool = False
    """User explicitly wants remote-proctored assessments."""

    # ── Duration ───────────────────────────────────────────────────────
    duration_budget: Optional[int] = None
    """Maximum acceptable assessment duration in minutes."""

    # ── Comparison ─────────────────────────────────────────────────────
    compare_targets: list[str] = field(default_factory=list)
    """Product names the user wants to compare (≥2 triggers compare mode)."""

    # ── Conversation-level flags ───────────────────────────────────────
    user_confirmed_final: bool = False
    """User explicitly said the recommendations are good / enough."""

    off_topic: bool = False
    """User is asking something outside the SHL assessment domain."""

    # ── Derived helpers ────────────────────────────────────────────────

    @property
    def desired_keys(self) -> list[str]:
        """Map the boolean test-type flags to SHL catalog key strings.

        This keeps backward-compatibility with ``score_candidate()``
        while the user-facing interface uses simple booleans.
        """
        keys: list[str] = []
        if self.wants_personality:
            keys.append("Personality & Behavior")
        if self.wants_cognitive:
            keys.append("Ability & Aptitude")
        if self.wants_sjt:
            keys.append("Biodata & Situational Judgment")
        if self.wants_simulation:
            keys.append("Simulations")
        return keys

    def build_query_string(self) -> str:
        """Combine known fields into a single lowercased string for
        embedding-based search.
        """
        parts: list[str] = []
        if self.role_title:
            parts.append(self.role_title)
        if self.domain_keywords:
            parts.append(" ".join(self.domain_keywords))
        if self.seniority_text:
            parts.append(self.seniority_text)
        if self.language_required:
            parts.append(self.language_required)
        if self.desired_keys:
            parts.append(" ".join(self.desired_keys))
        return " ".join(parts).lower()

    def has_enough_info(self) -> bool:
        """Return ``True`` when we have enough to produce useful recs.

        Modeled after the sample conversations C1-C10: the agent should
        recommend only when it knows at least:
            1. The role family OR ≥2 domain keywords (what job is this for?)
            2. At least one of: seniority, a test-type preference, or a
               duration budget (what *kind* of assessment is needed?)

        This prevents noisy early recommendations while keeping the bar
        low enough that specific queries ("I need a Python test for
        graduates under 30 min") pass on the first turn.
        """
        has_role = bool(self.role_title) or len(self.domain_keywords) >= 2
        has_constraint = bool(
            self.seniority_text
            or self.wants_personality
            or self.wants_cognitive
            or self.wants_sjt
            or self.wants_simulation
            or self.duration_budget is not None
            or self.language_required
        )
        return has_role and has_constraint


# ───────────────────────────────────────────────────────────────────────────
# Regex patterns for keyword extraction
# ───────────────────────────────────────────────────────────────────────────

# Seniority — matches phrases like "graduate", "entry-level", "CXO", etc.
_SENIORITY_RE = re.compile(
    r"\b(junior|senior|mid[- ]?level|entry[- ]?level|graduate|intern|trainee"
    r"|manager|director|executive|cxo|ceo|cfo|cto|supervisor|lead"
    r"|experienced|beginner|fresher|fresh)\b",
    re.IGNORECASE,
)

# Duration — "30 min", "under 20 minutes", etc.
_DURATION_EXPLICIT_RE = re.compile(
    r"\b(\d{1,3})\s*(?:min(?:ute)?s?|mins?)\b",
    re.IGNORECASE,
)

# Duration — indirect phrases that imply a short test
_QUICK_SCREEN_RE = re.compile(
    r"\b(quickly?\s+screen|short\s+test|brief\s+assessment|fast\s+screen"
    r"|quick\s+assessment|short\s+assessment|rapid\s+screen)\b",
    re.IGNORECASE,
)

# Remote
_REMOTE_RE = re.compile(
    r"\b(remote|online|virtual|proctored\s+remotely|unproctored)\b",
    re.IGNORECASE,
)

# Language — common assessment languages
_LANGUAGE_RE = re.compile(
    r"\b(?:in\s+|spoken\s+)?"
    r"(english(?:\s*\(?\s*(?:us|usa|uk|international)\s*\)?)?|spanish|french"
    r"|german|portuguese|chinese|japanese|dutch|arabic|hindi|mandarin)\b",
    re.IGNORECASE,
)

# Personality / behavior
_PERSONALITY_RE = re.compile(
    r"\b(personality|behavio(?:u?r)|opq|occupational\s+personality"
    r"|motivation\s+questionnaire|mq\b|big\s*five|trait|temperament)\b",
    re.IGNORECASE,
)

# Cognitive / aptitude / reasoning
_COGNITIVE_RE = re.compile(
    r"\b(cognitive|aptitude|reasoning|numerical|verbal|inductive"
    r"|deductive|abstract|ability\s+test|verify|g\+|general\s+ability"
    r"|logical|analytical|problem[- ]?solving)\b",
    re.IGNORECASE,
)

# Situational Judgment Tests
_SJT_RE = re.compile(
    r"\b(sjt|situational\s+judg(?:e?ment)|biodata|judgment\s+test)\b",
    re.IGNORECASE,
)

# Simulations
_SIMULATION_RE = re.compile(
    r"\b(simulation|role[- ]?play|inbox|in[- ]?tray|assessment\s+exercise"
    r"|case\s+study|group\s+exercise|presentation\s+exercise)\b",
    re.IGNORECASE,
)

# Off-topic / refusal patterns — legal, compliance, general hiring strategy
_OFF_TOPIC_RE = re.compile(
    r"\b(legal\s+(?:require|complian|advi|risk|liability)|labor\s+law"
    r"|employment\s+law|discrimination\s+law|gdpr|hipaa|eeoc|ada\s+complian"
    r"|how\s+(?:to|should\s+(?:i|we))\s+(?:hire|recruit|interview|fire|terminate)"
    r"|hiring\s+(?:strategy|process|best\s+practice|tip|guide)"
    r"|salary\s+negotiat|compensation\s+structure|offer\s+letter"
    r"|background\s+check\s+(?:legal|law))",
    re.IGNORECASE,
)

# Prompt injection / adversarial patterns
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

# Compare / difference request
_COMPARE_RE = re.compile(
    r"\b(differ(?:ence|ent|s)?(?:\s+between)?|compar(?:e|ing|ison)"
    r"|versus|vs\.?|how\s+(?:does|do|is|are)\s+\S+\s+(?:differ|compare))\b",
    re.IGNORECASE,
)

# Refinement cues — user wants to adjust, not start over
_REFINE_RE = re.compile(
    r"\b(actually|also\s+add|add\s+(?:personality|cognitive|sjt|simulation)"
    r"|remove|drop|instead|replace|swap|switch|exclude|don'?t\s+include"
    r"|without|no\s+(?:personality|cognitive|opq|verify|sjt)"
    r"|narrow\s+down|more\s+(?:focused|specific)|fewer|shorten"
    r"|can\s+you\s+also|what\s+about\s+adding|include\s+(?:also|too))\b",
    re.IGNORECASE,
)

# Confirmation that user is satisfied
_CONFIRMED_RE = re.compile(
    r"\b((?:that(?:'s|\s+is)\s+)?(?:perfect|great|good|enough|fine|excellent)"
    r"|thanks?,?\s+(?:that|this)\s+(?:is\s+)?(?:helpful|what\s+i\s+needed)"
    r"|(?:i(?:'m|\s+am)\s+)?(?:satisfied|happy)\s+with"
    r"|no\s+(?:more|further)\s+(?:questions?|changes?|refinements?)"
    r"|looks?\s+good|that\s+works|we(?:'re|\s+are)\s+good)\b",
    re.IGNORECASE,
)

# Stopwords for domain-keyword extraction
_STOPWORDS: set[str] = {
    "the", "and", "for", "with", "that", "this", "are", "was", "were",
    "can", "you", "need", "want", "looking", "find", "test", "tests",
    "assessment", "assessments", "please", "would", "like", "could",
    "about", "from", "have", "has", "our", "their", "some", "your",
    "should", "also", "any", "will", "not", "but", "than", "more",
    "into", "been", "being", "what", "which", "who", "how", "why",
    "remote", "online", "minutes", "min", "mins", "minute",
    "use", "using", "recommend", "suggestion", "suggest", "help",
    "good", "best", "right", "suitable", "appropriate", "there",
    "does", "don", "don't", "doesn", "doesn't", "didn", "didn't",
    "its", "they", "them", "then", "when", "where", "here",
    "just", "very", "really", "actually", "add", "remove", "drop",
    "shl", "product", "products", "catalog", "catalogue",
}


# ───────────────────────────────────────────────────────────────────────────
# Core extraction function
# ───────────────────────────────────────────────────────────────────────────

def extract_state_from_messages(
    messages: list[ChatMessage],
) -> ConversationState:
    """Extract structured hiring constraints from **all** user messages.

    This scans every user turn in the conversation (not just the last
    message) and applies keyword / regex heuristics to fill each field
    of :class:`ConversationState`.

    Heuristics
    ----------
    - **Seniority**: "graduate", "junior", "entry-level", "mid-level",
      "senior", "manager", "director", "CXO", etc.
    - **Duration**: explicit numbers like "30 min" or indirect phrases
      like "quickly screen" → defaults to 20 min budget.
    - **Language**: "English", "English (US)", "Spanish", etc.
    - **Test types**: each category has its own regex:
        - personality → "personality", "OPQ", "behavior", "trait"
        - cognitive → "aptitude", "reasoning", "Verify", "numerical"
        - SJT → "situational judgment", "SJT", "biodata"
        - simulation → "simulation", "role-play", "inbox", "in-tray"
    - **Remote**: "remote", "online", "virtual"
    - **Off-topic**: legal questions, hiring strategy, prompt injection,
      non-SHL assessments
    - **Compare**: "difference between", "compare", "vs"
    - **Confirmed**: "that's perfect", "looks good", "no more changes"

    Parameters
    ----------
    messages:
        Full conversation history (both user and assistant turns).

    Returns
    -------
    ConversationState
    """
    state = ConversationState()

    # Collect all user text for cumulative scanning
    user_texts: list[str] = [
        m.content for m in messages if m.role.value == "user"
    ]
    all_user_text: str = " ".join(user_texts)
    last_user_text: str = user_texts[-1] if user_texts else ""

    # ── Off-topic / refusal detection ─────────────────────────────────
    # (check last message primarily — earlier context may have been
    #  on-topic before the user went off-track)
    if _OFF_TOPIC_RE.search(last_user_text):
        state.off_topic = True
    if _INJECTION_RE.search(last_user_text):
        state.off_topic = True
    if _NON_SHL_RE.search(last_user_text):
        state.off_topic = True

    # ── Seniority ─────────────────────────────────────────────────────
    sen_match = _SENIORITY_RE.search(all_user_text)
    if sen_match:
        state.seniority_text = sen_match.group(1).lower()

    # ── Duration ──────────────────────────────────────────────────────
    dur_match = _DURATION_EXPLICIT_RE.search(all_user_text)
    if dur_match:
        state.duration_budget = int(dur_match.group(1))
    elif _QUICK_SCREEN_RE.search(all_user_text):
        # "quickly screen" implies a short test — default to 20 min
        state.duration_budget = 20

    # ── Language ──────────────────────────────────────────────────────
    lang_match = _LANGUAGE_RE.search(all_user_text)
    if lang_match:
        raw_lang: str = lang_match.group(1).strip()
        # Normalize "english (us)" → "English (USA)" etc.
        state.language_required = _normalize_language(raw_lang)

    # ── Remote ────────────────────────────────────────────────────────
    if _REMOTE_RE.search(all_user_text):
        state.wants_remote = True

    # ── Test-type booleans ────────────────────────────────────────────
    if _PERSONALITY_RE.search(all_user_text):
        state.wants_personality = True
    if _COGNITIVE_RE.search(all_user_text):
        state.wants_cognitive = True
    if _SJT_RE.search(all_user_text):
        state.wants_sjt = True
    if _SIMULATION_RE.search(all_user_text):
        state.wants_simulation = True

    # "spoken English" often implies simulation preference (C9 pattern)
    if re.search(r"\bspoken\s+english\b", all_user_text, re.IGNORECASE):
        state.wants_simulation = True

    # ── Compare targets ───────────────────────────────────────────────
    if _COMPARE_RE.search(last_user_text):
        state.compare_targets = _extract_product_names(last_user_text)

    # ── Confirmation ──────────────────────────────────────────────────
    if _CONFIRMED_RE.search(last_user_text):
        state.user_confirmed_final = True

    # ── Domain keywords ───────────────────────────────────────────────
    words = re.findall(r"[a-zA-Z+#.]{2,}", all_user_text.lower())
    state.domain_keywords = [
        w for w in dict.fromkeys(words) if w not in _STOPWORDS
    ][:15]

    # ── Role title ────────────────────────────────────────────────────
    # Use the first user message if it's short enough to be a role query
    if user_texts:
        first_msg = user_texts[0].strip()
        # If the first message is compact (≤25 words) and looks like a
        # role query, treat it as a role title / job description snippet
        if len(first_msg.split()) <= 25:
            state.role_title = first_msg

    # ── Handle refinement negations ───────────────────────────────────
    # If user says "remove personality" / "no OPQ" / "drop personality",
    # toggle the flag back off.  This must run *after* the cumulative
    # scan so that "add personality … actually remove personality" works.
    _apply_negations(state, last_user_text)

    return state


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def _normalize_language(raw: str) -> str:
    """Normalize a language match to a consistent form."""
    low = raw.lower().strip()
    if low.startswith("english"):
        if "us" in low:
            return "English (USA)"
        if "uk" in low:
            return "English (UK)"
        if "international" in low:
            return "English (International)"
        return "English"
    return raw.strip().title()


def _extract_product_names(text: str) -> list[str]:
    """Extract candidate SHL product names from a compare / vs request.

    Uses heuristics: quoted strings, or capitalized multi-word phrases
    that look like product names (e.g. "OPQ32r", "Verify G+").
    """
    names: list[str] = []

    # Quoted names first: "OPQ32r" or 'Verify G+'
    quoted = re.findall(r'["\']([^"\']{3,})["\']', text)
    names.extend(quoted)

    # Known SHL product prefixes
    known = re.findall(
        r"\b(OPQ\s*\S*(?:\s+\S+){0,3}|Verify\s+\S+(?:\s+\S+){0,2}"
        r"|MQ\s+\S+(?:\s+\S+){0,2}|SJT\s*\S*"
        r"|Universal\s+Competency\s+\S+(?:\s+\S+){0,2})",
        text,
        re.IGNORECASE,
    )
    names.extend(known)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for n in names:
        n_clean = n.strip()
        if n_clean.lower() not in seen:
            seen.add(n_clean.lower())
            unique.append(n_clean)
    return unique


_NEGATION_RE = re.compile(
    r"\b(?:remove|drop|exclude|no|don'?t\s+include|without|skip)\s+",
    re.IGNORECASE,
)


def _apply_negations(state: ConversationState, text: str) -> None:
    """Handle "remove X" / "no X" in the latest message to toggle flags off.

    This preserves the cumulative-scan approach while letting the user
    undo a previously-set preference in a refinement turn.
    """
    low = text.lower()
    if not _NEGATION_RE.search(low):
        return

    if re.search(r"\b(?:remove|drop|no|without|skip|exclude|don'?t\s+include)\s+(?:.*?)(?:personality|opq|behavio)", low):
        state.wants_personality = False
    if re.search(r"\b(?:remove|drop|no|without|skip|exclude|don'?t\s+include)\s+(?:.*?)(?:cognitive|aptitude|reasoning|verify)", low):
        state.wants_cognitive = False
    if re.search(r"\b(?:remove|drop|no|without|skip|exclude|don'?t\s+include)\s+(?:.*?)(?:sjt|situational|judgment)", low):
        state.wants_sjt = False
    if re.search(r"\b(?:remove|drop|no|without|skip|exclude|don'?t\s+include)\s+(?:.*?)(?:simulation|role.?play|inbox)", low):
        state.wants_simulation = False


# ───────────────────────────────────────────────────────────────────────────
# Merge utility (for refinement: "update the state, don't start over")
# ───────────────────────────────────────────────────────────────────────────

def merge_state(
    old: ConversationState,
    new: ConversationState,
) -> ConversationState:
    """Merge *new* into *old*, preserving prior constraints.

    - Scalars: prefer ``new`` when non-None, else keep ``old``.
    - Booleans: ``True`` in *new* overrides ``False`` in *old* (additive).
      Explicit negations should have already set *new* to ``False``
      via :func:`_apply_negations`.
    - Lists: non-empty ``new`` replaces ``old``.

    This implements the refinement rule: "update, don't start over."
    """
    return ConversationState(
        role_title=new.role_title or old.role_title,
        domain_keywords=new.domain_keywords or old.domain_keywords,
        seniority_text=new.seniority_text or old.seniority_text,
        language_required=new.language_required or old.language_required,
        wants_personality=new.wants_personality or old.wants_personality,
        wants_cognitive=new.wants_cognitive or old.wants_cognitive,
        wants_sjt=new.wants_sjt or old.wants_sjt,
        wants_simulation=new.wants_simulation or old.wants_simulation,
        wants_remote=new.wants_remote or old.wants_remote,
        duration_budget=(
            new.duration_budget
            if new.duration_budget is not None
            else old.duration_budget
        ),
        compare_targets=new.compare_targets or old.compare_targets,
        user_confirmed_final=new.user_confirmed_final,
        off_topic=new.off_topic,
    )

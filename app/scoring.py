"""
Scoring utilities for the SHL assessment agent.

Provides helpers that translate free-text job requirements into the
canonical labels used in the SHL product catalog, and a composite
``score_candidate`` function used by the retrieval layer to rank
catalog items against a user's requirements.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.catalog_loader import CatalogItem
    from app.state_extraction import ConversationState


# ---------------------------------------------------------------------------
# Canonical SHL job-level labels (as they appear in the catalog)
# ---------------------------------------------------------------------------
_ENTRY_LEVEL = "Entry-Level"
_GRADUATE = "Graduate"
_FRONT_LINE_MANAGER = "Front Line Manager"
_SUPERVISOR = "Supervisor"
_MID_PROFESSIONAL = "Mid-Professional"
_PROFESSIONAL_IC = "Professional Individual Contributor"
_MANAGER = "Manager"
_DIRECTOR = "Director"
_EXECUTIVE = "Executive"
_GENERAL_POPULATION = "General Population"


# ---------------------------------------------------------------------------
# Keyword → canonical labels mapping (order matters: first match wins)
# ---------------------------------------------------------------------------
# Each tuple is (set of trigger keywords/phrases, list of labels to return).
# The matching is performed against a *lowercased* version of the input.
_SENIORITY_MAP: list[tuple[list[str], list[str]]] = [
    # C-suite / executive  (must come before "director")
    # Short acronyms (cto, cio …) use word-boundary matching so they
    # don't accidentally match inside longer words like "director".
    (
        [r"\bcxo\b", r"\bceo\b", r"\bcfo\b", r"\bcoo\b", r"\bcto\b",
         r"\bcio\b", r"\bevp\b", r"\bsvp\b",
         "c-suite", "chief", "vice president", "vp "],
        [_EXECUTIVE, _DIRECTOR],
    ),
    # Director  (plain "director" without an executive-level prefix)
    (
        ["director"],
        [_DIRECTOR],
    ),
    # Team lead / supervisor / front-line manager
    # (must come *before* generic "manager" so that "frontline manager"
    # doesn't match the manager rule first)
    (
        ["team lead", "team leader", "front line", "frontline", "shift lead"],
        [_FRONT_LINE_MANAGER, _SUPERVISOR],
    ),
    (
        ["supervisor", "foreman"],
        [_SUPERVISOR],
    ),
    # Manager (senior / general)
    (
        ["senior manager", "head of", "general manager"],
        [_MANAGER, _DIRECTOR],
    ),
    (
        ["manager"],
        [_MANAGER],
    ),
    # Mid-level professional
    (
        [
            "mid-level", "mid level", "midlevel",
            "mid-professional", "professional individual contributor",
            "senior engineer", "senior developer", "senior analyst",
            "experienced", "specialist", "senior",
        ],
        [_MID_PROFESSIONAL, _PROFESSIONAL_IC],
    ),
    # Graduate / trainee / intern
    (
        ["graduate", "trainee", "intern", "fresh", "junior"],
        [_GRADUATE, _ENTRY_LEVEL],
    ),
    # Entry level / associate
    (
        ["entry-level", "entry level", "associate", "beginner"],
        [_ENTRY_LEVEL],
    ),
]


def map_seniority_to_job_levels(seniority: str) -> list[str]:
    """Map a free-text seniority description to SHL catalog job-level labels.

    Parameters
    ----------
    seniority:
        A natural-language string describing the target seniority, e.g.
        ``"graduate"``, ``"CXO"``, ``"mid-level engineer"``.

    Returns
    -------
    list[str]
        Matching canonical job-level labels.  Falls back to
        ``["General Population"]`` when no keyword matches.

    Examples
    --------
    >>> map_seniority_to_job_levels("graduate")
    ['Graduate', 'Entry-Level']

    >>> map_seniority_to_job_levels("CXO")
    ['Executive', 'Director']

    >>> map_seniority_to_job_levels("mid-level engineer")
    ['Mid-Professional', 'Professional Individual Contributor']
    """
    text = seniority.strip().lower()

    if not text:
        return [_GENERAL_POPULATION]

    for keywords, labels in _SENIORITY_MAP:
        for kw in keywords:
            # Keywords starting with \b are regex patterns (for word-
            # boundary matching of short acronyms); others use plain
            # substring matching.
            if kw.startswith(r"\b"):
                if re.search(kw, text):
                    return labels
            elif kw in text:
                return labels

    # No match – return the broadest bucket
    return [_GENERAL_POPULATION]


# ---------------------------------------------------------------------------
# Composite candidate scoring
# ---------------------------------------------------------------------------

# Weight allocation (must sum to 1.0)
_W_SEMANTIC: float = 0.60
_W_JOB_LEVEL: float = 0.10
_W_DURATION: float = 0.08
_W_REMOTE: float = 0.07
_W_LANGUAGE: float = 0.08
_W_KEYS: float = 0.07


def score_candidate(
    item: CatalogItem,
    state: ConversationState,
    cosine_sim: float,
) -> float:
    """Compute a composite relevance score for a single catalog item.

    Parameters
    ----------
    item:
        The :class:`CatalogItem` to evaluate.
    state:
        Current :class:`ConversationState` with the user's requirements.
    cosine_sim:
        Pre-computed cosine similarity (range ``[-1, 1]``) between the
        query embedding and the item's embedding.

    Returns
    -------
    float
        Blended score in the range ``[0, 1]`` (higher = better match).
    """

    # 1. Semantic similarity component (clamp to [0, 1])
    sem: float = max(0.0, min(1.0, (cosine_sim + 1.0) / 2.0))

    # 2. Job-level overlap
    jl_score: float = 0.5  # neutral default when user hasn't specified
    if state.seniority_text:
        wanted: set[str] = set(map_seniority_to_job_levels(state.seniority_text))
        item_levels: set[str] = set(item.job_levels)
        if wanted and item_levels:
            overlap = len(wanted & item_levels)
            jl_score = overlap / len(wanted)
        elif not item_levels:
            jl_score = 0.3  # no levels listed → slight penalty

    # 3. Duration budget
    dur_score: float = 0.5  # neutral
    if state.duration_budget is not None:
        if item.duration_minutes is None:
            dur_score = 0.3  # unknown duration → slight penalty
        elif item.duration_minutes <= state.duration_budget:
            dur_score = 1.0
        else:
            # Linearly penalise – up to 2× over budget → score 0
            overshoot: float = item.duration_minutes / state.duration_budget
            dur_score = max(0.0, 1.0 - (overshoot - 1.0))

    # 4. Remote support
    # wants_remote is a bool (False = not specified / don't care).
    rem_score: float = 0.5  # neutral when not specified
    if state.wants_remote:
        rem_score = 1.0 if item.remote_supported else 0.0

    # 5. Language match
    lang_score: float = 0.5  # neutral
    if state.language_required:
        target: str = state.language_required.lower()
        available: list[str] = [l.lower() for l in item.languages]
        if any(target in lang for lang in available):
            lang_score = 1.0
        elif not available:
            lang_score = 0.3  # no language info → slight penalty
        else:
            lang_score = 0.0

    # 6. Key / test-type overlap
    key_score: float = 0.5  # neutral
    if state.desired_keys:
        wanted_keys: set[str] = {k.lower() for k in state.desired_keys}
        item_keys: set[str] = {k.lower() for k in item.keys}
        if item_keys:
            overlap_k = len(wanted_keys & item_keys)
            key_score = overlap_k / len(wanted_keys) if wanted_keys else 0.5
        else:
            key_score = 0.3

    # Weighted blend
    score: float = (
        _W_SEMANTIC * sem
        + _W_JOB_LEVEL * jl_score
        + _W_DURATION * dur_score
        + _W_REMOTE * rem_score
        + _W_LANGUAGE * lang_score
        + _W_KEYS * key_score
    )

    return round(score, 6)


# app/scoring.py
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.catalog_loader import CatalogItem
    from app.state_extraction import ConversationState

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

_SENIORITY_MAP: list[tuple[list[str], list[str]]] = [
    (
        [r"\bcxo\b", r"\bceo\b", r"\bcfo\b", r"\bcoo\b", r"\bcto\b",
         r"\bcio\b", r"\bevp\b", r"\bsvp\b",
         "c-suite", "chief", "vice president", "vp "],
        [_EXECUTIVE, _DIRECTOR],
    ),
    (
        ["director"],
        [_DIRECTOR],
    ),
    (
        ["team lead", "team leader", "front line", "frontline", "shift lead"],
        [_FRONT_LINE_MANAGER, _SUPERVISOR],
    ),
    (
        ["supervisor", "foreman"],
        [_SUPERVISOR],
    ),
    (
        ["senior manager", "head of", "general manager"],
        [_MANAGER, _DIRECTOR],
    ),
    (
        ["manager"],
        [_MANAGER],
    ),
    (
        [
            "mid-level", "mid level", "midlevel",
            "mid-professional", "professional individual contributor",
            "senior engineer", "senior developer", "senior analyst",
            "experienced", "specialist", "senior",
        ],
        [_MID_PROFESSIONAL, _PROFESSIONAL_IC],
    ),
    (
        ["graduate", "trainee", "intern", "fresh", "junior"],
        [_GRADUATE, _ENTRY_LEVEL],
    ),
    (
        ["entry-level", "entry level", "associate", "beginner"],
        [_ENTRY_LEVEL],
    ),
]


def map_seniority_to_job_levels(seniority: str) -> list[str]:
    text = seniority.strip().lower()
    if not text:
        return [_GENERAL_POPULATION]
    for keywords, labels in _SENIORITY_MAP:
        for kw in keywords:
            if kw.startswith(r"\b"):
                if re.search(kw, text):
                    return labels
            elif kw in text:
                return labels
    return [_GENERAL_POPULATION]


_W_SEMANTIC: float = 0.58
_W_JOB_LEVEL: float = 0.10
_W_DURATION: float = 0.07
_W_REMOTE: float = 0.06
_W_LANGUAGE: float = 0.07
_W_KEYS: float = 0.07
_W_NEW_VARIANT: float = 0.05  # boost for (New) / Interactive variants


def score_candidate(
    item: "CatalogItem",
    state: "ConversationState",
    cosine_sim: float,
) -> float:
    sem: float = max(0.0, min(1.0, (cosine_sim + 1.0) / 2.0))

    jl_score: float = 0.5
    if state.seniority_text:
        wanted: set[str] = set(map_seniority_to_job_levels(state.seniority_text))
        item_levels: set[str] = set(item.job_levels)
        if wanted and item_levels:
            jl_score = len(wanted & item_levels) / len(wanted)
        elif not item_levels:
            jl_score = 0.3

    dur_score: float = 0.5
    if state.duration_budget is not None:
        if item.duration_minutes is None:
            dur_score = 0.3
        elif item.duration_minutes <= state.duration_budget:
            dur_score = 1.0
        else:
            overshoot: float = item.duration_minutes / state.duration_budget
            dur_score = max(0.0, 1.0 - (overshoot - 1.0))

    rem_score: float = 0.5
    if state.wants_remote:
        rem_score = 1.0 if item.remote_supported else 0.0

    lang_score: float = 0.5
    if state.language_required:
        target: str = state.language_required.lower()
        available: list[str] = [lg.lower() for lg in item.languages]
        if any(target in lang for lang in available):
            lang_score = 1.0
        elif not available:
            lang_score = 0.3
        else:
            lang_score = 0.0

    key_score: float = 0.5
    if state.desired_keys:
        wanted_keys: set[str] = {k.lower() for k in state.desired_keys}
        item_keys: set[str] = {k.lower() for k in item.keys}
        if item_keys and wanted_keys:
            key_score = len(wanted_keys & item_keys) / len(wanted_keys)
        elif not item_keys:
            key_score = 0.3

    # Boost modern / next-generation variants so they rank above legacy
    # equivalents that have similar semantic embeddings.
    # "(New)" items are the current catalog generation; "Interactive" products
    # are the computerised-adaptive successors of the paper-based Verify suite.
    new_score: float = 0.0
    name_lower: str = item.name.lower()
    if "(new)" in name_lower or "interactive" in name_lower or "365" in name_lower:
        new_score = 1.0

    score: float = (
        _W_SEMANTIC * sem
        + _W_JOB_LEVEL * jl_score
        + _W_DURATION * dur_score
        + _W_REMOTE * rem_score
        + _W_LANGUAGE * lang_score
        + _W_KEYS * key_score
        + _W_NEW_VARIANT * new_score
    )
    return round(score, 6)
# app/battery.py
"""
TASKS 3, 4, 6 — Battery composition, retrieval diversification,
and catalog-limitation detection.

Lightweight, deterministic helpers that post-process the scored candidate
list returned by semantic_search.  No new frameworks; no ontology.
"""
from __future__ import annotations

import re
from typing import Optional

from app.catalog_loader import CatalogItem
from app.state_extraction import ConversationState

# ---------------------------------------------------------------------------
# TASK 3 — Lightweight category tags
# ---------------------------------------------------------------------------

# Map SHL catalog key strings → internal category tags
_KEY_TO_CATEGORY: dict[str, str] = {
    "Ability & Aptitude": "cognitive",
    "Personality & Behavior": "personality",
    "Knowledge & Skills": "technical",
    "Simulations": "simulation",
    "Biodata & Situational Judgment": "sjt",
}

# "report" tag: items whose names suggest a leadership/360 report
_REPORT_RE = re.compile(
    r"\b(360|report|feedback\s+report|leadership\s+report"
    r"|ucf|universal\s+competency)\b",
    re.IGNORECASE,
)


def categorise(item: CatalogItem) -> str:
    """Return the primary category tag for a catalog item."""
    for key, cat in _KEY_TO_CATEGORY.items():
        if key in item.keys:
            return cat
    if _REPORT_RE.search(item.name) or _REPORT_RE.search(item.description or ""):
        return "report"
    return "other"


# ---------------------------------------------------------------------------
# TASK 3 — Assembly rules per hiring context
# ---------------------------------------------------------------------------

_GRADUATE_RE = re.compile(
    r"\b(graduate|intern|trainee|fresh|junior|entry.level)\b", re.IGNORECASE
)
_LEADERSHIP_RE = re.compile(
    r"\b(leader(?:ship)?|executive|director|manager|head\s+of|vp\b|ceo|cfo|cto)\b",
    re.IGNORECASE,
)
_ENGINEERING_RE = re.compile(
    r"\b(engineer|developer|programmer|software|data\s+scientist|devops"
    r"|backend|frontend|full.?stack|sre|swe)\b",
    re.IGNORECASE,
)
_SCREENING_RE = re.compile(
    r"\b(screen(?:ing)?|volume|bulk|large\s+(?:pool|batch)|many\s+candidates)\b",
    re.IGNORECASE,
)


def _preferred_order(state: ConversationState) -> list[str]:
    """Return category priority order for the hiring context."""
    q = (state.build_query_string() + " " + (state.role_title or "")).lower()
    if _LEADERSHIP_RE.search(q):
        return ["personality", "report", "cognitive", "sjt", "simulation", "technical", "other"]
    if _GRADUATE_RE.search(q) or (state.seniority_text and _GRADUATE_RE.search(state.seniority_text)):
        return ["cognitive", "personality", "sjt", "simulation", "technical", "report", "other"]
    if _ENGINEERING_RE.search(q):
        return ["technical", "cognitive", "simulation", "personality", "sjt", "report", "other"]
    if _SCREENING_RE.search(q):
        return ["cognitive", "sjt", "personality", "technical", "simulation", "report", "other"]
    # Default balanced order
    return ["cognitive", "personality", "technical", "sjt", "simulation", "report", "other"]


# ---------------------------------------------------------------------------
# TASK 4 — Retrieval diversification (reranking)
# ---------------------------------------------------------------------------

# Family grouping: items whose names share a common prefix are "same family"
def _family(item: CatalogItem) -> str:
    """Derive a coarse family key to prevent near-duplicate recommendations."""
    name = item.name.lower()
    # OPQ variants → same family
    if re.match(r"opq", name):
        return "opq"
    # Verify variants
    if re.match(r"verify", name):
        return "verify"
    # MQ variants
    if re.match(r"motivation questionnaire|mq\b", name):
        return "mq"
    # SJT variants
    if re.match(r"sjt|situational", name):
        return "sjt"
    # Otherwise use first word as family
    return name.split()[0] if name.split() else name


def diversify(
    scored: list[tuple[float, CatalogItem]],
    state: ConversationState,
    max_per_family: int = 2,
    max_per_category: int = 3,
    top_n: int = 20,
) -> list[tuple[float, CatalogItem]]:
    """Rerank candidates for category diversity and family deduplication.

    Steps:
    1. Apply a small category-diversity bonus based on assembly rules.
    2. Re-sort by adjusted score.
    3. Suppress families beyond max_per_family.
    4. Suppress categories beyond max_per_category.
    """
    if not scored:
        return scored

    priority = _preferred_order(state)

    # Apply category bonus: 0.03 per position up in priority list
    adjusted: list[tuple[float, CatalogItem]] = []
    for score, item in scored:
        cat = categorise(item)
        try:
            rank = priority.index(cat)
        except ValueError:
            rank = len(priority)
        bonus = max(0.0, (len(priority) - rank - 1) * 0.015)
        # If state has explicit test-type prefs, boost matching categories more
        if (cat == "cognitive" and state.wants_cognitive) or \
           (cat == "personality" and state.wants_personality) or \
           (cat == "sjt" and state.wants_sjt) or \
           (cat == "simulation" and state.wants_simulation):
            bonus += 0.04
        adjusted.append((score + bonus, item))

    adjusted.sort(key=lambda t: t[0], reverse=True)

    # Suppress duplicates: per-family and per-category caps
    family_count: dict[str, int] = {}
    category_count: dict[str, int] = {}
    seen_ids: set[str] = set()
    result: list[tuple[float, CatalogItem]] = []

    for score, item in adjusted:
        if item.id in seen_ids:
            continue
        fam = _family(item)
        cat = categorise(item)
        if family_count.get(fam, 0) >= max_per_family:
            continue
        if category_count.get(cat, 0) >= max_per_category:
            continue
        seen_ids.add(item.id)
        family_count[fam] = family_count.get(fam, 0) + 1
        category_count[cat] = category_count.get(cat, 0) + 1
        result.append((score, item))
        if len(result) >= top_n:
            break

    return result


# ---------------------------------------------------------------------------
# TASK 6 — Catalog limitation detection
# ---------------------------------------------------------------------------

# Languages we know the SHL catalog covers reasonably well
_SUPPORTED_LANGUAGES = {
    "english", "english (usa)", "english (uk)", "english (international)",
    "spanish", "french", "german", "dutch", "portuguese", "chinese",
    "japanese", "arabic",
}

# Domains where the SHL catalog has limited coverage
_THIN_DOMAIN_RE = re.compile(
    r"\b(blockchain|web3|solidity|rust\s+lang|flutter|dart"
    r"|quantum\s+computing|embedded\s+systems|fpga|verilog"
    r"|mainframe|cobol|fortran|assembly)\b",
    re.IGNORECASE,
)


def detect_catalog_limitations(
    state: ConversationState,
    candidates: list[tuple[float, CatalogItem]],
) -> Optional[str]:
    """Return a limitation notice string if the catalog cannot fully satisfy
    the user's constraints; else None.

    Never hallucinate: only warn, then let the agent fall back to best-match.
    """
    notices: list[str] = []

    # Language check
    if state.language_required:
        lang_key = state.language_required.lower().strip()
        if lang_key not in _SUPPORTED_LANGUAGES:
            notices.append(
                f"The SHL catalog may have limited coverage for assessments "
                f"in {state.language_required}. I'll show the closest available options."
            )
        else:
            # Check whether any top candidates actually support it
            lang_hits = sum(
                1 for _, item in candidates[:10]
                if any(lang_key in l.lower() for l in item.languages)
            )
            if lang_hits == 0 and candidates:
                notices.append(
                    f"I couldn't find assessments in {state.language_required} "
                    f"that match all your criteria. Showing the nearest alternatives."
                )

    # Domain/tech check
    query = state.build_query_string()
    if _THIN_DOMAIN_RE.search(query) or _THIN_DOMAIN_RE.search(state.role_title or ""):
        notices.append(
            "SHL's catalog focuses on general cognitive, personality, and "
            "widely-used technical domains — coverage for this specific technology "
            "may be limited. I'm showing the closest relevant assessments."
        )

    # Duration check: if budget is very tight and no candidates fit
    if state.duration_budget is not None:
        fitting = [
            item for _, item in candidates[:20]
            if item.duration_minutes is not None and item.duration_minutes <= state.duration_budget
        ]
        if not fitting and candidates:
            notices.append(
                f"No assessments under {state.duration_budget} minutes exactly match "
                f"your other criteria. Showing the shortest available options."
            )

    return " ".join(notices) if notices else None

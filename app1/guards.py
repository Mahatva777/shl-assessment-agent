# app/guards.py
from __future__ import annotations

import logging
import re
from typing import Optional

from app.catalog_loader import CatalogItem
from app.schemas import ChatResponse, Recommendation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test-type derivation
# ---------------------------------------------------------------------------

_KEY_PRIORITY: list[tuple[str, str]] = [
    ("Knowledge & Skills", "K"),
    ("Ability & Aptitude", "A"),
    ("Personality & Behavior", "P"),
    ("Biodata & Situational Judgment", "B"),
    ("Simulations", "S"),
]


def derive_test_type_from_keys(keys: list[str]) -> str:
    for catalog_key, code in _KEY_PRIORITY:
        if catalog_key in keys:
            return code
    return "A"


# ---------------------------------------------------------------------------
# Catalog item lookup
# ---------------------------------------------------------------------------

def find_catalog_item(
    name: str,
    catalog: list[CatalogItem],
) -> Optional[CatalogItem]:
    name_lower = name.strip().lower()
    if not name_lower:
        return None
    for item in catalog:
        if item.name.lower() == name_lower:
            return item
    for item in catalog:
        if name_lower in item.name.lower():
            return item
    for item in catalog:
        if item.name.lower() in name_lower:
            return item
    return None


# ---------------------------------------------------------------------------
# Recommendation builders
# ---------------------------------------------------------------------------

def build_recommendations_from_scores(
    scored: list[tuple[float, CatalogItem]],
    top_n: int = 10,
) -> list[Recommendation]:
    recs: list[Recommendation] = []
    seen_ids: set[str] = set()
    for _score, item in scored[:top_n]:
        if item.id in seen_ids:
            continue
        seen_ids.add(item.id)
        recs.append(Recommendation(
            name=item.name,
            url=item.url,
            test_type=derive_test_type_from_keys(item.keys),
        ))
    return recs


def build_recommendations_from_names(
    chosen_names: list[str],
    candidate_pool: list[CatalogItem],
    top_n: int = 10,
) -> list[Recommendation]:
    recs: list[Recommendation] = []
    seen_ids: set[str] = set()
    for name in chosen_names:
        item = find_catalog_item(name, candidate_pool)
        if item is None:
            logger.warning("LLM chose unknown name, dropping: '%s'", name)
            continue
        if item.id in seen_ids:
            continue
        seen_ids.add(item.id)
        recs.append(Recommendation(
            name=item.name,
            url=item.url,
            test_type=derive_test_type_from_keys(item.keys),
        ))
        if len(recs) >= top_n:
            break
    return recs


# ---------------------------------------------------------------------------
# Output sanitisation
# ---------------------------------------------------------------------------

def sanitise_reply(reply: str) -> str:
    reply = re.sub(
        r"\[SYSTEM[^\]]*\].*?(?=\n\n|\Z)",
        "",
        reply,
        flags=re.DOTALL | re.IGNORECASE,
    )
    reply = re.sub(
        r"https?://(?!www\.shl\.com)\S+",
        "[link removed]",
        reply,
    )
    return reply.strip()


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------

def validate_chat_response(response: ChatResponse) -> ChatResponse:
    recs = response.recommendations[:10]
    reply = response.reply or "I'm sorry, I was unable to generate a response. Please try again."
    return ChatResponse(
        reply=reply,
        recommendations=recs,
        end_of_conversation=response.end_of_conversation,
    )
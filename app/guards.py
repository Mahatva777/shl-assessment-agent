"""
Guards for the SHL Assessment Agent.

Provides validation and safety enforcement:
    - Schema validation for API responses.
    - Catalog-only enforcement (never let the LLM fabricate product names).
    - Output sanitisation.

These guards run *after* the LLM generates a reply and *before* the
response is returned to the caller, ensuring every recommendation maps
to a real catalog entry.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.catalog_loader import CatalogItem
from app.schemas import Recommendation

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Catalog-only enforcement
# ───────────────────────────────────────────────────────────────────────────

def find_catalog_item(
    name: str,
    catalog: list[CatalogItem],
) -> Optional[CatalogItem]:
    """Find a catalog item by exact or fuzzy name match.

    Tries exact match first (case-insensitive), then falls back to
    substring matching.  Returns ``None`` if no match is found.

    Parameters
    ----------
    name:
        Product name to look up (possibly from LLM output).
    catalog:
        The full product catalog.

    Returns
    -------
    CatalogItem | None
    """
    name_lower: str = name.strip().lower()
    if not name_lower:
        return None

    # Exact match (case-insensitive)
    for item in catalog:
        if item.name.lower() == name_lower:
            return item

    # Substring match — name appears within a catalog item name
    for item in catalog:
        if name_lower in item.name.lower():
            return item

    # Reverse substring — catalog item name appears within the query name
    for item in catalog:
        if item.name.lower() in name_lower:
            return item

    return None


def validate_recommendations(
    recs: list[Recommendation],
    catalog: list[CatalogItem],
) -> list[Recommendation]:
    """Ensure every recommendation maps to a real catalog item.

    Drops fabricated names and fixes URLs / fields from catalog data.

    Parameters
    ----------
    recs:
        Raw recommendations (potentially from LLM-generated data).
    catalog:
        The canonical product catalog.

    Returns
    -------
    list[Recommendation]
        Validated recommendations with correct URLs and metadata.
    """
    validated: list[Recommendation] = []
    seen_ids: set[str] = set()

    for rec in recs:
        item = find_catalog_item(rec.product_name, catalog)
        if item is None:
            logger.warning(
                "Dropping fabricated recommendation: '%s'",
                rec.product_name,
            )
            continue

        if item.id in seen_ids:
            continue  # deduplicate
        seen_ids.add(item.id)

        # Rebuild from catalog data — never trust LLM-generated URLs
        validated.append(Recommendation(
            product_name=item.name,
            url=item.url,
            duration_minutes=item.duration_minutes,
            remote_supported=item.remote_supported,
            adaptive=None,  # set from catalog if available
            description=item.description[:200] if item.description else "",
        ))

    return validated


def build_recommendations_from_scores(
    scored: list[tuple[float, CatalogItem]],
    top_n: int = 10,
) -> list[Recommendation]:
    """Convert scored catalog items to validated Recommendation objects.

    This is the preferred path: the ranker produces scored items, and
    this function converts them to API-ready objects.  No LLM-generated
    URLs or names can leak through.

    Parameters
    ----------
    scored:
        ``(score, CatalogItem)`` pairs sorted best-first.
    top_n:
        Maximum number of recommendations to return.

    Returns
    -------
    list[Recommendation]
    """
    recs: list[Recommendation] = []
    seen_ids: set[str] = set()

    for _score, item in scored[:top_n]:
        if item.id in seen_ids:
            continue
        seen_ids.add(item.id)

        recs.append(Recommendation(
            product_name=item.name,
            url=item.url,
            duration_minutes=item.duration_minutes,
            remote_supported=item.remote_supported,
            adaptive=None,
            description=item.description[:200] if item.description else "",
        ))

    return recs


# ───────────────────────────────────────────────────────────────────────────
# Output sanitisation
# ───────────────────────────────────────────────────────────────────────────

def sanitise_reply(reply: str) -> str:
    """Remove system-prompt leaks and dangerous content from LLM output.

    Parameters
    ----------
    reply:
        Raw LLM reply text.

    Returns
    -------
    str
        Cleaned reply.
    """
    # Remove any [SYSTEM] blocks that leaked into the reply
    reply = re.sub(
        r"\[SYSTEM[^\]]*\].*?(?=\n\n|\Z)",
        "",
        reply,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove raw URLs that don't come from the catalog
    # (catalog URLs are injected separately via Recommendation objects)
    reply = re.sub(
        r"https?://(?!www\.shl\.com)\S+",
        "[link removed]",
        reply,
    )

    return reply.strip()

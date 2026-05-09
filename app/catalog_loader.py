"""
Catalog loader for SHL product catalog.

Reads data/shl_product_catalog.json and produces a list of normalised
CatalogItem objects ready for embedding / retrieval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CatalogItem:
    """Normalised representation of one SHL assessment product."""

    id: str
    name: str
    url: str
    job_levels: list[str]
    duration_minutes: Optional[int]
    remote_supported: bool
    languages: list[str]
    keys: list[str]
    description: str
    search_blob: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_job_levels(raw: dict) -> list[str]:
    """Return a clean, deduplicated list of job-level strings.

    Prefers the already-parsed ``job_levels`` list when it exists and is
    non-empty; otherwise falls back to splitting ``job_levels_raw``.
    """
    levels: list[str] = raw.get("job_levels") or []

    # Fallback: parse from comma-separated raw string
    if not levels:
        raw_str: str = raw.get("job_levels_raw", "")
        levels = [s.strip() for s in raw_str.split(",") if s.strip()]

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for lvl in levels:
        if lvl not in seen:
            seen.add(lvl)
            unique.append(lvl)
    return unique


_DURATION_RE = re.compile(r"(\d+)")


def _parse_duration(raw: dict) -> Optional[int]:
    """Extract an integer number of minutes from *duration* / *duration_raw*.

    Handles common formats found in the catalog:
    - ``"30 minutes"``
    - ``"Approximate Completion Time in minutes = 30"``
    - ``"Approximate Completion Time in minutes = max 45"``
    - ``"Approximate Completion Time in minutes = 7 minutes"``
    - ``"Variable"`` → ``None``
    - ``""`` → ``None``
    """
    for key in ("duration", "duration_raw"):
        value: str = raw.get(key, "").strip()
        if not value or value.lower() == "variable":
            continue
        match = _DURATION_RE.search(value)
        if match:
            return int(match.group(1))
    return None


def _parse_remote(raw: dict) -> bool:
    """Return ``True`` when the product supports remote proctoring."""
    return raw.get("remote", "").strip().lower() == "yes"


def _normalise_languages(raw: dict) -> list[str]:
    """Return a clean list of languages.

    Prefers the parsed ``languages`` list; falls back to ``languages_raw``.
    """
    langs: list[str] = raw.get("languages") or []
    if not langs:
        raw_str: str = raw.get("languages_raw", "")
        langs = [s.strip() for s in raw_str.split(",") if s.strip()]
    return langs


def _build_search_blob(raw: dict, name: str, description: str, keys: list[str]) -> str:
    """Combine key fields into a single lowercased string for text search."""
    parts: list[str] = [
        name,
        description,
        " ".join(keys),
        raw.get("job_levels_raw", ""),
        raw.get("languages_raw", ""),
    ]
    return " ".join(parts).lower()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_catalog(path: str) -> list[CatalogItem]:
    """Read the SHL product catalog JSON and return normalised items.

    Parameters
    ----------
    path:
        File-system path (absolute or relative) to the JSON catalog.

    Returns
    -------
    list[CatalogItem]
        One entry per product in the catalog, fully normalised and ready for
        embedding / retrieval.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    items: list[CatalogItem] = []
    for entry in data:
        name: str = entry.get("name", "")
        description: str = entry.get("description", "")
        keys: list[str] = entry.get("keys", [])

        item = CatalogItem(
            id=entry.get("entity_id", ""),
            name=name,
            url=entry.get("link", ""),
            job_levels=_normalise_job_levels(entry),
            duration_minutes=_parse_duration(entry),
            remote_supported=_parse_remote(entry),
            languages=_normalise_languages(entry),
            keys=keys,
            description=description,
            search_blob=_build_search_blob(entry, name, description, keys),
            embedding=None,
        )
        items.append(item)

    return items

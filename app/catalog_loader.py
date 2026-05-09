# app/catalog_loader.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CatalogItem:
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
    adaptive: Optional[bool] = None
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


def _normalise_job_levels(raw: dict) -> list[str]:
    levels: list[str] = raw.get("job_levels") or []
    if not levels:
        raw_str: str = raw.get("job_levels_raw", "")
        levels = [s.strip() for s in raw_str.split(",") if s.strip()]
    seen: set[str] = set()
    unique: list[str] = []
    for lvl in levels:
        if lvl not in seen:
            seen.add(lvl)
            unique.append(lvl)
    return unique


_DURATION_RE = re.compile(r"(\d+)")


def _parse_duration(raw: dict) -> Optional[int]:
    for key in ("duration", "duration_raw"):
        value: str = raw.get(key, "").strip()
        if not value or value.lower() == "variable":
            continue
        match = _DURATION_RE.search(value)
        if match:
            return int(match.group(1))
    return None


def _parse_remote(raw: dict) -> bool:
    return raw.get("remote", "").strip().lower() == "yes"


def _parse_adaptive(raw: dict) -> Optional[bool]:
    val = raw.get("adaptive", "").strip().lower()
    if val == "yes":
        return True
    if val == "no":
        return False
    return None


def _normalise_languages(raw: dict) -> list[str]:
    langs: list[str] = raw.get("languages") or []
    if not langs:
        raw_str: str = raw.get("languages_raw", "")
        langs = [s.strip() for s in raw_str.split(",") if s.strip()]
    return langs


def _build_search_blob(raw: dict, name: str, description: str, keys: list[str]) -> str:
    parts: list[str] = [
        name,
        description,
        " ".join(keys),
        raw.get("job_levels_raw", ""),
        raw.get("languages_raw", ""),
    ]
    return " ".join(parts).lower()


def load_catalog(path: str) -> list[CatalogItem]:
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
            adaptive=_parse_adaptive(entry),
            languages=_normalise_languages(entry),
            keys=keys,
            description=description,
            search_blob=_build_search_blob(entry, name, description, keys),
            embedding=None,
        )
        items.append(item)
    return items
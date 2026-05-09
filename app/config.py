"""
Application configuration.

All settings are read from environment variables (with sensible defaults)
so the service can be configured via ``.env``, Docker env, or CI secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application settings populated from the environment."""

    # ── LLM ────────────────────────────────────────────────────────────
    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "gemini-2.0-flash"),
    )
    google_api_key: str = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", "AIzaSyCVPP9Ql36fIryxxZqvYozD8aTCApkIJLI"),
    )

    # ── Catalog / data ─────────────────────────────────────────────────
    catalog_path: str = field(
        default_factory=lambda: os.getenv("CATALOG_PATH", "data/shl_product_catalog.json"),
    )

    # ── Embedding ──────────────────────────────────────────────────────
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "models/text-embedding-004"),
    )

    # ── Retrieval ──────────────────────────────────────────────────────
    top_k: int = field(
        default_factory=lambda: int(os.getenv("TOP_K", "10")),
    )

    # ── Server ─────────────────────────────────────────────────────────
    host: str = field(
        default_factory=lambda: os.getenv("HOST", "0.0.0.0"),
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("PORT", "8000")),
    )
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() in ("1", "true", "yes"),
    )


def get_settings() -> Settings:
    """Return a freshly-constructed :class:`Settings` instance.

    Each call re-reads environment variables so tests can patch
    ``os.environ`` and get updated values.
    """
    return Settings()

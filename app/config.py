# app/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Settings:
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""),
    )
    google_api_key: str = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""),
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "claude-sonnet-5"),
    )
    catalog_path: str = field(
        default_factory=lambda: os.getenv("CATALOG_PATH", "data/shl_product_catalog.json"),
    )
    embeddings_path: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDINGS_PATH", "data/catalog_embeddings.npy"
        ),
    )
    metadata_path: str = field(
        default_factory=lambda: os.getenv(
            "METADATA_PATH", "data/catalog_metadata.json"
        ),
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )
    top_k: int = field(
        default_factory=lambda: int(os.getenv("TOP_K", "20")),
    )
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
    return Settings()
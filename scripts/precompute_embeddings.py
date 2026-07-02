#!/usr/bin/env python3
"""
scripts/precompute_embeddings.py
---------------------------------
Pre-compute SentenceTransformer embeddings for every item in the SHL product
catalog and save them as static files so the app never has to re-encode the
catalog at request time.

Outputs
-------
data/catalog_embeddings.npy   – float32 numpy array, shape [N, 384]
data/catalog_metadata.json    – list of {name, url, test_type, description}
                                 dicts in the same order as the .npy rows.

Usage
-----
    python scripts/precompute_embeddings.py

Re-run only when data/shl_product_catalog.json changes.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths (relative to repo root; script is expected to run from there)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
CATALOG_PATH = REPO_ROOT / "data" / "shl_product_catalog.json"
EMBEDDINGS_OUT = REPO_ROOT / "data" / "catalog_embeddings.npy"
METADATA_OUT = REPO_ROOT / "data" / "catalog_metadata.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Text builder – must stay in sync with catalog_loader._build_search_blob()
# ---------------------------------------------------------------------------

def _build_document_text(entry: dict) -> str:
    """
    Build the text string used as the embedding input for a catalog item.

    This replicates catalog_loader._build_search_blob() **exactly** so that
    the pre-computed embeddings match what retrieval.py's CatalogItem.search_blob
    holds at runtime.

    Order: name · description · keys/categories · job_levels
    (languages_raw appended last, same as _build_search_blob)
    """
    name: str = entry.get("name", "")
    description: str = entry.get("description", "")
    keys: list[str] = entry.get("keys", [])
    keys_str: str = " ".join(keys)
    job_levels_raw: str = entry.get("job_levels_raw", "")
    languages_raw: str = entry.get("languages_raw", "")

    parts = [name, description, keys_str, job_levels_raw, languages_raw]
    return " ".join(parts).lower()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.monotonic()

    # ── 1. Load catalog ──────────────────────────────────────────────────────
    if not CATALOG_PATH.exists():
        print(f"ERROR: catalog not found at {CATALOG_PATH}", file=sys.stderr)
        sys.exit(1)

    raw_items: list[dict] = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(raw_items)} items from {CATALOG_PATH}")

    # ── 2. Build text blobs ──────────────────────────────────────────────────
    texts: list[str] = [_build_document_text(item) for item in raw_items]

    # ── 3. Load embedding model and encode ───────────────────────────────────
    print(f"Loading SentenceTransformer('{EMBEDDING_MODEL}')…")
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Encoding catalog items…")
    embeddings: np.ndarray = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=64,
    ).astype(np.float32)

    # ── 4. Save embeddings ───────────────────────────────────────────────────
    EMBEDDINGS_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(EMBEDDINGS_OUT), embeddings)
    print(f"Saved embeddings → {EMBEDDINGS_OUT}  (shape={embeddings.shape})")

    # ── 5. Build and save parallel metadata ─────────────────────────────────
    # Each entry mirrors what retrieval.py needs for the recommendation output:
    # name, url, test_type (derived from 'keys'), description
    metadata: list[dict] = []
    for entry in raw_items:
        keys: list[str] = entry.get("keys", [])
        test_type: str = keys[0] if keys else ""
        metadata.append(
            {
                "name": entry.get("name", ""),
                "url": entry.get("link", ""),
                "test_type": test_type,
                "description": entry.get("description", ""),
            }
        )

    METADATA_OUT.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved metadata  → {METADATA_OUT}  ({len(metadata)} entries)")

    # ── 6. Sanity check ──────────────────────────────────────────────────────
    assert embeddings.shape[0] == len(metadata), (
        f"Shape mismatch: embeddings has {embeddings.shape[0]} rows "
        f"but metadata has {len(metadata)} entries."
    )
    assert embeddings.shape[1] == 384, (
        f"Unexpected embedding dimension: {embeddings.shape[1]} (expected 384)."
    )
    assert embeddings.dtype == np.float32, (
        f"Unexpected dtype: {embeddings.dtype}"
    )

    elapsed = time.monotonic() - t0
    print()
    print("=" * 60)
    print(f"  Item count      : {embeddings.shape[0]}")
    print(f"  Embedding shape : {embeddings.shape}")
    print(f"  Embedding dtype : {embeddings.dtype}")
    print(f"  Elapsed         : {elapsed:.1f}s")
    print("=" * 60)
    print("Done. Commit data/catalog_embeddings.npy and data/catalog_metadata.json.")


if __name__ == "__main__":
    main()

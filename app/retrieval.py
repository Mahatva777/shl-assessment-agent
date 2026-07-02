# app/retrieval.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from app.catalog_loader import CatalogItem
from app.scoring import score_candidate
from app.state_extraction import ConversationState

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Module-level globals – populated once by load_sentence_model() and
# load_catalog() which are called from the FastAPI lifespan hook in main.py.
# Nothing is loaded lazily at request time.
# ---------------------------------------------------------------------------

_model: Optional[object] = None          # SentenceTransformer instance
_emb_matrix: Optional[NDArray[np.float32]] = None   # shape [N, 384]
_metadata: Optional[list[dict]] = None  # parallel list of {name,url,test_type,description}


# ---------------------------------------------------------------------------
# Loaders – called ONCE from the lifespan hook, never from request handlers
# ---------------------------------------------------------------------------

def load_sentence_model() -> None:
    """Load SentenceTransformer model into the module-level _model global."""
    global _model
    if _model is not None:
        return  # already loaded (idempotent)
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    logger.info("Loading SentenceTransformer('%s')…", _EMBEDDING_MODEL_NAME)
    _model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    logger.info("SentenceTransformer loaded.")


def load_catalog_embeddings(
    embeddings_path: str,
    metadata_path: str,
) -> None:
    """
    Load pre-computed embeddings and metadata into module-level globals.

    Parameters
    ----------
    embeddings_path : str
        Path to data/catalog_embeddings.npy  (float32, shape [N, 384])
    metadata_path : str
        Path to data/catalog_metadata.json  (list of {name,url,test_type,description})
    """
    global _emb_matrix, _metadata

    emb_path = Path(embeddings_path)
    meta_path = Path(metadata_path)

    if not emb_path.exists():
        raise FileNotFoundError(
            f"Pre-computed embeddings not found: {emb_path}\n"
            "Run: python scripts/precompute_embeddings.py"
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Catalog metadata not found: {meta_path}\n"
            "Run: python scripts/precompute_embeddings.py"
        )

    logger.info("Loading pre-computed embeddings from '%s'…", emb_path)
    _emb_matrix = np.load(str(emb_path)).astype(np.float32)
    logger.info("Embeddings loaded: shape=%s, dtype=%s", _emb_matrix.shape, _emb_matrix.dtype)

    logger.info("Loading catalog metadata from '%s'…", meta_path)
    _metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    logger.info("Metadata loaded: %d entries.", len(_metadata))

    if _emb_matrix.shape[0] != len(_metadata):
        raise ValueError(
            f"Embedding row count ({_emb_matrix.shape[0]}) does not match "
            f"metadata entry count ({len(_metadata)})."
        )


# ---------------------------------------------------------------------------
# Legacy helpers – kept for backward compatibility with agent.py's import of
# build_embeddings.  They are now no-ops because embeddings are pre-computed.
# ---------------------------------------------------------------------------

def build_embeddings(catalog: list[CatalogItem]) -> None:
    """
    No-op shim retained for backward compatibility.

    Embeddings are now loaded from data/catalog_embeddings.npy at startup.
    CatalogItem.embedding fields are NOT populated; semantic_search() uses
    the pre-loaded _emb_matrix instead.
    """
    logger.debug(
        "build_embeddings() called but embeddings are already loaded from disk. "
        "This is a no-op."
    )


# ---------------------------------------------------------------------------
# Query-time retrieval
# ---------------------------------------------------------------------------

def semantic_search(
    query: str,
    state: ConversationState,
    catalog: list[CatalogItem],
    top_k: int = 50,
) -> list[tuple[float, CatalogItem]]:
    """
    Retrieve top-K catalog items by composite score (semantic + constraints).
    Returns (score, CatalogItem) pairs sorted descending.

    The catalog list must be parallel to the pre-loaded _emb_matrix / _metadata
    (same order as shl_product_catalog.json → catalog_embeddings.npy).
    """
    if not catalog:
        return []

    if _model is None:
        raise RuntimeError(
            "SentenceTransformer model not loaded. "
            "Ensure load_sentence_model() was called in the lifespan hook."
        )
    if _emb_matrix is None or _metadata is None:
        raise RuntimeError(
            "Pre-computed embeddings not loaded. "
            "Ensure load_catalog_embeddings() was called in the lifespan hook."
        )

    # Encode the query string only (1 string, not 370) – fast < 100 ms
    state_query: str = state.build_query_string()
    combined_query: str = f"{state_query} {query}".strip() if state_query else query
    if not combined_query:
        combined_query = "assessment"

    query_vec: NDArray[np.float32] = _model.encode(  # type: ignore[union-attr]
        [combined_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    # Matrix multiply against pre-loaded embeddings (shape [N, 384] @ [384])
    cosine_sims: NDArray[np.float32] = _emb_matrix @ query_vec

    scored: list[tuple[float, CatalogItem]] = []
    n = min(len(catalog), len(cosine_sims))
    if n < len(catalog):
        logger.warning(
            "Catalog has %d items but embedding matrix has %d rows; "
            "only scoring first %d items. Re-run precompute_embeddings.py to fix.",
            len(catalog), len(cosine_sims), n,
        )
    for idx in range(n):
        item = catalog[idx]
        cos_sim: float = float(cosine_sims[idx])
        composite: float = score_candidate(item, state, cos_sim)
        scored.append((composite, item))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:top_k]
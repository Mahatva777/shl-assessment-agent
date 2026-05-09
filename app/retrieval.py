# app/retrieval.py
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from app.catalog_loader import CatalogItem
from app.scoring import score_candidate
from app.state_extraction import ConversationState

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
_model: Optional[object] = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SentenceTransformer model '%s'…", _EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _model


def build_embeddings(catalog: list[CatalogItem]) -> None:
    """Encode every catalog item's search_blob in-place."""
    if not catalog:
        return
    model = _get_model()
    blobs: list[str] = [item.search_blob for item in catalog]
    logger.info("Encoding %d catalog items…", len(blobs))
    embeddings: NDArray[np.float32] = model.encode(
        blobs,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    for item, emb in zip(catalog, embeddings):
        item.embedding = emb
    logger.info("Embeddings built (dim=%d).", embeddings.shape[1])


def _build_embedding_matrix(catalog: list[CatalogItem]) -> NDArray[np.float32]:
    vecs: list[NDArray[np.float32]] = []
    for item in catalog:
        if item.embedding is None:
            raise ValueError(
                f"Item '{item.name}' has no embedding. Call build_embeddings() first."
            )
        vecs.append(item.embedding)
    return np.vstack(vecs)


def semantic_search(
    query: str,
    state: ConversationState,
    catalog: list[CatalogItem],
    top_k: int = 50,
) -> list[tuple[float, CatalogItem]]:
    """
    Retrieve top-K catalog items by composite score (semantic + constraints).
    Returns (score, CatalogItem) pairs sorted descending.
    """
    if not catalog:
        return []

    model = _get_model()

    state_query: str = state.build_query_string()
    combined_query: str = f"{state_query} {query}".strip() if state_query else query
    if not combined_query:
        combined_query = "assessment"

    query_vec: NDArray[np.float32] = model.encode(
        [combined_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    emb_matrix: NDArray[np.float32] = _build_embedding_matrix(catalog)
    cosine_sims: NDArray[np.float32] = emb_matrix @ query_vec

    scored: list[tuple[float, CatalogItem]] = []
    for idx, item in enumerate(catalog):
        cos_sim: float = float(cosine_sims[idx])
        composite: float = score_candidate(item, state, cos_sim)
        scored.append((composite, item))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:top_k]
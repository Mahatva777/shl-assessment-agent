"""
Retrieval layer for the SHL Assessment Agent.

Provides:
    - A lazily-loaded SentenceTransformer model for encoding.
    - ``build_embeddings()`` to pre-compute and cache item embeddings.
    - ``semantic_search()`` to rank catalog items against a
      :class:`ConversationState` using cosine similarity + constraint scoring.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from app.catalog_loader import CatalogItem
from app.scoring import score_candidate
from app.state_extraction import ConversationState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded SentenceTransformer singleton
# ---------------------------------------------------------------------------
_EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
_model: Optional[object] = None  # will hold the SentenceTransformer instance


def _get_model():
    """Return the shared :class:`SentenceTransformer` model, loading it once.

    The import and model load are deferred so that the rest of the
    application can start even if ``sentence-transformers`` is not yet
    installed (e.g. during early development / linting).
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading SentenceTransformer model '%s' …", _EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
        logger.info("Model loaded.")
    return _model


# ---------------------------------------------------------------------------
# Embedding builder
# ---------------------------------------------------------------------------

def build_embeddings(catalog: list[CatalogItem]) -> None:
    """Encode every catalog item's ``search_blob`` and store the result.

    After this call, each item's :pyattr:`CatalogItem.embedding` field
    is a unit-length ``np.ndarray`` of shape ``(dim,)``.

    Parameters
    ----------
    catalog:
        The list of catalog items (mutated in-place).
    """
    model = _get_model()
    blobs: list[str] = [item.search_blob for item in catalog]

    logger.info("Encoding %d catalog items …", len(blobs))
    embeddings: NDArray[np.float32] = model.encode(
        blobs,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit-length → dot = cosine sim
    )

    for item, emb in zip(catalog, embeddings):
        item.embedding = emb

    logger.info("Embeddings built (dim=%d).", embeddings.shape[1])


# ---------------------------------------------------------------------------
# Embedding matrix cache
# ---------------------------------------------------------------------------

def _build_embedding_matrix(catalog: list[CatalogItem]) -> NDArray[np.float32]:
    """Stack all item embeddings into a single ``(N, dim)`` matrix.

    Raises
    ------
    ValueError
        If any item has no embedding yet (call ``build_embeddings`` first).
    """
    vecs: list[NDArray[np.float32]] = []
    for item in catalog:
        if item.embedding is None:
            raise ValueError(
                f"Item '{item.name}' (id={item.id}) has no embedding. "
                "Call build_embeddings() first."
            )
        vecs.append(item.embedding)
    return np.vstack(vecs)


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

def semantic_search(
    query: str,
    state: ConversationState,
    catalog: list[CatalogItem],
    top_k: int = 50,
) -> list[tuple[float, CatalogItem]]:
    """Retrieve the top-K catalog items most relevant to the user's request.

    Steps
    -----
    1. Build a rich query string from the :class:`ConversationState`.
    2. Encode the query via the SentenceTransformer model.
    3. Compute cosine similarity to every item embedding (vectorised).
    4. For each candidate, call :func:`score_candidate` to blend
       semantic similarity with structured-constraint bonuses.
    5. Return the ``top_k`` items sorted by descending composite score.

    Parameters
    ----------
    query:
        The raw user query (used as fallback if the state is sparse).
    state:
        Structured :class:`ConversationState` extracted from the chat.
    catalog:
        Pre-embedded list of :class:`CatalogItem` objects.
    top_k:
        Maximum number of results to return.

    Returns
    -------
    list[tuple[float, CatalogItem]]
        ``(score, item)`` pairs sorted best-first.
    """
    model = _get_model()

    # --- 1. Build the search query ---
    state_query: str = state.build_query_string()
    combined_query: str = f"{state_query} {query}".strip() if state_query else query
    if not combined_query:
        combined_query = query or "assessment"

    # --- 2. Encode the query ---
    query_vec: NDArray[np.float32] = model.encode(
        [combined_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]  # shape (dim,)

    # --- 3. Cosine similarity (embeddings are unit-length → dot product) ---
    emb_matrix: NDArray[np.float32] = _build_embedding_matrix(catalog)
    cosine_sims: NDArray[np.float32] = emb_matrix @ query_vec  # shape (N,)

    # --- 4. Score every candidate ---
    scored: list[tuple[float, CatalogItem]] = []
    for idx, item in enumerate(catalog):
        cos_sim: float = float(cosine_sims[idx])
        composite: float = score_candidate(item, state, cos_sim)
        scored.append((composite, item))

    # --- 5. Sort & truncate ---
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:top_k]

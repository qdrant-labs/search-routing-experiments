"""The dense + sparse embedding recipe every composition-side collection
indexes against, so both retrieval directions run off one collection."""

from __future__ import annotations

from qdrant_client.models import Distance

from hybrid_search_rrf_dataset.indexer import EmbeddingConfig

DENSE_MODEL_ID = "BAAI/bge-small-en-v1.5"
DENSE_MODEL_SIZE = 384
SPARSE_MODEL_ID = "Qdrant/bm25"


def default_embeddings() -> list[EmbeddingConfig]:
    """The two-slot recipe: dense bge-small + sparse BM25."""
    return [
        EmbeddingConfig(
            name="dense", 
            model_id=DENSE_MODEL_ID, 
            kind="dense",
            size=DENSE_MODEL_SIZE,
            distance=Distance.COSINE,
        ),
        EmbeddingConfig(
            name="sparse", model_id=SPARSE_MODEL_ID, kind="sparse",
        ),
    ]

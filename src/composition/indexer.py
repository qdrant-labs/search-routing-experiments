"""Qdrant indexer for the d32/d33 selection artifact — one point per
training query, named vectors for dense + sparse so the labeling stage
can drive both retrieval directions off the same collection.

Follows `hybrid_search_rrf_dataset.indexer.CorpusIndexer`: UUID5 point
ids over `dataset:query_id` (Qdrant requires int or UUID), the raw
identifiers preserved in payload for filters."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict
from qdrant_client.models import Distance

from hybrid_search_rrf_dataset.indexer import (
    BaseIndexer,
    EmbeddingConfig,
)

DENSE_MODEL_ID = "BAAI/bge-small-en-v1.5"
DENSE_MODEL_SIZE = 384
SPARSE_MODEL_ID = "Qdrant/bm25"


class SelectionQuery(BaseModel):
    """One row of `data/composition/selection.parquet`."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    query_id: str
    query: str
    slice: str
    floors: list[str]
    checkable: bool
    label_lane: str


class SelectionIndexer(BaseIndexer[SelectionQuery]):
    """Indexes selection rows with `dense` + `sparse` vector slots — the
    two directions any downstream retrieval will run against."""

    item_type = SelectionQuery

    def item_id(self, item: SelectionQuery) -> str:
        # namespaced key so query_ids stay unique across source datasets
        return str(uuid.uuid5(
            uuid.NAMESPACE_DNS, f"{item.dataset}:{item.query_id}",
        ))

    def item_text(self, item: SelectionQuery) -> str:
        return item.query

    def item_payload(self, item: SelectionQuery) -> dict[str, Any]:
        return {
            "dataset": item.dataset,
            "query_id": item.query_id,
            "query": item.query,
            "slice": item.slice,
            "floors": item.floors,
            "checkable": item.checkable,
            "label_lane": item.label_lane,
        }


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

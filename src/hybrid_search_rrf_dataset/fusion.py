from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import ClassVar

import numpy as np
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Fusion,
    FusionQuery,
    Prefetch,
    Rrf,
    RrfQuery,
    ScoredPoint,
    SparseVector,
)

from hybrid_search_rrf_dataset.indexer import EmbeddingConfig


class StrategyName(StrEnum):
    PURE_RRF = "pure_rrf"
    WEIGHTED_RRF = "weighted_rrf"
    WEIGHTED_DBSF = "weighted_dbsf"


def dbsf_normalize(scores: list[float]) -> list[float]:
    """3-sigma DBSF: z-score, clip to [-3, 3], shift to [0, 1]."""
    if not scores:
        return []
    arr = np.asarray(scores, dtype=np.float64)
    if arr.std() == 0:
        return [0.5] * len(scores)
    z = np.clip((arr - arr.mean()) / arr.std(), -3.0, 3.0)
    return ((z + 3.0) / 6.0).tolist()


class FusionStrategy(ABC):
    """Fetches candidates from Qdrant and returns a fused ranking as {doc_id: score}."""

    name: ClassVar[StrategyName]

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_cfg: EmbeddingConfig,
        sparse_cfg: EmbeddingConfig,
        fetch_limit: int = 1000,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.dense_cfg = dense_cfg
        self.sparse_cfg = sparse_cfg
        self.fetch_limit = fetch_limit
        self._dense_model: TextEmbedding | None = None
        self._sparse_model: SparseTextEmbedding | None = None

    def _dense(self, text: str) -> list[float]:
        if self._dense_model is None:
            self._dense_model = TextEmbedding(
                self.dense_cfg.model_id, providers=self.dense_cfg.providers
            )
        return next(iter(self._dense_model.embed([text]))).tolist()

    def _sparse(self, text: str) -> SparseVector:
        if self._sparse_model is None:
            self._sparse_model = SparseTextEmbedding(
                self.sparse_cfg.model_id, providers=self.sparse_cfg.providers
            )
        s = next(iter(self._sparse_model.embed([text])))
        return SparseVector(indices=s.indices.tolist(), values=s.values.tolist())

    def _dense_hits(self, query: str) -> list[ScoredPoint]:
        return self.client.query_points(
            collection_name=self.collection_name,
            query=self._dense(query),
            using=self.dense_cfg.name,
            limit=self.fetch_limit,
            with_payload=True,
        ).points

    def _sparse_hits(self, query: str) -> list[ScoredPoint]:
        return self.client.query_points(
            collection_name=self.collection_name,
            query=self._sparse(query),
            using=self.sparse_cfg.name,
            limit=self.fetch_limit,
            with_payload=True,
        ).points

    def _prefetch_pair(self, query: str) -> list[Prefetch]:
        return [
            Prefetch(
                query=self._dense(query),
                using=self.dense_cfg.name,
                limit=self.fetch_limit,
            ),
            Prefetch(
                query=self._sparse(query),
                using=self.sparse_cfg.name,
                limit=self.fetch_limit,
            ),
        ]

    @abstractmethod
    def rank(
        self, query: str, dense_weight: float, sparse_weight: float
    ) -> dict[str, float]: ...


class WeightedFusionStrategy(FusionStrategy, ABC):
    """Marker base: strategies whose output actually depends on weights.

    GoldenSetBuilder and LLMFusionBuilder require this subtype so that
    weight-agnostic strategies (PureRRF) can't be passed by mistake.
    """


class PureRRFStrategy(FusionStrategy):
    """Baseline: Qdrant native RRF fusion in a single call. Weights are ignored."""

    name: ClassVar[StrategyName] = StrategyName.PURE_RRF

    def rank(
        self, query: str, dense_weight: float, sparse_weight: float
    ) -> dict[str, float]:
        hits = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=self._prefetch_pair(query),
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self.fetch_limit,
            with_payload=True,
        ).points
        return {h.payload["doc_id"]: h.score for h in hits if h.payload}


class WeightedRRFStrategy(WeightedFusionStrategy):
    """Qdrant-native weighted RRF via RrfQuery(rrf=Rrf(weights=[...]))."""

    name: ClassVar[StrategyName] = StrategyName.WEIGHTED_RRF

    def rank(
        self, query: str, dense_weight: float, sparse_weight: float
    ) -> dict[str, float]:
        hits = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=self._prefetch_pair(query),
            query=RrfQuery(rrf=Rrf(weights=[dense_weight, sparse_weight])),
            limit=self.fetch_limit,
            with_payload=True,
        ).points
        return {h.payload["doc_id"]: h.score for h in hits if h.payload}


class WeightedDBSFStrategy(WeightedFusionStrategy):
    """DBSF-normalize each source, then combine linearly with weights (client-side)."""

    name: ClassVar[StrategyName] = StrategyName.WEIGHTED_DBSF

    def rank(
        self, query: str, dense_weight: float, sparse_weight: float
    ) -> dict[str, float]:
        dense_hits = self._dense_hits(query)
        sparse_hits = self._sparse_hits(query)

        dense_ids = [h.payload["doc_id"] for h in dense_hits if h.payload]
        sparse_ids = [h.payload["doc_id"] for h in sparse_hits if h.payload]
        dense_scores = dbsf_normalize([h.score for h in dense_hits])
        sparse_scores = dbsf_normalize([h.score for h in sparse_hits])

        merged: dict[str, float] = {}
        for doc_id, score in zip(dense_ids, dense_scores, strict=True):
            merged[doc_id] = dense_weight * score
        for doc_id, score in zip(sparse_ids, sparse_scores, strict=True):
            merged[doc_id] = merged.get(doc_id, 0.0) + sparse_weight * score
        return merged

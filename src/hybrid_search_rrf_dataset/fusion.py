from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import ClassVar

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Fusion,
    FusionQuery,
    Prefetch,
    ScoredPoint,
    SparseVector,
)

from hybrid_search_rrf_dataset.indexer import EmbeddingConfig


class StrategyName(StrEnum):
    """The router's decision surface — exactly three routes.

    Weighted variants (weighted_rrf, weighted_dbsf) were removed 2026-07-28
    per SPEC d37: a continuous dense/sparse weight is not a label the router
    can emit, so tuning one served no downstream consumer.
    """

    DENSE_ONLY = "dense_only"
    PURE_RRF = "pure_rrf"
    SPARSE_ONLY = "sparse_only"


class FusionStrategy(ABC):
    """Fetches candidates from Qdrant and returns a ranking as {doc_id: score}."""

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

    @staticmethod
    def _ranking(hits: list[ScoredPoint]) -> dict[str, float]:
        return {h.payload["doc_id"]: h.score for h in hits if h.payload}

    @abstractmethod
    def rank(self, query: str) -> dict[str, float]: ...


class DenseOnlyStrategy(FusionStrategy):
    """Dense retrieval only, raw cosine similarity preserved."""

    name: ClassVar[StrategyName] = StrategyName.DENSE_ONLY

    def rank(self, query: str) -> dict[str, float]:
        return self._ranking(self._dense_hits(query))


class SparseOnlyStrategy(FusionStrategy):
    """Sparse (BM25) retrieval only, raw BM25 score preserved."""

    name: ClassVar[StrategyName] = StrategyName.SPARSE_ONLY

    def rank(self, query: str) -> dict[str, float]:
        return self._ranking(self._sparse_hits(query))


class PureRRFStrategy(FusionStrategy):
    """Qdrant-native RRF fusion in a single call — matches production `Fusion::Rrf`.

    RRF fuses by rank position, which dilutes a confident top-1: a doc ranked
    first by dense and 40th by sparse loses to one ranked third by both. That
    is the behaviour the router exists to avoid paying for (SPEC d37g).
    """

    name: ClassVar[StrategyName] = StrategyName.PURE_RRF

    def rank(self, query: str) -> dict[str, float]:
        hits = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=self._prefetch_pair(query),
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self.fetch_limit,
            with_payload=True,
        ).points
        return self._ranking(hits)

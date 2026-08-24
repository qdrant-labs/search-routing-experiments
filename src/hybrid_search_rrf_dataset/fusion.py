from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import StrEnum
from typing import ClassVar

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Document,
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


TIE_TOLERANCE = 1e-9
"""Two route scores within this are the same score (SPEC d41e: ties are
exact; near-ties stay `routes_differ` with a thin margin)."""

SERVING_COST: dict[StrategyName, int] = {
    StrategyName.SPARSE_ONLY: 0,
    StrategyName.DENSE_ONLY: 1,
    StrategyName.PURE_RRF: 2,
}
"""Relative query-time cost (SPEC d41c). pure_rrf above each component is
structural — it runs both plus fusion; sparse < dense (no query-side
transformer pass) is an assumption until the latency benchmark lands. A flip
re-derives the route column in seconds via `RouteLabels.rederive`."""


def derive_route(
    scores: Mapping[str, float], tolerance: float = TIE_TOLERANCE
) -> StrategyName:
    """The serving decision for one query — quality first, cost second.

    The cheapest route among those achieving the maximal score (SPEC d41b).
    Cost never overrides quality: it only arbitrates exact ties, so sparse at
    0.0 cannot take a row whose tied-best is {dense 1.0, rrf 1.0}. All-zero
    scores degenerate to the cheapest route overall — whether that is a
    *label* is the caller's concern (labels.py stores null there).
    """
    best = max(scores.values())
    winners = [
        StrategyName(name) for name, score in scores.items()
        if best - score <= tolerance
    ]
    return min(winners, key=SERVING_COST.__getitem__)


class FusionStrategy(ABC):
    """Fetches candidates from Qdrant and returns a ranking as {doc_id: score}."""

    name: ClassVar[StrategyName]

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_cfg: EmbeddingConfig,
        sparse_cfg: EmbeddingConfig,
        fetch_limit: int = 50,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.dense_cfg = dense_cfg
        self.sparse_cfg = sparse_cfg
        self.fetch_limit = fetch_limit
        self._dense_model: TextEmbedding | None = None
        self._sparse_model: SparseTextEmbedding | None = None

    def _dense(self, text: str) -> list[float] | Document:
        # query-side role marker, paired with the indexer's doc_prompt; "" for
        # bge/bm25, so leg-1 queries embed exactly as before
        text = self.dense_cfg.query_prompt + text
        if self.dense_cfg.cloud:  # no local ONNX build; Qdrant embeds server-side
            return Document(
                text=text, model=self.dense_cfg.model_id,
                options=self.dense_cfg.provider_options,
            )
        if self._dense_model is None:
            self._dense_model = TextEmbedding(
                self.dense_cfg.model_id, providers=self.dense_cfg.providers
            )
        return next(iter(self._dense_model.embed([text]))).tolist()

    def _sparse(self, text: str) -> SparseVector | Document:
        if self.sparse_cfg.cloud:
            return Document(
                text=text, model=self.sparse_cfg.model_id,
                options=self.sparse_cfg.provider_options,
            )
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
            with_payload=["doc_id"],
        ).points

    def _sparse_hits(self, query: str) -> list[ScoredPoint]:
        return self.client.query_points(
            collection_name=self.collection_name,
            query=self._sparse(query),
            using=self.sparse_cfg.name,
            limit=self.fetch_limit,
            with_payload=["doc_id"],
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
            with_payload=["doc_id"],
        ).points
        return self._ranking(hits)

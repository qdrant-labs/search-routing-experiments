from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (
    Document,
    Fusion,
    FusionQuery,
    Prefetch,
    ScoredPoint,
    SparseVector,
)
from tqdm.auto import tqdm

from hybrid_search_rrf_dataset.indexer import (
    _RETRYABLE_STATUS,
    EmbeddingConfig,
    st_sparse_vectors,
)

from hybrid_search_rrf_dataset.routes import (  # re-exported: fusion stays the public spelling
    SERVING_COST,
    TIE_TOLERANCE,
    StrategyName,
)

_QUERY_MAX_ATTEMPTS = 4
_QUERY_BACKOFF_S = 5.0
"""Searches share the upsert path's failure modes — a saturated cluster returns
500 'Operation Search timed out' rather than queueing, and cloud-inference
embeds ride inside the request — so they get the same transient-only retry
with doubling backoff (5s, 10s, 20s). The sleep also drains concurrent
pressure, which is usually what caused the timeout."""



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
        if self.sparse_cfg.engine == "sentence_transformers":  # learned-sparse query side
            # cpu: the query encode is tokenizer + IDF weights, and keeping MPS
            # out of the search path means a wedged GPU can't stall a labelling run
            return st_sparse_vectors(self.sparse_cfg.model_id, [text],
                                     is_query=True, device="cpu")[0]
        if self._sparse_model is None:
            self._sparse_model = SparseTextEmbedding(
                self.sparse_cfg.model_id, providers=self.sparse_cfg.providers,
                **(self.sparse_cfg.model_options or {}),
            )
        # query_embed, not embed: fastembed's embed() is the doc side (BM25 TF
        # saturation + length norm); the query side must stay unweighted
        s = next(iter(self._sparse_model.query_embed([text])))
        return SparseVector(indices=s.indices.tolist(), values=s.values.tolist())

    def _query_with_retry(self, **kwargs: Any) -> list[ScoredPoint]:
        for attempt in range(_QUERY_MAX_ATTEMPTS):
            try:
                return self.client.query_points(
                    collection_name=self.collection_name, **kwargs
                ).points
            except (UnexpectedResponse, ResponseHandlingException) as exc:
                status = getattr(exc, "status_code", None)
                transient = (
                    isinstance(exc, ResponseHandlingException)
                    or status in _RETRYABLE_STATUS
                )
                if not transient or attempt == _QUERY_MAX_ATTEMPTS - 1:
                    raise
                delay = _QUERY_BACKOFF_S * (2**attempt)
                tqdm.write(
                    f"{self.collection_name}: search {status or type(exc).__name__} "
                    f"(attempt {attempt + 1}/{_QUERY_MAX_ATTEMPTS}), "
                    f"retrying in {delay:.0f}s"
                )
                time.sleep(delay)
        raise AssertionError("unreachable")  # the loop returns or raises

    def _dense_hits(self, query: str) -> list[ScoredPoint]:
        return self._query_with_retry(
            query=self._dense(query),
            using=self.dense_cfg.name,
            limit=self.fetch_limit,
            with_payload=["doc_id"],
        )

    def _sparse_hits(self, query: str) -> list[ScoredPoint]:
        return self._query_with_retry(
            query=self._sparse(query),
            using=self.sparse_cfg.name,
            limit=self.fetch_limit,
            with_payload=["doc_id"],
        )

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
        hits = self._query_with_retry(
            prefetch=self._prefetch_pair(query),
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self.fetch_limit,
            with_payload=["doc_id"],
        )
        return self._ranking(hits)

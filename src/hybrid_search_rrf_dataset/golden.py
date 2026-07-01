from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

import numpy as np
import pandas as pd
import requests
from fastembed import SparseTextEmbedding, TextEmbedding
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
from ranx import Qrels, Run, evaluate
from tqdm.auto import tqdm

from hybrid_search_rrf_dataset.indexer import EmbeddingConfig
from hybrid_search_rrf_dataset.retrieval import RetrievalDataset


class FusionRow(BaseModel):
    """Shared row schema for any hybrid-fusion result.

    Concrete row types (GoldenDataset, LLMFusionDataset) inherit without
    adding fields — the class identity marks how `alpha` was chosen.
    """

    id: int
    query_id: str
    dataset_name: str
    query: str
    qdrant_answer: list[str]
    gold_qrel: dict[str, int]
    metric: float
    metric_name: str
    alpha: float


class GoldenDataset(FusionRow):
    """Alpha chosen by NDCG grid sweep — the ground-truth optimum."""


class LLMFusionDataset(FusionRow):
    """Alpha predicted by an external LLM scorer."""


T = TypeVar("T", bound=FusionRow)


def dbsf_normalize(scores: list[float]) -> list[float]:
    """3-sigma DBSF: z-score, clip to [-3, 3], shift to [0, 1]."""
    if not scores:
        return []
    arr = np.asarray(scores, dtype=np.float64)
    if arr.std() == 0:
        return [0.5] * len(scores)
    z = np.clip((arr - arr.mean()) / arr.std(), -3.0, 3.0)
    return ((z + 3.0) / 6.0).tolist()


def weighted_combine(
    dense: dict[str, float],
    sparse: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    """final = (1-alpha)*dense + alpha*sparse. alpha->0: semantic, alpha->1: sparse."""
    merged: dict[str, float] = {}
    for doc_id, score in dense.items():
        merged[doc_id] = (1.0 - alpha) * score
    for doc_id, score in sparse.items():
        merged[doc_id] = merged.get(doc_id, 0.0) + alpha * score
    return merged


def ndcg_score(
    merged: dict[str, float], gold_qrel: dict[str, int], top_k: int
) -> float:
    qrels = Qrels({"q": gold_qrel})
    run = Run({"q": merged})
    return float(evaluate(qrels, run, f"ndcg@{top_k}"))


def top_k_ids(merged: dict[str, float], k: int) -> list[str]:
    return [
        doc_id for doc_id, _ in sorted(merged.items(), key=lambda kv: -kv[1])[:k]
    ]


class FusionBuilder(ABC, Generic[T]):
    """Base for hybrid-fusion row builders.

    Owns Qdrant search + embedding + DBSF machinery. Subclasses decide how
    alpha is chosen per query (grid sweep vs external predictor) and which
    concrete row type they emit.
    """

    row_type: ClassVar[type[FusionRow]]
    default_dir: ClassVar[Path] = Path("data/fusion")

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_cfg: EmbeddingConfig,
        sparse_cfg: EmbeddingConfig,
        top_k: int = 10,
        fetch_limit: int = 1000,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.dense_cfg = dense_cfg
        self.sparse_cfg = sparse_cfg
        self.top_k = top_k
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

    def _fetch(self, query: str) -> tuple[dict[str, float], dict[str, float]]:
        dense_hits = self.client.query_points(
            collection_name=self.collection_name,
            query=self._dense(query),
            using=self.dense_cfg.name,
            limit=self.fetch_limit,
            with_payload=True,
        ).points
        sparse_hits = self.client.query_points(
            collection_name=self.collection_name,
            query=self._sparse(query),
            using=self.sparse_cfg.name,
            limit=self.fetch_limit,
            with_payload=True,
        ).points

        dense_ids = [h.payload["doc_id"] for h in dense_hits if h.payload]
        sparse_ids = [h.payload["doc_id"] for h in sparse_hits if h.payload]
        dense_scores = dbsf_normalize([h.score for h in dense_hits])
        sparse_scores = dbsf_normalize([h.score for h in sparse_hits])
        return (
            dict(zip(dense_ids, dense_scores, strict=True)),
            dict(zip(sparse_ids, sparse_scores, strict=True)),
        )

    def _iter_queries(
        self, dataset: RetrievalDataset, start_id: int
    ) -> Iterator[tuple[int, str, str, dict[str, int]]]:
        queries_df = dataset.queries()
        qrels_by_query: dict[str, dict[str, int]] = {
            str(qid): dict(
                zip(
                    g["doc_id"].astype(str),
                    g["relevance"].astype(int),
                    strict=True,
                )
            )
            for qid, g in dataset.qrels().groupby("query_id")
        }
        for i, q in enumerate(
            tqdm(
                queries_df.itertuples(index=False),
                total=len(queries_df),
                desc=f"{type(self).__name__.lower()}:{dataset.name}",
            )
        ):
            qid = str(q.query_id)
            gold = qrels_by_query.get(qid, {})
            if not gold:
                continue
            yield start_id + i, qid, str(q.text), gold

    @abstractmethod
    def build_row(
        self,
        row_id: int,
        query_id: str,
        query: str,
        gold_qrel: dict[str, int],
        dataset_name: str,
    ) -> T: ...

    def build(self, dataset: RetrievalDataset, start_id: int = 0) -> list[T]:
        return [
            self.build_row(rid, qid, q, gold, dataset.name)
            for rid, qid, q, gold in self._iter_queries(dataset, start_id)
        ]

    def save(self, rows: list[T], path: Path | str | None = None) -> Path:
        out = Path(path or self.default_dir)
        out.mkdir(parents=True, exist_ok=True)
        file = out / "rows.parquet"
        pd.DataFrame([r.model_dump() for r in rows]).to_parquet(file, index=False)
        return file

    @classmethod
    def load(cls, path: Path | str | None = None) -> list[T]:
        file = Path(path or cls.default_dir) / "rows.parquet"
        df = pd.read_parquet(file)
        return [cls.row_type(**row) for row in df.to_dict("records")]

    def build_or_load(
        self,
        dataset: RetrievalDataset,
        path: Path | str | None = None,
        start_id: int = 0,
    ) -> list[T]:
        """Load if the parquet exists, otherwise build and save."""
        file = Path(path or self.default_dir) / "rows.parquet"
        if file.exists():
            return type(self).load(path)
        rows = self.build(dataset, start_id=start_id)
        self.save(rows, path)
        return rows


class GoldenSetBuilder(FusionBuilder[GoldenDataset]):
    """Sweep alpha over [0, 1] and pick the alpha that maximises NDCG@k."""

    row_type: ClassVar[type[FusionRow]] = GoldenDataset
    default_dir: ClassVar[Path] = Path("data/golden")

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_cfg: EmbeddingConfig,
        sparse_cfg: EmbeddingConfig,
        alpha_step: float = 0.1,
        top_k: int = 10,
        fetch_limit: int = 1000,
    ) -> None:
        super().__init__(
            client, collection_name, dense_cfg, sparse_cfg, top_k, fetch_limit
        )
        self.alpha_step = alpha_step

    def build_row(
        self,
        row_id: int,
        query_id: str,
        query: str,
        gold_qrel: dict[str, int],
        dataset_name: str,
    ) -> GoldenDataset:
        dense, sparse = self._fetch(query)
        best_alpha, best_ndcg, best_ranking = 0.0, -1.0, []
        for alpha in np.arange(0.0, 1.0 + self.alpha_step / 2, self.alpha_step):
            merged = weighted_combine(dense, sparse, float(alpha))
            score = ndcg_score(merged, gold_qrel, self.top_k)
            if score > best_ndcg:
                best_ndcg = score
                best_alpha = float(alpha)
                best_ranking = top_k_ids(merged, self.top_k)

        return GoldenDataset(
            id=row_id,
            query_id=query_id,
            dataset_name=dataset_name,
            query=query,
            qdrant_answer=best_ranking,
            gold_qrel=gold_qrel,
            metric=best_ndcg,
            metric_name=f"NDCG@{self.top_k}",
            alpha=best_alpha,
        )


class LLMFusionBuilder(FusionBuilder[LLMFusionDataset]):
    """Ask an external LLM service for a 0-9 score, normalize to alpha in [0, 1]."""

    row_type: ClassVar[type[FusionRow]] = LLMFusionDataset
    default_dir: ClassVar[Path] = Path("data/llm_fusion")

    _LLM_SCORE_MAX = 9

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_cfg: EmbeddingConfig,
        sparse_cfg: EmbeddingConfig,
        api_url: str | None = None,
        api_key: str | None = None,
        top_k: int = 10,
        fetch_limit: int = 1000,
        request_timeout: float = 30.0,
    ) -> None:
        super().__init__(
            client, collection_name, dense_cfg, sparse_cfg, top_k, fetch_limit
        )
        url = api_url or os.getenv("QDRANT_LLM_FUSION_URL")
        key = api_key or os.getenv("QDRANT_LLM_FUSION_KEY")
        if not url or not key:
            raise RuntimeError(
                "LLMFusionBuilder needs api_url + api_key (or "
                "QDRANT_LLM_FUSION_URL + QDRANT_LLM_FUSION_KEY env vars)."
            )
        self._api_url = url
        self._api_key = key
        self._timeout = request_timeout

    def _predict_alpha(self, query: str) -> float:
        response = requests.post(
            self._api_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"text": query},
            timeout=self._timeout,
        )
        response.raise_for_status()
        score = int(response.json()["score"])
        return score / self._LLM_SCORE_MAX

    def build_row(
        self,
        row_id: int,
        query_id: str,
        query: str,
        gold_qrel: dict[str, int],
        dataset_name: str,
    ) -> LLMFusionDataset:
        alpha = self._predict_alpha(query)
        dense, sparse = self._fetch(query)
        merged = weighted_combine(dense, sparse, alpha)
        metric = ndcg_score(merged, gold_qrel, self.top_k)
        ranking = top_k_ids(merged, self.top_k)

        return LLMFusionDataset(
            id=row_id,
            query_id=query_id,
            dataset_name=dataset_name,
            query=query,
            qdrant_answer=ranking,
            gold_qrel=gold_qrel,
            metric=metric,
            metric_name=f"NDCG@{self.top_k}",
            alpha=alpha,
        )

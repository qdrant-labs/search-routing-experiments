from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel
from ranx import Qrels, Run, evaluate
from tqdm.auto import tqdm

from hybrid_search_rrf_dataset.fusion import (
    FusionStrategy,
    StrategyName,
    WeightedFusionStrategy,
)
from hybrid_search_rrf_dataset.retrieval import RetrievalDataset


class FusionRow(BaseModel):
    """Shared row schema for any hybrid-fusion result.

    Concrete row types (GoldenDataset, LLMFusionDataset, BaselineDataset)
    inherit without adding fields — the class identity marks how `alpha`
    was chosen. `strategy_name` marks which fusion algorithm produced the ranking.
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
    strategy_name: StrategyName


class GoldenDataset(FusionRow):
    """Alpha chosen by NDCG grid sweep — the ground-truth optimum."""


class LLMFusionDataset(FusionRow):
    """Alpha predicted by an external LLM scorer."""


class BaselineDataset(FusionRow):
    """No alpha selection — strategy evaluated at fixed weights."""


class HybridRoutingDataset(FusionRow):
    """LLM classifier score routes each query to Dense-only, Hybrid (RRF), or Sparse-only.

    `strategy_name` records the actual retriever picked per query (dense_only /
    pure_rrf / sparse_only). `alpha` is the *effective* alpha of that route
    (0.0 / 0.5 / 1.0) so the alpha diagnostic plots remain interpretable.
    """


class GoldenRoutingDataset(FusionRow):
    """Best of the three routing endpoints per query — oracle counterpart to
    HybridRoutingDataset on the same discrete decision surface.

    Same alpha convention as HybridRoutingDataset: Dense→0.0, Hybrid→0.5,
    Sparse→1.0 — a continuous golden alpha of, say, 0.2 falls in the Dense
    bucket and the row stores 0.0.
    """


T = TypeVar("T", bound=FusionRow)


class LLMScoreClient:
    """Client for the query-classification LLM API returning an integer score in [0, 9].

    Same endpoint that Qdrant page-search production hits (`fusion.qdrant.tech/v1/classify`,
    per rust_search/skills/fusion.rs). Shared by builders that consume the score
    in different ways: LLMFusionBuilder normalizes to alpha, HybridRoutingBuilder
    routes on the raw integer.
    """

    SCORE_MAX = 9

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        request_timeout: float = 30.0,
    ) -> None:
        url = api_url or os.getenv("QDRANT_LLM_FUSION_URL")
        key = api_key or os.getenv("QDRANT_LLM_FUSION_KEY")
        if not url or not key:
            raise RuntimeError(
                "LLMScoreClient needs api_url + api_key (or "
                "QDRANT_LLM_FUSION_URL + QDRANT_LLM_FUSION_KEY env vars)."
            )
        self._api_url = url
        self._api_key = key
        self._timeout = request_timeout

    def score(self, query: str) -> int:
        response = requests.post(
            self._api_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"text": query},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return int(response.json()["score"])


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

    Delegates all Qdrant + embedding + fusion math to the injected
    `FusionStrategy`. Subclasses decide only how alpha (weight split) is chosen.
    """

    row_type: ClassVar[type[FusionRow]]
    default_dir: ClassVar[Path] = Path("data/fusion")

    def __init__(self, strategy: FusionStrategy, top_k: int = 10) -> None:
        self.strategy = strategy
        self.top_k = top_k

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
        rows: list[T] = []
        for rec in df.to_dict("records"):
            # Parquet stores dict columns as STRUCT, unifying keys across all rows
            # and filling missing entries with None (ints also come back as floats).
            # Strip that padding so gold_qrel matches its dict[str, int] contract.
            gq = rec.get("gold_qrel")
            if isinstance(gq, dict):
                rec["gold_qrel"] = {k: int(v) for k, v in gq.items() if v is not None}
            rows.append(cls.row_type(**rec))
        return rows

    def build_or_load(
        self,
        dataset: RetrievalDataset,
        path: Path | str | None = None,
        start_id: int = 0,
    ) -> list[T]:
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
        strategy: WeightedFusionStrategy,
        alpha_step: float = 0.1,
        top_k: int = 10,
    ) -> None:
        super().__init__(strategy, top_k)
        self.alpha_step = alpha_step

    def build_row(
        self,
        row_id: int,
        query_id: str,
        query: str,
        gold_qrel: dict[str, int],
        dataset_name: str,
    ) -> GoldenDataset:
        best_alpha, best_ndcg, best_ranking = 0.0, -1.0, []
        for alpha in np.arange(0.0, 1.0 + self.alpha_step / 2, self.alpha_step):
            merged = self.strategy.rank(query, 1.0 - float(alpha), float(alpha))
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
            strategy_name=self.strategy.name,
        )


class LLMFusionBuilder(FusionBuilder[LLMFusionDataset]):
    """Ask an external LLM service for a 0-9 score, normalize to alpha in [0, 1]."""

    row_type: ClassVar[type[FusionRow]] = LLMFusionDataset
    default_dir: ClassVar[Path] = Path("data/llm_fusion")

    def __init__(
        self,
        strategy: WeightedFusionStrategy,
        client: LLMScoreClient | None = None,
        top_k: int = 10,
    ) -> None:
        super().__init__(strategy, top_k)
        self._client = client or LLMScoreClient()

    def build_row(
        self,
        row_id: int,
        query_id: str,
        query: str,
        gold_qrel: dict[str, int],
        dataset_name: str,
    ) -> LLMFusionDataset:
        alpha = self._client.score(query) / LLMScoreClient.SCORE_MAX
        merged = self.strategy.rank(query, 1.0 - alpha, alpha)
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
            strategy_name=self.strategy.name,
        )


class BaselineBuilder(FusionBuilder[BaselineDataset]):
    """No alpha selection — run the strategy at fixed weights.

    For PureRRFStrategy the weights are ignored, giving the true baseline.
    For weighted strategies this is "no alpha tuning, equal weights".
    """

    row_type: ClassVar[type[FusionRow]] = BaselineDataset
    default_dir: ClassVar[Path] = Path("data/baseline")

    def __init__(
        self,
        strategy: FusionStrategy,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        top_k: int = 10,
    ) -> None:
        super().__init__(strategy, top_k)
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def build_row(
        self,
        row_id: int,
        query_id: str,
        query: str,
        gold_qrel: dict[str, int],
        dataset_name: str,
    ) -> BaselineDataset:
        merged = self.strategy.rank(query, self.dense_weight, self.sparse_weight)
        metric = ndcg_score(merged, gold_qrel, self.top_k)
        ranking = top_k_ids(merged, self.top_k)

        return BaselineDataset(
            id=row_id,
            query_id=query_id,
            dataset_name=dataset_name,
            query=query,
            qdrant_answer=ranking,
            gold_qrel=gold_qrel,
            metric=metric,
            metric_name=f"NDCG@{self.top_k}",
            alpha=self.sparse_weight,
            strategy_name=self.strategy.name,
        )


class GoldenRoutingBuilder(FusionBuilder[GoldenRoutingDataset]):
    """Sweep the three routing endpoints per query and pick the metric-maximizing one.

    Direct analog of GoldenSetBuilder: same sweep-then-pick loop, but the sweep
    space is the router's finite decision surface {Dense, Hybrid, Sparse} rather
    than the continuous alpha grid. Regret of HybridRoutingBuilder against this
    oracle isolates classifier quality from the coarseness of the routing surface
    itself.

    The Hybrid endpoint should be `PureRRFStrategy` to match production and
    HybridRoutingBuilder — pass anything else only for ablation experiments.
    """

    row_type: ClassVar[type[FusionRow]] = GoldenRoutingDataset
    default_dir: ClassVar[Path] = Path("data/golden_routing")

    def __init__(
        self,
        dense_strategy: FusionStrategy,
        hybrid_strategy: FusionStrategy,
        sparse_strategy: FusionStrategy,
        top_k: int = 10,
    ) -> None:
        # Base needs *a* strategy; hybrid is the natural default and self.strategy
        # is not read by this builder (build_row picks per-route below).
        super().__init__(hybrid_strategy, top_k)
        self._routes: list[tuple[FusionStrategy, float]] = [
            (dense_strategy, 0.0),
            (hybrid_strategy, 0.5),
            (sparse_strategy, 1.0),
        ]

    def build_row(
        self,
        row_id: int,
        query_id: str,
        query: str,
        gold_qrel: dict[str, int],
        dataset_name: str,
    ) -> GoldenRoutingDataset:
        best_strategy, best_alpha = self._routes[0]
        best_ndcg, best_ranking = -1.0, []
        for strategy, effective_alpha in self._routes:
            merged = strategy.rank(query, 0.0, 0.0)  # routing endpoints ignore weights
            score = ndcg_score(merged, gold_qrel, self.top_k)
            if score > best_ndcg:
                best_ndcg = score
                best_strategy = strategy
                best_alpha = effective_alpha
                best_ranking = top_k_ids(merged, self.top_k)

        return GoldenRoutingDataset(
            id=row_id,
            query_id=query_id,
            dataset_name=dataset_name,
            query=query,
            qdrant_answer=best_ranking,
            gold_qrel=gold_qrel,
            metric=best_ndcg,
            metric_name=f"NDCG@{self.top_k}",
            alpha=best_alpha,
            strategy_name=best_strategy.name,
        )


class HybridRoutingBuilder(FusionBuilder[HybridRoutingDataset]):
    """Route each query to Dense / Hybrid / Sparse based on the LLM classifier score.

    Matches Qdrant page-search production (rust_search/src/skills/fusion.rs):
    integer score → { 0..dense_max: Dense, dense_max+1..hybrid_max: Hybrid, else: Sparse }.
    Any score outside [0, hybrid_max] falls through to Sparse — mirrors the Rust
    `_ => Bm25` catch-all, which handles both the 7..=9 tail and out-of-range values.
    Defaults (2, 6) reproduce production thresholds.

    Hybrid should be `PureRRFStrategy` to match production (`Fusion::Rrf`); pass a
    different strategy only for ablation experiments.
    """

    row_type: ClassVar[type[FusionRow]] = HybridRoutingDataset
    default_dir: ClassVar[Path] = Path("data/hybrid_routing")

    def __init__(
        self,
        dense_strategy: FusionStrategy,
        hybrid_strategy: FusionStrategy,
        sparse_strategy: FusionStrategy,
        dense_max: int = 2,
        hybrid_max: int = 6,
        client: LLMScoreClient | None = None,
        top_k: int = 10,
    ) -> None:
        if not 0 <= dense_max < hybrid_max <= LLMScoreClient.SCORE_MAX:
            raise ValueError(
                f"Thresholds must satisfy 0 <= dense_max < hybrid_max <= {LLMScoreClient.SCORE_MAX}, "
                f"got dense_max={dense_max}, hybrid_max={hybrid_max}."
            )
        # The base needs *a* strategy; hybrid is the natural default and self.strategy
        # is not read by this builder (routing picks per-query below).
        super().__init__(hybrid_strategy, top_k)
        self._dense = dense_strategy
        self._hybrid = hybrid_strategy
        self._sparse = sparse_strategy
        self._dense_max = dense_max
        self._hybrid_max = hybrid_max
        self._client = client or LLMScoreClient()

    def _route(self, score: int) -> tuple[FusionStrategy, float]:
        """Return (strategy, effective alpha) for a given classifier score."""
        if 0 <= score <= self._dense_max:
            return self._dense, 0.0
        if self._dense_max < score <= self._hybrid_max:
            return self._hybrid, 0.5
        return self._sparse, 1.0

    def build_row(
        self,
        row_id: int,
        query_id: str,
        query: str,
        gold_qrel: dict[str, int],
        dataset_name: str,
    ) -> HybridRoutingDataset:
        score = self._client.score(query)
        strategy, alpha = self._route(score)
        merged = strategy.rank(query, 0.0, 0.0)  # all routed strategies ignore weights
        metric = ndcg_score(merged, gold_qrel, self.top_k)
        ranking = top_k_ids(merged, self.top_k)

        return HybridRoutingDataset(
            id=row_id,
            query_id=query_id,
            dataset_name=dataset_name,
            query=query,
            qdrant_answer=ranking,
            gold_qrel=gold_qrel,
            metric=metric,
            metric_name=f"NDCG@{self.top_k}",
            alpha=alpha,
            strategy_name=strategy.name,
        )

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from hybrid_search_rrf_dataset.fusion import (
    FusionStrategy,
    StrategyName,
    WeightedFusionStrategy,
)
from hybrid_search_rrf_dataset.objective import Objective, RouterObjective
from hybrid_search_rrf_dataset.qrels import QrelStore
from hybrid_search_rrf_dataset.retrieval import RetrievalDataset


class FusionRow(BaseModel):
    """Shared row schema for any hybrid-fusion result.

    Concrete row types (GoldenDataset, LLMFusionDataset, BaselineDataset)
    inherit without adding fields — the class identity marks how `alpha`
    was chosen. `strategy_name` marks which fusion algorithm produced the ranking.

    Judgments deliberately live outside the row, in `QrelStore`: as a per-row
    dict they became a parquet struct with one field per distinct doc_id in the
    file. Rejoin on (dataset_name, query_id) when a metric needs recomputing.
    """

    id: int
    query_id: str
    dataset_name: str
    query: str
    qdrant_answer: list[str]
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

    Unlike the other row types this keeps the roads not taken, so a changed
    objective, a latency margin, or a tie-detection rule can be re-derived
    without touching Qdrant again. Both fields are keyed by strategy name —
    three stable keys, which parquet encodes as a fixed struct; keying
    anything by doc_id would unify every doc in the file into one schema.
    Defaulted so rows written before this field existed still load.
    """

    route_scores: dict[str, float] = Field(default_factory=dict)
    route_rankings: dict[str, list[str]] = Field(default_factory=dict)


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


class FusionBuilder(ABC, Generic[T]):
    """Base for hybrid-fusion row builders.

    Delegates all Qdrant + embedding + fusion math to the injected
    `FusionStrategy`, and all scoring to the injected `Objective`. Subclasses
    decide only how alpha (weight split) is chosen.
    """

    row_type: ClassVar[type[FusionRow]]
    default_dir: ClassVar[Path] = Path("data/fusion")

    def __init__(
        self,
        strategy: FusionStrategy,
        *,
        objective: Objective | None = None,
    ) -> None:
        self.strategy = strategy
        self.objective = objective or RouterObjective()

    def _iter_queries(
        self,
        dataset: RetrievalDataset,
        start_id: int,
        qrels: QrelStore | None = None,
    ) -> Iterator[tuple[int, str, str, dict[str, int]]]:
        store = qrels or QrelStore.from_dataset(dataset)
        by_query = store.lookup(dataset.name)
        queries_df = dataset.queries()
        for i, q in enumerate(
            tqdm(
                queries_df.itertuples(index=False),
                total=len(queries_df),
                desc=f"{type(self).__name__.lower()}:{dataset.name}",
            )
        ):
            qid = str(q.query_id)
            gold = by_query.get(qid, {})
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

    def build(
        self,
        dataset: RetrievalDataset,
        start_id: int = 0,
        qrels: QrelStore | None = None,
    ) -> list[T]:
        """Score `dataset`'s queries. Pass `qrels` to judge against a store
        other than the dataset's own — an LLM lane, or human and LLM merged."""
        return [
            self.build_row(rid, qid, q, gold, dataset.name)
            for rid, qid, q, gold in self._iter_queries(dataset, start_id, qrels)
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
        # Drop columns the schema no longer carries, so artifacts written before
        # judgments moved into QrelStore still load.
        fields = set(cls.row_type.model_fields)
        return [
            cls.row_type(**{k: v for k, v in rec.items() if k in fields})
            for rec in df.to_dict("records")
        ]

    def build_or_load(
        self,
        dataset: RetrievalDataset,
        path: Path | str | None = None,
        start_id: int = 0,
        qrels: QrelStore | None = None,
    ) -> list[T]:
        file = Path(path or self.default_dir) / "rows.parquet"
        if file.exists():
            rows = type(self).load(path)
            stale = {r.metric_name for r in rows} - {self.objective.name}
            if stale:
                raise ValueError(
                    f"{file} holds rows scored with {sorted(stale)}, but this "
                    f"builder is configured for {self.objective.name!r}. Delete "
                    f"the file to rebuild, or point at a different path — "
                    f"mixing objectives silently would corrupt every comparison."
                )
            return rows
        rows = self.build(dataset, start_id=start_id, qrels=qrels)
        self.save(rows, path)
        return rows


class GoldenSetBuilder(FusionBuilder[GoldenDataset]):
    """Sweep alpha over [0, 1] and pick the alpha that maximises the objective."""

    row_type: ClassVar[type[FusionRow]] = GoldenDataset
    default_dir: ClassVar[Path] = Path("data/golden")

    def __init__(
        self,
        strategy: WeightedFusionStrategy,
        *,
        objective: Objective | None = None,
        alpha_step: float = 0.1,
    ) -> None:
        super().__init__(strategy, objective=objective)
        self.alpha_step = alpha_step

    def build_row(
        self,
        row_id: int,
        query_id: str,
        query: str,
        gold_qrel: dict[str, int],
        dataset_name: str,
    ) -> GoldenDataset:
        best_alpha, best_score, best_ranking = 0.0, -1.0, []
        for alpha in np.arange(0.0, 1.0 + self.alpha_step / 2, self.alpha_step):
            merged = self.strategy.rank(query, 1.0 - float(alpha), float(alpha))
            score = self.objective.score(merged, gold_qrel)
            if score > best_score:
                best_score = score
                best_alpha = float(alpha)
                best_ranking = self.objective.ordered(merged)

        return GoldenDataset(
            id=row_id,
            query_id=query_id,
            dataset_name=dataset_name,
            query=query,
            qdrant_answer=best_ranking,
            metric=best_score,
            metric_name=self.objective.name,
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
        *,
        objective: Objective | None = None,
    ) -> None:
        super().__init__(strategy, objective=objective)
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

        return LLMFusionDataset(
            id=row_id,
            query_id=query_id,
            dataset_name=dataset_name,
            query=query,
            qdrant_answer=self.objective.ordered(merged),
            metric=self.objective.score(merged, gold_qrel),
            metric_name=self.objective.name,
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
        *,
        objective: Objective | None = None,
    ) -> None:
        super().__init__(strategy, objective=objective)
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

        return BaselineDataset(
            id=row_id,
            query_id=query_id,
            dataset_name=dataset_name,
            query=query,
            qdrant_answer=self.objective.ordered(merged),
            metric=self.objective.score(merged, gold_qrel),
            metric_name=self.objective.name,
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
        *,
        objective: Objective | None = None,
    ) -> None:
        # Base needs *a* strategy; hybrid is the natural default and self.strategy
        # is not read by this builder (build_row picks per-route below).
        super().__init__(hybrid_strategy, objective=objective)
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
        scores: dict[str, float] = {}
        rankings: dict[str, list[str]] = {}
        for strategy, _ in self._routes:
            merged = strategy.rank(query, 0.0, 0.0)  # routing endpoints ignore weights
            scores[strategy.name] = self.objective.score(merged, gold_qrel)
            rankings[strategy.name] = self.objective.ordered(merged)

        # Strict `>` over routes in [dense, hybrid, sparse] order, so a tie
        # resolves to the cheapest route to serve. Deliberate: hybrid pays for
        # two retrievals plus fusion and should not win on an exact tie.
        best_strategy, best_alpha = max(
            self._routes,
            key=lambda route: scores[route[0].name],
        )

        return GoldenRoutingDataset(
            id=row_id,
            query_id=query_id,
            dataset_name=dataset_name,
            query=query,
            qdrant_answer=rankings[best_strategy.name],
            metric=scores[best_strategy.name],
            metric_name=self.objective.name,
            alpha=best_alpha,
            strategy_name=best_strategy.name,
            route_scores=scores,
            route_rankings=rankings,
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
        *,
        objective: Objective | None = None,
    ) -> None:
        if not 0 <= dense_max < hybrid_max <= LLMScoreClient.SCORE_MAX:
            raise ValueError(
                f"Thresholds must satisfy 0 <= dense_max < hybrid_max <= {LLMScoreClient.SCORE_MAX}, "
                f"got dense_max={dense_max}, hybrid_max={hybrid_max}."
            )
        # The base needs *a* strategy; hybrid is the natural default and self.strategy
        # is not read by this builder (routing picks per-query below).
        super().__init__(hybrid_strategy, objective=objective)
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

        return HybridRoutingDataset(
            id=row_id,
            query_id=query_id,
            dataset_name=dataset_name,
            query=query,
            qdrant_answer=self.objective.ordered(merged),
            metric=self.objective.score(merged, gold_qrel),
            metric_name=self.objective.name,
            alpha=alpha,
            strategy_name=strategy.name,
        )

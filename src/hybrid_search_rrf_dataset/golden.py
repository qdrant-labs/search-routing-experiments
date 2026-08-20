from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, Field
from tqdm.auto import tqdm

from hybrid_search_rrf_dataset.fusion import (
    FusionStrategy,
    StrategyName,
    derive_route,
)
from hybrid_search_rrf_dataset.objective import Objective, RouterObjective
from hybrid_search_rrf_dataset.qrels import QrelStore
from hybrid_search_rrf_dataset.retrieval import RetrievalDataset


class FusionRow(BaseModel):
    """Shared row schema for any retrieval-route result.

    Concrete row types add no fields — the class identity records *how* the
    route was chosen (fixed, classifier, or oracle). `strategy_name` records
    which route actually produced the ranking.

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
    strategy_name: StrategyName
    provenance: str = "natural"
    """The query's own origin (`RetrievalDataset.provenance()`) — natural
    unless the source declares otherwise. Defaulted so rows written before
    this field existed still load."""


class BaselineDataset(FusionRow):
    """One fixed route applied to every query — the constant-route baseline a
    router has to beat (SPEC d37h)."""


class HybridRoutingDataset(FusionRow):
    """Route picked per query by the LLM classifier score."""


class GoldenRoutingDataset(FusionRow):
    """Best of the three routes per query — the oracle ceiling.

    Unlike the other row types this keeps the roads not taken, so a changed
    objective, a latency margin, or a tie-detection rule can be re-derived
    without touching Qdrant again. Both fields are keyed by strategy name —
    three stable keys, which parquet encodes as a fixed struct; keying anything
    by doc_id would unify every doc in the file into one schema. Defaulted so
    rows written before these fields existed still load.
    """

    route_scores: dict[str, float] = Field(default_factory=dict)
    route_rankings: dict[str, list[str]] = Field(default_factory=dict)
    route_raw_scores: dict[str, list[float]] = Field(default_factory=dict)
    """Each route's own retrieval score (cosine/BM25/RRF-fused), parallel to
    `route_rankings[route]` by position"""


REGIME_SUFFIX = ".provenance.json"
"""Sidecar naming, shared with `scripts/rederive_labels.py`."""


class ScoringRegime(BaseModel):
    """The scoring inputs `metric_name` cannot show — the relevance threshold
    and each route's fetch depth.

    Written beside a cache and compared on load, because `Objective.name` is
    invariant to both: nfcorpus' `min_relevance` 1→2 and `fetch_limit`
    1000→50 each moved scores under an unchanged name.
    """

    model_config = ConfigDict(frozen=True)

    objective: str
    min_relevance: int
    fetch_limit: dict[str, int]


class QueryContext(BaseModel):
    """One query's inputs to a builder.

    Threaded whole rather than as five positional parameters that every
    `build_row` passed straight through to the row constructor.
    """

    model_config = ConfigDict(frozen=True)

    row_id: int
    query_id: str
    query: str
    gold_qrel: dict[str, int]
    dataset_name: str
    provenance: str = "natural"


T = TypeVar("T", bound=FusionRow)


class LLMScoreClient:
    """Client for the query-classification LLM API returning an integer in [0, 9].

    The same endpoint Qdrant page-search production hits
    (`fusion.qdrant.tech/v1/classify`, per rust_search/skills/fusion.rs).
    Consumed by `HybridRoutingBuilder`, which routes on the raw integer.
    """

    SCORE_MAX = 9
    MAX_BYTES = 4096
    """The server's cap is on encoded bytes, not characters — measured: every
    payload it refused was >4096 bytes, none was under."""

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

    @classmethod
    def fit(cls, query: str) -> str:
        """`query` cut to the server's byte cap, never mid-character."""
        return query.encode()[: cls.MAX_BYTES].decode(errors="ignore")

    def score(self, query: str) -> int:
        response = requests.post(
            self._api_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"text": self.fit(query)},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return int(response.json()["score"])


class FusionBuilder(ABC, Generic[T]):
    """Base for route-result builders.

    Owns query iteration, scoring, and the artifact round-trip. Subclasses
    decide only which route(s) to run. Deliberately holds no strategy: two of
    the three builders evaluate three routes, not one.
    """

    row_type: ClassVar[type[FusionRow]]
    default_dir: ClassVar[Path] = Path("data/fusion")

    def __init__(
        self,
        *,
        objective: Objective | None = None,
        excluded: Mapping[str, Collection[str]] | None = None,
    ) -> None:
        self.objective = objective or RouterObjective()
        self._excluded = {
            str(qid): frozenset(str(d) for d in docs)
            for qid, docs in (excluded or {}).items()
        }

    @property
    @abstractmethod
    def strategies(self) -> list[FusionStrategy]:
        """Every route this builder queries."""

    @property
    def regime(self) -> ScoringRegime:
        return ScoringRegime(
            objective=self.objective.name,
            min_relevance=self.objective.min_relevance,
            fetch_limit={str(s.name): s.fetch_limit for s in self.strategies},
        )

    def _ranked(self, strategy: FusionStrategy, ctx: QueryContext) -> dict[str, float]:
        """Rank, then drop the query's excluded docs before scoring. Exclusion is per-query: a doc excluded here
        can be another query's gold, so the corpus keeps it and the strategy's
        own `fetch_limit` refills the cutoff."""
        ranking = strategy.rank(ctx.query)
        banned = self._excluded.get(ctx.query_id)
        if not banned:
            return ranking
        return {doc: score for doc, score in ranking.items() if doc not in banned}

    def _iter_queries(
        self,
        dataset: RetrievalDataset,
        start_id: int,
        qrels: QrelStore | None = None,
    ) -> Iterator[QueryContext]:
        store = qrels or QrelStore.from_dataset(dataset)
        by_query = store.lookup(dataset.name)
        prov_df = dataset.provenance()
        provenance = dict(zip(prov_df["query_id"].astype(str), prov_df["provenance"]))
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
            yield QueryContext(
                row_id=start_id + i,
                query_id=qid,
                query=str(q.text),
                gold_qrel=gold,
                dataset_name=dataset.name,
                provenance=provenance.get(qid, "natural"),
            )

    def _row(
        self,
        ctx: QueryContext,
        *,
        score: float,
        top_k: list[str],
        strategy_name: StrategyName,
        **extra: Any,
    ) -> T:
        """Assemble a row from an already-scored ranking. `extra` carries
        subclass-only fields (the routing oracle's per-route detail)."""
        return self.row_type(  # type: ignore[return-value]
            id=ctx.row_id,
            query_id=ctx.query_id,
            dataset_name=ctx.dataset_name,
            query=ctx.query,
            qdrant_answer=top_k,
            metric=score,
            metric_name=self.objective.name,
            strategy_name=strategy_name,
            provenance=ctx.provenance,
            **extra,
        )

    @abstractmethod
    def build_row(self, ctx: QueryContext) -> T: ...

    def build(
        self,
        dataset: RetrievalDataset,
        start_id: int = 0,
        qrels: QrelStore | None = None,
    ) -> list[T]:
        """Score `dataset`'s queries. Pass `qrels` to judge against a store
        other than the dataset's own — an LLM lane, or lanes merged."""
        return [
            self.build_row(ctx)
            for ctx in self._iter_queries(dataset, start_id, qrels)
        ]

    def save(self, rows: list[T], path: Path | str | None = None) -> Path:
        out = Path(path or self.default_dir)
        out.mkdir(parents=True, exist_ok=True)
        file = out / "rows.parquet"
        pd.DataFrame([r.model_dump() for r in rows]).to_parquet(file, index=False)
        file.with_suffix(REGIME_SUFFIX).write_text(
            self.regime.model_dump_json(indent=2) + "\n"
        )
        return file

    @classmethod
    def load(cls, path: Path | str | None = None) -> list[T]:
        file = Path(path or cls.default_dir) / "rows.parquet"
        df = pd.read_parquet(file)
        # Drop columns the schema no longer carries, so artifacts written before
        # judgments moved into QrelStore (and before `alpha` was removed) load.
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
            self._assert_reusable(file, rows)
            return rows
        rows = self.build(dataset, start_id=start_id, qrels=qrels)
        self.save(rows, path)
        return rows

    def _assert_reusable(self, file: Path, rows: list[T]) -> None:
        """Refuse a cache scored under a different objective or regime."""
        stale = {r.metric_name for r in rows} - {self.objective.name}
        if stale:
            raise ValueError(
                f"{file} holds rows scored with {sorted(stale)}, but this "
                f"builder is configured for {self.objective.name!r}. Delete "
                f"the file to rebuild, or point at a different path — "
                f"mixing objectives silently would corrupt every comparison."
            )
        sidecar = file.with_suffix(REGIME_SUFFIX)
        if not sidecar.exists():
            warnings.warn(
                f"{file} carries no {REGIME_SUFFIX} sidecar, so its "
                f"min_relevance and fetch_limit cannot be checked against "
                f"{self.regime.model_dump()} — reuse is unverified. Rebuild "
                f"to record them.",
                stacklevel=3,
            )
            return
        cached = ScoringRegime.model_validate_json(sidecar.read_text())
        if cached != self.regime:
            raise ValueError(
                f"{file} was scored under {cached.model_dump()}, but this "
                f"builder is configured for {self.regime.model_dump()}. Both "
                f"move the score while leaving metric_name untouched. Delete "
                f"the file to rebuild, or point at a different path."
            )


class SingleStrategyBuilder(FusionBuilder[T], ABC):
    """Base for builders that run exactly one route for every query."""

    def __init__(
        self,
        strategy: FusionStrategy,
        *,
        objective: Objective | None = None,
        excluded: Mapping[str, Collection[str]] | None = None,
    ) -> None:
        super().__init__(objective=objective, excluded=excluded)
        self.strategy = strategy

    @property
    def strategies(self) -> list[FusionStrategy]:
        return [self.strategy]


class RoutingBuilder(FusionBuilder[T], ABC):
    """Base for builders that choose among the three routes per query.

    Route order in `_routes` is iteration order only — selection among
    tied-best routes belongs to `fusion.derive_route` (SPEC d41), never to
    list position.
    """

    def __init__(
        self,
        dense_strategy: FusionStrategy,
        hybrid_strategy: FusionStrategy,
        sparse_strategy: FusionStrategy,
        *,
        objective: Objective | None = None,
        excluded: Mapping[str, Collection[str]] | None = None,
    ) -> None:
        super().__init__(objective=objective, excluded=excluded)
        self._routes = [dense_strategy, hybrid_strategy, sparse_strategy]

    @property
    def strategies(self) -> list[FusionStrategy]:
        return self._routes

    def _by_name(self, name: StrategyName) -> FusionStrategy:
        return next(s for s in self.strategies if s.name == name)


class BaselineBuilder(SingleStrategyBuilder[BaselineDataset]):
    """Run one route for every query — no per-query decision at all.

    Constant-dense and constant-sparse instances are the bar a router must
    clear (SPEC d37h): on trec-dl the oracle picked `dense_only` on 82% of
    queries, so a router scoring below that loses to one line of code.
    """

    row_type: ClassVar[type[FusionRow]] = BaselineDataset
    default_dir: ClassVar[Path] = Path("data/baseline")

    def build_row(self, ctx: QueryContext) -> BaselineDataset:
        score, top_k = self.objective.assess(
            self._ranked(self.strategy, ctx), ctx.gold_qrel
        )
        return self._row(
            ctx, score=score, top_k=top_k, strategy_name=self.strategy.name
        )


class GoldenRoutingBuilder(RoutingBuilder[GoldenRoutingDataset]):
    """Run all three routes per query and keep the objective-maximizing one.

    The oracle ceiling: regret of any candidate against these rows isolates
    decision quality from the coarseness of the three-route surface itself.
    Per-route scores and rankings are retained — see `GoldenRoutingDataset`.

    The hybrid endpoint should be `PureRRFStrategy` to match production; pass
    anything else only for ablation.

    `strategy_name` is the serving choice from `derive_route` (SPEC d41):
    the cheapest route among the tied-best. On all-zero rows it degenerates
    to the cheapest route overall and names the ranking the row carries, not
    a label — labels.py stores route = null there.
    """

    row_type: ClassVar[type[FusionRow]] = GoldenRoutingDataset
    default_dir: ClassVar[Path] = Path("data/golden_routing")

    def build_row(self, ctx: QueryContext) -> GoldenRoutingDataset:
        rankings = {s.name: self._ranked(s, ctx) for s in self.strategies}
        assessed = {
            name: self.objective.assess(ranking, ctx.gold_qrel)
            for name, ranking in rankings.items()
        }
        scores = {name: s for name, (s, _) in assessed.items()}
        serve = derive_route(scores)
        score, top_k = assessed[serve]
        return self._row(
            ctx,
            score=score,
            top_k=top_k,
            strategy_name=serve,
            route_scores=scores,
            route_rankings={name: r for name, (_, r) in assessed.items()},
            route_raw_scores={
                name: [rankings[name][doc] for doc in r]
                for name, (_, r) in assessed.items()
            },
        )


class HybridRoutingBuilder(RoutingBuilder[HybridRoutingDataset]):
    """Route each query on the LLM classifier score, reproducing production.

    Matches Qdrant page-search (rust_search/src/skills/fusion.rs): integer
    score → { 0..dense_max: Dense, dense_max+1..hybrid_max: Hybrid, else:
    Sparse }. Anything outside [0, hybrid_max] falls through to Sparse,
    mirroring the Rust `_ => Bm25` catch-all. Defaults (2, 6) are production's.
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
                "Thresholds must satisfy 0 <= dense_max < hybrid_max <= "
                f"{LLMScoreClient.SCORE_MAX}, got dense_max={dense_max}, "
                f"hybrid_max={hybrid_max}."
            )
        super().__init__(
            dense_strategy, hybrid_strategy, sparse_strategy, objective=objective
        )
        self._dense_max = dense_max
        self._hybrid_max = hybrid_max
        self._client = client or LLMScoreClient()

    def _route(self, score: int) -> FusionStrategy:
        dense, hybrid, sparse = self.strategies
        if 0 <= score <= self._dense_max:
            return dense
        if self._dense_max < score <= self._hybrid_max:
            return hybrid
        return sparse

    def build_row(self, ctx: QueryContext) -> HybridRoutingDataset:
        strategy = self._route(self._client.score(ctx.query))
        score, top_k = self.objective.assess(
            self._ranked(strategy, ctx), ctx.gold_qrel
        )
        return self._row(
            ctx, score=score, top_k=top_k, strategy_name=strategy.name
        )

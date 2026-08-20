"""Per-query scoring objectives that route builders maximize.

`assess(ranking, gold_qrel)` collapses one ranking into the float a builder
argmaxes over the three routes, plus the top-k ids the row stores. Graded qrels
are binarized at `min_relevance` first: ranx pins its own relevance threshold
to 1, so a TREC-DL grade 1 ("related but does not answer") would otherwise
count as a hit and hand a perfect score to a route that answered nothing.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

from pydantic import BaseModel, ConfigDict, Field
from ranx import Qrels, Run, evaluate


# metaclass=ABCMeta declares the abstractness explicitly without a second
# base class; at runtime pydantic's ModelMetaclass (an ABCMeta subclass)
# still wins metaclass resolution, so behavior is unchanged.
class Objective(BaseModel, metaclass=ABCMeta):
    model_config = ConfigDict(frozen=True)

    top_k: int = Field(default=10, gt=0)
    min_relevance: int = Field(default=1, ge=1)
    """Grades below this are irrelevant. TREC-DL's 0-3 scale needs 2; qrels
    that are already binary need 1. Set per dataset, not globally — so it
    cannot enter `name` without splitting comparability per lane; a cache's
    value is fingerprinted by `golden.ScoringRegime` instead."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stored on every row as `metric_name`; must be stable, since
        `evaluation.compare` refuses to mix rows scored differently."""

    @property
    @abstractmethod
    def decisive_margin(self) -> float:
        """Smallest winner-vs-runner-up score gap that certifies the winner
        hit rank 1 while the runner-up missed it (SPEC d41d). `inf` when the
        objective's score ranges cannot certify a top-1 separation at all."""

    @abstractmethod
    def assess(
        self, ranking: dict[str, float], gold_qrel: dict[str, int]
    ) -> tuple[float, list[str]]:
        """Return (score, top-k doc ids) from a single ordering pass.

        Combined rather than split because every caller needs both, and a
        separate `score()` implementation would order the ranking a second
        time — up to `fetch_limit` candidates re-sorted per route per query.
        """

    def score(self, ranking: dict[str, float], gold_qrel: dict[str, int]) -> float:
        return self.assess(ranking, gold_qrel)[0]

    def relevant(self, gold_qrel: dict[str, int]) -> dict[str, int]:
        return {d: r for d, r in gold_qrel.items() if r >= self.min_relevance}

    def ordered(self, ranking: dict[str, float]) -> list[str]:
        """Top-k doc ids, best first. Stable sort, so docs tied on score keep
        the retriever's own order rather than an arbitrary reshuffle."""
        ranked = sorted(ranking.items(), key=lambda kv: -kv[1])
        return [doc_id for doc_id, _ in ranked[: self.top_k]]

    def ndcg(self, ordered: list[str], relevant: dict[str, int]) -> float:
        """NDCG@k over exactly the persisted top-k ids, best first.

        Scored from the same `ordered()` list a row stores, not the raw
        ranking dict: ranx sorts its input with numpy's unstable sort, which
        scrambles score ties differently from `ordered()`'s stable sort, so
        passing the dict let a tied gold doc count toward the score while
        falling outside the stored top-k (the parquet round-trip bug). Synthetic
        descending scores reproduce `ordered`'s order under ranx; NDCG reads
        ranks, not score magnitudes, so non-tie values are unchanged.

        Shared by `NDCGObjective` and `RouterObjective`'s tie-breaker so the
        two cannot drift apart. ranx applies linear gain, so a grade-2 doc
        counts double a grade-1 rather than the exponential 2^g - 1.

        IDCG is computed from the *judged* relevant set, so judging only the
        union of route top-k inflates the value. IDCG is a property of the
        query and identical across routes, so it divides out and the argmax is
        unaffected — but absolute values are not comparable to published NDCG.
        """
        if not relevant or not ordered:
            return 0.0
        run = {doc: float(len(ordered) - i) for i, doc in enumerate(ordered)}
        return float(
            evaluate(
                Qrels({"q": relevant}),
                Run({"q": run}),
                f"ndcg@{self.top_k}",
            )
        )


class RouterObjective(Objective):
    """`hit_weight`·HitRate@1 + `ndcg_weight`·NDCG@k.

    While hit_weight > ndcg_weight the two score ranges are disjoint — a route
    whose rank-1 doc is relevant scores at least hit_weight, one whose isn't
    scores at most ndcg_weight — so this is lexicographic rather than a blend:
    top-1 decides, and NDCG@k only discriminates *within* each group. Setting
    ndcg_weight >= hit_weight gives that up and lets deeper ranks outvote a
    correct top hit.

    NDCG@k is the tie-breaker because both cheaper alternatives go blind on
    data we actually have. MRR@k is 1.0 for every route with a relevant rank-1
    doc, so it cannot tell a route that surfaced 1 of 4 relevant docs from one
    that surfaced 4 of 4, and the label falls through to tie-break order.
    Recall@k collapses to two values whenever a query has a single relevant doc
    — rank 2 and rank 10 score identically — which covers the majority of the
    corpus (msmarco-dev, rarb, most crumb, orcas). NDCG@k handles both.
    """

    hit_weight: float = Field(default=0.7, ge=0.0)
    ndcg_weight: float = Field(default=0.3, ge=0.0)

    @property
    def name(self) -> str:
        return f"{self.hit_weight:g}*HitRate@1+{self.ndcg_weight:g}*NDCG@{self.top_k}"

    @property
    def decisive_margin(self) -> float:
        # Lexicographic regime only: a rank-1 hit scores >= hit_weight, a miss
        # <= ndcg_weight, so a gap of hit_weight - ndcg_weight is reachable
        # exactly when the winner hit and the runner-up missed. With
        # hit_weight <= ndcg_weight the ranges overlap and no gap certifies it.
        if self.hit_weight <= self.ndcg_weight:
            return float("inf")
        return self.hit_weight - self.ndcg_weight

    def assess(
        self, ranking: dict[str, float], gold_qrel: dict[str, int]
    ) -> tuple[float, list[str]]:
        top = self.ordered(ranking)
        relevant = self.relevant(gold_qrel)
        if not relevant:
            return 0.0, top
        hit = self.hit_weight if top and top[0] in relevant else 0.0
        return hit + self.ndcg_weight * self.ndcg(top, relevant), top


class NDCGObjective(Objective):
    """Bare NDCG@k — no top-1 term.

    Retained so rows written before the router objective (metric_name
    "NDCG@10") stay reproducible, and so the cost of adding the top-1 term is
    measurable: dropping HitRate@1 makes the objective a plain blend again,
    which is what the earlier trec-dl artifacts were scored with.
    """

    @property
    def name(self) -> str:
        return f"NDCG@{self.top_k}"

    @property
    def decisive_margin(self) -> float:
        return float("inf")

    def assess(
        self, ranking: dict[str, float], gold_qrel: dict[str, int]
    ) -> tuple[float, list[str]]:
        top = self.ordered(ranking)
        return self.ndcg(top, self.relevant(gold_qrel)), top

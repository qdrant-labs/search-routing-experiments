"""The classify payload's cap, the assess score/ranking round-trip invariant,
and the cache-reuse guard."""

import time
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from hybrid_search_rrf_dataset.fusion import FusionStrategy, StrategyName
from hybrid_search_rrf_dataset.indexer import EmbeddingConfig
from hybrid_search_rrf_dataset.golden import (
    BaselineBuilder,
    BaselineDataset,
    GoldenRoutingBuilder,
    LLMScoreClient,
)
from hybrid_search_rrf_dataset.objective import RouterObjective

GOOAQ_ORACLE = Path(__file__).resolve().parents[1] / "src/data/route_labels/gooaq_oracle"
GOOAQ_QRELS = Path(__file__).resolve().parents[1] / "src/data/gooaq/qrels.parquet"


def reconstruct(ranking: list[str]) -> dict[str, float]:
    """A ranking dict recovered from a stored top-k id list alone: strictly
    descending synthetic scores that preserve the stored order."""
    n = len(ranking)
    return {doc: float(n - i) for i, doc in enumerate(ranking)}


def test_assess_score_is_reproducible_from_its_own_ranking():
    """The parquet round-trip bug: a big block of docs tied on score, with the
    gold beyond `ordered()`'s top-k. ranx's unstable sort pulled it into the
    scored top-k while the stored list left it out — so the score credited a
    doc the ranking cannot show. `assess` must score only what it persists."""
    objective = RouterObjective(top_k=10)
    ranking = {f"t{i:03d}": 0.5 for i in range(16)}  # exact-tie block
    gold = {"t010": 1}  # inside ranx's top-10, outside ordered()'s

    score, top = objective.assess(ranking, gold)

    assert objective.score(reconstruct(top), gold) == pytest.approx(score)


@pytest.mark.skipif(
    not (GOOAQ_ORACLE / "rows.parquet").exists() or not GOOAQ_QRELS.exists(),
    reason="gooaq oracle cache / qrels not materialized",
)
def test_gooaq_oracle_scores_round_trip_from_stored_rankings():
    """Every stored route score must be recomputable from its stored top-k, on
    a per-lane sample plus the traced reproducer (gooaq 139935)."""
    objective = RouterObjective()
    q = pd.read_parquet(GOOAQ_QRELS)
    gold = {
        qid: dict(zip(g["doc_id"].astype(str), g["relevance"].astype(int)))
        for qid, g in q.astype({"query_id": str}).groupby("query_id")
    }
    rows = GoldenRoutingBuilder.load(GOOAQ_ORACLE)

    reproducer = [r for r in rows if r.query_id == "139935"]
    assert reproducer, "traced reproducer 139935 missing from the cache"
    for row in reproducer + rows[:500]:
        judged = gold.get(row.query_id)
        if judged is None:
            continue
        for route, ranking in row.route_rankings.items():
            recomputed = objective.score(reconstruct(ranking), judged)
            assert recomputed == pytest.approx(row.route_scores[route]), (
                row.query_id, route
            )


def test_short_ascii_is_untouched():
    query = "what are fossilized bones made of?"

    assert LLMScoreClient.fit(query) == query


def test_multibyte_payload_stays_within_the_byte_cap():
    """4,096 characters of a 3-byte character is a 12KB payload — the measured 400."""
    query = "気" * 5000

    assert len(LLMScoreClient.fit(query).encode()) <= LLMScoreClient.MAX_BYTES
    assert len(query[: LLMScoreClient.MAX_BYTES].encode()) > LLMScoreClient.MAX_BYTES


def test_truncation_never_splits_a_character():
    head = "a" * (LLMScoreClient.MAX_BYTES - 1)

    assert LLMScoreClient.fit(head + "é" + "tail") == head


class _FixedDepth(FusionStrategy):
    """A route that carries only the regime's inputs — fetch depth and vector
    slots — with no client and no retrieval."""

    name: ClassVar[StrategyName] = StrategyName.DENSE_ONLY

    def __init__(self, fetch_limit: int, dense_model: str = "dense/v1") -> None:
        super().__init__(
            None,  # type: ignore[arg-type]
            "lane",
            EmbeddingConfig(name="dense", model_id=dense_model, kind="dense"),
            EmbeddingConfig(name="sparse", model_id="sparse/v1", kind="sparse"),
            fetch_limit,
        )

    def rank(self, query: str) -> dict[str, float]:
        return {}


def _cached(
    out: Path,
    objective: RouterObjective,
    fetch_limit: int,
    dense_model: str = "dense/v1",
) -> None:
    """One saved single-route cache, sidecar included."""
    row = BaselineDataset(
        id=0,
        query_id="q1",
        dataset_name="lane",
        query="q",
        qdrant_answer=["d1"],
        metric=1.0,
        metric_name=objective.name,
        strategy_name=StrategyName.DENSE_ONLY,
    )
    BaselineBuilder(
        _FixedDepth(fetch_limit, dense_model), objective=objective
    ).save([row], out)


def test_reuse_refuses_a_cache_scored_at_another_min_relevance(tmp_path):
    """beir-nfcorpus went 1→2 (6b8bf4b) nine hours after the labels were
    written and metric_name never moved, so the guard saw nothing."""
    _cached(tmp_path, RouterObjective(min_relevance=1), 50)
    builder = BaselineBuilder(
        _FixedDepth(50), objective=RouterObjective(min_relevance=2)
    )

    with pytest.raises(ValueError, match="min_relevance"):
        builder.build_or_load(None, tmp_path)  # type: ignore[arg-type]


def test_reuse_refuses_a_cache_fetched_at_another_depth(tmp_path):
    """fetch_limit 1000→50 (2443aa1) moved score_pure_rrf on 7% of rows, and
    it lives on the strategy — nothing the objective can see."""
    _cached(tmp_path, RouterObjective(), 1000)
    builder = BaselineBuilder(_FixedDepth(50), objective=RouterObjective())

    with pytest.raises(ValueError, match="fetch_limit"):
        builder.build_or_load(None, tmp_path)  # type: ignore[arg-type]


def test_reuse_refuses_a_cache_scored_by_another_encoder(tmp_path):
    """A swapped dense model moves every dense rank, and it lives on the
    strategy's vector slots — invisible to objective and fetch depth alike."""
    _cached(tmp_path, RouterObjective(), 50, dense_model="dense/v1")
    builder = BaselineBuilder(
        _FixedDepth(50, dense_model="dense/v2"), objective=RouterObjective()
    )

    with pytest.raises(ValueError, match="dense/v1"):
        builder.build_or_load(None, tmp_path)  # type: ignore[arg-type]


def test_reuse_accepts_a_cache_scored_under_the_same_regime(tmp_path):
    _cached(tmp_path, RouterObjective(), 50)
    builder = BaselineBuilder(_FixedDepth(50), objective=RouterObjective())

    assert len(builder.build_or_load(None, tmp_path)) == 1  # type: ignore[arg-type]


def test_a_cache_without_a_sidecar_still_loads_but_says_so(tmp_path):
    """The 42 shipped oracle caches predate the sidecar: they must keep
    loading, unverified but not silently."""
    _cached(tmp_path, RouterObjective(), 50)
    (tmp_path / "rows.provenance.json").unlink()
    builder = BaselineBuilder(
        _FixedDepth(50), objective=RouterObjective(min_relevance=3)
    )

    with pytest.warns(UserWarning, match="sidecar"):
        assert len(builder.build_or_load(None, tmp_path)) == 1  # type: ignore[arg-type]


class _EchoStrategy:
    """Ranks by query text, with per-query sleep — a naive concurrent
    implementation that returns completion order instead of submission order
    would reorder rows whose delays are deliberately mismatched."""

    fetch_limit = 50
    dense_cfg = EmbeddingConfig(name="dense", model_id="dense/v1", kind="dense")
    sparse_cfg = EmbeddingConfig(name="sparse", model_id="sparse/v1", kind="sparse")

    def __init__(
        self, delays: dict[str, float], name: StrategyName = StrategyName.DENSE_ONLY
    ) -> None:
        self._delays = delays
        self.name = name

    def rank(self, query: str) -> dict[str, float]:
        time.sleep(self._delays.get(query, 0))
        return {f"doc-{query}": 1.0}


class _StubDataset:
    name = "lane"

    def __init__(self, query_ids: list[str]) -> None:
        self._ids = query_ids

    def queries(self) -> pd.DataFrame:
        return pd.DataFrame({"query_id": self._ids, "text": self._ids})

    def qrels(self) -> pd.DataFrame:
        return pd.DataFrame({
            "query_id": self._ids, "doc_id": [f"doc-{q}" for q in self._ids],
            "relevance": [1] * len(self._ids),
        })

    def provenance(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["query_id", "provenance"])


def test_build_stays_serial_by_default():
    ids = ["a", "b", "c"]
    dataset = _StubDataset(ids)
    builder = BaselineBuilder(_EchoStrategy({}), objective=RouterObjective())

    rows = builder.build(dataset)
    assert [r.query_id for r in rows] == ids


def test_concurrent_build_preserves_submission_order_not_completion_order():
    """The regression this guards: ThreadPoolExecutor.map yields results in
    input order regardless of which finishes first — verified, not assumed,
    by making the FIRST query's rank() call the SLOWEST."""
    ids = ["a", "b", "c", "d"]
    delays = {"a": 0.05, "b": 0.0, "c": 0.0, "d": 0.0}
    dataset = _StubDataset(ids)
    builder = BaselineBuilder(_EchoStrategy(delays), objective=RouterObjective())

    rows = builder.build(dataset, max_workers=4)
    assert [r.query_id for r in rows] == ids
    assert [r.qdrant_answer for r in rows] == [[f"doc-{q}"] for q in ids]


def test_concurrent_scoring_does_not_crash_numbas_threading_layer():
    """The regression this guards: ranx's ndcg is `@njit(parallel=True)`, and
    numba's default workqueue layer aborts the WHOLE PROCESS ('Fatal Python
    error: Aborted') the moment it's entered from more than one Python thread
    at once — reproduced live, independent of JIT warm-up. GoldenRoutingBuilder
    calls `objective.assess` three times per row (once per route), the highest-
    contention real pattern; `Objective.ndcg`'s lock (objective.py) is what
    must prevent this. A regression here does not fail cleanly — it aborts
    the interpreter, which is itself the loudest possible signal."""
    ids = [f"q{i}" for i in range(150)]
    dataset = _StubDataset(ids)
    builder = GoldenRoutingBuilder(
        _EchoStrategy({}, StrategyName.DENSE_ONLY),
        _EchoStrategy({}, StrategyName.PURE_RRF),
        _EchoStrategy({}, StrategyName.SPARSE_ONLY),
        objective=RouterObjective(),
    )

    rows = builder.build(dataset, max_workers=16)
    assert [r.query_id for r in rows] == ids

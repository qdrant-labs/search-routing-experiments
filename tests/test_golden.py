"""The classify payload's cap, and the assess score/ranking round-trip invariant."""

from pathlib import Path

import pandas as pd
import pytest

from hybrid_search_rrf_dataset.golden import GoldenRoutingBuilder, LLMScoreClient
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

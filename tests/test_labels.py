import pandas as pd
import pytest

from hybrid_search_rrf_dataset.labels import (
    AcceptabilityLabels,
    RouteLabels,
    route_label,
)
from hybrid_search_rrf_dataset.objective import NDCGObjective, RouterObjective


@pytest.fixture
def labels(tmp_path):
    """Two lanes with hand-computable scores.

    lane-a r1: dense 1.0 / rrf 0.15 / sparse 0.0  -> decisive, dense wins
    lane-a r2: dense 0.0 / rrf 0.90 / sparse 0.1  -> decisive, rrf wins
    lane-b r3: all zero                            -> all_zero, not decisive
    """
    frame = pd.DataFrame(
        {
            "dataset": ["lane-a", "lane-a", "lane-b"],
            "query_id": ["1", "2", "3"],
            "score_dense_only": [1.0, 0.0, 0.0],
            "score_pure_rrf": [0.15, 0.9, 0.0],
            "score_sparse_only": [0.0, 0.1, 0.0],
        }
    )
    out_dir = tmp_path / "route_labels"
    out_dir.mkdir()
    frame.to_parquet(out_dir / "labels.parquet", index=False)
    selection = pd.DataFrame({"dataset": [], "query_id": []})
    return RouteLabels(selection, out_dir=out_dir)


def test_decisive_margin():
    assert RouterObjective().decisive_margin == pytest.approx(0.4)
    assert RouterObjective(hit_weight=0.3, ndcg_weight=0.7).decisive_margin == float(
        "inf"
    )
    assert NDCGObjective().decisive_margin == float("inf")


def test_headroom_per_lane_and_pooled(labels):
    table = labels.headroom().set_index("dataset")

    lane_a = table.loc["lane-a"]
    # lane-a means: dense 0.5, rrf 0.525, sparse 0.05 -> best constant rrf
    assert lane_a["best_constant"] == "pure_rrf"
    assert lane_a["constant"] == pytest.approx(0.525)
    assert lane_a["oracle"] == pytest.approx(0.95)  # (1.0 + 0.9) / 2
    assert lane_a["headroom"] == pytest.approx(0.425)
    assert lane_a["decisive_share"] == pytest.approx(1.0)
    assert lane_a["all_zero_share"] == pytest.approx(0.0)

    lane_b = table.loc["lane-b"]
    assert lane_b["headroom"] == pytest.approx(0.0)
    assert lane_b["headroom_pct"] == pytest.approx(0.0)  # guarded 0/0
    assert lane_b["all_zero_share"] == pytest.approx(1.0)

    pooled = table.loc["POOLED"]
    assert pooled["labelled"] == 3
    # pooled means: dense 1/3, rrf 0.35, sparse 0.0333 -> best constant rrf
    assert pooled["best_constant"] == "pure_rrf"
    assert pooled["constant"] == pytest.approx(0.35)
    assert pooled["oracle"] == pytest.approx((1.0 + 0.9 + 0.0) / 3, abs=1e-3)


def test_headroom_decomposition_levels(labels):
    levels = labels.headroom_decomposition()
    assert list(levels["score"]) == [
        pytest.approx(0.35),  # global constant (pure_rrf pooled mean)
        pytest.approx((2 * 0.525 + 1 * 0.0) / 3),  # per-collection, n-weighted
        pytest.approx((1.0 + 0.9 + 0.0) / 3),  # oracle
    ]
    assert pd.isna(levels["gain_vs_previous_pct"].iloc[0])


def test_decisive_winners_split(labels):
    table = labels.decisive_winners().set_index("dataset")
    assert table.loc["lane-a", "dense_only"] == 1
    assert table.loc["lane-a", "pure_rrf"] == 1
    assert table.loc["lane-a", "sparse_only"] == 0
    assert "lane-b" not in table.index  # no decisive rows there
    assert table.loc["POOLED"].sum() == 2


@pytest.fixture
def shaped_frame():
    """One row per interesting shape, scores as (dense, rrf, sparse)."""
    return pd.DataFrame(
        {
            "score_dense_only": [1.0, 0.2, 1.0, 0.0, 1.0, 1.0],
            "score_pure_rrf": [0.6, 0.1, 1.0, 0.0, 0.7, 1.0],
            "score_sparse_only": [1.0, 0.9, 1.0, 0.0, 0.5, 0.2],
        },
        index=["near_top", "sparse_wins", "all_tied", "all_zero",
               "hit_parity", "two_way_tie"],
    )


def test_tolerance_zero_reproduces_stored_route(shaped_frame):
    # the d60 must-pass: at tolerance=0 the view's serve IS today's label,
    # including the two-way tie resolving dense (cheapest of the tied best).
    view = AcceptabilityLabels(shaped_frame, tolerance=0.0).frame()
    score_cols = [c for c in shaped_frame.columns if c.startswith("score_")]
    for row_name, row in shaped_frame.iterrows():
        scores = {c.removeprefix("score_"): row[c] for c in score_cols}
        stored = route_label(scores)
        served = view.loc[row_name, "serve"]
        # both nulls (all_zero) count as agreement — parquet stores each as NaN
        assert (pd.isna(served) and stored is None) or served == stored, row_name


def test_hit_parity_default_tolerance(shaped_frame):
    view = AcceptabilityLabels(shaped_frame)
    assert view.tolerance == pytest.approx(RouterObjective().ndcg_weight)


def test_acceptability_per_shape(shaped_frame):
    view = AcceptabilityLabels(shaped_frame).frame()  # tolerance 0.3

    tied = view.loc["all_tied"]
    assert bool(tied.ok_dense_only) and bool(tied.ok_pure_rrf)
    assert bool(tied.ok_sparse_only) and tied.serve == "sparse_only"

    zero = view.loc["all_zero"]
    assert pd.isna(zero.ok_dense_only) and pd.isna(zero.serve)

    # 0.7 sits exactly at hit parity (1.0 - 0.3): same top-1 band, acceptable
    parity = view.loc["hit_parity"]
    assert bool(parity.ok_pure_rrf) and not bool(parity.ok_sparse_only)
    assert parity.serve == "dense_only"

    # sparse alone clears; the others miss by more than the tolerance
    assert view.loc["sparse_wins", "serve"] == "sparse_only"
    assert not bool(view.loc["sparse_wins", "ok_dense_only"])

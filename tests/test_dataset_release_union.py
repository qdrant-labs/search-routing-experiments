"""Tests for dataset_release.union — pins the D1 collision rule and row count."""

from pathlib import Path

import pandas as pd
import pytest

from dataset_release import _cascade_union, build_91k, build_union_233k
from dataset_release.union import KEY

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "data"


def _row(dataset, qid, tag, route=None):
    # `tag` lets us tell which source a surviving row came from.
    return {
        "dataset": dataset,
        "query_id": qid,
        "query": f"q-{tag}",
        "route": route,
        "score_dense_only": 0.1,
        "score_sparse_only": 0.2,
        "score_pure_rrf": 0.3,
        "shape": "routes_differ",
        "metric_name": f"metric-{tag}",
        "scored_against": f"pool-{tag}",
        "provenance": tag,
    }


def test_cascade_wins_and_unique_rows_survive_unit():
    # Overlap on ("dataset", "query_id") = (A, 1): every source contributes it,
    # cascade must be the surviving row. Every source also contributes a
    # non-overlapping row that must appear in the union.
    v2 = pd.DataFrame([_row("A", "1", "v2"), _row("A", "2", "v2")])
    v3 = pd.DataFrame([_row("A", "1", "v3"), _row("B", "3", "v3")])
    v3_aug = pd.DataFrame([_row("A", "1", "v3_aug"), _row("B", "4", "v3_aug")])
    cascade = pd.DataFrame([_row("A", "1", "cascade", route="dense_only"), _row("C", "5", "cascade")])

    union = _cascade_union(cascade, v2, v3, v3_aug)

    assert len(union) == 5
    keys = set(zip(union["dataset"], union["query_id"], strict=True))
    assert keys == {("A", "1"), ("A", "2"), ("B", "3"), ("B", "4"), ("C", "5")}
    collision = union[(union["dataset"] == "A") & (union["query_id"] == "1")].iloc[0]
    assert collision["provenance"] == "cascade"
    assert collision["route"] == "dense_only"
    expected_cols = {
        "dataset", "query_id", "query", "route",
        "score_dense_only", "score_sparse_only", "score_pure_rrf",
        "shape", "metric_name", "scored_against", "provenance",
    }
    assert expected_cols.issubset(set(union.columns))


CASCADE_PARQUET = DATA_DIR / "rungs" / "100k-v2" / "labeling" / "labels.parquet"


@pytest.mark.skipif(not CASCADE_PARQUET.exists(), reason="D1 source parquets not present")
def test_build_union_233k_row_count_integration():
    assert len(build_union_233k(DATA_DIR)) == 233_247


@pytest.mark.skipif(not CASCADE_PARQUET.exists(), reason="D1 source parquets not present")
def test_build_91k_is_text_bearing_cascade_inside_the_union_integration():
    cascade = build_91k(DATA_DIR)

    assert len(cascade) == 91_080
    assert cascade["query"].astype(str).str.strip().ne("").all()
    # 91K is the union's collision winner, so every one of its keys must survive.
    keys = set(zip(cascade["dataset"], cascade["query_id"], strict=True))
    assert keys <= set(zip(*(build_union_233k(DATA_DIR)[c] for c in KEY), strict=True))

"""Public-seam tests for the lightweight Rung B assessment."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from rungs.rung_b import RungB, RungBIntegrityError
from scripts.validate_rung_b import main as rung_b_main


def _plan() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["A:1", "A:2", "B:1", "B:2"],
            "dataset": ["A", "A", "B", "B"],
            "query_id": ["1", "2", "1", "2"],
            "provenance": ["natural", "augmented", "synthetic", "natural"],
            "corpus_regime": ["natural", "natural", "synthetic", "natural"],
        }
    )


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["A", "A", "B", "B"],
            "query_id": ["1", "2", "1", "2"],
            "route": ["dense_only", "sparse_only", None, "sparse_only"],
            "score_dense_only": [1.0, 0.5, 0.0, 0.1],
            "score_pure_rrf": [0.2, 0.5, 0.0, 0.2],
            "score_sparse_only": [0.1, 0.5, 0.0, 0.9],
            "trust_tier": ["measurement", "measurement", "ungrounded", "r1_assisted"],
        }
    )


def test_assess_derives_distribution_and_separation_without_mutating_inputs():
    plan = _plan()
    labels = _labels()
    original_plan = plan.copy(deep=True)
    original_labels = labels.copy(deep=True)

    report = RungB(decisive_margin=0.4).assess(plan, labels)

    assert report.winner_mix == {
        "dense_only": 1,
        "pure_rrf": 0,
        "sparse_only": 2,
        "null": 1,
    }
    assert report.winner_share == {
        "dense_only": 0.25,
        "pure_rrf": 0.0,
        "sparse_only": 0.5,
        "null": 0.25,
    }
    assert report.outcome_shape_mix == {
        "routes_differ": 2,
        "all_tied": 1,
        "all_zero": 1,
    }
    assert report.separation_summary == {
        "threshold": 0.4,
        "decisive_count": 2,
        "decisive_share": 0.5,
        "exact_tie_count": 1,
        "all_zero_count": 1,
        "null_route_count": 1,
        "decisive_winner_mix": {
            "dense_only": 1,
            "pure_rrf": 0,
            "sparse_only": 1,
            "null": 0,
        },
        "decisive_winner_share": {
            "dense_only": 0.5,
            "pure_rrf": 0.0,
            "sparse_only": 0.5,
            "null": 0.0,
        },
    }
    assert report.margin_distribution == [0.8, 0.0, 0.0, 0.7]
    assert_frame_equal(plan, original_plan)
    assert_frame_equal(labels, original_labels)


def test_assess_accepts_legacy_plan_with_unique_identity_and_no_row_id():
    report = RungB().assess(_plan().drop(columns="row_id"), _labels())

    assert report.counts == {"planned": 4, "completed": 4}


def test_assess_uses_selected_labels_when_legacy_plan_carries_stale_scores():
    plan = _plan().assign(
        route="pure_rrf",
        score_dense_only=0.2,
        score_pure_rrf=0.2,
        score_sparse_only=0.2,
    )

    report = RungB().assess(plan, _labels())

    assert report.winner_mix == {
        "dense_only": 1,
        "pure_rrf": 0,
        "sparse_only": 2,
        "null": 1,
    }
    assert report.outcome_shape_mix == {
        "routes_differ": 2,
        "all_tied": 1,
        "all_zero": 1,
    }


def test_assess_rejects_overlapping_label_shards():
    labels = pd.concat([_labels(), _labels().iloc[[0]]], ignore_index=True)

    with pytest.raises(RungBIntegrityError, match="labels have duplicate"):
        RungB().assess(_plan(), labels)


def test_assess_stratifies_separation_by_batch_dimensions():
    report = RungB(decisive_margin=0.4).assess(_plan(), _labels())

    lane_a = report.breakdowns["lane"]["A"]
    assert lane_a["count"] == 2
    assert lane_a["winner_mix"] == {
        "dense_only": 1,
        "pure_rrf": 0,
        "sparse_only": 1,
        "null": 0,
    }
    assert lane_a["decisive_count"] == 1
    assert lane_a["decisive_share"] == 0.5

    synthetic = report.breakdowns["query_origin"]["synthetic"]
    assert synthetic["count"] == 1
    assert synthetic["outcome_shape_mix"] == {"all_zero": 1}

    assert report.breakdowns["corpus_regime"]["synthetic"]["all_zero_count"] == 1
    assert report.breakdowns["trust_source"]["ungrounded"]["null_route_count"] == 1


def test_assess_compares_named_reference_on_the_matched_cohort_only():
    reference = pd.DataFrame(
        {
            "dataset": ["A", "A", "B", "outside"],
            "query_id": ["1", "2", "1", "9"],
            "route": ["sparse_only", "sparse_only", "dense_only", "dense_only"],
            "score_dense_only": [0.0, 0.5, 0.8, 1.0],
            "score_pure_rrf": [0.1, 0.5, 0.0, 0.0],
            "score_sparse_only": [1.0, 0.5, 0.0, 0.0],
        }
    )

    report = RungB(decisive_margin=0.4).assess(
        _plan(),
        _labels(),
        references={"v3": reference},
    )

    comparison = report.reference_comparisons["v3"]
    assert comparison["matched_count"] == 3
    assert comparison["current"]["decisive_share"] == 1 / 3
    assert comparison["reference"]["decisive_share"] == 2 / 3
    assert comparison["delta"]["decisive_share"] == -1 / 3
    assert comparison["delta"]["all_zero_share"] == 1 / 3
    assert comparison["delta"]["winner_share"] == {
        "dense_only": 0.0,
        "pure_rrf": 0.0,
        "sparse_only": -1 / 3,
        "null": 1 / 3,
    }


def test_assess_snapshots_execution_accounting_and_fingerprints_the_report():
    accounting = {
        "cache": {"hits": 3, "misses": 1},
        "stages": {"l1": {"executed": 1, "reused": 3}},
        "providers": {"qdrant": {"queries": 1}},
        "cost": {"estimated_usd": 1.25},
    }

    report = RungB().assess(_plan(), _labels(), accounting=accounting)
    report_fp = report.fingerprint()
    accounting["cache"]["hits"] = 999

    assert report.execution_accounting["cache"] == {"hits": 3, "misses": 1}
    assert len(report_fp) == 64
    assert report_fp == report.fingerprint()
    assert report_fp != RungB(decisive_margin=0.5).assess(
        _plan(), _labels(), accounting={"cache": {"hits": 3, "misses": 1}}
    ).fingerprint()


def test_cli_writes_fingerprinted_assessment_with_reference_and_accounting(tmp_path):
    plan_path = tmp_path / "plan.parquet"
    labels_path = tmp_path / "labels-a.parquet"
    labels_path_b = tmp_path / "labels-b.parquet"
    reference_path = tmp_path / "v3.parquet"
    accounting_path = tmp_path / "accounting.json"
    report_path = tmp_path / "rung_b_report.json"
    _plan().to_parquet(plan_path, index=False)
    _labels().iloc[:2].to_parquet(labels_path, index=False)
    _labels().iloc[2:].to_parquet(labels_path_b, index=False)
    _labels().iloc[:3].to_parquet(reference_path, index=False)
    accounting_path.write_text(json.dumps({"cache": {"hits": 4, "misses": 0}}))

    exit_code = rung_b_main(
        [
            "--plan",
            str(plan_path),
            "--labels",
            str(labels_path),
            "--labels",
            str(labels_path_b),
            "--reference",
            f"v3={reference_path}",
            "--accounting",
            str(accounting_path),
            "--decisive-margin",
            "0.4",
            "--out",
            str(report_path),
        ]
    )

    payload = json.loads(report_path.read_text())
    assert exit_code == 0
    assert len(payload["report_fp"]) == 64
    assert payload["winner_mix"]["dense_only"] == 1
    assert payload["reference_comparisons"]["v3"]["matched_count"] == 3
    assert payload["execution_accounting"]["cache"] == {"hits": 4, "misses": 0}

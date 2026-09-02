"""spec verification (11): Rung B membership is invariant under every
post-label outcome permutation. Changing a route/margin/depth/evidence field
must not add or remove rows or change their order."""

from __future__ import annotations

import pandas as pd
import pytest

from rungs.rung_b import RungB, RungBIntegrityError


def _plan() -> pd.DataFrame:
    return pd.DataFrame({
        "row_id": ["A:1", "A:2", "B:1"],
        "dataset": ["A", "A", "B"],
        "query_id": ["1", "2", "1"],
    })


def _labels(**overrides) -> pd.DataFrame:
    base = pd.DataFrame({
        "dataset": ["A", "A", "B"],
        "query_id": ["1", "2", "1"],
        "route_class": ["dense", "hybrid", "sparse"],
        "margin": [0.3, 0.1, 0.5],
        "depth": [3, 1, 2],
        "kind": ["decisive", "genuine_tie", "decisive"],
    })
    for column, values in overrides.items():
        base[column] = values
    return base


def test_row_membership_unchanged_by_route_permutation():
    plan = _plan()
    labels_a = _labels()
    labels_b = _labels(route_class=["sparse", "dense", "hybrid"])
    completed_a = RungB().join(plan, labels_a)
    completed_b = RungB().join(plan, labels_b)
    assert (
        completed_a.rows[["dataset", "query_id"]].values.tolist()
        == completed_b.rows[["dataset", "query_id"]].values.tolist()
    )


def test_row_membership_unchanged_by_margin_permutation():
    plan = _plan()
    completed = RungB().join(plan, _labels(margin=[10.0, -5.0, 999.0]))
    ids = completed.rows[["dataset", "query_id"]].values.tolist()
    assert ids == plan[["dataset", "query_id"]].values.tolist()


def test_missing_label_fails_the_run():
    plan = _plan()
    labels = _labels().iloc[:2]  # drop the B:1 row
    with pytest.raises(RungBIntegrityError, match="no label"):
        RungB().join(plan, labels)


def test_duplicate_plan_identity_fails_the_run():
    plan = pd.DataFrame({
        "row_id": ["A:1", "A:1"],
        "dataset": ["A", "A"],
        "query_id": ["1", "1"],
    })
    labels = _labels()
    with pytest.raises(RungBIntegrityError, match="duplicate"):
        RungB().join(plan, labels)

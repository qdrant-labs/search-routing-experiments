"""Run-scoped labeling contract for a frozen Rung A plan."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rungs.label_run import LabelRegime, LabelRunError, RungLabelRun
from scripts.label_rung_plan import main
from tests.rungs_fixtures import make_row


def _plan(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _label(row: dict[str, object]) -> dict[str, object]:
    return {
        "dataset": row["dataset"],
        "query_id": row["query_id"],
        "route": "dense_only",
        "shape": "routes_differ",
        "score_dense_only": 1.0,
        "score_pure_rrf": 0.5,
        "score_sparse_only": 0.0,
    }


def test_partition_routes_each_answer_regime():
    rows = [
        make_row("A:n", "A", "natural", ["cell_x"]),
        make_row(
            "A:a",
            "A",
            "augmented",
            ["cell_x"],
            provenance="augmented",
            operator="decorate",
        ),
        make_row(
            "A:l",
            "A",
            "lane generated",
            ["cell_x"],
            provenance="synthetic",
            operator="lane_synthesize",
        ),
        make_row(
            "A:s",
            "A",
            "constructed",
            ["cell_x"],
            provenance="synthetic",
            operator="synthesize",
        ),
    ]

    partitions = RungLabelRun(_plan(rows), out_dir=Path("unused")).partitions()

    assert partitions[LabelRegime.NATURAL]["query_id"].tolist() == ["n"]
    assert partitions[LabelRegime.SUPPLEMENTED]["query_id"].tolist() == ["a", "l"]
    assert partitions[LabelRegime.CONSTRUCTED]["query_id"].tolist() == ["s"]


def test_duplicate_plan_identity_is_rejected(tmp_path: Path):
    row = make_row("A:1", "A", "one", ["cell_x"])
    with pytest.raises(LabelRunError, match="duplicate"):
        RungLabelRun(_plan([row, row]), out_dir=tmp_path)


def test_run_directory_refuses_a_different_plan(tmp_path: Path):
    first = RungLabelRun(
        _plan([make_row("A:1", "A", "one", ["cell_x"])]), out_dir=tmp_path
    )
    first.initialize(plan_path=Path("first/planned_set.parquet"))
    before = json.loads((tmp_path / "run.json").read_text())

    changed = RungLabelRun(
        _plan([make_row("A:2", "A", "two", ["cell_x"])]), out_dir=tmp_path
    )
    with pytest.raises(LabelRunError, match="different plan"):
        changed.initialize(plan_path=Path("second/planned_set.parquet"))

    assert json.loads((tmp_path / "run.json").read_text()) == before


def test_resume_refuses_unowned_label_shards(tmp_path: Path):
    row = make_row("A:1", "A", "one", ["cell_x"])
    shard_dir = tmp_path / LabelRegime.NATURAL.value
    shard_dir.mkdir(parents=True)
    pd.DataFrame([_label(row)]).to_parquet(shard_dir / "labels.parquet", index=False)
    run = RungLabelRun(_plan([row]), out_dir=tmp_path)

    with pytest.raises(LabelRunError, match="not owned"):
        run.remaining_partitions()


def test_resume_refuses_shards_owned_by_a_different_plan(tmp_path: Path):
    first = RungLabelRun(
        _plan([make_row("A:1", "A", "one", ["cell_x"])]), out_dir=tmp_path
    )
    first.initialize(plan_path=Path("first/planned_set.parquet"))

    changed = RungLabelRun(
        _plan([make_row("A:2", "A", "two", ["cell_x"])]), out_dir=tmp_path
    )
    with pytest.raises(LabelRunError, match="different plan"):
        changed.remaining_partitions()


def test_publish_requires_exactly_one_label_per_planned_row(tmp_path: Path):
    rows = [
        make_row("A:1", "A", "one", ["cell_x"]),
        make_row(
            "A:2",
            "A",
            "two",
            ["cell_x"],
            provenance="augmented",
            operator="decorate",
        ),
    ]
    run = RungLabelRun(_plan(rows), out_dir=tmp_path)
    run.initialize(plan_path=Path("planned_set.parquet"))
    natural_dir = tmp_path / LabelRegime.NATURAL.value
    natural_dir.mkdir()
    pd.DataFrame([_label(rows[0])]).to_parquet(natural_dir / "labels.parquet", index=False)

    incomplete = run.publish_if_complete()

    assert not incomplete.published
    assert incomplete.missing == 1
    assert not (tmp_path / "labels.parquet").exists()

    supplemented_dir = tmp_path / LabelRegime.SUPPLEMENTED.value
    supplemented_dir.mkdir()
    pd.DataFrame([_label(rows[1])]).to_parquet(
        supplemented_dir / "labels.parquet", index=False
    )

    complete = run.publish_if_complete()

    assert complete.published
    assert complete.missing == 0
    labels = pd.read_parquet(tmp_path / "labels.parquet")
    assert list(zip(labels["dataset"], labels["query_id"])) == [("A", "1"), ("A", "2")]


def test_publish_rejects_duplicate_labels_across_regimes(tmp_path: Path):
    row = make_row("A:1", "A", "one", ["cell_x"])
    run = RungLabelRun(_plan([row]), out_dir=tmp_path)
    run.initialize(plan_path=Path("planned_set.parquet"))
    for regime in (LabelRegime.NATURAL, LabelRegime.SUPPLEMENTED):
        shard_dir = tmp_path / regime.value
        shard_dir.mkdir()
        pd.DataFrame([_label(row)]).to_parquet(shard_dir / "labels.parquet", index=False)

    with pytest.raises(LabelRunError, match="duplicate"):
        run.publish_if_complete()


def test_resume_only_returns_rows_not_written_by_this_run(tmp_path: Path):
    rows = [
        make_row("A:1", "A", "one", ["cell_x"]),
        make_row("A:2", "A", "two", ["cell_x"]),
    ]
    run = RungLabelRun(_plan(rows), out_dir=tmp_path)
    run.initialize(plan_path=Path("planned_set.parquet"))
    natural_dir = tmp_path / LabelRegime.NATURAL.value
    natural_dir.mkdir()
    pd.DataFrame([_label(rows[0])]).to_parquet(natural_dir / "labels.parquet", index=False)

    remaining = run.remaining_partitions()

    assert remaining[LabelRegime.NATURAL]["query_id"].tolist() == ["2"]


def test_plan_only_cli_writes_nothing(tmp_path: Path, capsys):
    plan_path = tmp_path / "planned_set.parquet"
    pd.DataFrame([make_row("A:1", "A", "one", ["cell_x"])]).to_parquet(
        plan_path, index=False
    )

    code = main(["--plan", str(plan_path), "--plan-only"])

    assert code == 0
    assert "natural" in capsys.readouterr().out
    assert not (tmp_path / "labeling").exists()

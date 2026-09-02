"""Advisory spend check over frozen Rung A artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from rungs.rung_a import RungA
from scripts.check_rung_spend import main
from tests.rungs_fixtures import default_config, make_catalog, make_row


def _write_plan(plan_dir: Path) -> None:
    catalog = make_catalog([
        make_row("A:1", "A", "alpha query", ["cell_x"]),
        make_row("A:2", "A", "beta query", ["cell_x"]),
    ])
    config = default_config(catalog, floor_by_cell=2, floor_by_lane=1)
    artifacts = RungA().compose(catalog, config, catalog_fp="catalog-fp")
    artifacts.write(plan_dir)
    catalog.to_parquet(plan_dir / "candidate_catalog.parquet", index=False)


def test_cli_writes_deterministic_advisory_report(tmp_path: Path, capsys):
    _write_plan(tmp_path)

    first = main(["--plan-dir", str(tmp_path), "--theta0", "0.7"])
    first_bytes = (tmp_path / "spend_check.json").read_bytes()
    second = main(["--plan-dir", str(tmp_path), "--theta0", "0.7"])

    assert first == second == 0
    assert (tmp_path / "spend_check.json").read_bytes() == first_bytes
    artifact = json.loads(first_bytes)
    assert artifact["advisory"] is True
    assert artifact["verdict"] == "PASS"
    assert artifact["theta0"] == 0.7
    assert artifact["report"]["rows_planned"] == 2
    assert artifact["report"]["predicted_yield"]["status"] == "not_activated"
    assert "ADVISORY PASS" in capsys.readouterr().out


def test_advisory_fail_is_recorded_without_failing_the_command(
    tmp_path: Path, capsys
):
    _write_plan(tmp_path)
    coverage_path = tmp_path / "coverage_report.json"
    coverage = json.loads(coverage_path.read_text())
    reachable = next(
        line for line in coverage["strata"] if line["achievable"] >= line["floor"]
    )
    reachable["selected"] = 0
    coverage_path.write_text(json.dumps(coverage))

    code = main(["--plan-dir", str(tmp_path), "--theta0", "0.7"])

    assert code == 0
    artifact = json.loads((tmp_path / "spend_check.json").read_text())
    assert artifact["verdict"] == "FAIL"
    assert artifact["report"]["failed_conditions"]
    assert "ADVISORY FAIL" in capsys.readouterr().out


def test_malformed_provenance_fails_cleanly_without_a_report(
    tmp_path: Path, capsys
):
    _write_plan(tmp_path)
    (tmp_path / "provenance.json").write_text("[]")

    code = main(["--plan-dir", str(tmp_path), "--theta0", "0.7"])

    assert code == 1
    assert "provenance.json must contain an object" in capsys.readouterr().err
    assert not (tmp_path / "spend_check.json").exists()


def test_advisory_check_does_not_read_or_modify_labeling_artifacts(tmp_path: Path):
    _write_plan(tmp_path)
    labeling = tmp_path / "labeling"
    labeling.mkdir()
    checkpoint = labeling / "active-checkpoint.bin"
    checkpoint.write_bytes(b"labeling-in-progress")
    before = {path.relative_to(labeling): path.read_bytes() for path in labeling.iterdir()}

    code = main(["--plan-dir", str(tmp_path), "--theta0", "0.7"])

    after = {path.relative_to(labeling): path.read_bytes() for path in labeling.iterdir()}
    assert code == 0
    assert after == before

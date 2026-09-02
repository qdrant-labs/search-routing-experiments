"""Run an advisory pre-label spend check over frozen Rung A artifacts.

This command never reads labels, contacts Qdrant, or changes plan membership.
Its PASS/FAIL verdict is advisory: a completed assessment exits successfully;
only missing, malformed, or inconsistent inputs produce a non-zero exit.

    poetry run python src/scripts/check_rung_spend.py \
        --plan-dir src/data/rungs/production --theta0 0.7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from rungs.label_run import RungLabelRun
from rungs.rung_a import CoverageLine, CoverageReport
from rungs.spend_gate import SpendGate

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
OUTPUT_NAME = "spend_check.json"


class SpendCheckError(RuntimeError):
    """Frozen plan artifacts cannot support a trustworthy assessment."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        required=True,
        help="directory containing the frozen Rung A artifacts",
    )
    parser.add_argument(
        "--theta0",
        type=float,
        required=True,
        help="explicit initial novelty threshold used for this composition",
    )
    args = parser.parse_args(argv)
    plan_dir = _resolve(args.plan_dir)

    try:
        artifact = assess_plan_dir(plan_dir, theta0=args.theta0)
        _atomic_json(plan_dir / OUTPUT_NAME, artifact)
    except (OSError, ValueError, SpendCheckError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    verdict = artifact["verdict"]
    report = artifact["report"]
    print(
        f"ADVISORY {verdict}: planned={report['rows_planned']:,} "
        f"reachable_met={report['reachable_floors_met']:,} "
        f"reachable_unmet={len(report['reachable_floors_unmet']):,} "
        f"unreachable_debt={len(report['unreachable_floors']):,}"
    )
    for condition in report["failed_conditions"]:
        print(f"- {condition}")
    print(f"report: {plan_dir / OUTPUT_NAME}")
    print("predicted yield: not activated; this is a structural sanity check")
    return 0


def assess_plan_dir(plan_dir: Path, *, theta0: float) -> dict[str, object]:
    """Load one immutable plan directory and return its advisory artifact."""
    paths = {
        "plan": plan_dir / "planned_set.parquet",
        "trace": plan_dir / "selection_trace.parquet",
        "coverage": plan_dir / "coverage_report.json",
        "catalog": plan_dir / "candidate_catalog.parquet",
        "provenance": plan_dir / "provenance.json",
    }
    missing = [path.name for path in paths.values() if not path.exists()]
    if missing:
        raise SpendCheckError(f"{plan_dir} missing required artifacts {missing}")

    planned = pd.read_parquet(paths["plan"])
    trace = pd.read_parquet(paths["trace"])
    catalog = pd.read_parquet(paths["catalog"])
    coverage = _read_coverage(paths["coverage"])
    provenance = json.loads(paths["provenance"].read_text())
    if not isinstance(provenance, dict):
        raise SpendCheckError(f"{paths['provenance'].name} must contain an object")
    _validate_inputs(planned, trace, catalog, coverage, provenance)

    plan_fp = RungLabelRun(planned, out_dir=plan_dir / "labeling").plan_fp
    report = SpendGate().assess(
        planned,
        trace,
        coverage,
        catalog=catalog,
        theta0=theta0,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "advisory": True,
        "plan_fp": plan_fp,
        "catalog_fp": provenance.get("catalog_fp"),
        "config_fp": provenance.get("config_fp"),
        "theta0": theta0,
        "verdict": report.verdict,
        "report_fp": report.fingerprint(),
        "limitations": [
            "predicted downstream label yield is not activated",
            "concentration and novelty have no calibrated warning thresholds",
            "this report does not alter or stop labeling",
        ],
        "report": report.to_dict(),
    }


def _read_coverage(path: Path) -> CoverageReport:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise SpendCheckError(f"{path} must contain an object")
    try:
        strata = [CoverageLine(**line) for line in raw.pop("strata")]
        return CoverageReport(strata=strata, **raw)
    except (KeyError, TypeError) as error:
        raise SpendCheckError(f"{path} has an invalid coverage schema: {error}") from error


def _validate_inputs(
    planned: pd.DataFrame,
    trace: pd.DataFrame,
    catalog: pd.DataFrame,
    coverage: CoverageReport,
    provenance: dict[str, object],
) -> None:
    if coverage.planned_size != len(planned):
        raise SpendCheckError(
            "coverage_report planned_size does not match planned_set "
            f"({coverage.planned_size} != {len(planned)})"
        )
    if provenance.get("planned_size") != len(planned):
        raise SpendCheckError(
            "provenance planned_size does not match planned_set "
            f"({provenance.get('planned_size')} != {len(planned)})"
        )
    for name, frame in (("planned_set", planned), ("selection_trace", trace)):
        if "row_id" not in frame.columns:
            raise SpendCheckError(f"{name} missing row_id")
        if frame["row_id"].astype(str).duplicated().any():
            raise SpendCheckError(f"{name} contains duplicate row_id values")
    planned_ids = set(planned["row_id"].astype(str))
    trace_ids = set(trace["row_id"].astype(str))
    if trace_ids != planned_ids:
        raise SpendCheckError(
            "selection_trace identities do not exactly match planned_set "
            f"(missing={len(planned_ids - trace_ids)}, extra={len(trace_ids - planned_ids)})"
        )
    if "row_id" not in catalog.columns:
        raise SpendCheckError("candidate_catalog missing row_id")
    catalog_ids = set(catalog["row_id"].astype(str))
    missing_from_catalog = planned_ids - catalog_ids
    if missing_from_catalog:
        raise SpendCheckError(
            f"candidate_catalog is missing {len(missing_from_catalog)} planned identities"
        )


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(payload)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())

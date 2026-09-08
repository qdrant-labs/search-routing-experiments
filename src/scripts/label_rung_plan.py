"""Label one immutable Rung A planned set into a run-scoped artifact.

No historical identity-only cache is consulted. Re-running the command resumes
only labels checkpointed under the same output directory and plan fingerprint.
SpendGate and Rung B are separate commands and are never invoked here.

    poetry run python src/scripts/label_rung_plan.py \
        --plan src/data/rungs/production/planned_set.parquet --plan-only

    poetry run python src/scripts/label_rung_plan.py \
        --plan src/data/rungs/production/planned_set.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from rungs.label_run import LabelRegime, LabelRunError, RungLabelRun

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        help="run directory; defaults to <plan-dir>/labeling",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print the run-scoped workload without Qdrant access or writes",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        metavar="LANE",
        help="execute only named lanes; the combined artifact waits for all lanes",
    )
    args = parser.parse_args(argv)

    plan_path = _resolve(args.plan)
    out = _resolve(args.out) if args.out is not None else plan_path.parent / "labeling"
    try:
        run = RungLabelRun.from_path(plan_path, out_dir=out)
        remaining = run.remaining_partitions()
    except (OSError, LabelRunError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    workload = _workload(run.partitions(), remaining)
    print(workload.to_string(index=False))
    print(
        json.dumps(
            {
                "plan_fp": run.plan_fp,
                "planned": len(run.plan),
                "remaining": int(sum(len(frame) for frame in remaining.values())),
                "out": str(out),
            },
            indent=2,
        )
    )
    if args.plan_only:
        return 0

    try:
        run.initialize(plan_path=plan_path)
        failures = _execute(run, remaining, only=tuple(args.only) if args.only else None)
        completion = run.publish_if_complete()
    except (OSError, LabelRunError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "planned": completion.planned,
                "labelled": completion.labelled,
                "missing": completion.missing,
                "published": completion.published,
                "labels": str(out / "labels.parquet") if completion.published else None,
            },
            indent=2,
        )
    )
    if failures:
        print(f"failed lanes: {sorted(failures)}", file=sys.stderr)
        return 1
    if args.only:
        return 0
    return 0 if completion.published else 1


def _execute(
    run: RungLabelRun,
    remaining: dict[LabelRegime, pd.DataFrame],
    *,
    only: tuple[str, ...] | None,
) -> set[str]:
    from dotenv import load_dotenv
    from qdrant_client import QdrantClient

    from rungs.retrieval_labeling import ConstructedLabelSweep, LaneLabelSweep

    load_dotenv()
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60,
    )
    failures: set[str] = set()
    for regime in (LabelRegime.NATURAL, LabelRegime.SUPPLEMENTED):
        selection = _only(remaining[regime], only)
        if selection.empty:
            continue
        failures.update(
            LaneLabelSweep(
                client,
                selection,
                out_dir=run.regime_dir(regime),
                supplemented=regime is LabelRegime.SUPPLEMENTED,
                run_fp=run.plan_fp,
            ).run(only=only)
        )
    selection = _only(remaining[LabelRegime.CONSTRUCTED], only)
    if not selection.empty:
        failures.update(
            ConstructedLabelSweep(
                client,
                selection,
                out_dir=run.regime_dir(LabelRegime.CONSTRUCTED),
                run_fp=run.plan_fp,
            ).run(only=only)
        )
    return failures


def _only(frame: pd.DataFrame, lanes: tuple[str, ...] | None) -> pd.DataFrame:
    if not lanes:
        return frame
    return frame[frame["dataset"].astype(str).isin(lanes)].reset_index(drop=True)


def _workload(
    planned: dict[LabelRegime, pd.DataFrame],
    remaining: dict[LabelRegime, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for regime in LabelRegime:
        planned_counts = planned[regime]["dataset"].value_counts()
        remaining_counts = remaining[regime]["dataset"].value_counts()
        for lane in sorted(planned_counts.index.astype(str)):
            rows.append({
                "regime": regime.value,
                "dataset": lane,
                "planned": int(planned_counts.get(lane, 0)),
                "checkpointed": int(
                    planned_counts.get(lane, 0) - remaining_counts.get(lane, 0)
                ),
                "remaining": int(remaining_counts.get(lane, 0)),
            })
    return pd.DataFrame(
        rows,
        columns=["regime", "dataset", "planned", "checkpointed", "remaining"],
    )


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())

"""Cost planner CLI: partition a frozen Rung A plan into cache hits and
misses under the full fingerprint key. Fails hard when misses exceed the
paid-label budget (spec:156-159).

    poetry run python src/scripts/plan_labels_v2.py \\
        --plan src/data/rungs/smoke/planned_set.parquet \\
        --fingerprints src/data/rungs/smoke/label_fingerprints.parquet \\
        --labels src/data/route_labels/labels.parquet --budget 5000 \\
        --out src/data/rungs/smoke/cost_partition
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from rungs.cache import FP_COLUMNS, LabelCache
from rungs.cost_planner import BudgetExceeded, CostPlanner

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_fingerprints(path: Path) -> dict[tuple[str, str], tuple[str, str, str, str]]:
    frame = pd.read_parquet(path).astype({"dataset": str, "query_id": str})
    missing = [c for c in FP_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} missing fingerprint columns {missing}; per spec:151-153 "
            "absent fingerprints are cache misses, so run with --strict to fail."
        )
    return {
        (row["dataset"], row["query_id"]): tuple(str(row[c]) for c in FP_COLUMNS)
        for _, row in frame.iterrows()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--fingerprints",
        type=Path,
        help="parquet of (dataset, query_id, query_fp, corpus_fp, qrels_fp, retrieval_stack_fp)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        action="append",
        default=[],
        help="one or more label parquet files (each must carry all 4 fingerprints)",
    )
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    plan_path = args.plan if args.plan.is_absolute() else REPO_ROOT / args.plan
    plan = pd.read_parquet(plan_path).astype({"dataset": str, "query_id": str})
    fps = (
        _load_fingerprints(args.fingerprints)
        if args.fingerprints
        else {(d, q): ("", "", "", "") for d, q in zip(plan["dataset"], plan["query_id"])}
    )
    cache = LabelCache(args.labels)

    planner = CostPlanner(cache)
    try:
        partition = planner.partition(plan, fingerprints=fps, budget=args.budget)
    except BudgetExceeded as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    partition.hits.to_parquet(out / "hits.parquet", index=False)
    partition.misses.to_parquet(out / "misses.parquet", index=False)
    planner.write_report(partition, out / "cost_report.parquet")
    summary = {
        "plan_size": partition.plan_size,
        "hits": len(partition.hits),
        "misses": len(partition.misses),
        "budget": partition.budget,
        "spend": partition.spend,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

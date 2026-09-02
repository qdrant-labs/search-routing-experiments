"""Rung B: validate a completed planned set. Read-only join of planned_set
against labels and assess its empirical route distribution as one JSON file.

    poetry run python src/scripts/validate_rung_b.py \\
        --plan src/data/rungs/smoke/planned_set.parquet \\
        --labels src/data/route_labels/labels.parquet \\
        --hits 0 --misses 0 \\
        --out src/data/rungs/smoke/rung_b_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from rungs.rung_b import RungB, RungBIntegrityError

REPO_ROOT = Path(__file__).resolve().parents[2]


def _log(t0: float, message: str) -> None:
    print(f"[+{time.perf_counter() - t0:6.1f}s] {message}", file=sys.stderr, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        action="append",
        required=True,
        help="label parquet shard; repeat to assess an immutable shard union",
    )
    parser.add_argument("--hits", type=int, default=0)
    parser.add_argument("--misses", type=int, default=0)
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="named reference label parquet; repeat for v2/v3 matched-cohort comparisons",
    )
    parser.add_argument(
        "--accounting",
        type=Path,
        help="JSON containing cache, stage, provider, and cost accounting",
    )
    parser.add_argument("--decisive-margin", type=float, default=0.4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    plan_path = args.plan if args.plan.is_absolute() else REPO_ROOT / args.plan
    _log(t0, f"Rung B: loading plan {plan_path}")
    plan = pd.read_parquet(plan_path).astype({"dataset": str, "query_id": str})
    _log(t0, f"Rung B: loaded {len(plan):,} planned rows across {plan['dataset'].nunique()} lanes")
    label_paths = [path if path.is_absolute() else REPO_ROOT / path for path in args.labels]
    _log(t0, f"Rung B: loading {len(label_paths):,} label shard(s)")
    labels = pd.concat(
        [
            pd.read_parquet(path).astype({"dataset": str, "query_id": str})
            for path in label_paths
        ],
        ignore_index=True,
    )
    _log(t0, f"Rung B: loaded {len(labels):,} label rows")
    try:
        references = _load_references(args.reference)
        accounting = _load_accounting(args.accounting)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    try:
        _log(t0, "Rung B: validating and assessing the immutable completed set")
        report = RungB(decisive_margin=args.decisive_margin).assess(
            plan,
            labels,
            cost_hits=args.hits,
            cost_misses=args.misses,
            references=references,
            accounting=accounting,
        )
    except RungBIntegrityError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    _log(t0, f"Rung B: writing report to {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    payload = {**report.to_dict(), "report_fp": report.fingerprint()}
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(out)
    _log(t0, "Rung B: complete")
    print(f"wrote {out}")
    print(json.dumps(report.counts, indent=2))
    return 0


def _load_references(values: Sequence[str]) -> dict[str, pd.DataFrame]:
    references: dict[str, pd.DataFrame] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"invalid --reference {value!r}; expected NAME=PATH")
        if name in references:
            raise ValueError(f"duplicate --reference name {name!r}")
        path = Path(raw_path)
        path = path if path.is_absolute() else REPO_ROOT / path
        references[name] = pd.read_parquet(path).astype(
            {"dataset": str, "query_id": str}
        )
    return references


def _load_accounting(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    resolved = path if path.is_absolute() else REPO_ROOT / path
    payload = json.loads(resolved.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{resolved} must contain a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())

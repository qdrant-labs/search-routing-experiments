"""Compose v4 targets from the pre-label candidate catalog — diversity coverage
only, no route labels in the decision.

    poetry run python src/scripts/compose_v4.py [--force]
    poetry run python src/scripts/compose_v4.py --target 5000 --out /tmp/v4_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from composition.composer_v4 import ComposerV4, CoverageReport
from composition.recipe import Recipe

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG = REPO_ROOT / "src" / "data" / "v4" / "catalog_v4.parquet"


def _log(t0: float, message: str) -> None:
    print(f"[+{time.perf_counter() - t0:6.1f}s] {message}", file=sys.stderr, flush=True)


def _summary(report: CoverageReport) -> str:
    met = sum(1 for r in report.requirements if r.missing == 0)
    return (
        f"|D|={report.size:,} target={report.target:,} lanes={report.lanes_seen} "
        f"| requirements {met}/{len(report.requirements)} met "
        f"| {len(report.shortages)} shortages | families={report.families_used:,} "
        f"| lane_capped={report.lane_capped}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=200_000)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--out", type=Path, default=Path("src/data/v4"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out = out.resolve()
    targets_path = out / "composition_targets_v4.parquet"
    provenance_path = out / "composition_targets_v4.provenance.json"
    if not args.force:
        for path in (targets_path, provenance_path):
            if path.exists():
                print(f"error: {path} already exists; pass --force to overwrite", file=sys.stderr)
                return 1
    out.mkdir(parents=True, exist_ok=True)

    catalog_path = args.catalog if args.catalog.is_absolute() else REPO_ROOT / args.catalog
    if not catalog_path.exists():
        print(f"error: {catalog_path} not found; run build_v4_catalog.py first", file=sys.stderr)
        return 1

    recipe = Recipe()
    config = {
        "target": args.target,
        "rho": args.rho,
        "kappa": recipe.target_lane_share,
        "stratum_floor": recipe.stratum_floor,
    }
    t0 = time.perf_counter()
    _log(t0, f"loading catalog {catalog_path}")
    catalog = pd.read_parquet(catalog_path)
    catalog["cells"] = catalog["cells"].map(frozenset)
    _log(t0, f"loaded {len(catalog):,} candidates across {catalog['dataset'].nunique()} lanes")

    _log(t0, f"composing target={args.target:,} rho={args.rho} kappa={recipe.target_lane_share}")
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        frame, report = ComposerV4().compose(
            catalog, args.target, rho=args.rho,
            kappa=recipe.target_lane_share, stratum_floor=recipe.stratum_floor,
            on_progress=lambda done, tgt: _log(t0, f"  selected {done:,}/{tgt:,}"),
        )
    except (AssertionError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finished_at = datetime.now(timezone.utc).isoformat()
    _log(t0, f"composed |D|={report.size:,}")

    targets_tmp = targets_path.with_suffix(targets_path.suffix + ".tmp")
    provenance_tmp = provenance_path.with_suffix(provenance_path.suffix + ".tmp")
    frame.assign(cells=frame["cells"].map(sorted)).to_parquet(targets_tmp, index=False)
    provenance_tmp.write_text(
        json.dumps(
            {
                "config": config,
                "started_at": started_at,
                "finished_at": finished_at,
                "report": asdict(report),
            },
            indent=2,
        )
        + "\n"
    )
    os.replace(targets_tmp, targets_path)
    os.replace(provenance_tmp, provenance_path)
    _log(t0, f"wrote {targets_path.name} and {provenance_path.name} to {out}")
    print(_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

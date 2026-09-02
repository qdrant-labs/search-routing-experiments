"""Rung A: build the version-neutral candidate catalog and compose a
label-blind plan. All artifacts land under --out; catalog and plan carry
fingerprints so a byte-identical rerun is verifiable.

    poetry run python src/scripts/compose_rung_a.py \\
        --planned-size 5000 --theta0 0.7 --theta-step 0.1 --rho 0.5 \\
        --stratum-floor 25 --lane-cap-frac 0.2 --out src/data/rungs/smoke
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

from composition.strata import STRATA
from rungs.catalog import CandidateCatalog, CatalogConfig
from rungs.rung_a import RungA, RungAConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def _log(t0: float, message: str) -> None:
    print(f"[+{time.perf_counter() - t0:6.1f}s] {message}", file=sys.stderr, flush=True)


def _floors_from_recipe(
    catalog: pd.DataFrame, *, stratum_floor: int, rho: float
) -> dict[tuple[str, str], int]:
    """Build the floor table from explicit run dials: cells scale with
    rho*catalog-size/cell-count; every other axis stays at the flat floor."""
    floors: dict[tuple[str, str], int] = {}
    cells = sorted({c for cs in catalog["cells"] for c in cs})
    for cell in cells:
        # spec:180 — the exact scaled value stays configurable, not an
        # implementation constant.
        scaled = max(stratum_floor, rho * len(catalog) / max(1, len(cells)))
        floors[("cell", cell)] = int(scaled)
    for lane in sorted(catalog["dataset"].astype(str).unique()):
        floors[("lane", lane)] = stratum_floor
    for axis, bands in STRATA.items():
        for band in bands:
            floors[(axis, band)] = stratum_floor
    return floors


def _lane_budgets(
    catalog: pd.DataFrame, *, planned_size: int, lane_cap_frac: float
) -> dict[str, int]:
    cap = math.ceil(lane_cap_frac * planned_size)
    return {lane: cap for lane in catalog["dataset"].astype(str).unique()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planned-size", type=int, required=True)
    parser.add_argument("--theta0", type=float, required=True)
    parser.add_argument("--theta-step", type=float, required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--stratum-floor", type=int, required=True)
    parser.add_argument("--lane-cap-frac", type=float, required=True)
    parser.add_argument("--per-lane-cap", type=int, default=None)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--residual-fill", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    planned_path = out / "planned_set.parquet"
    if planned_path.exists() and not args.force:
        print(f"error: {planned_path} exists; pass --force to overwrite", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    catalog_config = CatalogConfig(
        default_per_lane_cap=args.per_lane_cap,
        include_augmentation=not args.no_augmentation,
        seed=args.seed,
    )
    _log(t0, "building candidate catalog")
    catalog_df, manifests, catalog_fp = CandidateCatalog(catalog_config).build()
    _log(
        t0,
        f"catalog: {len(catalog_df):,} candidates across "
        f"{catalog_df['dataset'].nunique()} lanes (fp={catalog_fp[:12]})",
    )
    _log(
        t0,
        f"dropped: natural={catalog_df.attrs.get('dropped_natural',0):,} "
        f"generated={catalog_df.attrs.get('dropped_generated',0):,} "
        f"dedup={catalog_df.attrs.get('dropped_dedup',0):,}",
    )

    floors = _floors_from_recipe(
        catalog_df, stratum_floor=args.stratum_floor, rho=args.rho
    )
    budgets = _lane_budgets(
        catalog_df, planned_size=args.planned_size, lane_cap_frac=args.lane_cap_frac
    )
    config = RungAConfig(
        planned_size_ceiling=args.planned_size,
        floors=floors,
        lane_budgets=budgets,
        theta0=args.theta0,
        theta_step=args.theta_step,
        residual_fill=args.residual_fill,
        seed=args.seed,
    )

    _log(t0, f"composing plan (ceiling={args.planned_size:,})")
    artifacts = RungA().compose(
        catalog_df, config, catalog_fp=catalog_fp,
        on_progress=lambda done, ceil: _log(t0, f"  picked {done:,}/{ceil:,}"),
    )
    _log(
        t0,
        f"planned: |D|={artifacts.coverage_report.planned_size:,} "
        f"lanes={artifacts.coverage_report.lanes_used} "
        f"debt={len(artifacts.generation_debt):,}",
    )

    _log(t0, f"writing Rung A artifacts under {out}")
    out.mkdir(parents=True, exist_ok=True)
    artifacts.write(out)
    tmp = out / "candidate_manifests.parquet.tmp"
    manifests.to_parquet(tmp, index=False)
    tmp.replace(out / "candidate_manifests.parquet")
    tmp = out / "candidate_catalog.parquet.tmp"
    catalog_df.assign(cells=catalog_df["cells"].map(sorted)).to_parquet(tmp, index=False)
    tmp.replace(out / "candidate_catalog.parquet")

    summary_path = out / "summary.json"
    summary = {
        "catalog_fp": catalog_fp,
        "planned_size": artifacts.coverage_report.planned_size,
        "lanes_used": artifacts.coverage_report.lanes_used,
        "families_used": artifacts.coverage_report.families_used,
        "provenance_mix": artifacts.coverage_report.provenance_mix,
        "debt": len(artifacts.generation_debt),
        "wrote": str(out),
    }
    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    tmp.replace(summary_path)
    _log(t0, f"wrote artifacts under {out}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

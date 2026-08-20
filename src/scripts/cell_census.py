"""Per-cell census over the labelled pool: what each cell claims, and whether
its `predicts` prior survives contact with the labels.

Answers the two questions a repair-or-extend decision needs — which cells are
empty, which are inverted, and where the pool's mass sits with no cell to claim
it. No retrieval, no labelling; reads the re-derived pool and the v3 catalog.

    poetry run python src/scripts/cell_census.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from composition.cells import CELLS, CELLS_BY_NAME
from composition.floors import with_derived
from hybrid_search_rrf_dataset.objective import RouterObjective

DATA = Path(__file__).resolve().parent.parent / "data"
POOL = DATA / "v3" / "labels_rederived.parquet"
CATALOG = DATA / "v3" / "catalog_v3.parquet"
OUT = DATA / "v3" / "cell_census.parquet"

SCORES = ["score_dense_only", "score_pure_rrf", "score_sparse_only"]
ROUTE_TO_CLASS = {"dense_only": "dense", "sparse_only": "sparse", "pure_rrf": "hybrid"}
MARGIN = RouterObjective().decisive_margin


def _classified(pool: pd.DataFrame) -> pd.DataFrame:
    ordered = np.sort(pool[SCORES].to_numpy(), axis=1)
    oracle, runner, low = ordered[:, -1], ordered[:, -2], ordered[:, 0]
    winner = pool[SCORES].idxmax(axis=1).str.replace("score_", "")
    decisive = (oracle > 1e-9) & (oracle - low > 1e-9) & (oracle - runner >= MARGIN)
    return pool.assign(
        cls=np.where(decisive, winner.map(ROUTE_TO_CLASS).fillna(""), ""),
        answered=oracle > 1e-9,
    )


def census() -> tuple[pd.DataFrame, pd.Series]:
    """One row per cell, plus the per-row count of cells claiming it."""
    pool = _classified(pd.read_parquet(POOL).astype({"query_id": str}))
    catalog = with_derived(pd.read_parquet(CATALOG).astype({"query_id": str}))
    joined = catalog.merge(
        pool[["dataset", "query_id", "cls", "answered"]],
        on=["dataset", "query_id"], how="inner",
    )

    rows, claims = [], np.zeros(len(joined), dtype=int)
    for cell in CELLS_BY_NAME.values():
        missing = [b.column for b in cell.bands if b.column not in joined.columns]
        if missing:
            rows.append({"cell": cell.name, "claimed": 0, "note": f"no column {missing[0]}"})
            continue
        mask = cell.select(joined).to_numpy()
        claims += mask
        sub = joined[mask]
        decided = sub[sub["cls"] != ""]
        mix = decided["cls"].value_counts()
        top = str(mix.index[0]) if len(mix) else ""
        rows.append({
            "cell": cell.name,
            "v": "v2" if cell.name in {c.name for c in CELLS} else "v3",
            "claimed": int(mask.sum()),
            "share": round(float(mask.mean()), 5),
            "decisive": len(decided),
            "dense": int(mix.get("dense", 0)),
            "sparse": int(mix.get("sparse", 0)),
            "hybrid": int(mix.get("hybrid", 0)),
            "top_route": top,
            "predicts": "|".join(sorted(cell.predicts_class)),
            "prior_holds": top in cell.predicts_class if top else None,
            "lanes": int(sub["dataset"].nunique()),
            "note": "",
        })
    return pd.DataFrame(rows).set_index("cell"), pd.Series(claims, index=joined.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    frame, claims = census()

    empty = frame[frame["claimed"] == 0]
    print(f"49 cells | claimed nothing: {len(empty)} -> {list(empty.index)}")
    thin = frame[(frame["claimed"] > 0) & (frame["decisive"] == 0)]
    print(f"claimed rows but NO decisive row: {len(thin)} -> {list(thin.index)}")

    priors = frame[frame["prior_holds"].notna()]
    broken = priors[~priors["prior_holds"].astype(bool)]
    print(f"\nprior INVERTED (top route not in predicts): {len(broken)}/{len(priors)}")
    print(broken[["v", "decisive", "dense", "sparse", "hybrid", "top_route", "predicts"]]
          .sort_values("decisive", ascending=False).head(args.top).to_string())

    print(f"\nrows claimed by NO cell: {int((claims == 0).sum()):,} "
          f"({(claims == 0).mean():.1%} of the labelled pool)")
    print(f"rows claimed by >1 cell: {int((claims > 1).sum()):,} "
          f"({(claims > 1).mean():.1%}) — overlapping cells double-draw")

    print("\nwidest cells by share of the pool:")
    print(frame.nlargest(args.top, "share")[
        ["v", "claimed", "share", "decisive", "top_route", "predicts", "prior_holds"]
    ].to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()

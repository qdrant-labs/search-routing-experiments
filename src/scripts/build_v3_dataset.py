"""The v3 dataset — the actual deliverable, not a feasibility report: the
corrected selector's row set (certified + tier-0 top-up + waste, `selected.
parquet`) joined with query text, labels, cell membership and corruption
degree, plus a real per-cell K_cap=5 quota/breach report. No new selection
logic and no new spend — everything here already exists on disk.

    poetry run python src/scripts/build_v3_dataset.py
"""

from __future__ import annotations

import pandas as pd

from composition.cellfill import MassCap
from composition.recipe import Recipe
from scripts.select_v3_prototype import (
    DATA,
    OUT,
    SelectorRecipe,
    _active_cells,
    _load_labels,
    attach_strata,
    classify,
)

DATASET_OUT = DATA / "v3" / "dataset_v3.parquet"
CELL_REPORT_OUT = DATA / "v3" / "cell_k_cap_report.parquet"


def full_pool(recipe: SelectorRecipe) -> pd.DataFrame:
    """The same classified, strata-attached pool the selector itself drew
    from — re-derived here (not re-read from a thin projection) so the
    dataset carries query text, cell tags and corruption degree too."""
    return attach_strata(classify(_load_labels(), recipe))


def cell_k_cap_report(pool: pd.DataFrame, recipe: Recipe) -> pd.DataFrame:
    """Real K_cap=5 per cell (5.1): natural prevalence against the WHOLE
    labelled pool (mirrors CellFill._plan's own natural/catalog-size
    convention), not just the rows this build happened to select — a cell's
    cap is a fact about the corpus, not about today's draw."""
    n = len(pool)
    rows = []
    for cell in _active_cells():
        natural = int(pool["cells"].map(lambda s, name=cell.name: name in s).sum())
        mass = MassCap(natural, n, recipe)
        rows.append({
            "cell": cell.name,
            "natural_rows": natural,
            "p_natural": round(mass.p_natural, 6),
            "quota": mass.quota,
            "mass_cap": mass.cap,
            "generation_only": mass.generation_only,
        })
    return pd.DataFrame(rows).sort_values("p_natural").reset_index(drop=True)


def main() -> None:
    recipe = SelectorRecipe()
    pool = full_pool(recipe)
    selected = pd.read_parquet(OUT / "selected.parquet").astype({"query_id": str})

    # start from the selector's OWN verdict (route_class/certified, including
    # its waste recode) and left-join everything else off the pool — never
    # the reverse, or the pool's pre-recode route_class/certified would win
    pool_cols = [c for c in pool.columns if c not in ("route_class", "certified")]
    dataset = selected.merge(pool[pool_cols], on=["dataset", "query_id"], how="left")
    assert dataset["query"].notna().all(), "a selected row is missing from the labelled pool"
    assert len(dataset) == len(selected), "selected keys missing from pool"

    cap_report = cell_k_cap_report(pool, Recipe(k_cap=5.0))

    dataset["cells"] = dataset["cells"].map(sorted)  # frozenset isn't parquet-serializable
    DATASET_OUT.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(DATASET_OUT, index=False)
    cap_report.to_parquet(CELL_REPORT_OUT, index=False)

    print(f"v3 dataset: {len(dataset):,} rows x {dataset.shape[1]} cols -> {DATASET_OUT}")
    print(dataset["route_class"].value_counts().to_string())
    breached = cap_report[cap_report["generation_only"]]
    print(
        f"\nK_cap=5 (5.1): {len(breached)}/{len(cap_report)} cells are "
        f"generation-only (capped allocation cannot reach the floor of "
        f"{recipe.per_dataset_floor} organically) -> {CELL_REPORT_OUT}"
    )
    print(breached[["cell", "p_natural", "quota"]].to_string(index=False))


if __name__ == "__main__":
    main()

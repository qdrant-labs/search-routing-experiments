"""Every number of the cell fill lives here, nowhere else. Frozen so a
build's artifact can be trusted to match its recipe; changing a value means
a new build (`--force`)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FloorRules(BaseModel):
    """d33a floor mechanics — how evidence is counted and capped."""

    model_config = ConfigDict(frozen=True)

    cap_frac: float = Field(
        default=0.5,
        description=(
            "d29 anti-monoculture cap: no dataset supplies more than this "
            "fraction of any floor (waived for sole suppliers)."
        ),
    )


class Recipe(BaseModel):
    """d48/d49 cell-fill values — one field per SPEC decision."""

    model_config = ConfigDict(frozen=True)

    seed: int = 0
    floor_rules: FloorRules = Field(default_factory=FloorRules)
    """d33a floor mechanics — see `FloorRules`."""
    n_per_route: int = Field(
        default=200,
        description=(
            "Rows the cell fill quotas per route in EVERY cell, dense and "
            "sparse alike — never narrowed by what current labels happen to "
            "show. A cell's draw size is twice this. THE dial between "
            "labelling budget and dataset size: 32 cells x n x 2 routes is the "
            "decisive-row target, and the draw scales with it."
        ),
    )
    k_cap: float | None = Field(
        default=None,
        description=(
            "Policy cap on a cell's allocation: at most this multiple of the "
            "cell's own share of the catalog, times `certified_total`. None "
            "keeps the flat `n_per_route` every cell drew before, so a v2 "
            "rebuild under default arguments is unchanged."
        ),
    )
    certified_total: int = Field(
        default=4_247,
        description=(
            "The total the cap is a share of: the certified-tier feasible total "
            "the v3 selector reports at its target split (v3_feasibility/"
            "report.md, binding class sparse). Measured, not chosen."
        ),
    )
    cell_floor: int = Field(
        default=25,
        description=(
            "Rows below which a cell is no longer a usable stratum (the "
            "selector's per-dataset floor). A cell whose capped allocation "
            "cannot reach it has no organic path and is reported "
            "generation-only."
        ),
    )
    target_lane_share: float = Field(
        default=0.2,
        description=(
            "The row share of a cell a single lane should stay under. Not the "
            "enforced cap: that is computed per cell from its own lane counts, "
            "and this is only what the computed cap is measured against."
        ),
    )
    control_share: float = Field(
        default=0.2,
        description=(
            "Control rows, as a share of the cell rows — drawn from queries no "
            "cell claimed and never trained on. Provisional: the defensible "
            "size needs the labels these rows do not have yet."
        ),
    )
    min_natural_share: float = Field(
        default=0.85,
        description=(
            "d42i/d43 review: minimum share of the composition that is "
            "natural provenance. Enforced by the cell fill as an admission "
            "ceiling — natural_rows*(1-m)/m augmented rows at most."
        ),
    )

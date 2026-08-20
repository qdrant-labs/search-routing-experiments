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
    """d48/d49 cell-fill values plus the v3 composition dials — one field per
    decision. Bare `Recipe()` keeps every v2 default; `Recipe.v3()` is the v3
    build's configuration."""

    model_config = ConfigDict(frozen=True)

    seed: int = 0
    floor_rules: FloorRules = Field(default_factory=FloorRules)
    """d33a floor mechanics — see `FloorRules`."""
    n_per_route: int = Field(
        default=200,
        description=(
            "v2 CellFill only — the flat per-route quota every cell drew "
            "before K_cap; v3 targets are per-axis marginals, never this. "
            "A cell's draw size is twice this."
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
            "The total the cap is a share of. Frozen v2-era constant only: a "
            "v3 build computes its own feasible total live and reports it; "
            "this default exists so MassCap stays usable standalone."
        ),
    )
    stratum_floor: int = Field(
        default=25,
        description=(
            "Rows below which ANY diversity stratum (cell, lane, corruption "
            "degree) is no longer usable — one floor for every marginal axis. "
            "A cell whose capped allocation cannot reach it has no organic "
            "path and is reported generation-only."
        ),
    )
    target_lane_share: float = Field(
        default=0.2,
        description=(
            "The row share a single lane should stay under. Two enforcement "
            "modes, one dial: v2's per-cell LaneCap relaxes it to the tightest "
            "achievable share; v3's feasible_total shrinks the class total "
            "until the cap holds strictly."
        ),
    )
    target_split: tuple[float, float, float] = Field(
        default=(0.45, 0.45, 0.10),
        description="dense/sparse/hybrid class shares of the selected tiers.",
    )
    waste_cap: float = Field(
        default=0.05,
        description=(
            "Dictated waste (fake_tie + all_zero) share of the tier-0 total — "
            "a policy spend, decided 2026-08-20, never inferred."
        ),
    )
    eval_reserve_frac: float = Field(
        default=0.2,
        description=(
            "Certified rows frozen for ablation BEFORE any selection, "
            "stratified (lane x route class); excluded with their near-dup "
            "cluster-mates."
        ),
    )
    genuine_tie_depth: int = Field(
        default=2,
        description=(
            "Judged-doc count below which an all-tied row is a fake tie (a "
            "qrels artifact), not genuine hybrid supply."
        ),
    )
    class_margin: float = Field(
        default=0.4,
        description=(
            "oracle - runner_up at or above this certifies a decisive route "
            "class; margin 0 is the tier-0 (routes_differ) bar."
        ),
    )
    target_total: int = Field(
        default=200_000,
        description="The extrapolation target the generation debt is sized against.",
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

    @classmethod
    def v3(cls, **overrides) -> Recipe:
        """The v3 composition configuration: K_cap allocation ON; everything
        else is already the shared default."""
        return cls(**{"k_cap": 5.0, **overrides})

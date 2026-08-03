"""SPEC d32/d33 recipe — every number of the fill lives here, nowhere
else. One `total_rows` splits into the four slices by `SliceShares` (the
d32 60/20/20 macro-split with its 80/20 sub-split inside span-evidence =
48/12/20/20); row counts are derived properties, never stored. Frozen so
a build's artifact can be trusted to match its recipe; changing a value
means a new build (`--force`)."""

from __future__ import annotations

from collections.abc import Mapping
from math import isclose

from pydantic import BaseModel, ConfigDict, Field, model_validator

from query_taxonomy.core import AmbiguityTier

_DEFAULT_DISCOUNTS = {
    AmbiguityTier.RIGID: 1.0,
    AmbiguityTier.MODERATE: 0.75,
    AmbiguityTier.AMBIGUOUS: 0.5,
}

_CHAMPIONS = {
    "orcas": 0.5,
    "msmarco-passage-dev": 0.3,
    "crumb-legal-qa": 0.2,
}


class SliceShares(BaseModel):
    """How `total_rows` splits across the four d32 slices — owns the
    sum-to-1 invariant."""

    model_config = ConfigDict(frozen=True)

    span_target_share: float = Field(
        default=0.48,
        description=(
            "Slice A — weakest-first over the span-type evidence floors "
            "(d32: 60% span-evidence × 80% type-targeted)."
        ),
    )
    no_preference_share: float = Field(
        default=0.12,
        description=(
            "Slice B — draws FIRST (d33c), one uniform sample over span rows (d32: 60% × 20%)."
        ),
    )
    stat_strata_share: float = Field(
        default=0.20,
        description="Slice C — zero-span rows over the raw scalar bands (d33b).",
    )
    dark_forest_share: float = Field(
        default=0.20,
        description="Slice D — feature-blind uniform draws from the champions.",
    )

    @model_validator(mode="after")
    def validate_shares_sum_to_one(self) -> "SliceShares":
        total = (
            self.span_target_share
            + self.no_preference_share
            + self.stat_strata_share
            + self.dark_forest_share
        )
        if not isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"Shares must sum to 1.0, got {total}")
        return self


class FloorRules(BaseModel):
    """d33a floor mechanics — how evidence is counted and capped."""

    model_config = ConfigDict(frozen=True)

    span_floor_weight: float = Field(
        default=1_000.0,
        description=(
            "d33a: evidence amount per span-type floor, sized BELOW the "
            "slice budget so ambiguity-discount inflation is absorbed by "
            "slack. (Stat-band floors derive their amount instead: "
            "stat rows / band count.)"
        ),
    )
    discounts: Mapping[AmbiguityTier, float] = Field(
        default=_DEFAULT_DISCOUNTS,
        description=(
            "d33a: evidence a row earns a floor, by the most rigid "
            "qualifying bank — guesses are worth less per row."
        ),
    )
    cap_frac: float = Field(
        default=0.5,
        description=(
            "d29 anti-monoculture cap: no dataset supplies more than this "
            "fraction of any floor (waived for sole suppliers); doubles as "
            "the dark-forest per-champion ceiling (d33d)."
        ),
    )


class Recipe(BaseModel):
    """d32 macro-split + d33 fill values — one field per SPEC decision.
    Provisional values (discounts, floor weight) are flagged in TODOS —
    revisit after the first fill's order sheet."""

    model_config = ConfigDict(frozen=True)

    seed: int = 0
    total_rows: int = 50_000
    """The d32 target size; every slice is a share of this."""
    shares: SliceShares = Field(default_factory=SliceShares)
    """The four slice shares — see `SliceShares` for the d32 mapping."""
    floor_rules: FloorRules = Field(default_factory=FloorRules)
    """d33a floor mechanics — see `FloorRules`."""
    champions: Mapping[str, float] = _CHAMPIONS
    """d33d: dark-forest sources and their shares of that slice."""
    min_natural_share: float = Field(
        default=0.85,
        description=(
            "d42i/d43 review: minimum share of the composition that is "
            "natural provenance. Enforced by the mini-fill as an admission "
            "ceiling — natural_rows*(1-m)/m augmented rows at most (8,824 "
            "over the 50K base; the full current order sheet ~7.4K fits)."
        ),
    )

    @property
    def span_target_rows(self) -> int:
        return round(self.total_rows * self.shares.span_target_share)

    @property
    def no_preference_rows(self) -> int:
        return round(self.total_rows * self.shares.no_preference_share)

    @property
    def stat_strata_rows(self) -> int:
        return round(self.total_rows * self.shares.stat_strata_share)

    @property
    def dark_forest_rows(self) -> int:
        """The remainder slice: absorbs share-rounding drift so the four
        slices always sum to exactly `total_rows`."""
        return self.total_rows - (
            self.span_target_rows
            + self.no_preference_rows
            + self.stat_strata_rows
        )

    @model_validator(mode="after")
    def _coherent(self) -> "Recipe":
        if not isclose(
            sum(self.champions.values()), 1.0, rel_tol=0.0, abs_tol=1e-9,
        ):
            raise ValueError("champion shares must sum to 1.0")
        cap = self.floor_rules.cap_frac
        over = [
            name for name, share in self.champions.items()
            if share > cap + 1e-9
        ]
        if over:
            raise ValueError(f"champions over the ≤{cap} cap: {over}")
        return self

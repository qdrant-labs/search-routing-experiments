"""The four d32 slices, run in d33c order B → A → C → D by the composer.
Every slice returns rows tagged (slice, floors, label_lane); A and C also
return their order-sheet lines and ledger for the summary."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from composition.fill import (
    DEFERRED,
    QRELS,
    FloorLedger,
    Shortfall,
    WeakestFirstFill,
)
from composition.floors import (
    FloorSpec,
    StatFloorDeriver,
)
from composition.recipe import Recipe

ID_COLUMNS = ["dataset", "query_id", "checkable"]


class SliceResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    frame: pd.DataFrame
    shortfalls: list[Shortfall] = Field(default_factory=list)
    ledger: FloorLedger | None = None


def _tag(
    rows: pd.DataFrame,
    name: str,
    floors: list[FloorSpec] | None,
    lanes: list[str] | None = None,
) -> pd.DataFrame:
    frame = rows[ID_COLUMNS].copy()
    frame["slice"] = name
    frame["label_lane"] = (
        lanes if lanes is not None
        else np.where(frame["checkable"], QRELS, DEFERRED)
    )
    frame["floors"] = _memberships(rows, floors)
    return frame


def _memberships(
    rows: pd.DataFrame, floors: list[FloorSpec] | None,
) -> pd.Series:
    """Floor keys each row belongs to — evidence attribution, not ledger
    bookkeeping (a row lists a floor even where the cap stopped the
    credit). Slice D stays deliberately unattributed (feature-blind)."""
    if floors is None:
        return pd.Series([[]] * len(rows), index=rows.index)
    member = pd.DataFrame(
        {floor.key: floor.credit.reindex(rows.index) > 0 for floor in floors},
    )
    keys = np.array(member.columns)
    return pd.Series(
        [list(keys[row]) for row in member.to_numpy()], index=rows.index,
    )


class NoPreferenceSlice:
    """B (d33c: draws FIRST) — one seeded uniform sample over the whole
    span pool; no lane preference, no type targeting, no mass bias. The
    lane flag is recorded post-hoc from `checkable`."""

    name = "B"

    def __init__(self, recipe: Recipe) -> None:
        self._recipe = recipe

    def run(
        self,
        pool: pd.DataFrame,
        floors: list[FloorSpec],
        rng: np.random.Generator,
    ) -> SliceResult:
        take = min(self._recipe.no_preference_rows, len(pool))
        rows = pool.sample(n=take, random_state=rng).sort_index()
        return SliceResult(frame=_tag(rows, self.name, floors))


class SpanTargetSlice:
    """A — weakest-first over the span-type evidence floors (d33a)."""

    name = "A"

    def __init__(self, recipe: Recipe) -> None:
        self._recipe = recipe

    def run(
        self,
        pool: pd.DataFrame,
        floors: list[FloorSpec],
        rng: np.random.Generator,
    ) -> SliceResult:
        engine = WeakestFirstFill(pool, floors, self._recipe, rng)
        result = engine.run(min(self._recipe.span_target_rows, len(pool)))
        rows = pool.iloc[result.picked]
        return SliceResult(
            frame=_tag(rows, self.name, floors, result.lanes),
            shortfalls=result.shortfalls,
            ledger=result.ledger,
        )


class StatStrataSlice:
    """C — zero-span rows through the same engine over the raw scalar
    band floors (d33b)."""

    name = "C"

    def __init__(self, recipe: Recipe) -> None:
        self._recipe = recipe

    def run(
        self, pool: pd.DataFrame, rng: np.random.Generator,
    ) -> SliceResult:
        floors = StatFloorDeriver(self._recipe).derive(pool)
        engine = WeakestFirstFill(pool, floors, self._recipe, rng)
        result = engine.run(min(self._recipe.stat_strata_rows, len(pool)))
        rows = pool.iloc[result.picked]
        return SliceResult(
            frame=_tag(rows, self.name, floors, result.lanes),
            shortfalls=result.shortfalls,
            ledger=result.ledger,
        )


class DarkForestSlice:
    """D — feature-blind uniform draws from the champions at their d33d
    shares; a champion short on remaining rows spills proportionally to
    the others, never past the ≤50% cap."""

    name = "D"

    def __init__(self, recipe: Recipe) -> None:
        self._recipe = recipe

    def run(
        self, remaining: pd.DataFrame, rng: np.random.Generator,
    ) -> SliceResult:
        total = self._recipe.dark_forest_rows
        cap = int(self._recipe.floor_rules.cap_frac * total)
        pools = {
            name: remaining[remaining["dataset"] == name]
            for name in self._recipe.champions
        }
        targets = self._targets(pools, total, cap)
        rows = pd.concat([
            pools[name].sample(n=take, random_state=rng)
            for name, take in targets.items() if take
        ]).sort_index()
        return SliceResult(frame=_tag(rows, self.name, floors=None))

    def _targets(
        self, pools: dict[str, pd.DataFrame], total: int, cap: int,
    ) -> dict[str, int]:
        shares = self._recipe.champions
        targets = {
            name: min(round(share * total), cap, len(pools[name]))
            for name, share in shares.items()
        }
        # spill any deficit to champions with headroom, cap re-checked
        while (deficit := total - sum(targets.values())) > 0:
            headroom = {
                name: min(cap, len(pools[name])) - targets[name]
                for name in targets
            }
            open_names = [name for name, room in headroom.items() if room > 0]
            if not open_names:
                break
            for name in open_names:
                extra = min(
                    headroom[name],
                    max(1, deficit // len(open_names)),
                    deficit,
                )
                targets[name] += extra
                deficit -= extra
                if deficit == 0:
                    break
        return targets

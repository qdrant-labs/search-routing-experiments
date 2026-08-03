"""d30b weakest-first fill over evidence floors (d33a), two label lanes
(d33d): qrels rows first, deferred rows top up.

Marginal gain for a pick = the row's total credit toward OPEN, UNCAPPED
floors — no min-clamp against the deficit. The clamp only matters within
one row's credit of a floor closing (<=1.0 against 1,000-weight floors);
dropping it makes gains event-driven (recomputed only when a floor closes
or a cap binds) instead of per-pick, which is what keeps 24K picks over a
30K-row pool in seconds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from composition.floors import FloorSpec
from composition.recipe import Recipe

QRELS = "qrels"
DEFERRED = "deferred"


class Shortfall(BaseModel):
    """One order-sheet line (d33a): the floor's unmet evidence and why.
    `exhausted` = supply ran out; `capped` = supply remains but every
    supplier hit the d29 cap; `budget` = the slice budget ran out."""

    model_config = ConfigDict(frozen=True)

    key: str
    amount: float
    credit: float
    reason: str

    @property
    def missing(self) -> float:
        return self.amount - self.credit


class FloorLedger:
    """Weight-currency accounting: per floor, per (floor, dataset), per
    label lane. Caps are checked against the STATIC floor amount (d29 —
    a running-share cap can deadlock when one dataset remains)."""

    def __init__(
        self, floors: list[FloorSpec], datasets: list[str], cap_frac: float,
    ) -> None:
        self.keys = [floor.key for floor in floors]
        self.amounts = np.array([floor.amount for floor in floors])
        self._cap = cap_frac * self.amounts
        self._waived = np.array([floor.cap_waived for floor in floors])
        self.datasets = datasets
        self.credit = np.zeros(len(floors))
        self.by_dataset = np.zeros((len(floors), len(datasets)))
        self.by_lane = {
            QRELS: np.zeros(len(floors)),
            DEFERRED: np.zeros(len(floors)),
        }

    def ratios(self) -> np.ndarray:
        return self.credit / self.amounts

    def uncapped(self) -> np.ndarray:
        """(floors x datasets) bool — where crediting is still allowed."""
        return self._waived[:, None] | (self.by_dataset < self._cap[:, None])

    def add(self, credits: np.ndarray, dataset: int, lane: str) -> bool:
        """Credit one selected row to every uncapped floor it belongs to.
        Returns True when the pick closed a floor or bound a cap — the
        engine's signal to refresh its gain vector."""
        eligible = (credits > 0) & self.uncapped()[:, dataset]
        open_before = self.credit < self.amounts
        column_before = self.by_dataset[:, dataset] < self._cap
        gained = credits * eligible
        self.credit += gained
        self.by_dataset[:, dataset] += gained
        self.by_lane[lane] += gained
        return bool(
            ((self.credit < self.amounts) != open_before).any()
            | ((self.by_dataset[:, dataset] < self._cap) != column_before).any()
        )


class FillResult(BaseModel):
    """Picked positions (into the pool given to the engine), their label
    lanes, the final ledger, and the order-sheet lines."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    picked: list[int]
    lanes: list[str]
    ledger: FloorLedger
    shortfalls: list[Shortfall]


class WeakestFirstFill:
    """d30b: each pick feeds the floor with the lowest fill/floor ratio,
    choosing the member row with the highest marginal gain. Deterministic:
    the pool arrives sorted by (dataset, query_id), ties resolve to the
    first maximum, and the only randomness is the seeded top-up draw."""

    def __init__(
        self,
        pool: pd.DataFrame,
        floors: list[FloorSpec],
        recipe: Recipe,
        rng: np.random.Generator,
    ) -> None:
        self._pool = pool
        self._floors = floors
        self._rng = rng
        self._credit = np.column_stack([
            floor.credit.reindex(pool.index).to_numpy(dtype=float)
            for floor in floors
        ])
        self._members = self._credit > 0
        codes, names = pd.factorize(pool["dataset"], sort=True)
        self._dataset_codes = codes
        self._checkable = pool["checkable"].to_numpy(dtype=bool)
        self._ledger = FloorLedger(
            floors, list(names), recipe.floor_rules.cap_frac,
        )

    def run(self, budget: int, *, top_up: bool = True) -> FillResult:
        """`top_up=False` is the start-from-base mode (d42i/d43a): budget
        is a ceiling, not a target — picks happen only while they feed an
        open floor, never as a blind budget-spending draw."""
        selected = np.zeros(len(self._pool), dtype=bool)
        picked: list[int] = []
        lanes: list[str] = []
        for lane, lane_mask in self._lanes():
            budget = self._fill_lane(
                lane, lane_mask, selected, budget, picked, lanes,
            )
        if top_up:
            for lane, lane_mask in self._lanes():
                budget = self._top_up(
                    lane, lane_mask, selected, budget, picked, lanes,
                )
        return FillResult(
            picked=picked,
            lanes=lanes,
            ledger=self._ledger,
            shortfalls=self._shortfalls(selected),
        )

    def _lanes(self) -> tuple[tuple[str, np.ndarray], ...]:
        return ((QRELS, self._checkable), (DEFERRED, ~self._checkable))

    def _fill_lane(
        self,
        lane: str,
        lane_mask: np.ndarray,
        selected: np.ndarray,
        budget: int,
        picked: list[int],
        lanes: list[str],
    ) -> int:
        gain = self._gain_vector()
        exhausted = np.zeros(len(self._floors), dtype=bool)
        while budget > 0:
            ratios = self._ledger.ratios()
            floors_open = np.flatnonzero((ratios < 1.0) & ~exhausted)
            if floors_open.size == 0:
                break
            weakest = floors_open[np.argmin(ratios[floors_open])]
            candidates = (
                self._members[:, weakest]
                & lane_mask
                & ~selected
                & self._ledger.uncapped()[weakest][self._dataset_codes]
            )
            if not candidates.any():
                exhausted[weakest] = True
                continue
            row = int(np.argmax(np.where(candidates, gain, -np.inf)))
            selected[row] = True
            picked.append(row)
            lanes.append(lane)
            budget -= 1
            refresh = self._ledger.add(
                self._credit[row], self._dataset_codes[row], lane,
            )
            if refresh:
                gain = self._gain_vector()
        return budget

    def _gain_vector(self) -> np.ndarray:
        floors_open = self._ledger.ratios() < 1.0
        allowed = self._ledger.uncapped()[:, self._dataset_codes].T
        return (self._credit * floors_open[None, :] * allowed).sum(axis=1)

    def _top_up(
        self,
        lane: str,
        lane_mask: np.ndarray,
        selected: np.ndarray,
        budget: int,
        picked: list[int],
        lanes: list[str],
    ) -> int:
        if budget <= 0:
            return budget
        available = np.flatnonzero(lane_mask & ~selected)
        take = min(budget, available.size)
        if take == 0:
            return budget
        chosen = np.sort(self._rng.choice(available, size=take, replace=False))
        selected[chosen] = True
        picked.extend(int(row) for row in chosen)
        lanes.extend([lane] * take)
        # top-up rows still credit their floors — the ledger must keep
        # matching a recount over the final selection
        for row in chosen:
            self._ledger.add(
                self._credit[row], self._dataset_codes[row], lane,
            )
        return budget - take

    def _shortfalls(self, selected: np.ndarray) -> list[Shortfall]:
        out = []
        ratios = self._ledger.ratios()
        uncapped = self._ledger.uncapped()
        for index in np.flatnonzero(ratios < 1.0):
            remaining = self._members[:, index] & ~selected
            if not remaining.any():
                reason = "exhausted"
            elif not (
                remaining & uncapped[index][self._dataset_codes]
            ).any():
                reason = "capped"
            else:
                reason = "budget"
            out.append(Shortfall(
                key=self._ledger.keys[index],
                amount=float(self._ledger.amounts[index]),
                credit=float(self._ledger.credit[index]),
                reason=reason,
            ))
        return out

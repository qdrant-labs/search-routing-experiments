"""Floor accounting in weight currency (d33a) over the two label lanes
(d33d): qrels rows first, deferred rows top up."""

from __future__ import annotations

import numpy as np

from composition.floors import FloorSpec

QRELS = "qrels"
DEFERRED = "deferred"


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

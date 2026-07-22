"""SPEC d31 equal-weight percentile scale over the d29 feature-table
catalog — the canonical scale for comparing datasets on a scalar (see
CONTEXT.md). The raw catalog pool is ~90% msmarco+orcas rows, so any
unweighted pooled statistic silently becomes "the msmarco+orcas profile";
here every dataset contributes equal total weight (one 69-query pool query
weighs ~1450 msmarco queries), and a dataset's value on an axis is the
weighted percentile of its median query."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from composition.catalog_axes import stat_column


class EqualWeightScale:
    """Fitted once on the catalog for a fixed axis tuple; fingerprint
    charts ask it for raw per-dataset medians (cell text, axis keys) and
    the percentile matrix (color / polygon reach)."""

    def __init__(self, catalog: pd.DataFrame, axes: Sequence[str]) -> None:
        self.axes = tuple(axes)
        self._columns = {axis: stat_column(catalog, axis) for axis in self.axes}
        sizes = catalog["dataset"].value_counts()
        weights = 1.0 / catalog["dataset"].map(sizes).to_numpy(dtype=float)
        self._reference: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for axis, column in self._columns.items():
            values = catalog[column].to_numpy(dtype=float)
            order = np.argsort(values)
            cumulative = np.cumsum(weights[order])
            self._reference[axis] = (values[order], cumulative / cumulative[-1])
        self._medians = catalog.groupby("dataset")[
            list(self._columns.values())
        ].median()

    def dataset_medians(self, labels: Sequence[str]) -> np.ndarray:
        """Raw median per dataset per axis — what cell text and axis keys
        show; the percentile matrix is this passed through the reference."""
        unknown = [
            label for label in labels
            if _dataset_key(label) not in self._medians.index
        ]
        if unknown:
            raise ValueError(
                f"not in the fitted catalog: {unknown}; "
                f"catalog datasets: {sorted(self._medians.index)}"
            )
        return np.array(
            [
                [
                    self._medians.loc[_dataset_key(label), self._columns[axis]]
                    for axis in self.axes
                ]
                for label in labels
            ],
            dtype=float,
        )

    def percentile_matrix(self, labels: Sequence[str]) -> np.ndarray:
        """Each dataset's median passed through the equal-weight reference
        distribution — already in [0, 1], no further normalization."""
        medians = self.dataset_medians(labels)
        out = np.zeros_like(medians)
        for j, axis in enumerate(self.axes):
            values, cdf = self._reference[axis]
            last_leq = np.searchsorted(values, medians[:, j], side="right") - 1
            out[:, j] = np.where(
                last_leq >= 0, cdf[np.clip(last_leq, 0, None)], 0.0
            )
        return out


def _dataset_key(label: str) -> str:
    """Chart labels may carry a query-count suffix ("orcas (10405342)");
    the catalog keys don't."""
    return label.rsplit(" (", 1)[0] if label.endswith(")") else label

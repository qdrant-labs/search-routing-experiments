"""SPEC d28 small-multiples spider — one mini per dataset, all polygons
normalized against the same per-axis min-max across the FULL set so a
mini's reach on an axis reads as "how this dataset sits inside the
catalog". Use when N > 4 (overlay spider becomes unreadable)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from query_taxonomy.reporting import CorpusReport

from ._base import DOMAIN_PALETTE, ROUTER_SIGNAL_AXES
from ._matrix import build_matrix, per_column_normalize, stat_value
from .equal_weight import EqualWeightScale


class FingerprintSpiderGrid:
    """Small-multiples spider: `math.ceil(sqrt(N))` cols × as many rows
    as needed. Without a scale, minis normalize against the per-axis
    min-max of dataset means (one 252-word outlier flattens everyone
    else); with an `EqualWeightScale`, polygon reach is the equal-weight
    percentile of the dataset's median query (SPEC d31)."""

    def __init__(
        self,
        reports: dict[str, CorpusReport],
        axes: Sequence[str] = ROUTER_SIGNAL_AXES,
        scale: EqualWeightScale | None = None,
    ) -> None:
        if len(reports) < 2:
            raise ValueError(
                f"spider grid needs ≥2 datasets; got {len(reports)}"
            )
        self.reports = reports
        self.axes = tuple(axes)
        self.labels = list(reports)
        if scale is None:
            self.raw = build_matrix(reports, self.axes, stat_value)
            self.normalized = per_column_normalize(self.raw)
            self.scale_note = "per-axis min-max across the set"
        else:
            if scale.axes != self.axes:
                raise ValueError(
                    f"scale fitted for {scale.axes}, chart wants {self.axes}"
                )
            self.raw = scale.dataset_medians(self.labels)
            self.normalized = scale.percentile_matrix(self.labels)
            self.scale_note = (
                "equal-weight percentile scale, medians in axis key (d31)"
            )
        n = len(self.axes)
        self.angles = [i * 2 * math.pi / n for i in range(n)] + [0.0]
        self.cols = math.ceil(math.sqrt(len(self.labels)))
        self.rows = math.ceil(len(self.labels) / self.cols)

    def render(self) -> Figure:
        fig, axes = plt.subplots(
            self.rows, self.cols,
            figsize=(2.5 * self.cols, 2.5 * self.rows + 1.0),
            subplot_kw={"projection": "polar"},
        )
        flat = np.array(axes).flatten() if self.rows * self.cols > 1 else [axes]
        for i, label in enumerate(self.labels):
            color = DOMAIN_PALETTE[i % len(DOMAIN_PALETTE)]
            self._draw_mini(flat[i], self.normalized[i], label, color)
        for extra in flat[len(self.labels):]:
            extra.set_axis_off()
        fig.suptitle(
            f"Dataset fingerprints — {len(self.labels)} datasets "
            f"({self.scale_note})\n{self._axis_key()}",
            y=1.02, fontsize=10,
        )
        fig.tight_layout()
        return fig

    def _draw_mini(
        self, polar: plt.Axes, values: np.ndarray, label: str, color: str,
    ) -> None:
        closed = list(values) + [values[0]]
        polar.plot(self.angles, closed, color=color, linewidth=1.4)
        polar.fill(self.angles, closed, color=color, alpha=0.28)
        polar.set_theta_offset(math.pi / 2)
        polar.set_theta_direction(-1)
        polar.set_xticks(self.angles[:-1])
        polar.set_xticklabels(
            [_abbreviate(name) for name in self.axes], fontsize=7,
        )
        polar.set_ylim(0, 1)
        polar.set_yticks([0.5, 1.0])
        polar.set_yticklabels([""] * 2)
        polar.grid(True, alpha=0.3)
        polar.set_title(label, fontsize=9, pad=6)

    def _axis_key(self) -> str:
        """One line explaining the abbreviations + the shared ranges."""
        parts = []
        for j, name in enumerate(self.axes):
            lo, hi = self.raw[:, j].min(), self.raw[:, j].max()
            fmt = "{:.1f}" if hi >= 10 else "{:.2f}"
            parts.append(
                f"{_abbreviate(name)}={name} [{fmt.format(lo)}, {fmt.format(hi)}]"
            )
        return "  ·  ".join(parts)


def _abbreviate(name: str) -> str:
    """Short axis code: initials of underscore-split words."""
    return "".join(part[0].upper() for part in name.split("_") if part)

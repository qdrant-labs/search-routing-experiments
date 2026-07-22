"""Coverage small-multiples: the same (x band × span bucket) map drawn
once per stock x-axis. Length is the obvious primitive signal; the other
panels ask the same "what does the catalog cover, and who supplies it?"
question through identifier evidence, syntactic depth and NL shape — one
expectation-setting view per candidate signal, no algorithm yet."""

from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from catalog_axes import DEFAULT_AXES, CoverageAxis

from ._base import blank
from .coverage_map import CoverageMap


class CoverageMapGrid:
    """One `CoverageMap` per x-axis, `ceil(sqrt(N))` columns (same layout
    convention as the spider grid)."""

    def __init__(
        self,
        catalog: pd.DataFrame,
        x_axes: Sequence[CoverageAxis] = DEFAULT_AXES,
    ) -> None:
        if not x_axes:
            raise ValueError("CoverageMapGrid needs at least one x-axis")
        self.maps = [CoverageMap(catalog, x_axis) for x_axis in x_axes]
        self.cols = math.ceil(math.sqrt(len(self.maps)))
        self.rows = math.ceil(len(self.maps) / self.cols)

    def render(self) -> Figure:
        fig, panels = plt.subplots(
            self.rows, self.cols,
            figsize=(7.5 * self.cols, 3.8 * self.rows),
            squeeze=False,
        )
        flat = panels.flatten()
        for map_, panel in zip(self.maps, flat):
            map_.render(panel)
            panel.set_title(f"x = {map_.x_axis.title}", fontsize=10)
        for extra in flat[len(self.maps):]:
            extra.set_axis_off()
        fig.suptitle(
            "Coverage — catalog rows per cell (log color), "
            "equal-weight dominant source in cell (SPEC d31)",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig


def coverage_map_grid(
    catalog: pd.DataFrame,
    x_axes: Sequence[CoverageAxis] = DEFAULT_AXES,
) -> Figure:
    """Functional shim — blank chart on an empty catalog."""
    if catalog.empty:
        return blank(None, "no catalog rows to map")[0]
    return CoverageMapGrid(catalog, x_axes).render()

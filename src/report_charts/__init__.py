"""Matplotlib helpers for `CorpusReport` (SPEC d27, d28).

Package boundary — the query_taxonomy package stays stdlib-only per
d27's arch-validator verdict. Each chart module exposes either a class
(fingerprint views hold reusable computed state) or a pure render
function (donut / bar / signals) plus a functional shim for callers
that prefer one-shot invocation.
"""

from ._base import DOMAIN_PALETTE, ROUTER_SIGNAL_AXES
from composition.catalog_axes import (
    DEFAULT_AXES,
    CoverageAxis,
    SpanCountAxis,
    StatAxis,
)

from .coverage_map import CoverageMap, coverage_map
from .coverage_map_grid import CoverageMapGrid, coverage_map_grid
from .domain_donut import feature_donut
from .equal_weight import EqualWeightScale
from .fingerprint_heatmap import FingerprintHeatmap, fingerprint_heatmap
from .fingerprint_spider import FingerprintSpider, fingerprint_spider
from .fingerprint_spider_grid import FingerprintSpiderGrid
from .share_bar import domain_share_bar
from .signal_bars import signal_bars
from .target_composition import target_composition

__all__ = [
    "DEFAULT_AXES",
    "DOMAIN_PALETTE",
    "CoverageAxis",
    "CoverageMap",
    "CoverageMapGrid",
    "EqualWeightScale",
    "FingerprintHeatmap",
    "FingerprintSpider",
    "FingerprintSpiderGrid",
    "ROUTER_SIGNAL_AXES",
    "SpanCountAxis",
    "StatAxis",
    "coverage_map",
    "coverage_map_grid",
    "domain_share_bar",
    "feature_donut",
    "fingerprint_heatmap",
    "fingerprint_spider",
    "signal_bars",
    "target_composition",
]

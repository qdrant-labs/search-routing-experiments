"""Matplotlib helpers for `CorpusReport` (SPEC d27, d28).

Package boundary — the query_taxonomy package stays stdlib-only per
d27's arch-validator verdict. Each chart module exposes either a class
(fingerprint views hold reusable computed state) or a pure render
function (donut / bar / signals) plus a functional shim for callers
that prefer one-shot invocation.
"""

from ._base import DOMAIN_PALETTE, ROUTER_SIGNAL_AXES
from .domain_donut import feature_donut
from .fingerprint_heatmap import FingerprintHeatmap, fingerprint_heatmap
from .fingerprint_spider import FingerprintSpider, fingerprint_spider
from .share_bar import domain_share_bar
from .signal_bars import signal_bars

__all__ = [
    "DOMAIN_PALETTE",
    "FingerprintHeatmap",
    "FingerprintSpider",
    "ROUTER_SIGNAL_AXES",
    "domain_share_bar",
    "feature_donut",
    "fingerprint_heatmap",
    "fingerprint_spider",
    "signal_bars",
]

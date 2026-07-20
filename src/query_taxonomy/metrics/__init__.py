from query_taxonomy.metrics.core import MetricBank
from query_taxonomy.metrics.general import LengthBank, StopwordRatioBank

METRIC_BANKS: tuple[type[MetricBank], ...] = (
    LengthBank,
    StopwordRatioBank,
)

__all__ = [
    "METRIC_BANKS",
    "LengthBank",
    "MetricBank",
    "StopwordRatioBank",
]

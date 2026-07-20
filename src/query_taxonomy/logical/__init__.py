from query_taxonomy.logical.core import LogicalBank
from query_taxonomy.logical.general import OperatorSyntaxBank, TemporalRelativeBank

LOGICAL_BANKS: tuple[type[LogicalBank], ...] = (
    OperatorSyntaxBank,
    TemporalRelativeBank,
)

__all__ = [
    "LOGICAL_BANKS",
    "LogicalBank",
    "OperatorSyntaxBank",
    "TemporalRelativeBank",
]

from query_taxonomy.markers.core import MarkerBank, phrase_alternation
from query_taxonomy.markers.general import (
    AcronymBank,
    ComparativeBank,
    GreetingBank,
    InterjectionBank,
    NegationBank,
    PolitenessBank,
)
from query_taxonomy.taxonomy import SentenceMarker

MARKER_BANKS: tuple[type[MarkerBank], ...] = (
    NegationBank,
    GreetingBank,
    PolitenessBank,
    InterjectionBank,
    ComparativeBank,
    AcronymBank,
)

__all__ = [
    "MARKER_BANKS",
    "AcronymBank",
    "ComparativeBank",
    "GreetingBank",
    "InterjectionBank",
    "MarkerBank",
    "NegationBank",
    "PolitenessBank",
    "SentenceMarker",
    "phrase_alternation",
]

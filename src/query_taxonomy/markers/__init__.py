from query_taxonomy.markers.core import MarkerBank, phrase_alternation
from query_taxonomy.taxonomy import SentenceMarker
from query_taxonomy.markers.general import NegationBank

MARKER_BANKS: tuple[type[MarkerBank], ...] = (NegationBank,)

__all__ = [
    "MARKER_BANKS",
    "MarkerBank",
    "NegationBank",
    "SentenceMarker",
    "phrase_alternation",
]

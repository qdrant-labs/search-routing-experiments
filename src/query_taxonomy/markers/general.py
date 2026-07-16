from typing import override

from edify import RegexBuilder

from query_taxonomy.core import AmbiguityTier
from query_taxonomy.markers.core import MarkerBank, phrase_alternation
from query_taxonomy.taxonomy import SentenceMarker

NEGATION_PHRASES: tuple[str, ...] = (
    "not",
    "no",
    "never",
    "without",
    "except",
    "excluding",
    "excluded",
    "neither",
    "nor",
    "cannot",
    "don't",
    "doesn't",
    "isn't",
    "won't",
    "can't",
    "versus",
    "vs",
)


class NegationBank(MarkerBank):
    """Negation / exclusion cues — the constraint sparse bag-of-words cannot
    represent. "no" is a deliberate high-recall inclusion despite "no. 5"
    style false positives."""

    @property
    @override
    def name(self) -> SentenceMarker:
        return SentenceMarker.NEGATION

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return phrase_alternation(builder, NEGATION_PHRASES)

from abc import ABC, abstractmethod
from collections.abc import Sequence

from edify import RegexBuilder

from query_taxonomy.core import AmbiguityTier, FeatureGroup, RegexBank
from query_taxonomy.taxonomy import SentenceMarker


class MarkerBank(RegexBank, ABC):
    """
    One regex bank per SentenceMarker. Same claim mechanics as identifier
    banks but a separate group: marker spans never compete with identifier
    spans for text ranges.
    """

    @property
    def group(self) -> FeatureGroup:
        return FeatureGroup.SENTENCE_MARKERS

    @property
    @abstractmethod
    def ambiguity(self) -> AmbiguityTier:
        """Re-abstracted: every span bank must declare its tier explicitly."""

    @property
    @abstractmethod
    def name(self) -> SentenceMarker:
        """
        Name of the current bank
        """


def phrase_alternation(
    builder: RegexBuilder, phrases: Sequence[str]
) -> RegexBuilder:
    """Boundary-guarded, case-insensitive alternation over a closed phrase
    list. Longest phrase first: any_of is first-match, so longer forms must
    beat their own prefixes ("cannot" before "can't"-adjacent shapes)."""
    chain = builder.ignore_case().word_boundary().any_of()
    for phrase in sorted(phrases, key=len, reverse=True):
        chain = chain.string(phrase)
    return chain.end().word_boundary()

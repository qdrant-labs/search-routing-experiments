from abc import ABC, abstractmethod

from query_taxonomy.core import AmbiguityTier, FeatureGroup, RegexBank
from query_taxonomy.taxonomy import CorruptionKind


class CorruptionBank(RegexBank, ABC):
    """
    One regex bank per CorruptionKind. Corruption spans ignore word
    boundaries — damage happens mid-word.
    """

    @property
    def group(self) -> FeatureGroup:
        return FeatureGroup.CORRUPTION

    @property
    @abstractmethod
    def ambiguity(self) -> AmbiguityTier:
        """Re-abstracted: every span bank must declare its tier explicitly."""

    @property
    @abstractmethod
    def name(self) -> CorruptionKind:
        """
        Name of the current bank
        """

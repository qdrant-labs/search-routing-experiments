from abc import ABC, abstractmethod

from query_taxonomy.core import AmbiguityTier, FeatureGroup, RegexBank
from query_taxonomy.taxonomy import LogicalStructure


class LogicalBank(RegexBank, ABC):
    """
    One regex bank per LogicalStructure. Same claim mechanics as the other
    span groups; groups never compete for text ranges.
    """

    @property
    def group(self) -> FeatureGroup:
        return FeatureGroup.LOGICAL_STRUCTURES

    @property
    @abstractmethod
    def ambiguity(self) -> AmbiguityTier:
        """Re-abstracted: every span bank must declare its tier explicitly."""

    @property
    @abstractmethod
    def name(self) -> LogicalStructure:
        """
        Name of the current bank
        """

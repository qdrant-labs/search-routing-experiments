import re
from abc import ABC, abstractmethod

from edify import RegexBuilder

from query_taxonomy.core import AmbiguityTier, FeatureGroup, RegexBank
from query_taxonomy.taxonomy import Domain, StructuralIdentifier


def surface_form_pattern(value: str) -> re.Pattern[str]:
    """
    Exact-match pattern for one surface form, boundary-guarded so
    `42` does not match inside `426`
    """
    return re.compile(rf"(?<!\w){re.escape(value)}(?!\w)")


class IdentifierBank(RegexBank, ABC):
    """
    One regex bank per StructuralIdentifier — the Structured Identifiers
    group of the taxonomy.
    """

    @property
    def group(self) -> FeatureGroup:
        return FeatureGroup.STRUCTURED_IDENTIFIERS

    @property
    @abstractmethod
    def ambiguity(self) -> AmbiguityTier:
        """Re-abstracted: every span bank must declare its tier explicitly."""

    @property
    @abstractmethod
    def name(self) -> StructuralIdentifier:
        """
        Name of the current bank
        """

    @property
    @abstractmethod
    def domain(self) -> Domain:
        """Semantic domain of this bank — groups banks for validation and reporting."""


# Reusable char classes for subexpression() — edify's any_of fuses
# ranges/chars into a single [..] class.
HEX_DIGIT = RegexBuilder().any_of().range("0", "9").range("a", "f").range("A", "F").end()
UPPER_OR_UNDERSCORE = RegexBuilder().any_of().range("A", "Z").char("_").end()
ALNUM_OR_DOT = (
    RegexBuilder().any_of().range("0", "9").range("a", "z").range("A", "Z").char(".").end()
)
ALNUM = RegexBuilder().any_of().range("0", "9").range("a", "z").range("A", "Z").end()
LOWER_ALNUM = RegexBuilder().any_of().range("0", "9").range("a", "z").end()

# Strict IPv4 octet: 25[0-5] | 2[0-4]\d | 1\d\d | [1-9]?\d — rejects 999
IPV4_OCTET = (
    RegexBuilder()
    .any_of()
        .group().string("25").range("0", "5").end()
        .group().char("2").range("0", "4").digit().end()
        .group().char("1").exactly(2).digit().end()
        .group().optional().range("1", "9").digit().end()
    .end()
)

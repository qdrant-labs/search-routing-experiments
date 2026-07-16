import re
from abc import ABC, abstractmethod
from enum import IntEnum, StrEnum
from typing import Generic, NamedTuple, TypeVar

from edify import RegexBuilder

from query_taxonomy.taxonomy import FeatureGroup


class FeatureSpan(NamedTuple):
    text: str
    start: int
    end: int


class FeatureStat(NamedTuple):
    name: str
    value: float


class AmbiguityTier(IntEnum):
    """Pattern precision tier: lower value = higher claim priority during
    within-group span resolution."""

    RIGID = 0
    MODERATE = 1
    AMBIGUOUS = 2


EngineT = TypeVar("EngineT")
OutT = TypeVar("OutT", FeatureSpan, FeatureStat)


class GeneralBank(ABC, Generic[OutT, EngineT]):
    """
    One feature extractor. OutT is constrained to the two output shapes:
    spans or stats. The definition engine (RegexBuilder, tokenizer, model)
    is an implementation detail of `define` — typed at the method level by
    each concrete subclass, not exposed in the class signature. One bank per
    feature-enum member; each group's base class narrows the enum type of
    `name`.

    Claim resolution is within-group only: banks of different groups never
    compete for the same char ranges.
    """

    @property
    @abstractmethod
    def name(self) -> StrEnum:
        """Feature-enum member this bank detects (enum type set per group)."""

    @property
    @abstractmethod
    def group(self) -> FeatureGroup:
        """Parent group from query-taxonomy.csv (fixed per group base class)."""

    @property
    def ambiguity(self) -> AmbiguityTier:
        """Claim priority for span resolution. Stats are solid numbers, so
        stat banks keep this default; span-group bases re-abstract it to
        force an explicit tier per bank."""
        return AmbiguityTier.RIGID

    @abstractmethod
    def define(self, builder: EngineT) -> EngineT:
        """Chain the definition onto `builder` and return the result."""

    @abstractmethod
    def compute(self, text: str) -> list[OutT]:
        """Feature outputs for one text: spans (left to right,
        non-overlapping) or named stats."""


class RegexBank(GeneralBank[FeatureSpan, RegexBuilder], ABC):
    """
    REGEX-method span bank: `define` chains onto an edify RegexBuilder.
    Simple, dumb class that just does SRP - gives places where text has a
    certain pattern.
    """

    def __init__(self) -> None:
        super().__init__()
        # edify builders are immutable — every call returns a clone, so
        # `define` must return the chained builder, not mutate one in place.
        self._regex: re.Pattern[str] = self.define(RegexBuilder()).to_regex()

    @abstractmethod
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        """
        Chain the pattern onto `builder` and return the result.
        Quantifiers come before their element: `.exactly(4).digit()` -> \\d{4}
        """

    def compute(self, text: str) -> list[FeatureSpan]:
        return [
            FeatureSpan(m.group(0), *m.span())
            for m in self._regex.finditer(text)
        ]


class StatBank(GeneralBank[FeatureStat, EngineT], ABC):
    """
    Stat-emitting bank: named scalars over the whole text — never enters a
    claim registry; a histogram is many stats from one bank (POS profile).
    The engine stays open per bank family: a tokenizer pattern for
    length/ratio metrics, a pinned tagger for the POS profile.
    """

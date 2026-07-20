from typing import override

from edify import RegexBuilder

from query_taxonomy.core import AmbiguityTier
from query_taxonomy.logical.core import LogicalBank
from query_taxonomy.taxonomy import LogicalStructure


class OperatorSyntaxBank(LogicalBank):
    """Explicit boolean operators. Case-sensitive on purpose: lowercase
    and/or/not are ordinary function words, only the uppercase forms signal
    operator intent."""

    @property
    @override
    def name(self) -> LogicalStructure:
        return LogicalStructure.OPERATOR_SYNTAX

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of().string("AND").string("NOT").string("OR").end()
            .word_boundary()
        )


TEMPORAL_PHRASES: tuple[str, ...] = (
    "today",
    "tonight",
    "yesterday",
    "tomorrow",
    "now",
    "currently",
    "recently",
    "latest",
    "right now",
    "last week",
    "last month",
    "last year",
    "next week",
    "next month",
    "next year",
    "this week",
    "this month",
    "this year",
)


class TemporalRelativeBank(LogicalBank):
    """Relative temporal expressions — the slice absolute identifier banks
    (DATETIME, BUSINESS_TEMPORAL) cannot express. Covers a closed vocabulary
    plus "N <unit>(s) ago". Layered feature: a GLiNER2 backstop at the same
    feature name lands with the Gliner2Bank wrapper (SPEC decision 16)."""

    @property
    @override
    def name(self) -> LogicalStructure:
        return LogicalStructure.TEMPORAL

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        chain = (
            builder.ignore_case().word_boundary().any_of()
            # the consuming "N <unit>(s) ago" branch goes first
            .group()
                .one_or_more().digit()
                .one_or_more().whitespace_char()
                .any_of()
                    .string("minute").string("second").string("month")
                    .string("hour").string("week").string("year").string("day")
                .end()
                .optional().char("s")
                .one_or_more().whitespace_char()
                .string("ago")
            .end()
        )
        for phrase in sorted(TEMPORAL_PHRASES, key=len, reverse=True):
            chain = chain.string(phrase)
        return chain.end().word_boundary()

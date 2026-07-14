from typing import override

from edify import RegexBuilder

from query_taxonomy.banks.core import (
    AmbiguityTier,
    Domain,
    RegexBank,
    StructuralIdentifier,
)


class NumberBank(RegexBank):
    """Standalone integers, decimals and scientific notation."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.NUMBER

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @property
    @override
    def domain(self) -> Domain:
        return Domain.GENERAL

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .one_or_more().digit()
            .optional().group().char(".").one_or_more().digit().end()
            .optional().group()
                .any_of_chars("eE")
                .optional().any_of_chars("+-")
                .one_or_more().digit()
            .end()
            .word_boundary()
        )


class BusinessTemporalBank(RegexBank):
    """Q3 2026, FY25, CW28, Sprint 42. Bare quarters (Q3) without a year
    excluded as too generic."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.BUSINESS_TEMPORAL

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.GENERAL

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .char("Q")
                    .range("1", "4")
                    .optional().whitespace_char()
                    .exactly(4).digit()
                .end()
                .group()
                    .string("FY")
                    .optional().whitespace_char()
                    .between(2, 4).digit()
                .end()
                .group()
                    .string("CW")
                    .optional().whitespace_char()
                    .between(1, 2).digit()
                .end()
                .group()
                    .string("Sprint")
                    .optional().whitespace_char()
                    .between(1, 3).digit()
                .end()
            .end()
            .word_boundary()
        )


class DateTimeBank(RegexBank):
    """ISO 8601 dates with optional time/offset, or 10-digit unix epochs (2017-2033)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.DATETIME

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.GENERAL

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .exactly(4).digit().char("-")
                    .exactly(2).digit().char("-")
                    .exactly(2).digit()
                    .optional().group()
                        .any_of_chars("T ")
                        .exactly(2).digit().char(":").exactly(2).digit()
                        .optional().group().char(":").exactly(2).digit().end()
                        .optional().group()
                            .any_of()
                                .char("Z")
                                .group()
                                    .any_of_chars("+-")
                                    .exactly(2).digit()
                                    .optional().char(":")
                                    .exactly(2).digit()
                                .end()
                            .end()
                        .end()
                    .end()
                .end()
                .group().char("1").range("5", "9").exactly(8).digit().end()
            .end()
            .word_boundary()
        )

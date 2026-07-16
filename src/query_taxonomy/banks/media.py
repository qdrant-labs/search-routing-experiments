from typing import override

from edify import RegexBuilder

from query_taxonomy.banks.core import IdentifierBank
from query_taxonomy.core import AmbiguityTier
from query_taxonomy.taxonomy import Domain, StructuralIdentifier

_FEN_CHAR = (
    RegexBuilder().any_of().any_of_chars("rnbqkpRNBQKP").range("1", "8").end()
)


class ISOCodeBank(IdentifierBank):
    """BCP-47 locale forms (en-US). Bare currency/country codes (EUR, DE) are
    ceded to StockTickerBank — bare 2-3 caps are format-identical. Known FP:
    prose like 'ex-US'."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.ISO_CODE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @property
    @override
    def domain(self) -> Domain:
        return Domain.MEDIA

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .between(2, 3).range("a", "z")
            .char("-")
            .exactly(2).range("A", "Z")
            .word_boundary()
        )


class SocialHandleBank(IdentifierBank):
    """@handles and #hashtags; lookbehind rejects emails. Ordered after
    PackageCoordinateBank so @scope/name is not split."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.SOCIAL_HANDLE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.MEDIA

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .assert_not_behind().word().end()
            .any_of_chars("@#")
            .any_of().range("a", "z").range("A", "Z").char("_").end()
            .zero_or_more().word()
        )


class AcademicIdentifierBank(IdentifierBank):
    """DOIs (10.xxxx/...), arXiv ids (2104.08663), keyword-gated ISBNs.
    The arXiv shape can FP on 4.4-digit decimals — rare in practice."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.ACADEMIC_IDENTIFIER

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.MEDIA

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .string("10.")
                    .between(4, 9).digit()
                    .char("/")
                    .one_or_more().non_whitespace_char()
                .end()
                .group()
                    .exactly(4).digit()
                    .char(".")
                    .between(4, 5).digit()
                    .optional().group().char("v").one_or_more().digit().end()
                    .word_boundary()
                .end()
                .group()
                    .string("ISBN")
                    .optional().any_of_chars("- ")
                    .optional().group().string("13").end()
                    .optional().char(":")
                    .optional().whitespace_char()
                    .between(10, 17).any_of().range("0", "9").char("-").end()
                .end()
            .end()
        )


class ISSNBank(IdentifierBank):
    """ISSNs: 4 digits - 3 digits + check char. Year-guarded: the shape is
    identical to year ranges (1939-1945), so a second group that looks like
    a year (19xx/20xx) is rejected — costs a few real ISSNs, kills the
    dominant prose FP class."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.ISSN

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.MEDIA

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .exactly(4).digit()
            .char("-")
            .assert_not_ahead()
                .any_of().string("19").string("20").end()
                .exactly(2).digit()
                .word_boundary()
            .end()
            .exactly(3).digit()
            .any_of_chars("0123456789X")
            .word_boundary()
        )


class MusicWorkCodeBank(IdentifierBank):
    """ISRC (USRC17607839) and ISWC (T-034.524.680-1) work codes."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.MUSIC_WORK_CODE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.MEDIA

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .exactly(2).range("A", "Z")
                    .exactly(3).any_of().range("A", "Z").range("0", "9").end()
                    .exactly(7).digit()
                .end()
                .group()
                    .char("T")
                    .optional().char("-")
                    .exactly(3).digit()
                    .char(".")
                    .exactly(3).digit()
                    .char(".")
                    .exactly(3).digit()
                    .optional().char("-")
                    .digit()
                .end()
            .end()
            .word_boundary()
        )


class MediaDbIdBank(IdentifierBank):
    """IMDb ids: tt/nm/co/ev + 7-8 digits."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.MEDIA_DB_ID

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.MEDIA

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of().string("tt").string("nm").string("co").string("ev").end()
            .between(7, 8).digit()
            .word_boundary()
        )


class LibraryClassificationBank(IdentifierBank):
    """LCC call numbers (QA76.73.C15) and leading-zero Dewey (005.74).
    General Dewey (nnn.nn) excluded — it would shadow every 3-digit decimal."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.LIBRARY_CLASSIFICATION

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @property
    @override
    def domain(self) -> Domain:
        return Domain.MEDIA

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .between(1, 3).range("A", "Z")
                    .between(1, 4).digit()
                    .one_or_more().group()
                        .char(".")
                        .optional().range("A", "Z")
                        .between(1, 4).digit()
                    .end()
                .end()
                .group()
                    .char("0")
                    .exactly(2).digit()
                    .char(".")
                    .between(1, 6).digit()
                .end()
            .end()
            .word_boundary()
        )


class AstronomicalDesignationBank(IdentifierBank):
    """NGC/IC/HD/HR catalogs, Messier (M31), Kepler planets.
    Known FP: M-branch collides with motorways (M25) and rifles (M16)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.ASTRONOMICAL_DESIGNATION

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.MEDIA

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .any_of().string("NGC").string("IC").string("HD").string("HR").end()
                    .optional().whitespace_char()
                    .between(1, 6).digit()
                .end()
                .group()
                    .string("Kepler-")
                    .between(1, 4).digit()
                    .optional().range("a", "z")
                .end()
                .group()
                    .char("M")
                    .between(1, 3).digit()
                .end()
            .end()
            .word_boundary()
        )


class GameNotationBank(IdentifierBank):
    """Chess SAN piece moves (Nf3), castling (O-O), FEN position strings.
    Bare pawn moves (e4) excluded — two chars of pure cell reference."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.GAME_NOTATION

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.MEDIA

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                # FEN: 8 slash-separated ranks
                .group()
                    .exactly(7).group()
                        .one_or_more().subexpression(_FEN_CHAR)
                        .char("/")
                    .end()
                    .one_or_more().subexpression(_FEN_CHAR)
                .end()
                # castling
                .group()
                    .word_boundary()
                    .string("O-O")
                    .optional().group().string("-O").end()
                .end()
                # SAN piece move: Nf3, Qxe5+, Rad1
                .group()
                    .word_boundary()
                    .any_of_chars("KQRBN")
                    .optional().any_of_chars("abcdefgh")
                    .optional().any_of_chars("12345678")
                    .optional().char("x")
                    .range("a", "h")
                    .range("1", "8")
                    .optional().group().char("=").any_of_chars("QRBN").end()
                    .optional().any_of_chars("+#")
                .end()
            .end()
        )

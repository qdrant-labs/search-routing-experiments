from typing import override

from edify import RegexBuilder

from query_taxonomy.banks.core import (
    ALNUM_OR_DOT,
    HEX_DIGIT,
    IdentifierBank,
    UPPER_OR_UNDERSCORE,
)
from query_taxonomy.core import AmbiguityTier
from query_taxonomy.taxonomy import Domain, StructuralIdentifier


class CVEBank(IdentifierBank):
    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.CVE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return builder.string("CVE-").exactly(4).digit().char("-").at_least(4).digit()


class VersionStringBank(IdentifierBank):
    """Semver-ish versions with optional leading v and pre-release tag."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.VERSION_STRING

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .optional().char("v")
            .one_or_more().digit()
            .char(".")
            .one_or_more().digit()
            .optional().group().char(".").one_or_more().digit().end()
            .optional().group().char("-").one_or_more().subexpression(ALNUM_OR_DOT).end()
            .word_boundary()
        )


class FilePathBank(IdentifierBank):
    """Absolute unix paths (>=2 segments) and Windows drive paths."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.FILE_PATH

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                # unix: at least two /segment parts, so "and/or" and "/16" stay out
                .group()
                    .at_least(2).group()
                        .char("/")
                        .one_or_more().any_of()
                            .range("a", "z").range("A", "Z").range("0", "9")
                            .any_of_chars("._-")
                        .end()
                    .end()
                .end()
                # windows: C:\... up to whitespace
                .group()
                    .range("A", "Z")
                    .char(":")
                    .char("\\")
                    .one_or_more().non_whitespace_char()
                .end()
            .end()
        )


class UUIDBank(IdentifierBank):
    """Canonical 8-4-4-4-12 UUIDs, or bare 32-64 char hex digests (MD5/SHA-1/SHA-256)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.UUID

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .exactly(8).subexpression(HEX_DIGIT).char("-")
                    .exactly(4).subexpression(HEX_DIGIT).char("-")
                    .exactly(4).subexpression(HEX_DIGIT).char("-")
                    .exactly(4).subexpression(HEX_DIGIT).char("-")
                    .exactly(12).subexpression(HEX_DIGIT)
                .end()
                .group().between(32, 64).subexpression(HEX_DIGIT).end()
            .end()
            .word_boundary()
        )


class URIBank(IdentifierBank):
    """scheme://... URIs (http, s3, file, ...). Claims full URLs before HostPortBank."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.URI

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .range("a", "z")
            .zero_or_more().any_of()
                .range("a", "z").range("0", "9").char("+").char(".").char("-")
            .end()
            .string("://")
            .one_or_more().non_whitespace_char()
        )


class ErrorCodeBank(IdentifierBank):
    """Node ERR_* symbols and POSIX errno names. Known FP: the bare word ERROR."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.ERROR_CODE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group().string("ERR_").one_or_more().subexpression(UPPER_OR_UNDERSCORE).end()
                .group().char("E").between(3, 9).range("A", "Z").end()
            .end()
            .word_boundary()
        )


class EnvVarBank(IdentifierBank):
    """$UPPER_SNAKE env vars and --long-flags. The $ branch requires an
    underscore or >=6 chars so short cashtags ($AAPL) fall to STOCK_TICKER;
    the cost is that $HOME/$PATH-style short env vars are ceded too."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.ENV_VAR

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                # $WITH_UNDERSCORE first: alternation is first-match, and this
                # branch is the one that consumes the full underscored name
                .group()
                    .char("$")
                    .subexpression(UPPER_OR_UNDERSCORE)
                    .zero_or_more().any_of().range("A", "Z").range("0", "9").end()
                    .char("_")
                    .zero_or_more().any_of().range("A", "Z").range("0", "9").char("_").end()
                .end()
                # $LONGNAME: 6+ chars, no underscore needed
                .group()
                    .char("$")
                    .range("A", "Z")
                    .at_least(5).any_of().range("A", "Z").range("0", "9").end()
                .end()
                .group()
                    .string("--")
                    .range("a", "z")
                    .zero_or_more().any_of().range("a", "z").range("0", "9").char("-").end()
                .end()
            .end()
        )


class HexColorBank(IdentifierBank):
    """#RRGGBB and #RGB CSS colors."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.HEX_COLOR

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .char("#")
            .any_of()
                .group().exactly(6).subexpression(HEX_DIGIT).end()
                .group().exactly(3).subexpression(HEX_DIGIT).end()
            .end()
            .word_boundary()
        )


class CodeIdentifierBank(IdentifierBank):
    """camelCase, snake_case (lower/UPPER), dotted.paths (segments >= 2 chars,
    so `e.g` and `i.e` stay out). Known FP: prose camelCase brands (iPhone, eBay)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.CODE_IDENTIFIER

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                # dotted.path — each segment at least 2 chars
                .group()
                    .range("a", "z")
                    .one_or_more().any_of().range("a", "z").range("0", "9").char("_").end()
                    .one_or_more().group()
                        .char(".")
                        .range("a", "z")
                        .one_or_more().any_of().range("a", "z").range("0", "9").char("_").end()
                    .end()
                .end()
                # camelCase
                .group()
                    .one_or_more().range("a", "z")
                    .one_or_more().group()
                        .range("A", "Z")
                        .zero_or_more().any_of().range("a", "z").range("0", "9").end()
                    .end()
                .end()
                # snake_case (lower)
                .group()
                    .range("a", "z")
                    .zero_or_more().any_of().range("a", "z").range("0", "9").end()
                    .one_or_more().group()
                        .char("_")
                        .one_or_more().any_of().range("a", "z").range("0", "9").end()
                    .end()
                .end()
                # SNAKE_CASE (upper)
                .group()
                    .range("A", "Z")
                    .zero_or_more().any_of().range("A", "Z").range("0", "9").end()
                    .one_or_more().group()
                        .char("_")
                        .one_or_more().any_of().range("A", "Z").range("0", "9").end()
                    .end()
                .end()
            .end()
            .word_boundary()
        )


class PackageCoordinateBank(IdentifierBank):
    """npm scoped packages (@scope/name) and Maven-style group:artifact."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.PACKAGE_COORDINATE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                # @scope/name
                .group()
                    .char("@")
                    .one_or_more().any_of().range("a", "z").range("0", "9").char("-").end()
                    .char("/")
                    .one_or_more().any_of().range("a", "z").range("0", "9").any_of_chars("._-").end()
                .end()
                # group:artifact (letters after the colon, so host:port stays out)
                .group()
                    .word_boundary()
                    .range("a", "z")
                    .zero_or_more().any_of().range("a", "z").range("0", "9").any_of_chars("._-").end()
                    .char(":")
                    .range("a", "z")
                    .zero_or_more().any_of().range("a", "z").range("0", "9").any_of_chars("._-").end()
                    .word_boundary()
                .end()
            .end()
        )


_UNITS = (
    "GHz", "MHz", "kHz", "mAh", "fps", "dpi",
    "GB", "MB", "KB", "TB", "PB", "Hz",
    "mm", "cm", "km", "kg", "mg", "ml", "nm", "ms", "px",
    "kW", "mV", "mA", "W", "V",
)


class ValueWithUnitBank(IdentifierBank):
    """Quantity + whitelisted unit, optional space: 16GB, 3.5mm, 240 Hz."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.VALUE_WITH_UNIT

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        units = RegexBuilder().any_of()
        for unit in _UNITS:
            units = units.string(unit)
        units = units.end()

        return (
            builder
            .word_boundary()
            .one_or_more().digit()
            .optional().group().char(".").one_or_more().digit().end()
            .optional().whitespace_char()
            .subexpression(units)
            .word_boundary()
        )


class StandardsCitationBank(IdentifierBank):
    """RFC / ISO(-IEC) / IEEE citations: RFC 9110, ISO/IEC 27001, IEEE 802.11ax."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.STANDARDS_CITATION

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .string("RFC")
                    .optional().whitespace_char()
                    .between(3, 5).digit()
                .end()
                .group()
                    .string("ISO")
                    .optional().group().string("/IEC").end()
                    .optional().whitespace_char()
                    .between(4, 5).digit()
                    .optional().group().char("-").one_or_more().digit().end()
                .end()
                .group()
                    .string("IEEE")
                    .optional().whitespace_char()
                    .between(3, 4).digit()
                    .optional().group().char(".").one_or_more().digit().end()
                    .between(0, 2).range("a", "z")
                .end()
            .end()
            .word_boundary()
        )


class CryptoAddressBank(IdentifierBank):
    """bech32 (bc1...), ETH (0x + 40 hex), legacy base58 BTC. The base58 form
    can collide with random long alnum tokens — hence MODERATE, not RIGID."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.CRYPTO_ADDRESS

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .string("bc1")
                    .between(8, 87).any_of().range("a", "z").range("0", "9").end()
                .end()
                .group()
                    .string("0x")
                    .exactly(40).subexpression(HEX_DIGIT)
                .end()
                # legacy base58: [13] + 25-34 chars, no 0/O/I/l
                .group()
                    .any_of_chars("13")
                    .between(25, 34).any_of()
                        .range("1", "9").range("A", "H").range("J", "N")
                        .range("P", "Z").range("a", "k").range("m", "z")
                    .end()
                .end()
            .end()
            .word_boundary()
        )

from typing import override

from edify import RegexBuilder

from query_taxonomy.banks.core import (
    ALNUM,
    HEX_DIGIT,
    IPV4_OCTET,
    AmbiguityTier,
    Domain,
    RegexBank,
    StructuralIdentifier,
)

_HTTP_SEP_CHAR = RegexBuilder().any_of().any_of_chars("-/.").whitespace_char().end()

# IPv6 building blocks: h16 = 1-4 hex digits; guards replace \b (colons break it)
_H16 = RegexBuilder().between(1, 4).subexpression(HEX_DIGIT)
_IPV6_BOUNDARY_CHAR = (
    RegexBuilder().any_of().range("0", "9").range("a", "z").range("A", "Z").char(":").end()
)


class HTTPStatusCodeBank(RegexBank):
    """HTTP status codes in both directions: '200 OK', 'error 500', 'HTTP/1.1 201'."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.HTTP_STATUS_CODE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.NETWORK

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                # HTTP <sep> <code>: HTTP/1.1 201, HTTP - 200, HTTP 200
                .group()
                    .string("HTTP")
                    .any_of()
                        .group().one_or_more().non_whitespace_char().one_or_more().whitespace_char().end()
                        .group().zero_or_more().whitespace_char().zero_or_more().subexpression(_HTTP_SEP_CHAR).end()
                    .end()
                    .range("1", "5").exactly(2).digit()
                    .word_boundary()
                .end()
                # keyword before code: error 500, status 200
                .group()
                    .any_of().string("error").string("status").string("code").end()
                    .word_boundary()
                    .one_or_more().whitespace_char()
                    .range("1", "5").exactly(2).digit()
                    .word_boundary()
                .end()
                # code then reason word: 200 OK, 500 error
                .group()
                    .assert_not_behind().digit().end()
                    .range("1", "5").exactly(2).digit()
                    .one_or_more().whitespace_char()
                    .any_of()
                        .group().range("A", "Z").zero_or_more().any_of().range("A", "Z").range("a", "z").end().end()
                        .group().string("error").end()
                    .end()
                    .word_boundary()
                .end()
            .end()
        )


class CIDRBank(RegexBank):
    """IPv4 CIDR ranges; ordered before IPAddressBank so the /nn is claimed whole."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.CIDR

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.NETWORK

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .subexpression(IPV4_OCTET)
            .exactly(3).group().char(".").subexpression(IPV4_OCTET).end()
            .char("/")
            .between(1, 2).digit()
            .word_boundary()
        )


class IPAddressBank(RegexBank):
    """IPv4 with strict octets (rejects 999.x), plus pragmatic IPv6
    (full 8-group and ::-compressed forms — not the complete RFC 4291 grammar)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.IP_ADDRESS

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.NETWORK

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                # IPv4
                .group()
                    .word_boundary()
                    .subexpression(IPV4_OCTET)
                    .exactly(3).group().char(".").subexpression(IPV4_OCTET).end()
                    .word_boundary()
                .end()
                # IPv6 full: h16(:h16){7}
                .group()
                    .assert_not_behind().subexpression(_IPV6_BOUNDARY_CHAR).end()
                    .subexpression(_H16)
                    .exactly(7).group().char(":").subexpression(_H16).end()
                    .assert_not_ahead().subexpression(_IPV6_BOUNDARY_CHAR).end()
                .end()
                # IPv6 compressed: (h16:){1,6}:(h16(:h16){0,5})?
                .group()
                    .assert_not_behind().subexpression(_IPV6_BOUNDARY_CHAR).end()
                    .between(1, 6).group().subexpression(_H16).char(":").end()
                    .char(":")
                    .optional().group()
                        .subexpression(_H16)
                        .between(0, 5).group().char(":").subexpression(_H16).end()
                    .end()
                    .assert_not_ahead().subexpression(_IPV6_BOUNDARY_CHAR).end()
                .end()
                # IPv6 leading ::h16(:h16){0,6}
                .group()
                    .assert_not_behind().subexpression(_IPV6_BOUNDARY_CHAR).end()
                    .string("::")
                    .subexpression(_H16)
                    .between(0, 6).group().char(":").subexpression(_H16).end()
                    .assert_not_ahead().subexpression(_IPV6_BOUNDARY_CHAR).end()
                .end()
            .end()
        )


class MACAddressBank(RegexBank):
    """Colon- or dash-separated 6-group MAC addresses."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.MAC_ADDRESS

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.NETWORK

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .exactly(5).group()
                .exactly(2).subexpression(HEX_DIGIT)
                .any_of_chars(":-")
            .end()
            .exactly(2).subexpression(HEX_DIGIT)
            .word_boundary()
        )


class EmailBank(RegexBank):
    """RFC-5322-ish pragmatic emails: local@domain.tld."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.EMAIL

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.NETWORK

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .one_or_more().any_of()
                .range("a", "z").range("A", "Z").range("0", "9").any_of_chars("._%+-")
            .end()
            .char("@")
            .one_or_more().any_of()
                .range("a", "z").range("A", "Z").range("0", "9").any_of_chars(".-")
            .end()
            .char(".")
            .between(2, 24).any_of().range("a", "z").range("A", "Z").end()
            .word_boundary()
        )


class HostPortBank(RegexBank):
    """host:port, plus bare domains gated on a common-TLD whitelist.
    Until URIBank lands (batch 2), full URLs get partial bare-domain claims."""

    _TLDS = ("com", "cloud", "org", "net", "edu", "gov", "tech", "dev", "app", "io", "ai", "co")

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.HOST_PORT

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.NETWORK

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        tld_alternation = RegexBuilder().any_of()
        for tld in self._TLDS:
            tld_alternation = tld_alternation.string(tld)
        tld_alternation = tld_alternation.end()

        return (
            builder
            .any_of()
                # host:port — host must start with a letter, so 12:30 stays out
                .group()
                    .word_boundary()
                    .any_of().range("a", "z").range("A", "Z").end()
                    .zero_or_more().any_of()
                        .range("a", "z").range("A", "Z").range("0", "9").any_of_chars(".-")
                    .end()
                    .char(":")
                    .between(2, 5).digit()
                    .word_boundary()
                .end()
                # bare domain: label(.label)*.tld with whitelisted tld
                .group()
                    .word_boundary()
                    .one_or_more().group()
                        .one_or_more().any_of().range("a", "z").range("0", "9").char("-").end()
                        .char(".")
                    .end()
                    .subexpression(tld_alternation)
                    .word_boundary()
                .end()
            .end()
        )


class APIKeyBank(RegexBank):
    """Well-known secret prefixes: sk- (OpenAI), AKIA (AWS), ghp_ (GitHub), eyJ (JWT)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.API_KEY

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.NETWORK

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                .group()
                    .string("sk-")
                    .at_least(16).any_of()
                        .range("a", "z").range("A", "Z").range("0", "9").any_of_chars("_-")
                    .end()
                .end()
                .group()
                    .string("AKIA")
                    .exactly(16).any_of().range("A", "Z").range("0", "9").end()
                .end()
                .group()
                    .string("ghp_")
                    .at_least(20).subexpression(ALNUM)
                .end()
                .group()
                    .string("eyJ")
                    .at_least(10).any_of()
                        .range("a", "z").range("A", "Z").range("0", "9").any_of_chars("_.=+/-")
                    .end()
                .end()
            .end()
        )

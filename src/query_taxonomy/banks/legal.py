from typing import override

from edify import RegexBuilder

from query_taxonomy.banks.core import IdentifierBank
from query_taxonomy.core import AmbiguityTier
from query_taxonomy.taxonomy import Domain, StructuralIdentifier

_UPPER_ALNUM = RegexBuilder().any_of().range("A", "Z").range("0", "9").end()


class LegalCitationBank(IdentifierBank):
    """U.S.C. sections and F./S.Ct. reporters. Bare 'nn U.S. nn' reporter
    excluded — it FPs on prose like '1945 U.S. 200,000'."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.LEGAL_CITATION

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LEGAL

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                # 17 U.S.C. § 107
                .group()
                    .word_boundary()
                    .between(1, 3).digit()
                    .whitespace_char()
                    .string("U.S.C.")
                    .optional().group()
                        .optional().whitespace_char()
                        .char("§")
                        .optional().whitespace_char()
                        .one_or_more().digit()
                        .optional().range("a", "z")
                    .end()
                .end()
                # 539 F.3d 1024 / 100 S. Ct. 200
                .group()
                    .word_boundary()
                    .between(1, 4).digit()
                    .whitespace_char()
                    .any_of()
                        .group()
                            .string("F.")
                            .optional().any_of().string("2d").string("3d").string("4th").end()
                        .end()
                        .group().string("S.").optional().whitespace_char().string("Ct.").end()
                    .end()
                    .whitespace_char()
                    .between(1, 4).digit()
                    .word_boundary()
                .end()
            .end()
        )


class CourtDocketBank(IdentifierBank):
    """US federal docket numbers: 1:21-cv-02547."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.COURT_DOCKET

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LEGAL

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .digit()
            .char(":")
            .exactly(2).digit()
            .char("-")
            .any_of().string("cv").string("cr").string("mj").string("md").string("bk").end()
            .char("-")
            .between(4, 6).digit()
            .word_boundary()
        )


class NeutralCitationBank(IdentifierBank):
    """ECLI identifiers and UK neutral citations ([2019] UKSC 41)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.NEUTRAL_CITATION

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LEGAL

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                .group()
                    .string("ECLI:")
                    .exactly(2).range("A", "Z")
                    .char(":")
                    .one_or_more().subexpression(_UPPER_ALNUM)
                    .char(":")
                    .exactly(4).digit()
                    .char(":")
                    .one_or_more().digit()
                .end()
                .group()
                    .char("[")
                    .exactly(4).digit()
                    .char("]")
                    .whitespace_char()
                    .any_of()
                        .string("UKSC").string("UKHL").string("UKPC")
                        .string("EWCA").string("EWHC")
                    .end()
                    .whitespace_char()
                    .one_or_more().digit()
                .end()
            .end()
        )


class CELEXBank(IdentifierBank):
    """CELEX ids: sector digit + year + type letter + 4-digit number (32016R0679)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.CELEX

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LEGAL

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .digit()
            .any_of().string("19").string("20").end()
            .exactly(2).digit()
            .range("A", "Z")
            .exactly(4).digit()
            .word_boundary()
        )


class LegislativeCitationBank(IdentifierBank):
    """Pub. L. 117-58, 88 FR 12345, 100 Stat. 200."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.LEGISLATIVE_CITATION

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LEGAL

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                .group()
                    .string("Pub.")
                    .optional().whitespace_char()
                    .string("L.")
                    .optional().whitespace_char()
                    .between(2, 3).digit()
                    .char("-")
                    .between(1, 4).digit()
                .end()
                .group()
                    .word_boundary()
                    .between(1, 3).digit()
                    .whitespace_char()
                    .string("FR")
                    .whitespace_char()
                    .between(3, 6).digit()
                    .word_boundary()
                .end()
                .group()
                    .word_boundary()
                    .between(1, 3).digit()
                    .whitespace_char()
                    .string("Stat.")
                    .whitespace_char()
                    .between(1, 5).digit()
                .end()
            .end()
        )


class EURegulatoryCitationBank(IdentifierBank):
    """Regulation (EU) 2016/679 and Art. 6(1)(a) GDPR-style citations."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.EU_REGULATORY_CITATION

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LEGAL

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                .group()
                    .any_of().string("Regulation").string("Directive").string("Decision").end()
                    .whitespace_char()
                    .char("(")
                    .any_of().string("EU").string("EC").string("EEC").end()
                    .char(")")
                    .whitespace_char()
                    .optional().group().string("No").optional().whitespace_char().end()
                    .between(1, 4).digit()
                    .char("/")
                    .between(1, 4).digit()
                .end()
                .group()
                    .string("Art")
                    .optional().group().string("icle").end()
                    .optional().char(".")
                    .optional().whitespace_char()
                    .between(1, 3).digit()
                    .zero_or_more().group()
                        .char("(")
                        .one_or_more().any_of().range("0", "9").range("a", "z").end()
                        .char(")")
                    .end()
                    .whitespace_char()
                    .any_of()
                        .string("GDPR").string("TFEU").string("TEU").string("CRR")
                    .end()
                .end()
            .end()
        )


class PatentNumberBank(IdentifierBank):
    """Country prefix + digits + REQUIRED kind code (US10123456B2). The kind
    code disambiguates from VAT shapes — DE123456789 stays with TAX_ID."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.PATENT_NUMBER

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LEGAL

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .string("US").string("EP").string("WO").string("JP")
                .string("CN").string("KR").string("GB").string("DE")
            .end()
            .optional().whitespace_char()
            .between(6, 12).digit()
            .any_of_chars("ABU")
            .optional().digit()
            .word_boundary()
        )


class NationalIdBank(IdentifierBank):
    """SSN (123-45-6789) and UK NINO (QQ123456C) shapes."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.NATIONAL_ID

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LEGAL

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .exactly(3).digit()
                    .char("-")
                    .exactly(2).digit()
                    .char("-")
                    .exactly(4).digit()
                .end()
                .group()
                    .exactly(2).range("A", "Z")
                    .exactly(6).digit()
                    .range("A", "D")
                .end()
            .end()
            .word_boundary()
        )

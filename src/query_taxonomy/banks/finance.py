from typing import override

from edify import RegexBuilder

from query_taxonomy.banks.core import IdentifierBank
from query_taxonomy.core import AmbiguityTier
from query_taxonomy.taxonomy import Domain, StructuralIdentifier

_UPPER_ALNUM = RegexBuilder().any_of().range("A", "Z").range("0", "9").end()


class CurrencyAmountBank(IdentifierBank):
    """Symbol amounts (€499.99, $1,200) and code amounts (100 EUR)."""

    _CODES = ("EUR", "USD", "GBP", "JPY", "CHF", "INR", "CNY")

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.CURRENCY_AMOUNT

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.FINANCE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        codes = RegexBuilder().any_of()
        for code in self._CODES:
            codes = codes.string(code)
        codes = codes.end()

        return (
            builder
            .any_of()
                .group()
                    .any_of_chars("€$£¥")
                    .optional().whitespace_char()
                    .one_or_more().digit()
                    .zero_or_more().group().char(",").exactly(3).digit().end()
                    .optional().group().char(".").one_or_more().digit().end()
                    .word_boundary()
                .end()
                .group()
                    .word_boundary()
                    .one_or_more().digit()
                    .optional().group().char(".").one_or_more().digit().end()
                    .optional().whitespace_char()
                    .subexpression(codes)
                    .word_boundary()
                .end()
            .end()
        )


class IBANBank(IdentifierBank):
    """IBANs, spaced (DE89 3704 0044 ...) or compact (>=11 body chars, so
    short VAT IDs like DE123456789 stay with TAX_ID)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.IBAN

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.FINANCE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .exactly(2).range("A", "Z")
            .exactly(2).digit()
            .any_of()
                .group()
                    .between(2, 7).group().char(" ").exactly(4).subexpression(_UPPER_ALNUM).end()
                    .optional().group().char(" ").between(1, 4).subexpression(_UPPER_ALNUM).end()
                .end()
                .group().between(11, 30).subexpression(_UPPER_ALNUM).end()
            .end()
            .word_boundary()
        )


class BICBank(IdentifierBank):
    """8 or 11 char BIC/SWIFT. Known FP: any 8-letter all-caps word (PASSWORD)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.BIC

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @property
    @override
    def domain(self) -> Domain:
        return Domain.FINANCE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .exactly(4).range("A", "Z")
            .exactly(2).range("A", "Z")
            .exactly(2).subexpression(_UPPER_ALNUM)
            .optional().group().exactly(3).subexpression(_UPPER_ALNUM).end()
            .word_boundary()
        )


class SecuritiesIdBank(IdentifierBank):
    """ISIN (2 letters + 9 alnum + check digit) and CUSIP (9 alnum).
    All-numeric CUSIPs overlap ABA routing numbers — ordered before MARKET_CODE."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.SECURITIES_ID

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.FINANCE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .exactly(2).range("A", "Z")
                    .exactly(9).subexpression(_UPPER_ALNUM)
                    .digit()
                .end()
                .group()
                    .exactly(3).digit()
                    .exactly(5).subexpression(_UPPER_ALNUM)
                    .digit()
                .end()
            .end()
            .word_boundary()
        )


class StockTickerBank(IdentifierBank):
    """$CASHTAGS (1-5 caps) and bare 2-5 cap tickers (NVDA).
    The bare form matches any short all-caps word (USA, NATO) — worst FP
    generator in the bank; kept per full-coverage policy, measure per corpus."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.STOCK_TICKER

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @property
    @override
    def domain(self) -> Domain:
        return Domain.FINANCE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                .group()
                    .char("$")
                    .between(1, 5).range("A", "Z")
                    .word_boundary()
                .end()
                .group()
                    .word_boundary()
                    .between(2, 5).range("A", "Z")
                    .word_boundary()
                .end()
            .end()
        )


class LEIBank(IdentifierBank):
    """20-char Legal Entity Identifiers ending in two check digits."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.LEI

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.FINANCE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .exactly(18).subexpression(_UPPER_ALNUM)
            .exactly(2).digit()
            .word_boundary()
        )


class DerivativesSymbolBank(IdentifierBank):
    """OCC option symbols (AAPL240119C00150000) and futures codes (ESH25).
    The futures form (root + month letter + year) can FP on short caps words."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.DERIVATIVES_SYMBOL

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.FINANCE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .between(1, 5).range("A", "Z")
                    .exactly(6).digit()
                    .any_of_chars("CP")
                    .exactly(8).digit()
                .end()
                .group()
                    .between(1, 3).range("A", "Z")
                    .any_of_chars("FGHJKMNQUVXZ")
                    .between(1, 2).digit()
                .end()
            .end()
            .word_boundary()
        )


class MarketCodeBank(IdentifierBank):
    """X-prefixed MICs, 9-digit ABA routing numbers, BBG FIGIs.
    SEDOL deliberately excluded: 7 generic alnum chars carry no anchor —
    a regex for it is indistinguishable from noise."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.MARKET_CODE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @property
    @override
    def domain(self) -> Domain:
        return Domain.FINANCE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group().string("BBG").exactly(9).subexpression(_UPPER_ALNUM).end()
                .group().char("X").exactly(3).range("A", "Z").end()
                .group().exactly(9).digit().end()
            .end()
            .word_boundary()
        )


class IndustryCodeBank(IdentifierBank):
    """Keyword-gated NAICS / SIC / GICS codes."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.INDUSTRY_CODE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.FINANCE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of().string("NAICS").string("SIC").string("GICS").end()
            .optional().whitespace_char()
            .between(2, 8).digit()
            .word_boundary()
        )


class BusinessRegistrationBank(IdentifierBank):
    """Keyword-gated EIN / DUNS / CRN registration numbers."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.BUSINESS_REGISTRATION

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.FINANCE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .string("EIN")
                    .optional().whitespace_char()
                    .exactly(2).digit()
                    .optional().char("-")
                    .exactly(7).digit()
                .end()
                .group()
                    .string("DUNS")
                    .optional().whitespace_char()
                    .exactly(2).digit()
                    .optional().char("-")
                    .exactly(3).digit()
                    .optional().char("-")
                    .exactly(4).digit()
                .end()
                .group()
                    .string("CRN")
                    .optional().whitespace_char()
                    .exactly(8).digit()
                .end()
            .end()
            .word_boundary()
        )


class TaxIdBank(IdentifierBank):
    """EU-VAT-style CC+digits and bare EIN-shaped nn-nnnnnnn.
    Keyworded forms are claimed first by BUSINESS_REGISTRATION (RIGID)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.TAX_ID

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @property
    @override
    def domain(self) -> Domain:
        return Domain.FINANCE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .exactly(2).range("A", "Z")
                    .between(9, 12).digit()
                .end()
                .group()
                    .exactly(2).digit()
                    .char("-")
                    .exactly(7).digit()
                .end()
            .end()
            .word_boundary()
        )


class BettingOddsBank(IdentifierBank):
    """Fractional (5/2) and positive American (+150) odds. Decimal odds (3.50)
    deliberately excluded: format-identical to NUMBER. Negative American odds
    excluded: format-identical to negative integers in prose."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.BETTING_ODDS

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @property
    @override
    def domain(self) -> Domain:
        return Domain.FINANCE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                .group()
                    .word_boundary()
                    .between(1, 2).digit()
                    .char("/")
                    .between(1, 2).digit()
                    .word_boundary()
                .end()
                .group()
                    .assert_not_behind().word().end()
                    .char("+")
                    .exactly(3).digit()
                    .word_boundary()
                .end()
            .end()
        )


class TicketBank(IdentifierBank):
    """PREFIX-digits ticket/invoice refs (INV-2026-000123, TCK-84721, JIRA-style).
    CVE-... is shaped the same — CVEBank (RIGID) claims it first in resolution."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.TICKET

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.FINANCE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .between(2, 5).range("A", "Z")
            .between(1, 2).group()
                .char("-")
                .between(2, 10).digit()
            .end()
            .word_boundary()
        )

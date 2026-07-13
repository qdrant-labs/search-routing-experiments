from abc import ABC, abstractmethod
from enum import StrEnum
from typing import override

from edify import RegexBuilder
import re


class StructuralIdentifier(StrEnum):
    """
    Named Identifiers are set of classifications for
    the query corpora. Group: Structured Identifiers. Method: REGEX.
    """

    UUID = "uuid"
    """UUIDs, hashes, fingerprints — e.g. 550e8400-e29b-41d4-a716-446655440000, 3f1c2a9b7d..."""

    NUMBER = "number"
    """Plain integers, decimals, scientific notation — e.g. 42, 3.14159, 1e-9"""

    VERSION_STRING = "version_string"
    """Semver / PEP 440 strings — e.g. v1.2.3, 2.0.0-beta.1"""

    ERROR_CODE = "error_code"
    """POSIX / Node error symbols — e.g. EACCES, ERR_CONNECTION_REFUSED"""

    URI = "uri"
    """HTTP, S3, file and other URI schemes — e.g. https://qdrant.tech/docs, s3://bucket/path"""

    ISO_CODE = "iso_code"
    """BCP-47 locale, ISO 4217, ISO 3166 — e.g. en-US, EUR, DE"""

    SOCIAL_HANDLE = "social_handle"
    """Twitter/Instagram handles and hashtags — e.g. @qdrant_engine, #vectorsearch"""

    TICKET = "ticket"
    """Invoice/ticket reference numbers — e.g. INV-2026-000123, TCK-84721"""

    TAX_ID = "tax_id"
    """VAT/EIN/TIN registration numbers — e.g. DE123456789"""

    CRYPTO_ADDRESS = "crypto_address"
    """On-chain wallet addresses — e.g. bc1qxy2kgdyg..."""

    MEDICAL_CODE = "medical_code"
    """ICD codes, CAS numbers, gene/SNP IDs — e.g. J45.909, rs429358"""

    HTTP_STATUS_CODE = "http_status_code"
    """HTTP response codes mentioned in prose — e.g. error 503, HTTP 429"""

    LEGAL_CITATION = "legal_citation"
    """Court citations, statute sections — e.g. 17 U.S.C. § 107, 539 F.3d 1024"""

    ENV_VAR = "env_var"
    """Shell env vars and CLI flags — e.g. $QDRANT__SERVICE__PORT, --force"""

    CVE = "cve"
    """MITRE CVE and similar advisory IDs — e.g. CVE-2024-3094"""

    ACADEMIC_IDENTIFIER = "academic_identifier"
    """DOI, arXiv IDs, ISBN-13 — e.g. 10.1145/3539618, 2104.08663"""

    VALUE_WITH_UNIT = "value_with_unit"
    """Quantity + SI/colloquial unit — e.g. 16GB, 3.5mm, 240Hz"""

    CODE_IDENTIFIER = "code_identifier"
    """camelCase, snake_case, dotted.path tokens — e.g. getUserById, os.path.join"""

    CURRENCY_AMOUNT = "currency_amount"
    """Price with currency symbol/code — e.g. €499.99, $1,200"""

    PACKAGE_COORDINATE = "package_coordinate"
    """npm scoped packages, Maven GAV coordinates — e.g. @types/node, org.apache:kafka"""

    POSTAL_CODE = "postal_code"
    """ZIP / postcode strings — e.g. 10115, SW1A 1AA"""

    GEO_COORDINATE = "geo_coordinate"
    """Decimal degree lat/lon pairs — e.g. 52.5200, 13.4050"""

    BOOKING_REFERENCE = "booking_reference"
    """Airline PNR, record locators, flight numbers — e.g. LH1830, X4B9C2"""

    DATETIME = "datetime"
    """ISO 8601 dates, RFC 3339 timestamps, Unix epochs — e.g. 2026-07-08T10:00Z, 1751968800"""

    HEX_COLOR = "hex_color"
    """CSS/HTML hex colors — e.g. #FF5733"""

    BARCODE = "barcode"
    """EAN-13, UPC-A, GTIN, Amazon ASIN — e.g. B08N5WRWNW"""

    PHONE_NUMBER = "phone_number"
    """E.164 and local phone formats — e.g. +49 30 12345678"""

    SKU = "sku"
    """Manufacturer part/model strings — e.g. WH-1000XM5, RTX 4090"""

    SERIAL_NUMBER = "serial_number"
    """Device serial numbers, 15-digit IMEI — e.g. 356938035643809"""

    TRACKING_NUMBER = "tracking_number"
    """Carrier shipment tracking codes — e.g. 1Z999AA10123456784"""


class RegexBank(ABC):
    """One regex bank per StructuralIdentifier."""

    def __init__(self) -> None:
        super().__init__()
        # edify builders are immutable — every call returns a clone, so
        # `define` must return the chained builder, not mutate one in place.
        self._regex: re.Pattern[str] = self.define(RegexBuilder()).to_regex()

    @property
    @abstractmethod
    def name(self) -> StructuralIdentifier:
        """
        Name of the current bank
        """

    @abstractmethod
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        """
        Chain the identifier pattern onto `builder` and return the result.
        Quantifiers come before their element: `.exactly(4).digit()` -> \\d{4}
        """

    def matches(self, string: str) -> list[str]:
        """
        Matched surface forms, left to right, non-overlapping
        """
        return [m.group(0) for m in self._regex.finditer(string)]

    def spans(self, string: str) -> list[tuple[int, int]]:
        """
        (start, end) character spans of matches
        """
        return [m.span() for m in self._regex.finditer(string)]

    def number_for_line(self, string: str) -> int:
        return sum(1 for _ in self._regex.finditer(string))

    def coverage(self, string: str) -> float:
        """
        Fraction of characters covered by identifier matches, in [0, 1]
        """
        if not string:
            return 0.0
        return sum(end - start for start, end in self.spans(string)) / len(string)

    def df(self, docs: list[str]) -> int:
        """
        The document frequency for the current structural identifier:
        number of docs containing at least one match
        """
        return sum(1 for doc in docs if self._regex.search(doc))

    def value_df(self, value: str, docs: list[str]) -> int:
        """
        Document frequency of one exact surface form (e.g. a specific UUID),
        boundary-guarded so `42` does not count `426`
        """
        pattern = re.compile(rf"(?<!\w){re.escape(value)}(?!\w)")
        return sum(1 for doc in docs if pattern.search(doc))

    def generate_diversity(self, docs: list[str]) -> int:
        """
        Give how much a list of documents is diverse for the current structural
        identifier: the number of distinct surface forms across all docs
        """
        unique: set[str] = set()
        for doc in docs:
            unique.update(self.matches(doc))
        return len(unique)


# Reusable char classes for subexpression() — edify's any_of fuses
# ranges/chars into a single [..] class.
_HEX_DIGIT = RegexBuilder().any_of().range("0", "9").range("a", "f").range("A", "F").end()
_UPPER_OR_UNDERSCORE = RegexBuilder().any_of().range("A", "Z").char("_").end()
_ALNUM_OR_DOT = (
    RegexBuilder().any_of().range("0", "9").range("a", "z").range("A", "Z").char(".").end()
)


class CVEBank(RegexBank):
    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.CVE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return builder.string("CVE-").exactly(4).digit().char("-").at_least(4).digit()
    


class UUIDBank(RegexBank):
    """Canonical 8-4-4-4-12 UUIDs, or bare 32-64 char hex digests (MD5/SHA-1/SHA-256)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.UUID

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .exactly(8).subexpression(_HEX_DIGIT).char("-")
                    .exactly(4).subexpression(_HEX_DIGIT).char("-")
                    .exactly(4).subexpression(_HEX_DIGIT).char("-")
                    .exactly(4).subexpression(_HEX_DIGIT).char("-")
                    .exactly(12).subexpression(_HEX_DIGIT)
                .end()
                .group().between(32, 64).subexpression(_HEX_DIGIT).end()
            .end()
            .word_boundary()
        )


class NumberBank(RegexBank):
    """Standalone integers, decimals and scientific notation."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.NUMBER

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


class VersionStringBank(RegexBank):
    """Semver-ish versions with optional leading v and pre-release tag."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.VERSION_STRING

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
            .optional().group().char("-").one_or_more().subexpression(_ALNUM_OR_DOT).end()
            .word_boundary()
        )


class ErrorCodeBank(RegexBank):
    """Node ERR_* symbols and POSIX errno names. Known FP: the bare word ERROR."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.ERROR_CODE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group().string("ERR_").one_or_more().subexpression(_UPPER_OR_UNDERSCORE).end()
                .group().char("E").between(3, 9).range("A", "Z").end()
            .end()
            .word_boundary()
        )


class URIBank(RegexBank):
    """scheme://... URIs (http, s3, file, ...)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.URI

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


class EnvVarBank(RegexBank):
    """$UPPER_SNAKE env vars and --long-flags."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.ENV_VAR

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                .group()
                    .char("$")
                    .subexpression(_UPPER_OR_UNDERSCORE)
                    .zero_or_more().any_of().range("A", "Z").range("0", "9").char("_").end()
                .end()
                .group()
                    .string("--")
                    .range("a", "z")
                    .zero_or_more().any_of().range("a", "z").range("0", "9").char("-").end()
                .end()
            .end()
        )


class HTTPStatusCodeBank(RegexBank):
    """1xx-5xx codes in context: 'HTTP 429', 'error 503', 'status 200'."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.HTTP_STATUS_CODE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .ignore_case()
            .word_boundary()
            .any_of().string("HTTP").string("error").string("status").string("code").end()
            .one_or_more().whitespace_char()
            .range("1", "5")
            .exactly(2).digit()
            .word_boundary()
        )


class DateTimeBank(RegexBank):
    """ISO 8601 dates with optional time/offset, or 10-digit unix epochs (2017-2033)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.DATETIME

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


class HexColorBank(RegexBank):
    """#RRGGBB and #RGB CSS colors."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.HEX_COLOR

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .char("#")
            .any_of()
                .group().exactly(6).subexpression(_HEX_DIGIT).end()
                .group().exactly(3).subexpression(_HEX_DIGIT).end()
            .end()
            .word_boundary()
        )


class SocialHandleBank(RegexBank):
    """@handles and #hashtags; lookbehind rejects emails. Known FP: hex colors."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.SOCIAL_HANDLE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .assert_not_behind().word().end()
            .any_of_chars("@#")
            .any_of().range("a", "z").range("A", "Z").char("_").end()
            .zero_or_more().word()
        )


BANKS: tuple[type[RegexBank], ...] = (
    CVEBank,
    UUIDBank,
    NumberBank,
    VersionStringBank,
    ErrorCodeBank,
    URIBank,
    EnvVarBank,
    HTTPStatusCodeBank,
    DateTimeBank,
    HexColorBank,
    SocialHandleBank,
)
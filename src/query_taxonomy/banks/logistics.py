from typing import override

from edify import RegexBuilder

from query_taxonomy.banks.core import (
    AmbiguityTier,
    Domain,
    RegexBank,
    StructuralIdentifier,
)

_UPPER_ALNUM = RegexBuilder().any_of().range("A", "Z").range("0", "9").end()
# VIN charset excludes I, O, Q
_VIN_LETTER = (
    RegexBuilder().any_of().range("A", "H").range("J", "N").char("P").range("R", "Z").end()
)
_VIN_CHAR = (
    RegexBuilder()
    .any_of()
    .range("A", "H").range("J", "N").char("P").range("R", "Z").range("0", "9")
    .end()
)


class PhoneNumberBank(RegexBank):
    """International +CC numbers (needs >=2 digit groups after the country
    code, so betting odds like +150 stay out) and (555) 123-4567 US style."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.PHONE_NUMBER

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LOGISTICS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                .group()
                    .assert_not_behind().word().end()
                    .char("+")
                    .between(1, 3).digit()
                    .between(2, 4).group()
                        .optional().any_of_chars(" -")
                        .between(2, 8).digit()
                    .end()
                    .word_boundary()
                .end()
                .group()
                    .char("(")
                    .exactly(3).digit()
                    .char(")")
                    .optional().whitespace_char()
                    .exactly(3).digit()
                    .char("-")
                    .exactly(4).digit()
                    .word_boundary()
                .end()
            .end()
        )


class PostalCodeBank(RegexBank):
    """5-digit ZIPs (any 5-digit number — worst-case ambiguity) and UK postcodes."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.POSTAL_CODE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LOGISTICS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .between(1, 2).range("A", "Z")
                    .digit()
                    .optional().subexpression(_UPPER_ALNUM)
                    .whitespace_char()
                    .digit()
                    .exactly(2).range("A", "Z")
                .end()
                .group()
                    .exactly(5).digit()
                    .optional().group().char("-").exactly(4).digit().end()
                .end()
            .end()
            .word_boundary()
        )


class GeoCoordinateBank(RegexBank):
    """Decimal lat/lon pairs with >=3 decimals each — price pairs (1.99, 2.99)
    don't reach the precision bar."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.GEO_COORDINATE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LOGISTICS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .optional().char("-")
            .between(1, 3).digit()
            .char(".")
            .between(3, 8).digit()
            .char(",")
            .optional().whitespace_char()
            .optional().char("-")
            .between(1, 3).digit()
            .char(".")
            .between(3, 8).digit()
            .word_boundary()
        )


class AltGeocodingBank(RegexBank):
    """what3words (///a.b.c), Plus Codes (8FVC9G8F+6X), MGRS (33UUU9012)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.ALT_GEOCODING

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LOGISTICS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                .group()
                    .string("///")
                    .one_or_more().range("a", "z")
                    .char(".")
                    .one_or_more().range("a", "z")
                    .char(".")
                    .one_or_more().range("a", "z")
                .end()
                .group()
                    .word_boundary()
                    .between(4, 8).subexpression(_UPPER_ALNUM)
                    .char("+")
                    .between(2, 3).subexpression(_UPPER_ALNUM)
                .end()
                .group()
                    .word_boundary()
                    .between(1, 2).digit()
                    .range("C", "X")
                    .exactly(2).range("A", "Z")
                    .between(4, 10).digit()
                    .word_boundary()
                .end()
            .end()
        )


class BookingReferenceBank(RegexBank):
    """Flight numbers (LH1830) and 6-char PNRs with mixed letters+digits.
    The flight form matches any CAPS+digits token (WW2) — measure per corpus.
    Ordered before SKUBank so LH1830 reads as a flight, not a model number."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.BOOKING_REFERENCE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LOGISTICS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                # PNR: 6 upper-alnum with at least one digit and one letter
                .group()
                    .assert_ahead()
                        .zero_or_more().subexpression(_UPPER_ALNUM)
                        .digit()
                    .end()
                    .assert_ahead()
                        .zero_or_more().digit()
                        .range("A", "Z")
                    .end()
                    .exactly(6).subexpression(_UPPER_ALNUM)
                .end()
                # flight: LH1830
                .group()
                    .exactly(2).range("A", "Z")
                    .between(1, 4).digit()
                .end()
            .end()
            .word_boundary()
        )


class TrackingNumberBank(RegexBank):
    """UPS 1Z tracking codes. Digit-run carrier formats (FedEx/DHL) excluded —
    indistinguishable from serial numbers without a carrier keyword."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.TRACKING_NUMBER

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LOGISTICS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .string("1Z")
            .exactly(16).subexpression(_UPPER_ALNUM)
            .word_boundary()
        )


class BarcodeBank(RegexBank):
    """Amazon ASINs (B0 + 8 alnum) and 12-13 digit EAN/UPC runs (ambiguous)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.BARCODE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LOGISTICS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group().string("B0").exactly(8).subexpression(_UPPER_ALNUM).end()
                .group().between(12, 13).digit().end()
            .end()
            .word_boundary()
        )


class SKUBank(RegexBank):
    """CAPS prefix + 3-5 digits + optional alnum tail: WH-1000XM5, RTX 4090."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.SKU

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LOGISTICS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .between(2, 5).range("A", "Z")
            .optional().any_of_chars(" -")
            .between(3, 5).digit()
            .between(0, 4).subexpression(_UPPER_ALNUM)
            .word_boundary()
        )


class SerialNumberBank(RegexBank):
    """15-digit IMEIs and S/N-prefixed serials."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.SERIAL_NUMBER

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LOGISTICS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group().exactly(15).digit().end()
                .group()
                    .any_of().string("S/N").string("SN:").end()
                    .optional().char(":")
                    .optional().whitespace_char()
                    .between(5, 20).any_of()
                        .range("A", "Z").range("0", "9").char("-")
                    .end()
                .end()
            .end()
            .word_boundary()
        )


class VinContainerBank(RegexBank):
    """17-char VINs (needs a letter, I/O/Q excluded) and container numbers
    (4 letters + 7 digits)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.VIN_CONTAINER

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LOGISTICS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .assert_ahead()
                        .zero_or_more().digit()
                        .subexpression(_VIN_LETTER)
                    .end()
                    .exactly(17).subexpression(_VIN_CHAR)
                .end()
                .group()
                    .exactly(4).range("A", "Z")
                    .exactly(7).digit()
                .end()
            .end()
            .word_boundary()
        )


class LicensePlateBank(RegexBank):
    """German-style plates: B-QD 1234."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.LICENSE_PLATE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LOGISTICS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .between(1, 3).range("A", "Z")
            .char("-")
            .between(1, 2).range("A", "Z")
            .optional().whitespace_char()
            .between(1, 4).digit()
            .optional().any_of_chars("EH")
            .word_boundary()
        )


class AirportAirlineCodeBank(RegexBank):
    """Cue-word-gated IATA/ICAO codes (airport BER / BER airport / flight LH).
    Bare 2-4 cap codes are fully shadowed by StockTickerBank — gating is what
    keeps this bank alive at all."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.AIRPORT_AIRLINE_CODE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LOGISTICS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                .group()
                    .any_of().string("airport").string("Airport").end()
                    .whitespace_char()
                    .optional().char("(")
                    .between(3, 4).range("A", "Z")
                    .optional().char(")")
                    .word_boundary()
                .end()
                .group()
                    .word_boundary()
                    .between(3, 4).range("A", "Z")
                    .whitespace_char()
                    .any_of().string("airport").string("Airport").end()
                .end()
                .group()
                    .any_of().string("flight").string("Flight").string("airline").end()
                    .whitespace_char()
                    .between(2, 3).range("A", "Z")
                    .word_boundary()
                .end()
            .end()
        )


class AircraftVesselRegBank(RegexBank):
    """N-number tails (>=3 digits so N95 masks stay out), D- registrations,
    keyword-gated IMO / MMSI numbers."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.AIRCRAFT_VESSEL_REG

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LOGISTICS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .char("N")
                    .between(3, 5).digit()
                    .between(0, 2).range("A", "Z")
                .end()
                .group().string("D-").exactly(4).range("A", "Z").end()
                .group().string("IMO").optional().whitespace_char().exactly(7).digit().end()
                .group().string("MMSI").optional().whitespace_char().exactly(9).digit().end()
            .end()
            .word_boundary()
        )


class CustomsClassificationBank(RegexBank):
    """Keyword-gated HS / HTS tariff codes."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.CUSTOMS_CLASSIFICATION

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LOGISTICS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of().string("HTS").string("HS").end()
            .optional().whitespace_char()
            .between(4, 10).digit()
            .between(0, 3).group().char(".").exactly(2).digit().end()
            .word_boundary()
        )


class HazmatCodeBank(RegexBank):
    """UN numbers (UN1203), GHS H/P codes (H225 — known FP: H264, P450),
    keyword-gated NFPA."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.HAZMAT_CODE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LOGISTICS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group().string("UN").exactly(4).digit().end()
                .group().any_of_chars("HP").exactly(3).digit().end()
                .group().string("NFPA").optional().whitespace_char().between(1, 4).digit().end()
            .end()
            .word_boundary()
        )


class MaterialGradeBank(RegexBank):
    """AISI grades, aluminium tempers (6061-T6), titanium alloys (Ti-6Al-4V)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.MATERIAL_GRADE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LOGISTICS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .string("AISI")
                    .optional().whitespace_char()
                    .between(3, 4).digit()
                    .optional().range("A", "Z")
                .end()
                .group()
                    .exactly(4).digit()
                    .string("-T")
                    .between(1, 2).digit()
                .end()
                .group()
                    .string("Ti-")
                    .one_or_more().digit()
                    .range("A", "Z")
                    .optional().range("a", "z")
                    .zero_or_more().group()
                        .char("-")
                        .one_or_more().digit()
                        .range("A", "Z")
                        .optional().range("a", "z")
                    .end()
                .end()
            .end()
            .word_boundary()
        )


class NSNBank(RegexBank):
    """NATO Stock Numbers: 4-2-3-4 digit groups."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.NSN

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LOGISTICS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .exactly(4).digit().char("-")
            .exactly(2).digit().char("-")
            .exactly(3).digit().char("-")
            .exactly(4).digit()
            .word_boundary()
        )


class SimSubscriberIdBank(RegexBank):
    """ICCIDs: 89 + 17-18 digits. IMSI (15 digits) excluded — format-identical
    to IMEI, which SerialNumberBank owns."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.SIM_SUBSCRIBER_ID

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.LOGISTICS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .string("89")
            .between(17, 18).digit()
            .word_boundary()
        )

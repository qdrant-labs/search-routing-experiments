"""Lightweight core checks per bank: ~2 blatant positives / 2 blatant negatives.

Deliberately not exhaustive — these pin the core of each regex, not its edges.
"""

from enum import StrEnum

import pytest

from query_taxonomy import FEATURE_BANKS
from query_taxonomy.banks import BANKS, StructuralIdentifier
from query_taxonomy.features import FeatureExtractor
from query_taxonomy.taxonomy import FeatureGroup, SentenceMarker

# every registered bank of every group is subject to the case discipline
_BANKS = {
    cls().name: cls()
    for group_banks in FEATURE_BANKS.values()
    for cls in group_banks
}

# name -> (positives, negatives)
CASES: dict[StrEnum, tuple[list[str], list[str]]] = {
    SentenceMarker.NEGATION: (
        ["laptops without touchscreen", "NOT the Nokia one"],
        ["nothing knots canned", "annotated notation"],
    ),
    StructuralIdentifier.CVE: (
        ["CVE-2024-3094", "see CVE-2023-12345 advisory"],
        ["CVE-24-1", "cve"],
    ),
    StructuralIdentifier.DATETIME: (
        ["2026-07-08T10:00Z", "1751968800"],
        ["12-34-56", "1234567890123"],
    ),
    StructuralIdentifier.NUMBER: (
        ["42", "pi is 3.14159"],
        ["abc", "..."],
    ),
    StructuralIdentifier.VERSION_STRING: (
        ["v1.2.3", "upgrade to 2.0.0-beta.1"],
        ["v.", "version"],
    ),
    StructuralIdentifier.HTTP_STATUS_CODE: (
        ["error 503", "HTTP 429"],
        ["error 999", "HTTP 42"],
    ),
    StructuralIdentifier.CIDR: (
        ["192.168.0.0/16", "10.0.0.0/8"],
        ["192.168.0.0", "5/2"],
    ),
    StructuralIdentifier.IP_ADDRESS: (
        ["192.168.1.1", "2001:db8::1"],
        ["999.999.999.999", "1.2"],
    ),
    StructuralIdentifier.MAC_ADDRESS: (
        ["00:1B:44:11:3A:B7", "aa-bb-cc-dd-ee-ff"],
        ["00:1B:44", "GG:HH:II:JJ:KK:LL"],
    ),
    StructuralIdentifier.EMAIL: (
        ["jane.doe@example.com", "mail support@qdrant.tech now"],
        ["@qdrant_engine", "foo@bar"],
    ),
    StructuralIdentifier.HOST_PORT: (
        ["localhost:6333", "docs at qdrant.tech"],
        ["12:30", "config.yaml"],
    ),
    StructuralIdentifier.API_KEY: (
        ["AKIA1234567890ABCDEF", "ghp_abcdefghijklmnopqrstuvwxyz1234"],
        ["sk-", "AKIA123"],
    ),
    StructuralIdentifier.FILE_PATH: (
        ["/etc/qdrant/config.yaml", "C:\\Users\\andrei\\report.csv"],
        ["and/or", "/single"],
    ),
    StructuralIdentifier.UUID: (
        ["550e8400-e29b-41d4-a716-446655440000", "d41d8cd98f00b204e9800998ecf8427e"],
        ["hello-world", "deadbeef"],
    ),
    StructuralIdentifier.URI: (
        ["https://qdrant.tech/docs", "s3://bucket/path"],
        ["not a uri", "http:/oops"],
    ),
    StructuralIdentifier.ERROR_CODE: (
        ["EACCES", "got ERR_CONNECTION_REFUSED"],
        ["Error", "E42"],
    ),
    StructuralIdentifier.ENV_VAR: (
        ["$QDRANT__SERVICE__PORT", "run with --force"],
        ["$3", "-f"],
    ),
    StructuralIdentifier.HEX_COLOR: (
        ["#FF5733", "#fff"],
        ["#GGHHII", "#12"],
    ),
    StructuralIdentifier.CODE_IDENTIFIER: (
        ["getUserById", "os.path.join"],
        ["hello", "e.g"],
    ),
    StructuralIdentifier.PACKAGE_COORDINATE: (
        ["@types/node", "org.apache:kafka"],
        ["12:30", "plain words"],
    ),
    StructuralIdentifier.VALUE_WITH_UNIT: (
        ["16GB", "3.5mm"],
        ["GB", "16"],
    ),
    StructuralIdentifier.STANDARDS_CITATION: (
        ["RFC 9110", "ISO/IEC 27001"],
        ["RFC", "9110"],
    ),
    StructuralIdentifier.CRYPTO_ADDRESS: (
        [
            "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh",
            "0x52908400098527886E0F7030069857D2E4169EE7",
        ],
        ["bc1", "0x123"],
    ),
    StructuralIdentifier.CURRENCY_AMOUNT: (
        ["€499.99", "$1,200"],
        ["$AAPL", "499"],
    ),
    StructuralIdentifier.IBAN: (
        ["DE89 3704 0044 0532 0130 00", "DE89370400440532013000"],
        ["DE123456789", "DE89"],
    ),
    StructuralIdentifier.BIC: (
        ["DEUTDEFF", "MARKDEF1100"],
        ["DEUT", "deutdeff"],
    ),
    StructuralIdentifier.SECURITIES_ID: (
        ["US0378331005", "037833100"],
        ["US037", "12345"],
    ),
    StructuralIdentifier.STOCK_TICKER: (
        ["$AAPL", "NVDA"],
        ["nvda", "ABCDEFG"],
    ),
    StructuralIdentifier.LEI: (
        ["529900T8BM49AURSDO55", "5493001KJTIIGC8Y1R12"],
        ["SHORT123", "529900"],
    ),
    StructuralIdentifier.DERIVATIVES_SYMBOL: (
        ["AAPL240119C00150000", "ESH25"],
        ["AAPL", "240119"],
    ),
    StructuralIdentifier.MARKET_CODE: (
        ["XNYS", "021000021"],
        ["NYSE", "1234"],
    ),
    StructuralIdentifier.INDUSTRY_CODE: (
        ["NAICS 541511", "SIC 7372"],
        ["NAICS", "541511"],
    ),
    StructuralIdentifier.BUSINESS_REGISTRATION: (
        ["EIN 12-3456789", "DUNS 150483782"],
        ["EIN", "150483782"],
    ),
    StructuralIdentifier.TAX_ID: (
        ["DE123456789", "12-3456789"],
        ["DE12", "123"],
    ),
    StructuralIdentifier.BETTING_ODDS: (
        ["odds of 5/2", "+150"],
        ["3.50", "150"],
    ),
    StructuralIdentifier.TICKET: (
        ["INV-2026-000123", "TCK-84721"],
        ["INV-", "ABC"],
    ),
    StructuralIdentifier.LEGAL_CITATION: (
        ["17 U.S.C. § 107", "539 F.3d 1024"],
        ["U.S.C.", "539"],
    ),
    StructuralIdentifier.COURT_DOCKET: (
        ["1:21-cv-02547", "2:19-cr-00123"],
        ["1:21", "cv-02547"],
    ),
    StructuralIdentifier.NEUTRAL_CITATION: (
        ["ECLI:EU:C:2019:772", "[2019] UKSC 41"],
        ["ECLI:", "[2019]"],
    ),
    StructuralIdentifier.CELEX: (
        ["32016R0679", "31995L0046"],
        ["32016", "R0679"],
    ),
    StructuralIdentifier.LEGISLATIVE_CITATION: (
        ["Pub. L. 117-58", "88 FR 12345"],
        ["Pub. L.", "FR"],
    ),
    StructuralIdentifier.EU_REGULATORY_CITATION: (
        ["Regulation (EU) 2016/679", "Art. 6(1)(a) GDPR"],
        ["Regulation", "GDPR"],
    ),
    StructuralIdentifier.PATENT_NUMBER: (
        ["US10123456B2", "EP1234567A1"],
        ["DE123456789", "US12"],
    ),
    StructuralIdentifier.NATIONAL_ID: (
        ["123-45-6789", "QQ123456C"],
        ["123-456", "12-34-56"],
    ),
    StructuralIdentifier.MEDICAL_CODE: (
        ["J45.909", "rs429358"],
        ["J45", "rs12"],
    ),
    StructuralIdentifier.CLINICAL_CODING: (
        ["CPT 99213", "LOINC 4548-4"],
        ["CPT", "99213"],
    ),
    StructuralIdentifier.DRUG_ID: (
        ["NDC 0069-4200-83", "ATC A10BA02"],
        ["NDC", "A10BA02"],
    ),
    StructuralIdentifier.GENOMIC_ACCESSION: (
        ["ENSG00000139618", "NM_000546"],
        ["2024", "ENSG"],
    ),
    StructuralIdentifier.HGVS_VARIANT: (
        ["c.76A>T", "p.Lys76Asn"],
        ["c.76", "p.Lys"],
    ),
    StructuralIdentifier.CHEMICAL_ID: (
        ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "CHEMBL25"],
        ["BSYNRYMUTXBXSQ", "CID"],
    ),
    StructuralIdentifier.CLINICAL_TRIAL_ID: (
        ["NCT01234567", "0000-0002-1825-0097"],
        ["NCT123", "0000-0002"],
    ),
    StructuralIdentifier.HEALTHCARE_PROVIDER_ID: (
        ["NPI 1234567893", "DEA AB1234563"],
        ["NPI", "AB1234563"],
    ),
    StructuralIdentifier.PHONE_NUMBER: (
        ["+49 30 12345678", "(555) 123-4567"],
        ["+150", "12345"],
    ),
    StructuralIdentifier.POSTAL_CODE: (
        ["10115", "SW1A 1AA"],
        ["1011", "SW1A"],
    ),
    StructuralIdentifier.GEO_COORDINATE: (
        ["52.5200, 13.4050", "-33.8688, 151.2093"],
        ["1.99, 2.99", "52.52"],
    ),
    StructuralIdentifier.ALT_GEOCODING: (
        ["///filled.count.soap", "8FVC9G8F+6X"],
        ["///filled", "8FVC"],
    ),
    StructuralIdentifier.BOOKING_REFERENCE: (
        ["LH1830", "X4B9C2"],
        ["L1830", "ABCDEF"],
    ),
    StructuralIdentifier.TRACKING_NUMBER: (
        ["1Z999AA10123456784", "1ZA1B2C3D4E5F6G7H8"],
        ["1Z", "999AA10123456784"],
    ),
    StructuralIdentifier.BARCODE: (
        ["B08N5WRWNW", "4006381333931"],
        ["B08", "12345"],
    ),
    StructuralIdentifier.SKU: (
        ["WH-1000XM5", "RTX 4090"],
        ["MP3", "COVID-19"],
    ),
    StructuralIdentifier.SERIAL_NUMBER: (
        ["356938035643809", "S/N ABC123XY"],
        ["12345678901234", "SN"],
    ),
    StructuralIdentifier.VIN_CONTAINER: (
        ["WVWZZZ1JZXW000001", "MSKU1234567"],
        ["WVWZZZ", "12345678901234567"],
    ),
    StructuralIdentifier.LICENSE_PLATE: (
        ["B-QD 1234", "M-AB 123"],
        ["B-", "1234"],
    ),
    StructuralIdentifier.AIRPORT_AIRLINE_CODE: (
        ["airport BER", "flight LH"],
        ["BER", "airport"],
    ),
    StructuralIdentifier.AIRCRAFT_VESSEL_REG: (
        ["N12345", "D-AIMA"],
        ["N95", "D-AI"],
    ),
    StructuralIdentifier.CUSTOMS_CLASSIFICATION: (
        ["HS 090111", "HTS 0901.11.00"],
        ["HS", "090111"],
    ),
    StructuralIdentifier.HAZMAT_CODE: (
        ["UN1203", "H225"],
        ["UN 12", "H22"],
    ),
    StructuralIdentifier.MATERIAL_GRADE: (
        ["AISI 304", "6061-T6"],
        ["AISI", "6061"],
    ),
    StructuralIdentifier.NSN: (
        ["1234-56-789-0123", "5305-00-543-2100"],
        ["1234-56-789", "1234"],
    ),
    StructuralIdentifier.SIM_SUBSCRIBER_ID: (
        ["8901410321111851072", "89014103211118510720"],
        ["310150123456789", "8901"],
    ),
    StructuralIdentifier.ISO_CODE: (
        ["en-US", "pt-BR"],
        ["EUR", "en"],
    ),
    StructuralIdentifier.SOCIAL_HANDLE: (
        ["@qdrant_engine", "tag #vectorsearch"],
        ["user@example.com", "@ loose"],
    ),
    StructuralIdentifier.ACADEMIC_IDENTIFIER: (
        ["10.1145/3539618", "2104.08663"],
        ["3539618", "21.04"],
    ),
    StructuralIdentifier.ISSN: (
        ["2049-3630", "0028-0836"],
        ["1939-1945", "2049"],
    ),
    StructuralIdentifier.MUSIC_WORK_CODE: (
        ["USRC17607839", "T-034.524.680-1"],
        ["USRC", "034.524"],
    ),
    StructuralIdentifier.MEDIA_DB_ID: (
        ["tt0111161", "nm0000138"],
        ["tt01", "0111161"],
    ),
    StructuralIdentifier.LIBRARY_CLASSIFICATION: (
        ["005.74", "QA76.73.C15"],
        ["212.5", "QA"],
    ),
    StructuralIdentifier.ASTRONOMICAL_DESIGNATION: (
        ["NGC 224", "M31"],
        ["NGC", "224"],
    ),
    StructuralIdentifier.GAME_NOTATION: (
        ["Nf3", "O-O"],
        ["e4", "N"],
    ),
    StructuralIdentifier.BUSINESS_TEMPORAL: (
        ["Q3 2026", "FY25"],
        ["Q5 2026", "FY"],
    ),
}


def _params(index: int):
    return [
        pytest.param(name, text, id=f"{name.value}:{text[:30]}")
        for name, case in CASES.items()
        for text in case[index]
    ]


@pytest.mark.parametrize(("name", "text"), _params(0))
def test_positive(name: StructuralIdentifier, text: str) -> None:
    assert _BANKS[name].compute(text), f"{name} should match {text!r}"


@pytest.mark.parametrize(("name", "text"), _params(1))
def test_negative(name: StructuralIdentifier, text: str) -> None:
    assert not _BANKS[name].compute(text), f"{name} should not match {text!r}"


def test_every_bank_has_cases() -> None:
    assert set(CASES) == set(_BANKS), "add 2/2 cases for every new bank"


def test_module_placement_matches_domain() -> None:
    for cls in BANKS:
        bank = cls()
        assert cls.__module__.endswith(bank.domain.value), (
            f"{cls.__name__} has domain {bank.domain} but lives in {cls.__module__}"
        )


def test_registry_tier_priority() -> None:
    extractor = FeatureExtractor()
    by_type = extractor.resolve("scan 192.168.0.0/16 for v1.0.0 now").spans[FeatureGroup.STRUCTURED_IDENTIFIERS]

    assert [m.text for m in by_type[StructuralIdentifier.CIDR]] == ["192.168.0.0/16"]
    assert StructuralIdentifier.IP_ADDRESS not in by_type
    assert [m.text for m in by_type[StructuralIdentifier.VERSION_STRING]] == ["v1.0.0"]
    assert StructuralIdentifier.NUMBER not in by_type


def test_uri_claims_full_url_before_host_port() -> None:
    extractor = FeatureExtractor()
    by_type = extractor.resolve("see https://qdrant.tech/docs and qdrant.tech").spans[FeatureGroup.STRUCTURED_IDENTIFIERS]

    assert [m.text for m in by_type[StructuralIdentifier.URI]] == [
        "https://qdrant.tech/docs"
    ]
    assert [m.text for m in by_type[StructuralIdentifier.HOST_PORT]] == ["qdrant.tech"]


def test_finance_claim_order() -> None:
    extractor = FeatureExtractor()

    by_type = extractor.resolve("buy $AAPL, set $JAVA_HOME, fix CVE-2024-3094").spans[FeatureGroup.STRUCTURED_IDENTIFIERS]
    assert [m.text for m in by_type[StructuralIdentifier.STOCK_TICKER]] == ["$AAPL"]
    assert [m.text for m in by_type[StructuralIdentifier.ENV_VAR]] == ["$JAVA_HOME"]
    assert [m.text for m in by_type[StructuralIdentifier.CVE]] == ["CVE-2024-3094"]
    assert StructuralIdentifier.TICKET not in by_type

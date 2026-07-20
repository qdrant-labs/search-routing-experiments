"""The taxonomy itself — the code twin of query-taxonomy.csv: parent
groups and their member vocabularies. Extraction machinery lives in
core.py and the group packages (banks/, markers/, ...)."""

from enum import StrEnum


class FeatureGroup(StrEnum):
    """The 'Parent' rows of query-taxonomy.csv — every extractor belongs to
    exactly one group, orthogonal to its method (REGEX / ALGO / MODEL)."""

    STRUCTURED_IDENTIFIERS = "structured_identifiers"
    SENTENCE_MARKERS = "sentence_markers"
    LOGICAL_STRUCTURES = "logical_structures"
    STATISTICAL_METRICS = "statistical_metrics"
    CORRUPTION = "corruption"
    SEMANTICAL = "semantical"
    """Language identity and code-switching (SPEC decision 18); corpus-relative
    features (vocabulary mismatch, ambiguity, etc.) also land here eventually."""


class StructuralIdentifier(StrEnum):
    """
    Named Identifiers are set of classifications for
    the query corpora. Group: Structured Identifiers. Method: REGEX.
    """

    # --- general ---
    NUMBER = "number"
    """Plain integers, decimals, scientific notation — e.g. 42, 3.14159, 1e-9"""

    DATETIME = "datetime"
    """ISO 8601 dates, RFC 3339 timestamps, Unix epochs — e.g. 2026-07-08T10:00Z, 1751968800"""

    BUSINESS_TEMPORAL = "business_temporal"
    """Quarters, fiscal years, calendar weeks, sprints — e.g. Q3 2026, FY25, CW28, Sprint 42"""

    # --- network / infra ---
    IP_ADDRESS = "ip_address"
    """IPv4 / IPv6 addresses — e.g. 192.168.1.1, 2001:db8::1"""

    MAC_ADDRESS = "mac_address"
    """MAC addresses — e.g. 00:1B:44:11:3A:B7"""

    EMAIL = "email"
    """Email addresses — e.g. jane.doe@example.com, support@qdrant.tech"""

    HOST_PORT = "host_port"
    """host:port / bare domains — e.g. localhost:6333, qdrant.tech"""

    CIDR = "cidr"
    """CIDR ranges / subnets — e.g. 192.168.0.0/16, 10.0.0.0/8"""

    API_KEY = "api_key"
    """API keys / secret token patterns — e.g. sk-…, AKIA…, ghp_…, eyJhbGciOiJIUzI1NiJ9"""

    HTTP_STATUS_CODE = "http_status_code"
    """HTTP response codes mentioned in prose — e.g. error 503, HTTP 429"""

    # --- tech / dev ---
    UUID = "uuid"
    """UUIDs, hashes, fingerprints — e.g. 550e8400-e29b-41d4-a716-446655440000, 3f1c2a9b7d..."""

    VERSION_STRING = "version_string"
    """Semver / PEP 440 strings — e.g. v1.2.3, 2.0.0-beta.1"""

    ERROR_CODE = "error_code"
    """POSIX / Node error symbols — e.g. EACCES, ERR_CONNECTION_REFUSED"""

    URI = "uri"
    """HTTP, S3, file and other URI schemes — e.g. https://qdrant.tech/docs, s3://bucket/path"""

    ENV_VAR = "env_var"
    """Shell env vars and CLI flags — e.g. $QDRANT__SERVICE__PORT, --force"""

    CVE = "cve"
    """MITRE CVE and similar advisory IDs — e.g. CVE-2024-3094"""

    CODE_IDENTIFIER = "code_identifier"
    """camelCase, snake_case, dotted.path tokens — e.g. getUserById, os.path.join"""

    PACKAGE_COORDINATE = "package_coordinate"
    """npm scoped packages, Maven GAV coordinates — e.g. @types/node, org.apache:kafka"""

    HEX_COLOR = "hex_color"
    """CSS/HTML hex colors — e.g. #FF5733"""

    VALUE_WITH_UNIT = "value_with_unit"
    """Quantity + SI/colloquial unit — e.g. 16GB, 3.5mm, 240Hz"""

    STANDARDS_CITATION = "standards_citation"
    """Standards citations (RFC / ISO-IEC / IEEE) — e.g. RFC 9110, ISO/IEC 27001, IEEE 802.11ax"""

    FILE_PATH = "file_path"
    """Filesystem paths / filenames — e.g. /etc/qdrant/config.yaml, C:\\Users\\andrei\\report.csv"""

    CRYPTO_ADDRESS = "crypto_address"
    """On-chain wallet addresses — e.g. bc1qxy2kgdyg..."""

    GAME_NOTATION = "game_notation"
    """Chess PGN/FEN, Go SGF moves — e.g. Nf3, O-O, rnbqkbnr/pppppppp/8/..."""

    # --- finance / markets ---
    CURRENCY_AMOUNT = "currency_amount"
    """Price with currency symbol/code — e.g. €499.99, $1,200"""

    IBAN = "iban"
    """IBAN account numbers — e.g. DE89 3704 0044 0532 0130 00"""

    BIC = "bic"
    """BIC / SWIFT codes — e.g. DEUTDEFF, MARKDEF1100"""

    SECURITIES_ID = "securities_id"
    """Securities identifiers (ISIN / CUSIP) — e.g. US0378331005, 037833100"""

    STOCK_TICKER = "stock_ticker"
    """Stock tickers / cashtags — e.g. $AAPL, NVDA"""

    LEI = "lei"
    """Legal Entity Identifier — e.g. 529900T8BM49AURSDO55"""

    DERIVATIVES_SYMBOL = "derivatives_symbol"
    """OCC options / futures symbols — e.g. AAPL240119C00150000, ESH25"""

    MARKET_CODE = "market_code"
    """Market infrastructure codes (MIC / ABA routing / SEDOL / FIGI) — e.g. XNYS, 021000021, BBG000BLNNH6"""

    INDUSTRY_CODE = "industry_code"
    """Industry classification codes (NAICS / SIC / GICS) — e.g. NAICS 541511, SIC 7372"""

    BUSINESS_REGISTRATION = "business_registration"
    """Business registration numbers (EIN / CRN / DUNS) — e.g. EIN 12-3456789, DUNS 150483782"""

    TAX_ID = "tax_id"
    """VAT/EIN/TIN registration numbers — e.g. DE123456789"""

    BETTING_ODDS = "betting_odds"
    """Fractional / decimal / American odds — e.g. 5/2, 3.50, +150"""

    TICKET = "ticket"
    """Invoice/ticket reference numbers — e.g. INV-2026-000123, TCK-84721"""

    # --- legal / gov ---
    LEGAL_CITATION = "legal_citation"
    """Court citations, statute sections — e.g. 17 U.S.C. § 107, 539 F.3d 1024"""

    COURT_DOCKET = "court_docket"
    """Court docket numbers — e.g. 1:21-cv-02547, 2:19-cr-00123"""

    NEUTRAL_CITATION = "neutral_citation"
    """ECLI / UK neutral citations — e.g. ECLI:EU:C:2019:772, [2019] UKSC 41"""

    CELEX = "celex"
    """EU legal document IDs (CELEX) — e.g. 32016R0679"""

    LEGISLATIVE_CITATION = "legislative_citation"
    """Public Law / Statutes at Large / Federal Register — e.g. Pub. L. 117-58, 88 FR 12345"""

    EU_REGULATORY_CITATION = "eu_regulatory_citation"
    """EU regulatory citations — e.g. Art. 6(1)(a) GDPR, Regulation (EU) 2016/679"""

    PATENT_NUMBER = "patent_number"
    """Patent numbers — e.g. US10123456B2, EP1234567A1"""

    NATIONAL_ID = "national_id"
    """National identity numbers (SSN / NINO / passport) — e.g. 123-45-6789, QQ123456C"""

    # --- medical / bio ---
    MEDICAL_CODE = "medical_code"
    """ICD codes, CAS numbers, gene/SNP IDs — e.g. J45.909, rs429358"""

    CLINICAL_CODING = "clinical_coding"
    """CPT / HCPCS / SNOMED CT / LOINC / DRG — e.g. CPT 99213, SNOMED 22298006, LOINC 4548-4"""

    DRUG_ID = "drug_id"
    """Drug identifiers (NDC / ATC / RxNorm / UNII) — e.g. NDC 0069-4200-83, ATC A10BA02"""

    GENOMIC_ACCESSION = "genomic_accession"
    """UniProt / Ensembl / RefSeq / PDB accessions — e.g. P12345, ENSG00000139618, NM_000546, 1ABC"""

    HGVS_VARIANT = "hgvs_variant"
    """Genetic variant notation (HGVS) — e.g. c.76A>T, p.Lys76Asn"""

    CHEMICAL_ID = "chemical_id"
    """InChIKey / PubChem CID / ChEMBL (SMILES deliberately excluded: it is a
    grammar, not a token format — any regex would be a low-precision heuristic)
    — e.g. BSYNRYMUTXBXSQ-UHFFFAOYSA-N"""

    CLINICAL_TRIAL_ID = "clinical_trial_id"
    """NCT / PMID / ORCID registry IDs — e.g. NCT01234567, PMID 31978945, 0000-0002-1825-0097"""

    HEALTHCARE_PROVIDER_ID = "healthcare_provider_id"
    """NPI / DEA identifiers — e.g. NPI 1234567893, DEA AB1234563"""

    # --- logistics / transport / physical ---
    PHONE_NUMBER = "phone_number"
    """E.164 and local phone formats — e.g. +49 30 12345678"""

    POSTAL_CODE = "postal_code"
    """ZIP / postcode strings — e.g. 10115, SW1A 1AA"""

    GEO_COORDINATE = "geo_coordinate"
    """Decimal degree lat/lon pairs — e.g. 52.5200, 13.4050"""

    ALT_GEOCODING = "alt_geocoding"
    """what3words / Plus Codes / MGRS — e.g. ///filled.count.soap, 8FVC9G8F+6X, 33UUU9012"""

    BOOKING_REFERENCE = "booking_reference"
    """Airline PNR, record locators, flight numbers — e.g. LH1830, X4B9C2"""

    TRACKING_NUMBER = "tracking_number"
    """Carrier shipment tracking codes — e.g. 1Z999AA10123456784"""

    BARCODE = "barcode"
    """EAN-13, UPC-A, GTIN, Amazon ASIN — e.g. B08N5WRWNW"""

    SKU = "sku"
    """Manufacturer part/model strings — e.g. WH-1000XM5, RTX 4090"""

    SERIAL_NUMBER = "serial_number"
    """Device serial numbers, 15-digit IMEI — e.g. 356938035643809"""

    VIN_CONTAINER = "vin_container"
    """VIN / container numbers — e.g. WVWZZZ1JZXW000001, MSKU1234567"""

    LICENSE_PLATE = "license_plate"
    """License plates — e.g. B-QD 1234"""

    AIRPORT_AIRLINE_CODE = "airport_airline_code"
    """IATA / ICAO codes — e.g. BER, EDDB, LH, DLH"""

    AIRCRAFT_VESSEL_REG = "aircraft_vessel_reg"
    """Tail numbers / IMO / MMSI — e.g. N12345, D-AIMA, IMO 9074729"""

    CUSTOMS_CLASSIFICATION = "customs_classification"
    """HS / HTS / Schedule B codes — e.g. HS 090111, HTS 0901.11.00"""

    HAZMAT_CODE = "hazmat_code"
    """UN numbers / GHS / NFPA codes — e.g. UN1203, H225"""

    MATERIAL_GRADE = "material_grade"
    """AISI-SAE / aluminium temper / titanium grades — e.g. AISI 304, 6061-T6, Ti-6Al-4V"""

    NSN = "nsn"
    """NATO Stock Numbers — e.g. 1234-56-789-0123"""

    SIM_SUBSCRIBER_ID = "sim_subscriber_id"
    """ICCID / IMSI subscriber identifiers — e.g. 8901410321111851072, 310150123456789"""

    # --- media / knowledge / misc ---
    ISO_CODE = "iso_code"
    """BCP-47 locale, ISO 4217, ISO 3166 — e.g. en-US, EUR, DE"""

    SOCIAL_HANDLE = "social_handle"
    """Twitter/Instagram handles and hashtags — e.g. @qdrant_engine, #vectorsearch"""

    ACADEMIC_IDENTIFIER = "academic_identifier"
    """DOI, arXiv IDs, ISBN-13 — e.g. 10.1145/3539618, 2104.08663"""

    ISSN = "issn"
    """Serial publication IDs — e.g. 2049-3630"""

    MUSIC_WORK_CODE = "music_work_code"
    """ISRC / ISWC / ISAN codes — e.g. USRC17607839, T-034.524.680-1"""

    MEDIA_DB_ID = "media_db_id"
    """IMDb / TMDB identifiers — e.g. tt0111161, nm0000138"""

    LIBRARY_CLASSIFICATION = "library_classification"
    """Dewey / Library of Congress call numbers — e.g. 005.74, QA76.73.C15"""

    ASTRONOMICAL_DESIGNATION = "astronomical_designation"
    """NGC / Messier / HD / Kepler designations — e.g. NGC 224, M31, HD 209458, Kepler-186f"""

    # --- named entities (Method: MODEL — GLiNER2, audited labels only) ---
    PERSON = "person"
    """Person names in any casing — e.g. chef mike ward. Audited 0.87 @0.32."""

    LOCATION = "location"
    """Place names in any casing — e.g. dallas ga. Audited 0.84 @0.34."""

    PROPER_NOUN = "proper_noun"
    """Coarse name-like spans: titles, works, brands — the residue the four
    fine classes miss, and the coarse cover for dropped org/product.
    Audited 0.84 @0.31 (0.77 on all-lowercase text)."""


class Domain(StrEnum):
    """Semantic domain of an identifier type — validation/reporting axis.

    Module placement follows this enum: banks/<domain>.py contains exactly
    the banks of that domain (tested invariant).
    """

    GENERAL = "general"
    NETWORK = "network"
    TECH = "tech"
    FINANCE = "finance"
    LEGAL = "legal"
    MEDICAL = "medical"
    LOGISTICS = "logistics"
    MEDIA = "media"


class SentenceMarker(StrEnum):
    """
    Sentence Markers group of the taxonomy: lexical cues about phrasing
    register. Method: REGEX (closed word/phrase lists and shapes).
    """

    NEGATION = "negation"
    """Negation / exclusion words — e.g. "laptops without touchscreen"."""

    GREETING = "greeting"
    """Conversational openers — e.g. "hi how do I set up Qdrant"."""

    POLITENESS = "politeness"
    """Politeness / request markers — e.g. "please explain quantization"."""

    INTERJECTION = "interjection"
    """Interjections and exclamations — e.g. "ugh my container keeps crashing"."""

    COMPARATIVE = "comparative"
    """Comparative / superlative markers — irregulars and function words only
    (better, worst, more, than); -er/-est suffix morphology deliberately
    excluded as FP-prone (water, forest) — revisit as an ALGO stemmer feature."""

    ACRONYM = "acronym"
    """Cased acronym shapes — e.g. NASA, N.Y. Lowercase acronyms (tv, dna)
    are undetectable by shape; model path requires fine-tuning (SPEC d13)."""


class LogicalStructure(StrEnum):
    """
    Logical Structures group: query-logic constructs. Method: REGEX now;
    TEMPORAL is the first layered feature (GLiNER2 backstop pending).
    """

    OPERATOR_SYNTAX = "operator_syntax"
    """Explicit boolean operators — e.g. "cats AND dogs", "java NOT javascript".
    Uppercase-gated: lowercase and/or/not are ordinary function words."""

    TEMPORAL = "temporal"
    """Relative temporal expressions — e.g. "bitcoin price today", "3 days ago".
    Absolute forms (ISO dates, Q3 2026) belong to the identifier group
    (DATETIME / BUSINESS_TEMPORAL); this feature covers what they can't."""


class CorruptionKind(StrEnum):
    """
    Corruption group: text-damage signals. Method: REGEX for artifacts;
    typo/noise features are ALGO/MODEL and deferred.
    """

    ENCODING_ARTIFACT = "encoding_artifact"
    """Encoding junk — mojibake digraphs (â€™, Ã©), U+FFFD replacement char.
    Matches mid-word: artifacts ignore word boundaries."""


class StatisticalMetric(StrEnum):
    """
    Statistical Metrics group: named scalars per query. Method: ALGO
    (stat banks — no spans, no claim registry).
    """

    LENGTH = "length"
    """Query size — length_tokens, length_chars."""

    STOPWORD_RATIO = "stopword_ratio"
    """Function-word share — stopword_count, stopword_ratio. The REGEX
    fallback of the POS profile's closed_class_share (SPEC decision 14)."""

    POS_PROFILE = "pos_profile"
    """UD-17 part-of-speech histogram + derived shares (open/closed class,
    noun, verb presence, PROPN) via the pinned spaCy tagger (SPEC decision
    14). Needs the downloaded spaCy model."""

    MORPHOLOGY = "morphology"
    """Inflection profile via the pinned lemmatizer — inflected_count,
    inflected_share (tokens whose lemma differs from their surface form).
    The grammar-caused half of vocabulary mismatch."""

    SYNTACTIC_DEPTH = "syntactic_depth"
    """Compositional structure via the pinned parser — parse_depth,
    clause_count. Deep structure = meaning sparse bag-of-words loses."""


class SemanticFeature(StrEnum):
    """
    Semantical group: features that capture meaning-affecting register and
    cross-lingual properties. Method: MODEL (lang-id engine, pending library
    decision — see SPEC decision 18). Output: stat banks (named scalars).
    """

    LANGUAGE_SET = "language_set"
    """The set of languages detected in the query — one FeatureStat per BCP-47
    code present (detected_en, detected_de, ...) plus language_count. A query
    may carry multiple languages simultaneously (code-switching); this is
    multi-label, not single-class."""

    CODE_SWITCHING = "code_switching"
    """Whether the query mixes two or more languages as a natural, culturally
    embedded whole (e.g. Moldavian Romanian with Russian jargon, Egyptian Arabic
    with English class-marker terms). Stat: is_code_switched (0.0/1.0),
    language_count. Not segmentable — the mixing is the register, not a
    sequence of monolingual spans."""

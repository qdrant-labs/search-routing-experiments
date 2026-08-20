"""GATE 3 — the gate neither `cellfill.admit()` nor the route prior provides:
does a cell's predicate admit the archetype its own `looks_like` describes?
Positives were authored from `name` + `looks_like` ALONE, before any band was
read; negatives are blatant non-members of another archetype entirely.

A prose-literal positive its predicate rejects is a FINDING: it is marked
`xfail(strict=True)` and listed in docs/cells_v3_extension.md as a quarantine
candidate. Cases are never tuned into the bands — that would make this test
a restatement of the predicate, which is the circularity it exists to break.
"""

from __future__ import annotations

import pandas as pd
import pytest

from composition.cells import CELLS, CELLS_V3
from composition.mini_catalog import mini_catalog
from query_taxonomy.features import FeatureExtractor

ALL_CELLS = (*CELLS, *CELLS_V3)

# Recurring blatant non-members, reused as negatives wherever they are one.
PLAIN_CONCEPT = "coffee grinder"
PLAIN_QUESTION = "why do cats purr"
TELEGRAM = "postgres index bloat vacuum"
COURTESY = "hi there, could you please help me with my order, thanks"
UUID = "3f9a2c1e-77bb-4c0d-9a11-2f8e6d4b0a55"
CLEAN = "how do I reset my password"
TYPO = "how to confgure nginx reverse proxy"
MOJIBAKE = "what is the â€œbest way to"
PY_SNIPPET = "for i in range(10):\n    print(items[i].name)  # TODO fix"
PASTED = (
    "The migration ran for about forty minutes on the primary replica before "
    "the coordinator reported a lock timeout, and at that point the queue "
    "backed up behind the write path while the read replicas kept serving "
    "stale rows to the checkout service, which retried each request three "
    "times and eventually surfaced a generic error page to customers in the "
    "European region for roughly a quarter of an hour."
)

CASES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # ------------------------------------------------------- shape / length ---
    "bare_concept_token": (
        ("photosynthesis", "coffee grinder"),
        (PASTED, PY_SNIPPET),
    ),
    "short_grammatical_question": (
        ("why do cats purr", "how does rain form"),
        (TELEGRAM, UUID),
    ),
    "stopword_saturated_midlength": (
        (
            "i am trying to find out what the cause of the noise in my car is",
            "is there a way to get a copy of the report that was sent to me",
        ),
        (TELEGRAM, PLAIN_CONCEPT),
    ),
    "verbose_grammatical_request": (
        (
            "I have been trying to understand whether the annual maintenance "
            "plan that came with the appliance also covers the replacement of "
            "the filter, because the leaflet in the box says one thing and the "
            "website appears to say something rather different about it.",
            "Could you explain how the refund is calculated when an order is "
            "returned after the promotional period has ended, since the amount "
            "that was credited back to my card seems noticeably smaller than "
            "the price I actually paid for the item at the time of purchase.",
        ),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "extreme_length_pasted_query": (
        (
            PASTED,
            "Subject: recurring build failure. The nightly pipeline has failed "
            "on the packaging step for the last four nights. The step pulls the "
            "vendored wheels from the internal mirror, verifies the hashes, and "
            "then assembles the container image. Nothing in the manifest "
            "changed this week as far as anyone can tell. The logs mention a "
            "certificate that expired, but the same certificate is apparently "
            "still accepted by the staging mirror, so nobody is sure whether "
            "that is the real cause or simply noise in the output.",
        ),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "high_morphological_variation": (
        (
            "the workers were complaining about the reorganizations affecting "
            "their scheduled shifts",
            "she noticed the darkened images were becoming increasingly "
            "unreadable",
        ),
        (PLAIN_CONCEPT, UUID),
    ),
    "deep_nesting_single_sentence": (
        (
            "if the invoice that finance flagged was already paid, who decides "
            "whether the vendor we onboarded stays active",
            "the manager who approved the request that the team submitted "
            "before the policy changed said that the refund would be issued "
            "once the account was verified",
        ),
        (PLAIN_CONCEPT, TELEGRAM),
    ),
    "multi_statement_context_dump": (
        (
            "My laptop stopped charging last week. I already tried a different "
            "cable and a different outlet. Can someone tell me whether the "
            "battery is replaceable on this model?",
            "We upgraded the database on Friday. Since then the nightly export "
            "job fails halfway through. I need to know where the partial files "
            "are written so I can clean them up.",
        ),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "wide_flat_enumeration": (
        ("docker, kubernetes, terraform, ansible", "apples, oranges, pears and plums"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "keyword_telegram_short": (
        ("postgres index bloat vacuum", "espresso machine descaling vinegar"),
        (PLAIN_QUESTION, PASTED),
    ),
    "negation_bearing_question": (
        (
            "which laptops do not have a soldered battery",
            "how do I open the door without a key",
        ),
        (PLAIN_CONCEPT, TELEGRAM),
    ),
    "comparative_multi_entity": (
        ("iPhone vs Android: which is better", "React versus Vue for a small dashboard"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "conversational_courtesy_wrapper": (
        (
            "hi there, could you please tell me how to change the delivery "
            "address on my order, thanks",
            "hello, sorry to bother you, but I would really appreciate it if "
            "you could explain how the refund works, thank you",
        ),
        (TELEGRAM, PLAIN_CONCEPT),
    ),
    # ------------------------------------------------------ symbols / prose ---
    "bare_acronym": (
        ("NASA", "IMF report"),
        (PLAIN_QUESTION, PASTED),
    ),
    "acronym_inside_question": (
        (
            "how do I configure the DNS records for a new subdomain on my server",
            "what does the IMF actually do when a country cannot repay its loans",
        ),
        ("NASA", PLAIN_CONCEPT),
    ),
    "pasted_code_fragment": (
        (
            PY_SNIPPET,
            "at com.example.service.OrderService.process(OrderService.java:142)\n"
            "\tat com.example.web.Controller.handle(Controller.java:88)",
        ),
        (PLAIN_QUESTION, PLAIN_CONCEPT),
    ),
    "code_symbol_named_in_prose": (
        (
            "why does requests.get raise a timeout error on the second call",
            "how do I stop the user_profile_cache from growing without a bound",
        ),
        (PLAIN_CONCEPT, PY_SNIPPET),
    ),
    "symbol_pile_no_grammar": (
        (
            "NullPointerException OrderService.process",
            "malloc.c:2379 free(): invalid pointer",
        ),
        (PLAIN_QUESTION, COURTESY),
    ),
    "math_notation_present": (
        ("solve x^2 + 5x + 6 = 0", "E = mc^2 derivation"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "boolean_operator_query": (
        ("cats AND dogs NOT birds", '"exact phrase" AND site:example.com -advertising'),
        (PLAIN_QUESTION, PLAIN_CONCEPT),
    ),
    # ------------------------------------------------------------- temporal ---
    "relative_temporal_no_dates": (
        (
            "latest stable release of the driver",
            "what changed since the last deployment",
        ),
        ("release notes for 2024-01-15", PLAIN_CONCEPT),
    ),
    "datetime_token_present": (
        ("outage report 2024-03-17", "meeting notes 2023-11-02 summary"),
        (PLAIN_CONCEPT, "latest stable release of the driver"),
    ),
    "business_temporal_reference": (
        (
            "what were the revenue drivers in Q3 2023",
            "where are the sprint 42 retrospective notes kept",
        ),
        (PLAIN_CONCEPT, UUID),
    ),
    # ---------------------------------------------------- codes and numbers ---
    "version_pinned_technical": (
        ("django 4.2.1 migration error", "upgrade to node 18.16.0"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "status_code_idf_split": (
        (
            "why does my server keep returning a 502 error after the deploy",
            "what causes HTTP 429 responses when the rate limit is not exceeded",
        ),
        (PLAIN_CONCEPT, "502"),
    ),
    "rare_key_buried_in_chatter": (
        (
            "hey folks, not sure if this is the right channel but we were "
            "looking at that thing from last week and I think CVE-2021-44228 "
            "is the one that actually matters here",
            "so I half remember someone saying the fix went in under PROJ-4821 "
            "but I cannot find the ticket anywhere and it has been bugging me "
            "all morning",
        ),
        ("CVE-2021-44228", PLAIN_CONCEPT),
    ),
    "bare_number_token": (
        ("1984", "route 66"),
        (PLAIN_QUESTION, PASTED),
    ),
    "number_inside_natural_question": (
        (
            "how much did the population grow between 1990 and 2020 in this region",
            "is there a form 1099 requirement for a contractor who earned less "
            "than 600 dollars",
        ),
        ("1984", PLAIN_CONCEPT),
    ),
    "short_quantified_spec": (
        ("m3 screw 8mm length", "220v 60hz transformer"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "instance_value_in_intent": (
        (
            "what colors go well with #3b5998 for a website header",
            "is $1,299.99 a good price for this laptop right now",
        ),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "env_var_configuration": (
        (
            "why is DATABASE_URL ignored when I set it in the shell",
            "what does AWS_DEFAULT_REGION do if the profile already names a region",
        ),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    # --------------------------------------------------- opaque identifiers ---
    "bare_machine_token": (
        (UUID, "block the 10.0.0.0/8 range"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "registry_structured_identifier": (
        ("IBAN DE89370400440532013000 transfer failed", "verify BIC DEUTDEFF branch"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "logistics_catalog_token": (
        ("SKU ABC-12345-XY availability", "barcode 036000291452 lookup"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "legal_citation_canonical": (
        ("Roe v. Wade, 410 U.S. 113", "docket No. 1:21-cv-04184 filings"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "bio_clinical_identifier": (
        ("E11.9 diagnosis code", "accession NM_000546 variants"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "bibliographic_catalog_identifier": (
        ("doi 10.1038/nature12373 abstract", "ISSN 0028-0836 archive"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "travel_transport_code": (
        ("cheapest flights from LHR to JFK", "who owns tail number N12345 now"),
        (PLAIN_CONCEPT, "1984"),
    ),
    "capsword_shape_ambiguity": (
        ("is AAPL a good buy right now", "what does Nf3 open up for white"),
        (PLAIN_CONCEPT, UUID),
    ),
    "package_coordinate_dependency": (
        (
            "com.fasterxml.jackson.core:jackson-databind:2.13.0 vulnerability",
            "@types/node 18.11.9 breaking change",
        ),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "web_locator_token": (
        ("https://example.com/docs/getting-started", "mail to support@example.com bounced"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "standards_compliance_lookup": (
        ("does ISO 27001 require annual audits", "shipping UN1203 by air rules"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "geo_coordinate_postal": (
        ("weather in 94103 today", "what is at 48.8584, 2.2945"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "single_token_char_blob": (
        (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0",
        ),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    # ----------------------------------------------------------------- v3 ----
    "rare_term_query": (
        ("cholangiocarcinoma chemoembolization outcomes", "thermoluminescence dosimetry calibration"),
        ("how are you today", PLAIN_CONCEPT),
    ),
    "fragmented_query": (
        ("phosphorylation cascade regulation", "immunohistochemistry staining protocol"),
        ("cat food", "how are you today"),
    ),
    "typo_bearing_query": (
        (TYPO, "downlod the latest driver"),
        ("how to configure nginx reverse proxy", PLAIN_CONCEPT),
    ),
    "high_oov_query": (
        ("zorblax quantifier splines", "flurbicon telemetry mesh"),
        ("how are you today", PLAIN_CONCEPT),
    ),
    "artifact_corrupted_query": (
        ("what is the â€œbest way to", "how do I reset my <br> password &nbsp; on the"),
        (CLEAN, PLAIN_CONCEPT),
    ),
    # ------------------------------------- v3: the orphaned identifier banks ---
    "network_device_identity": (
        ("MAC aa:bb:cc:dd:ee:ff duplicate", "SIM 8944500102198765432 blocked"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "corporate_registry_code": (
        ("LEI 5493001KJTIIGC8Y1R12 lookup", "NAICS 541511 definition"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "statutory_instrument_citation": (
        ("ECLI:EU:C:2019:772 ruling summary", "Regulation (EU) 2016/679 article 17"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "clinical_chemical_code": (
        ("CPT 99213 billing rules", "NCT01234567 enrollment criteria"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "logistics_materials_code": (
        ("1Z999AA10123456784 delivery status", "AISI 304 stainless yield strength"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "media_catalog_code": (
        ("tt0111161 cast list", "ISRC USRC17607839 rights holder"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    # -------------------------------------------- v3: the corruption ladder ---
    "damage_free_query": (
        (CLEAN, "annual leave policy"),
        (TYPO, MOJIBAKE),
    ),
    "lightly_damaged_query": (
        (TYPO, "how do I reset my <br> password &nbsp; on the"),
        (CLEAN, "confgure the â€œsettings"),
    ),
    "heavily_damaged_query": (
        (MOJIBAKE, "confgure the â€œsettings"),
        (CLEAN, TYPO),
    ),
}

PROSE_CONFLICTS: dict[str, str] = {
    "deep_nesting_single_sentence": (
        "prose demands one sentence stacking relcl/advcl/ccomp six deep; those "
        "are the CLAUSAL_DEPS statement_count counts, so `statement_count < 2` "
        "rejects every instance (5 and 6 measured)"
    ),
    "wide_flat_enumeration": (
        "prose permits 'at most a thin frame'; the predicate requires "
        "natural_language_share >= 0.1, which a bare comma list cannot reach "
        "(0.0). An and-joined list passes only because `and` is a function word"
    ),
    "comparative_multi_entity": (
        "'X versus Y' — the prose's first named form and the cell's own name — "
        "yields 0 conj arcs under the pinned en_core_web_sm, so "
        "widest_list_size >= 2 is unreachable; 'versus' is also not a "
        "comparative marker"
    ),
    "pasted_code_fragment": (
        "prose asks for real traces; `for`/`in`/`at`/`if` are closed-class, so "
        "a Python snippet scores 0.3 and a Java trace 0.2 against a "
        "natural_language_share < 0.1 ceiling"
    ),
    "symbol_pile_no_grammar": (
        "prose names 'a pasted stack-trace fragment'; code_identifier fires on "
        "snake_case only — camelCase dotted frames and C traces claim 0 (the "
        "trace goes to host_port), so >= 2 is unreachable for that half"
    ),
    "boolean_operator_query": (
        "uppercase AND/OR tag CCONJ and NOT tags PART, all closed-class, so "
        "'cats AND dogs NOT birds' scores 0.4 against a "
        "natural_language_share < 0.25 ceiling — operator density inverts the "
        "predicate"
    ),
    "relative_temporal_no_dates": (
        "the prose's own example 'since the last release' emits no temporal "
        "span — TemporalBank is a closed list (latest / now / last week), so "
        "'since the last X' and 'before the change' have no path in"
    ),
    "short_quantified_spec": (
        "value_with_unit misses electrical units written closed-up: '220v' and "
        "'60hz' claim nothing while '8mm', '5kg' and '60 Hz' claim — the cell's "
        "spec-token axis is a unit-vocabulary lottery"
    ),
    "env_var_configuration": (
        "the env_var bank requires a `$` sigil, but the prose demands the "
        "documentation form (bare ALL_CAPS_UNDERSCORE) — which code_identifier "
        "claims instead. No prose-literal instance can satisfy env_var >= 1"
    ),
    "bare_machine_token": (
        "a bare CIDR measures 5 length_words on its own, so the cell's 'up to "
        "six words total' leaves room for one framing word, not the 'few' the "
        "prose promises ('block the 10.0.0.0/8 range' = 8 words)"
    ),
    "logistics_catalog_token": (
        "the prose's 'dash-joined alphanumerics' SKU goes to ticket_like, not "
        "sku (which needs the digits closed up: SKU12345), and ticket_like is "
        "in no branch of this cell"
    ),
    "legal_citation_canonical": (
        "a case-reporter citation — the prose's first named form — is claimed "
        "by http_status_code ('410 U') plus number; legal_citation only fires "
        "on the U.S.C. statute form. Explains p_natural = 0"
    ),
    "travel_transport_code": (
        "airport_airline_code is keyword-gated ('flight LHR'), so a natural "
        "travel query naming bare IATA codes is claimed by stock_ticker_like — "
        "a bank in no branch of this cell"
    ),
    "geo_coordinate_postal": (
        "the prose names 'a coordinate pair', but geo_coordinate is one of the "
        "26 orphaned banks and is in no branch; the coordinate also costs 4 "
        "length_words, breaching the cell's own 7-word ceiling"
    ),
    "business_temporal_reference": (
        "the prose names 'a sprint number' and 'a calendar week'; "
        "business_temporal claims only FY/quarter forms (FY2024, Q3 2023) — "
        "'sprint 42' and 'week 32' fall to number"
    ),
    "typo_bearing_query": (
        "TypoBank recall is partial: 'confgure' claims, 'downlod' and 'adress' "
        "do not — the prose's single-character-deletion recipe is not uniformly "
        "detectable"
    ),
}
"""Cells whose blind, prose-literal positive their own predicate rejects. Read
from the measured run and never repaired by reshaping the case: each entry is a
quarantine candidate, documented in docs/cells_v3_extension.md."""

OVER_ADMITS: dict[str, str] = {
    "rare_term_query": (
        "at two tokens rare_share is 0.5 whenever ONE word is uncommon, so "
        "'coffee grinder' — the archetypal common concept — is claimed"
    ),
}
"""Cells that admit a blatant non-member. The over-admission the membership
justification names, and the half a one-sided gate cannot see."""


@pytest.fixture(scope="module")
def measured() -> pd.DataFrame:
    """Every case text as a catalog row, extracted ONCE with the full engine
    set (the parser stats the cells band on need spaCy)."""
    texts = sorted({t for pos, neg in CASES.values() for t in (*pos, *neg)})
    pool = pd.DataFrame({
        "home_lane": "gate",
        "query_id": [str(i) for i in range(len(texts))],
        "query": texts,
    })
    columns = tuple({band.column for cell in ALL_CELLS for band in cell.bands})
    mini = mini_catalog(pool, FeatureExtractor(engines=None), columns=columns)
    return mini.set_axis(pd.Index(texts, name="query"))


def _params(quarantined: dict[str, str]) -> list:
    return [
        pytest.param(
            cell,
            id=cell.name,
            marks=(
                [pytest.mark.xfail(strict=True, reason=quarantined[cell.name])]
                if cell.name in quarantined
                else []
            ),
        )
        for cell in ALL_CELLS
    ]


def test_every_cell_has_two_positives_and_two_negatives():
    missing = [cell.name for cell in ALL_CELLS if cell.name not in CASES]
    assert not missing, missing
    assert all(len(side) == 2 for sides in CASES.values() for side in sides)
    assert set(CASES) == {cell.name for cell in ALL_CELLS}


@pytest.mark.parametrize("cell", _params(PROSE_CONFLICTS))
def test_prose_positives_satisfy_their_own_predicate(cell, measured):
    positives = list(CASES[cell.name][0])
    admitted = cell.select(measured.loc[positives])
    rejected = [q for q, ok in zip(positives, admitted) if not ok]
    assert not rejected, (
        f"{cell.name}: its looks_like describes queries its predicate "
        f"rejects: {rejected}"
    )


@pytest.mark.parametrize("cell", _params(OVER_ADMITS))
def test_blatant_non_members_are_rejected(cell, measured):
    negatives = list(CASES[cell.name][1])
    admitted = cell.select(measured.loc[negatives])
    claimed = [q for q, ok in zip(negatives, admitted) if ok]
    assert not claimed, f"{cell.name}: over-admits {claimed}"

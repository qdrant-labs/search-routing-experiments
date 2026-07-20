# SPEC: Diversified Query Dataset Pipeline

Vocabulary: see CONTEXT.md. Session date: 2026-07-15.

## Problem

Build a diversified query dataset for the Strategy Router (dense/sparse/hybrid)
by composing natural queries harvested from registered IR datasets with
generated/augmented ones — such that the full feature taxonomy
(`src/query_taxonomy/query-taxonomy.csv`) is well represented, every row stays
answerable against some corpus, and the dataset can later be labeled
empirically. Current focus is extractor quality and breadth; strategy labeling
is explicitly a later stage.

## Decisions

1. **Next extractor wave = REGEX + query-only ALGO features** (negation,
   operator syntax, stopword/function-word ratio, interjections, comparatives,
   politeness, greetings, artifacts, length, PMI, morphology, char-level
   typos). MODEL-tier and corpus-relative features are deferred, not dropped.
   — *Most Classification power per unit of new machinery; every extractor
   stays deterministic and bank-style testable.*

2. **Measured features are spans-based structs; targets are quantities.**
   `features` = `QueryIdentifiers`-style pydantic model (spans section +
   scalars section), always re-measured on final text — never trusted from a
   generator. `feature_targets` = per-feature counts/ratios/ranges.
   — *Spans exist only after generation; targets can only prescribe
   quantities. The verification loop checks measured-vs-target.*

3. **Diversified dataset row schema** (`queries.parquet`): `query_id`
   (namespaced), `text`, `provenance` (`natural | doc_grounded | synthetic`),
   `source_dataset` (None only for synthetic), `doc_ids`, `features`,
   `feature_targets` (None for natural), `strategy_label` (nullable,
   reserved). — *Targets-vs-measured is the generation audit trail; the
   nullable label means the labeling stage extends the table, not rebuilds it.*

4. **Synthetic documents live in a sibling `documents.parquet`** inside the
   artifact; `doc_grounded` rows reference their source corpus by
   `(source_dataset, doc_id)`, no copying. The registry stays an
   acquisition-only, queries-only catalog. — *Pipeline outputs don't belong in
   the acquisition registry.*

5. **Grounding: doc_grounded wherever a real corpus supports the target
   feature; synthetic query+doc pairs for the residue; never free-floating
   generation.** The verification loop guarantees feature fidelity;
   answerability is guaranteed by construction. — *Free-floating queries are
   permanently unlabelable; grounding is cheapest at generation time.*

6. **Composition = recipe + two steps.** The recipe (global quotas over
   features, within-feature strata, provenance mix with a minimum natural
   share) is a hand-owned, reviewable artifact. Step A builds the natural
   core from harvest-target-driven scan-and-filter over cached parquets;
   Step B diffs the core against the recipe and hands the deficit to
   Generation as the order sheet. Greedy quota-fill (one query may satisfy
   several quotas); ILP escalation only if quotas demonstrably conflict.
   — *All demo claims are marginal claims; greedy is auditable and a day of
   work.*

7. **Harvest targets are ranked, profiling-proposed, human-ratified.**
   Top-N features by prevalence relative to the cross-dataset average;
   ratification doubles as the extractor FP audit on fresh text; the result
   is committed as a declared code object (banks-style). Datasets rich in
   nothing serve the background stratum. — *Fully automatic comparative
   advantage would mine false positives at volume.*

8. **Within-feature strata are computed views over spans** — density,
   surface diversity, repetition, clustering — not separate extractors.

9. **Estimation and composition stay separate methods.** Uniform seeded
   sampling only for profiling (unbiased estimates); composition is
   constrained selection over full scans — rare strata are scanned-and-
   filtered, never sampled-and-hoped.

10. **Verification tooling is library-first**: `verify(text, targets)` over
    the banks/extractors, driving a scripted Claude-API generation loop
    (generate → verify → PASS/FAIL → retry). MCP server is a thin later
    wrapper if interactive generation is wanted. — *The order-sheet workflow
    is batch; library→MCP is the cheap migration direction.*

11. **Demo ladder**: (a) feature-coverage figures + (b) strategy-disagreement
    measurement (dense vs sparse vs RRF ranking divergence; needs corpora +
    Qdrant, needs no qrels) first; then (c) an LLM-router class inside
    `src/hybrid_search_rrf_dataset` whose prompt carries the taxonomy labels
    + examples, evaluated on NDCG against baselines — with the production
    hard classifier (0–2 dense / 3–6 RRF / 7–9 BM25) as a first-class
    baseline. — *(b) is the weakest claim that still justifies the project
    and is measurable without labels.*

12. **Taxonomy scoping**: taxonomy labels may serve as *prompt context* in
    the demo router (to prove the labels carry routing signal); they remain
    off-limits as learned/engineered features of a production router.
    — *Scopes the 2026-07-13 "not router features" rule.*

13. **MODEL-tier v0 = GLiNER2, named entities only, gated on a smoke eval**
    (arch-validator verdict + andrey-review amendment, 2026-07-15; user
    ratified). GLiNER2 (`fastino/gliner2-base-v1`, Apache 2.0) beat spaCy
    (closed 18-type vocabulary fails the evolving-taxonomy requirement) and
    tied GLiNER v1, tiebreakers favoring GLiNER2: classification head
    covers the deferred judgment-shaped features, built-in RegexValidator,
    family shared with the auto-fusion router. Review verdict:
    **measure-then-ship** — commitment is gated on a 200-query smoke eval
    (50 each msmarco / trec-dl / nfcorpus / miracl-en from the cached
    parquets; labels person/org/location/product/date; hand-audited).
    Acceptance: precision ≥ 0.8 on person/org/location at tuned thresholds;
    `text[start:end] == span.text` on every hit; lowercase variants within
    a few points of cased. Below the bar → same 200 through GLiNER v1
    before conceding to fine-tuning. Scope trim from review: **acronyms →
    regex bank** (shape-defined, AMBIGUOUS tier); **temporal → banks-first**
    (patterns + small closed vocabulary), model fallback only if the smoke
    eval's date label earns it. Guardrails: pin `gliner2>=1.2.4`
    (char-offset bug below); pinned weights + batch size (near-threshold
    float determinism); per-type confidence thresholds; MODEL spans never
    claim ranges from the banks' tier-priority registry; MODEL features
    nullable per-row (English-only); torch deps (~2–3 GB) in a dedicated
    Poetry group, never in the default install.

    **Outcome (2026-07-16, hand-audited smoke eval, user ratified): ADOPTED,
    label-scoped.** v0 label set `{person, location, proper noun}` at tuned
    thresholds `{0.32, 0.34, 0.31}` (audited precision 0.87 / 0.84 / 0.84 —
    proper noun holds 0.77 on the all-lowercase slice); `temporal` allowed
    as banks-first fallback (0.83 tuned @0.58). Dropped as ambiguous
    classes: `product` (0.49) and `organization` (0.30; threshold rescue
    keeps 7/33) — proper noun covers org/product mentions coarsely. GLiNER
    v1 cross-check waived. Acronym zero-shot rejected (lowercase slice
    0.43, high-confidence FPs → thresholds can't rescue; fine-tune with
    bank silver labels is the only model path). Closed-list markers all
    scored 0 — banks own them, now with data. Extra guardrail from the
    eval: GLiNER2 outputs are **schema-composition-dependent** (adding a
    label shifts other labels' scores), so the exact schema is part of the
    determinism pin alongside weights, library version, and batch size; the
    extractor wrapper must clamp span `end` to `len(text)` and drop spans
    failing the round-trip (trailing-period hallucination on
    state-abbreviation-shaped tails).

14. **POS profile = ALGO feature via pinned spaCy tagger; GLiNER schema
    unchanged** (grill-me 2026-07-16). New taxonomy row "POS profile (UD-17
    tagset)": full 17-tag histogram stored per query; recipe-facing scalars
    derived as computed views (open_class_share, closed_class_share,
    noun_share, verb_presence, propn_share) per decision 8. POS labels
    rejected as GLiNER entity prompts — violates decision 13's
    named-entities-only scope, zero-shot NER recall on function words is
    unproven-to-poor, and tagging is a solved task (~0.97 newswire) with no
    audit cost via a pinned tagger. Stopword/function-word ratio row demoted
    to REGEX fallback of closed_class_share (non-English, minimal installs);
    recipe quotas reference closed_class_share only. Domain-shift guardrail
    (taggers degrade on keyword telegrams): bank-style fixed cases + one-off
    closed_class_share↔stopword-ratio correlation diagnostic over the cached
    datasets (investigate if r < 0.8) — no hand audit, aggregate shares wash
    out single-token errors. Deps: spacy + exact-pinned en_core_web_sm in
    the `model` group; POS fields nullable per-row (English-only).

15. **Unified bank family + FeatureExtractor** (grill-me 2026-07-16). One
    family: `GeneralBank[EngineT, OutT]` where `OutT` is a *constrained*
    TypeVar over exactly `FeatureSpan | FeatureStat` — `matches()` and
    `compute()` unify as `compute(text) -> list[OutT]`; `ScalarExtractor`
    deleted; `FeatureStat = (name, value)` is the stats counterpart of
    `FeatureSpan`; `StatBank(GeneralBank[EngineT, FeatureStat])` keeps the
    engine generic (tokenizer pattern, spaCy Language). `ambiguity` gets a
    concrete RIGID default on `GeneralBank` ("stats are solid numbers");
    span-group bases (IdentifierBank, MarkerBank) re-abstract it so span
    banks still declare tiers explicitly. Registry:
    `FEATURE_BANKS: dict[FeatureGroup, tuple[type[GeneralBank], ...]]` in
    `query_taxonomy/__init__.py` (defined before any features re-export —
    import-order rule), group↔key consistency validated at
    `FeatureExtractor.__init__` (structural within-group claim resolution).
    `FeatureExtractor` replaces `CorpusIdentifierExtractor`:
    `resolve(text, *, groups: Iterable[FeatureGroup] | None = None)` and
    `extract(queries, *, groups=...)` — the class owns iteration, one text
    pass covers all requested groups. Output model: `QueryFeatures`
    (spans + stats sections, each nested by group), `SpanProfile` (was
    DocumentIdentifier; diversity/dfs unchanged), `StatProfile` (doc-keyed
    per-query stat values kept — recipe strata need them — with computed
    corpus aggregates), `CorpusFeatures` with per-group `summary()` (Domain
    sub-grouping only inside structured_identifiers). Old names die without
    aliases; `dataset_registry.profile()` updates in the same change.

16. **Extractor fan-out wave + engine doctrine** (grill-me 2026-07-16).
    Three engines: RegexBank (edify — closed shapes/lists, precision-first),
    SpacyBank (`StatBank[Language]` — grammatical signal, lands with
    decision 14's POS wave), Gliner2Bank (context entities, audited labels
    only, pinned schema per decision 13). **A Query Feature may need more
    than one signal**: multiple banks may share one feature name — the
    within-group claim registry plus AmbiguityTier ordering arbitrate
    (deterministic engine claims first, model engine backstops at
    AMBIGUOUS). First layered feature: `LogicalStructure.TEMPORAL` —
    relative-vocab regex bank now; GLiNER2 date backstop (threshold 0.58)
    lands with the Gliner2Bank wrapper. Language policy: English-v0
    everywhere; word lists are versioned code; coverage gaps are handled by
    process (ratification audits, model-vs-bank disagreement mining), not
    speculative engines; multilingual is one phase-2 sweep across all
    tiers. This wave (all regex/stat, 2-pos/2-neg cases each): markers
    GREETING/POLITENESS/INTERJECTION/COMPARATIVE (closed lists; comparative
    = markers + irregulars only, no -er/-est suffix matching) and ACRONYM
    (cased shape + dotted); logical/ OPERATOR_SYNTAX (case-sensitive
    uppercase AND/OR/NOT) and TEMPORAL (relative vocabulary only — absolute
    forms stay with identifier DATETIME/BUSINESS_TEMPORAL); corruption/
    ENCODING_ARTIFACT (mojibake digraphs, U+FFFD; no word boundaries);
    metrics/ LENGTH and STOPWORD_RATIO as the first StatBanks (token-regex
    engines; stat banks are exempt from span-case enforcement and get value
    assertions instead). Test keying stays name-based until the first
    layered bank ships, then re-keys per class. Deferred from the wave: POS
    profile (spaCy dependency wave, decision 14), corrupted-identifiers
    (needs design against banks' claim data), Gliner2Bank wrapper (offset
    clamping + schema pin).

17. **Engine as a first-class bank attribute; one unified registry**
    (grill-me 2026-07-17). `Engine` StrEnum (`regex | gliner_model |
    spacy_model`) as a `ClassVar` on every bank — including stat banks
    (LengthBank's engine is its tokenizer regex; PosProfileBank's is
    spaCy). Engine bases fix it; filtering happens **before
    instantiation**, so selecting regex-only never imports torch/spaCy.
    `full_feature_banks()` dies: FEATURE_BANKS holds ALL banks, and
    `FeatureExtractor` gains `engines: Iterable[Engine] | None` —
    **default `(Engine.REGEX,)`** (deterministic, dependency-light,
    keeps profiling and tests fast), `None` = every engine (requires the
    `model` + `nlp` groups). Layering composes: dropping GLINER removes
    the temporal backstop, keeps the regex layer. Stage-1 close-out from
    the CSV gap audit: implement Morphology (lemma≠token inflected share)
    and Syntactic Depth (parse depth + clause count) as spaCy stat banks
    over ONE shared cached pipeline (tagger+parser+lemmatizer, ner
    disabled); defer PMI (blocking question: background co-occurrence
    source), char-typos (blocking: dictionary source), mono/multilingual
    (blocking: lang-id dependency + phase-2 multilingual), corruption
    degree (derived view per d8); corpus-relative five stay blocked on
    the queries-only registry decision. Case enforcement scopes to
    engine == REGEX span banks; model banks get live tests in
    model_banks_test.py.

18. **Multilingual strategy + language engine** (grill-me 2026-07-17).
    Three-axis design: (a) **invariant banks** — 79 identifier banks are
    mostly language-invariant already (UUIDs, IBANs, CVEs are international
    standards); a small audit flags the English-gated exceptions;
    (b) **UD-routed grammatical banks** — spaCy POS/morphology/depth banks
    route to per-language pipelines (`de_core_news_sm`, etc.) via the
    `languages` parameter, zero architecture change; (c) **language-identity
    features** (`LANGUAGE_SET` + `CODE_SWITCHING`) join the **SEMANTICAL**
    group as stat banks. Engine: **lingua-py** (Apache 2.0, 75 languages,
    Rust-backed v2, `compute_language_confidence_values()` returns ranked
    multi-language scores — the only candidate with multi-label output,
    which is the hard requirement for code-switching detection; spacy-fastlang
    disqualified: exposes only `doc._.language` single string, no multi-label
    API). fastText direct is the fallback if the 4 missing MIRACL languages
    (Yoruba, Telugu, Swahili, Farsi) become blocking. Memory guardrail: scope
    `LanguageDetectorBuilder.from_languages([...])` to the MIRACL 14 actually
    covered, not all 75. Missing languages return nullable per-row (same
    precedent as English-only spaCy banks). `LLMBank` reserved as last-resort
    engine — explicitly deferred, not this stage. Language parameter:
    `resolve(text, *, languages=["en"])` — caller-declares, not detected;
    banks declare `supported_languages: ClassVar[frozenset[str] | None]`.
    Routing architecture for per-language spaCy pipelines: see decision 19.

19. **Per-language pipeline routing** (grill-me 2026-07-17). The shared
    spaCy caches become language-keyed: `_pipeline(lang)` loads the pinned
    per-language model (`en_core_web_sm`, `de_core_news_sm`, ...);
    `_doc(text, lang)` caches by both. Banks declare
    `supported_languages: ClassVar[frozenset[str] | None]` (None =
    language-invariant); the extractor skips non-supporting banks for the
    requested `languages` and runs supporting banks once per requested
    language, merging results. Missing pipeline downloads **fail loudly**
    with the download command in the error — no silent fallback to the
    weaker `xx` multilingual model (silent degradation would corrupt
    profiles exactly where multilingual data is the point).
    `spacy.util.get_installed_models()` lets the extractor warn at init
    about unservable languages. Multilingual monolingual models rejected
    as primary (option b) on accuracy; silent fallback rejected (option c)
    on integrity.

## Deferred questions

- Concrete recipe values: total size, per-feature quotas, strata quotas,
  minimum natural share, harvest-target N.
- Judgment-shaped MODEL features (word-order sensitivity, syntactic depth,
  corruption degree...) — GLiNER2 classification head is the default
  candidate, but behavioral/perturbation designs may fit better; own
  session.
- Multilingual MODEL tier: GLiNER v1 `gliner_multi` vs future mDeBERTa
  GLiNER2 — revisit when non-English profiling matters.
- Corpus-relative features (IDF profile, vocabulary mismatch, ambiguity,
  specificity, answerability): requires explicitly reopening the registry's
  queries-only decision.
- Strategy labeling stage: empirical dense/sparse/hybrid labels via NDCG in
  `src/hybrid_search_rrf_dataset`.
- ILP escalation for quota conflicts (solver choice, formulation).
- Next acquisitions: ORCAS (needs `recommended_sample`), BRIGHT, CLERC,
  further BEIR subsets.
- MCP wrapper around `verify()` for interactive generation.
- Demo (b) infrastructure: corpus indexing + local Qdrant
  (docker-compose.yml exists) for the disagreement measurement.

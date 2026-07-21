# SPEC: Diversified Query Dataset Pipeline

Vocabulary: see CONTEXT.md. Session dates: 2026-07-15 → 2026-07-20 (each
decision carries its own date).

## Problem

Build a diversified query dataset for the Strategy Router (dense/sparse/hybrid)
by composing natural queries harvested from registered IR datasets with
generated/augmented ones — such that the full feature taxonomy
(`query_taxonomy/query-taxonomy.csv` in the sibling package) is well
represented, every row stays answerable against some corpus, and the dataset
can later be labeled empirically. Current focus is extractor quality and
breadth; strategy labeling is explicitly a later stage.

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
    tied GLiNER v1; measure-then-ship gated on a hand-audited 200-query
    smoke eval (50 each msmarco/trec-dl/nfcorpus/miracl-en). See outcome
    below for what shipped.

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
    generic family `GeneralBank[EngineT, OutT]` with `OutT` constrained to
    exactly `FeatureSpan | FeatureStat` (`RegexBank`/`StatBank`/`Gliner2Bank`
    as engine bases). Registry `FEATURE_BANKS: dict[FeatureGroup,
    tuple[type[GeneralBank], ...]]` is group-partitioned; `FeatureExtractor`
    runs one pass per text with within-group claim resolution, exposing
    `resolve(text, *, groups=...)` and `extract(queries, *, groups=...)`.
    Output: `QueryFeatures` (spans + stats sections nested by group),
    aggregated into `CorpusFeatures` with per-group `summary()`.

16. **Extractor fan-out wave + engine doctrine** (grill-me 2026-07-16).
    Three engines: RegexBank (edify — closed shapes/lists, precision-first),
    SpacyBank (grammatical signal), Gliner2Bank (context entities, audited
    labels only, pinned schema per d13). **A Query Feature may need more
    than one signal**: multiple banks may share one feature name — the
    within-group claim registry plus AmbiguityTier ordering arbitrate
    (deterministic engine claims first, model engine backstops at AMBIGUOUS).
    First layered feature: `LogicalStructure.TEMPORAL` — relative-vocab
    regex bank + GLiNER2 date backstop (threshold 0.58). Language policy:
    English-v0 everywhere; word lists are versioned code; coverage gaps are
    handled by process (ratification audits, model-vs-bank disagreement
    mining), not speculative engines; multilingual is one phase-2 sweep.

17. **Engine as a first-class bank attribute; one unified registry**
    (grill-me 2026-07-17). `Engine` StrEnum (`regex | gliner_model |
    spacy_model`) as a `ClassVar` on every bank — engine bases fix it, so
    `FeatureExtractor(engines=[Engine.REGEX])` filters **before
    instantiation** and never imports torch/spaCy. Default is
    `(Engine.REGEX,)` (deterministic, dependency-light); `None` selects
    every engine (requires the `model` group + downloaded spaCy model).
    Layering composes: dropping GLINER removes the temporal backstop, keeps
    the regex layer. Case-enforcement invariant scopes to engine == REGEX
    span banks; model banks get live tests. Stage-1 CSV gap close-out
    shipped Morphology + Syntactic Depth as spaCy stat banks over ONE
    shared cached pipeline; PMI, char-typos, multilingual,
    corruption-degree, and the five corpus-relative features tracked in
    TODOS.

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

20. **Logical group expansion + coordination metric** (grill-me 2026-07-20).
    OPERATOR_SYNTAX stays narrow: uppercase word operators (AND/OR/NOT)
    only. Research grounding: boolean operators appear in ≤10% of web
    queries (Spink et al. 2002; ~1% for advanced syntax, White & Morris
    2007) with 50% of AND uses erroneous (Jansen et al. 2000), but usage is
    markedly higher among specialized/developer audiences (Jones et al.
    2000 CSTR; DIALOG 36%) — exactly Qdrant's population. Rare-but-real:
    the recipe quotas it. Symbolic forms (`!=`, `<>`, `!x`) are NOT
    attested search dialect — in a real query they are evidence of
    **embedded formal content**, two new LogicalStructure members:
    CODE_FRAGMENT (programming-language grammar: compound symbolic
    operators `!=`/`<>`/`=>`/`->`/`::`/`&&`/`||`/`===`, call syntax
    `identifier(`, keyword-gated bigrams like `SELECT … FROM`) and
    MATH_EXPRESSION (equation grammar: operand-operator-operand runs,
    `=` flanked by expressions). "Formula" dissolves: spreadsheet → code,
    physics → math, chemical ids → ChemicalIdBank, bare chemical formulas
    (H2SO4) deliberately excluded (letter-digit shapes collide with
    SKUs/tickers; rationale in the enum docstring). The SMILES precedent
    ("a grammar, not a token format") now has a home: grammars go to
    logical, token formats to identifiers. **Cue-claiming doctrine**: the
    regex banks (both MODERATE) claim high-precision evidence tokens, not
    fragment boundaries — full-fragment segmentation is out of scope by
    design; bare single-char `=`/`+`/`-`/`<`/`>` are never claimed.
    Precision-first is structurally correct here: recipe harvesting is
    precision-sensitive (FPs poison strata) and recall-tolerant (the order
    sheet fills deficits via generation); recall arrives later as a model
    layer (TEMPORAL precedent). Lowercase and/or/comma coordination is NOT
    a span feature — new `StatisticalMetric.COORDINATION`, a SpacyBank
    beside SyntacticDepthBank (shared cached pipeline) emitting
    coordination_count, max_conjunct_width, clausal_coordination_count,
    nominal_coordination_count (the parser separates the two "and"s:
    conj arcs between verbs = clausal, between nouns = enumeration;
    comma-coordination comes free). Not folded into SYNTACTIC_DEPTH: depth
    = nesting (subordination), coordination = breadth (parataxis) —
    orthogonal axes, separately quotable strata. Wave scope: 2 enum
    members + 1 metric member, 2 CSV rows + 1, three banks, registry
    entries, 2-pos/2-neg cases each (value assertions for the stat bank).

21. **Dataset acquisition catalog + wave-1 implementation** (2026-07-20).
    The candidate catalog lives in `docs/datasets.md`: card tuples along the
    six DatasetCard dimensions, verified acquisition pointers, and harvest
    hypotheses (priors for decision 7's profiling-proposes/human-ratifies
    loop). Wave 1 = ORCAS, BRIGHT, QUEST, CRUMB, RAR-b math/code pools,
    LIMIT, DBPedia-entity. Implementation plan:
    - New `DatasetName` members: `ORCAS`, `BRIGHT_<split>` (the 3
      taxonomy-relevant splits first: `leetcode`, `aops`,
      `theoremqa_questions`; the other 9 later), `QUEST`, `CRUMB_<task>` ×8,
      `RARB_MATH`, `RARB_CODE`, `LIMIT`, `DBPEDIA_ENTITY`. Parameterized
      classes (MiraclDev pattern) for BRIGHT/CRUMB keep it one class per
      source.
    - irds one-liners (ORCAS, DBPedia, ANTIQUE): subclass
      `IRDatasetsBacked`, set `irds_id`, write the card (~15 lines each).
    - hf fetchers: `load_dataset(repo, config, split=..., streaming=True)`,
      yield `Query(str(id), text)`; field names verified at registration
      (CRUMB and RAR-b schemas unconfirmed).
    - url fetchers (LIMIT, XOR-TyDi): `load_dataset("json",
      data_files=<raw url>, streaming=True)` — reuses the HF machinery;
      cards get `SourceKind.URL`, no new base class.
    - Sample caps (proposals, ratified with the first profile run):
      ORCAS 100K, GooAQ 50K, WebFAQ 50K/language.
    - Every registration ends with `registry.profile()` + harvest-target
      ratification (decision 7); the catalog's hypotheses are the priors.
    LIMIT registers but is excluded from harvest targets (its value is the
    strategy-labeling stage); MEMERAG is not a query source (MIRACL's
    queries, already registered). Open ratifications: MIRACL
    `llm_target`/`non_trivial` card vs the candidate-list coding;
    DBPedia scope G-vs-S.

22. **`-Like` suffix on non-RIGID span banks** (grill-me 2026-07-20,
    triggered by first-profile results). Rename every non-RIGID span bank
    with a `-Like` suffix — 7 regex banks (AMBIGUOUS: `StockTickerBank`,
    `BookingReferenceBank`; MODERATE: `DerivativesSymbolBank`, `TicketBank`,
    `GameNotationBank`, `AircraftVesselRegBank`, `ErrorCodeBank`) and 4
    MODEL banks (`PersonBank`, `LocationBank`, `ProperNounBank`,
    `Gliner2TemporalBank`). Cascade: class name → enum member in
    `StructuralIdentifier` / `LogicalStructure` / entity enum → emitted
    string in `SpanProfile`. RIGID regex banks and the regex `TemporalBank`
    unchanged. Motivation: profile results on scientific corpora — scifact
    fires `stock_ticker` on 29% of queries (DNA/TCR/PPAR — 100% gene names);
    MODERATE banks misfire at the same rate under domain shift; GLiNER
    audited precisions of 0.84–0.87 don't hold on real corpora (lowercased
    web queries, short medical titles). Combined with the taxonomy CSV's
    stated purpose for Structured Identifiers ("Helps to determine
    sparseness"), the non-RIGID banks are shape-guessers, not class-claimers
    — the emitted identifier should say so. Behavioral change: none (same
    regex, same claim range, same tier); test surface survives
    parametrization. Consumers filter by suffix or by tier to reject
    shape-guessed evidence when they need certified matches only.

23. **Profiling decouples from grounded snapshots** (grill-me 2026-07-20).
    `src/profile_datasets.py` reads full source query sets directly —
    `dataset._test_ds.queries_iter()` for TrecDL2022; existing
    `dataset.queries()` for NFCorpus/SciFact — no snapshot materialization
    before profiling. Motivation: the previous coupling cost 424/500 queries
    on TREC-DL 2022 (materialize streams the 138M-passage MSMARCO v2 corpus,
    keeps the first 30 000 judged docs by iteration order, drops queries
    whose qrels lose all supporting docs). Two-mode split: **mining** wants
    the full source distribution (this decision); **labeling** wants the
    grounded snapshot for per-query NDCG correlation (deferred). Snapshot
    machinery is unchanged for retrieval eval. Docstring in
    `profile_datasets.py` updated: profiles now characterize the SOURCE,
    not the labeling snapshot.

24. **POS profile schema legibility** (grill-me 2026-07-20). Three
    mechanical changes: (a) `PosProfileBank.compute` emits a new stat
    `residual_share` = `(count[PUNCT] + count[SYM] + count[X]) / total`, so
    `open_class_share + closed_class_share + residual_share ≈ 1` holds as a
    consumer-checkable invariant; (b) `LengthBank` renames emitted stat
    `length_tokens` → `length_words` — the count uses regex `\w+`
    tokenization (not spaCy's; differs by ~7% on nfcorpus), and the name
    now says so; (c) `PosProfileBank` renames every `pos_<tag>` →
    `pos_count_<tag>` (17 renames) so raw counts are lexically distinct
    from `_share` rates and `_presence` binaries in the same section. New
    CONTEXT.md entry documents the suffix convention (see Stat suffix
    convention). No new stats, no engine dependency change, regex-only
    install path unchanged.

25. **Cross-group co-firing is by design** (grill-me 2026-07-20). Same-token
    spans emitted by banks in *different* FeatureGroups — e.g. `acronym`
    from sentence_markers and `stock_ticker_like` from structured_identifiers
    both claiming `DNA` on scifact — are parallel independent layers per
    d15/d16, not double-counting bugs. Downstream reads them as evidence
    about the same token from two axes; the d22 `-Like` rename makes
    interpretation self-honest at the emit boundary (RIGID acronym +
    assumptive stock-ticker-like ≠ two independent facts). Documentation
    change only: CONTEXT.md's Layered banks entry updated to note that
    cross-group co-fires are expected. No code change.

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
- Model backstop for CODE_FRAGMENT/MATH_EXPRESSION recall (symbol-light
  formal content: "x squared plus y squared", prose pseudo-code) — layered
  bank; needs a code/math detection model choice (d20).
- Attested search-syntax extensions to OPERATOR_SYNTAX (quoted phrases,
  minus-exclusion, `site:`) — attested in query logs but precision-dangerous;
  own decision (d20).

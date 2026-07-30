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
    (2026-07-15; adopted label-scoped 2026-07-16 after a hand-audited
    200-query smoke eval; **dropped 2026-07-21**, same push as the d26
    prune: `entities/` deleted, `Engine` = regex | spacy only — GLiNER ran
    ~600× slower than regex and its entity spans answered no router
    question; profiles, demo, and benchmarks regenerated without it).
    What survives: the smoke-eval method (stratified sample, hand-audited
    precision, lowercase parity, offset integrity) remains the gate for
    any future MODEL-tier engine; closed-list markers scored 0 zero-shot —
    banks own them, now with data. Eval details (per-label thresholds,
    acronym rejection, the schema-composition determinism pin) live in
    this entry's git history.

14. **POS profile = ALGO feature via pinned spaCy tagger** (grill-me
    2026-07-16). Shipped as the tagger behind what d26 renamed
    `natural_language_share`; the 17-tag histogram and its derived views
    were retired by d26 (recompute from the shared spaCy doc if ever
    needed). Still live: stopword/function-word ratio is the REGEX
    fallback of `natural_language_share` (non-English, minimal installs;
    recipe quotas reference the spaCy scalar only); POS fields nullable
    per-row (English-only); and the domain-shift guardrail (taggers
    degrade on keyword telegrams) — bank-style fixed cases + a one-off
    `natural_language_share`↔stopword-ratio correlation diagnostic over
    the cached datasets (investigate if r < 0.8), still an open TODO gate.

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

24. **POS profile schema legibility** (grill-me 2026-07-20). Surviving
    piece: `LengthBank` emits `length_words` (regex `\w+` tokenization,
    not spaCy's; differs by ~7% on nfcorpus — the name says so). Parts (a)
    and (c) were retired with the POS histogram by d26. The Stat suffix
    convention (CONTEXT.md) stands.

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

26. **Metrics prune to the four router signals** (2026-07-21). The
    statistical-metrics group emits exactly the scalars that answer a
    question the router cares about — four signals, seven scalars:
    NL-shape (`natural_language_signal.natural_language_share`; REGEX
    fallback `stopword_ratio.stopword_ratio`), word variation
    (`morphology.word_variation_share`), structure
    (`syntactic_depth.nesting_depth` + `statement_count`), size
    (`length.length_words` + `length_chars`). Dropped: the 17
    `pos_count_*` histogram stats, `open_class_share`, `residual_share`,
    `noun_share`, `verb_presence`, `propn_share`, `inflected_count`,
    `stopword_count` — none answered a router question (`propn_share`
    duplicated the proper-noun span axis; counts are share × length).
    `StatisticalMetric.POS_PROFILE` renamed `NATURAL_LANGUAGE_SIGNAL`
    (single-stat bank; "profile" over-promised). Amends d14: the histogram
    and derived views are no longer stored — recompute from the shared
    spaCy doc if a corpus study ever needs them. Retires d24(a)/(c): the
    residual invariant and `pos_count_` renames existed to make the
    histogram legible, moot once it's gone (d24(b) `length_words` stands).
    Doctrine shipped with the prune: **signals are coordinates and
    acceptance filters, never label sources** — strata are boxes in signal
    space, synthetic queries are rejection-sampled against target
    signatures, and strategy labels always come from retrieval outcomes
    (labeling by signal would teach the router our heuristic back, and it
    could then never beat the production hard classifier). Joint-reading
    caveat: the parser hallucinates structure on non-sentences (the CVE
    telegram out-depths the cats question), so `nesting_depth` is
    meaningful only conditional on `natural_language_share` indicating
    natural language — the signals are one panel, not four independent
    columns.
    sanity-check 2026-07-21: stat names de-jargoned to meaning-first
    (`closed_class_share` → `natural_language_share`, `inflected_share` →
    `word_variation_share`, `parse_depth` → `nesting_depth`,
    `clause_count` → `statement_count`); scale-suffix convention kept, the
    computing mechanism lives in bank docstrings and `metrics/config.py`
    comments. Revisit if dataset columns must match an external
    NLP-standard vocabulary.
    Amendment (2026-07-21, d20 reconciliation): d20's COORDINATION ships
    as the FIFTH signal, pruned to ONE scalar by this decision's own
    doctrine — `widest_list_size` (de-jargoned from d20's
    `max_conjunct_width`): how many equal parts the longest and/or/comma
    chain strings together, via conj-arc chains. It answers a router
    question the four signals cannot: wide-but-flat enumerations are
    structure `nesting_depth` does not see, and enumeration-heavy strata
    are separately quotable (d20). d20's `coordination_count` (a raw
    count — share × length) and the clausal/nominal split (a second
    scalar per question) are not emitted. Five signals, eight scalars;
    the joint-reading caveat applies (parser output, condition on
    `natural_language_share`).

27. **Presentation layer: `CorpusReport` + domain rollup** (grill-me
    2026-07-21). New `reporting.py` in query_taxonomy: `CorpusReport`
    consumes `CorpusFeatures` and owns `.text()` (human-readable rewrite)
    plus chart-ready rollup data — stdlib only; matplotlib drawing lives
    with the consumer (parent repo, which already carries it).
    `CorpusFeatures.summary()` and its `__str__` are deleted, not wrapped.
    Headline chart: two-ring donut — inner ring the 8 `Domain`s sized by
    span mass (disjoint after claim resolution → honest parts-of-whole;
    `general` exploded into its member banks, a grab-bag slice explains
    nothing), outer ring each domain split certified vs `-like` so the
    Assumptive-bank doctrine survives into the viz. Query-share appears as
    a companion bar where router framing needs it (it does not sum to 100%,
    so it never gets a circle). Rollup is presentation-only: profiles JSON
    and audit surfaces stay bank-level (d7 ratification needs per-bank FP
    checks). — *82 banks as a list is unreadable; 8 domains with honesty
    stripes is one glance.*
    arch-validator 2026-07-21: matplotlib KEEP (HIGH) — GitHub's ipynb
    viewer strips JS, so plotly/altair would render blank in demo.ipynb;
    spider = ~30 lines of polar-projection DIY, no new dep. Revisit if the
    demo moves to a hosted page wanting interactivity.
    arch-validator 2026-07-21 (design): donut KEEP (MEDIUM) with a binding
    slice budget — total ≤10 slices; `general` explodes into top-3 banks +
    one `general·other`, never more (perception research holds pies to
    5–10 slices; part-whole estimation is where pies match bars). Fallback
    per chart instance: sorted stacked bars (certified/-like segments) if
    the story hides inside `general·other`.

28. **Dataset fingerprints: heatmap catalog view + spider comparison view**
    (grill-me 2026-07-21). Both are views over the existing
    `data/profiles/*.json` (d9 seeded sampling already gives unbiased
    shape — no new sampling design). Catalog view: one heatmap, rows =
    datasets, columns = 8 domain query-shares + 7 stat means, color
    normalized per column, raw value printed in each cell — scales to the
    full docs/datasets.md catalog. Comparison view: spider overlay for 2–4
    hand-picked datasets in the demo, the only regime where radar is
    readable (axis order is arbitrary and enclosed area exaggerates —
    never overlay the catalog). — *One matrix answers "which dataset is
    rich in what"; the spider keeps the storytelling moment.*
    arch-validator 2026-07-21 (design): KEEP (HIGH) — matches published
    radar guidance (5–8 axes, ≤4 overlays, shape-as-story; heatmap for
    many×many). Spider axis order must be a fixed global constant —
    reordering spokes changes the perceived shape.

29. **Composition mechanics: feature table + capped harvest-priority fill**
    (grill-me 2026-07-21). Confirms d6's greedy quota-fill; the per-dataset
    "optimization cycle until best subset" idea stays rejected (d6/d7 —
    ILP remains the escalation path, not the default). Greedy runs on a
    materialized feature table: one parquet of (dataset, query_id,
    per-bank span counts, stat scalars) from a single full extraction pass
    per dataset; selection reads the table, so nothing unneeded is ever
    taken (no prune phase) and recipe tweaks re-run selection without
    re-paying extraction (~2h spaCy for ORCAS-scale). The table doubles as
    the labeling-stage substrate and the audit trail. Fill order: each
    quota fills from its d7 harvest-target ranking with a per-quota
    per-dataset cap (default ≤50%, spill to next-ranked; waived when only
    one source carries the feature) so no quota becomes a single-dataset
    monoculture the router could learn as a register proxy. Cap value is a
    recipe value (deferred with the rest). — *Extraction is the expensive
    leg; selection must stay cheap to re-run.*
    arch-validator 2026-07-21: pyarrow+pandas KEEP (HIGH) — extraction
    (spaCy ~2h) dominates; selection at 10M rows is seconds in pandas and
    parquet keeps an engine swap free. Revisit (duckdb over the same
    files) if the concatenated catalog passes ~50M rows or fill logic
    turns relational.
    arch-validator 2026-07-21 (design): global greedy KEEP (HIGH) — quota
    coverage Σ min(count, quota) is monotone submodular, so greedy carries
    the classic Nemhauser (1−1/e) guarantee (caps: since quota sets
    overlap, per-quota source caps form an *intersection* of partition
    matroids — greedy's constant relaxes to ~1/(p+1); the monotone +
    diminishing-returns structure is unaffected); per-dataset
    optimization cycles score structurally worse (blind
    subproblems + a reconciliation pass that reinvents global greedy).
    ILP escalation trigger stays: greedy terminating with unfilled quotas
    despite available rows.

30. **Composition doctrine: floors, weakest-first fill, checkability,
    dark matter** (grill-me 2026-07-21). Amends d29's fill mechanics; the
    feature table, source caps, and harvest priority stand.
    (a) *Recipe numbers are amounts, not proportions.* Quotas become
    per-cell floors ("≥ T_c rows"), sized by the precision rule (cell
    score trustworthy to ~±1/√n: 400 rows ≈ ±5 points; labeling budget ≈
    cells × floor, so cells stay coarse). Representativeness is an
    eval-time weighting: score cells separately, weight by a workload's
    proportions — a real log's cell histogram when available (page-search
    logs are the acquisition to chase), several hypothesized mixes or the
    worst cell otherwise. Selection never claims to match real traffic:
    that claim is untestable in-house (the ORCAS-anchor trap) and stays a
    swappable input. Per-cell facts ("sparse wins UUID cells") are
    workload-invariant; only the headline aggregate needs proportions.
    (b) *Weakest-first (maximin) fill.* Each round feeds the cell with
    the lowest fill/floor ratio: removes quota-order dependence, balanced
    coverage at any budget cut, starved cell = exact per-cell conflict
    signal feeding the order sheet. Feasible case terminates identically
    to d29 greedy. The plain loop is a heuristic (min of submodular isn't
    submodular); SATURATE is the named fallback, as ILP is for quotas.
    Pilot: A/B both fill orders over the same feature table.
    (c) *Checkable-first.* A row counts toward a floor only if gradeable:
    ≥1 judged doc, or doc_grounded/synthetic (answerable by construction,
    d5). Ungradeable rows bounce back for replacement; exhausted cells go
    to the order sheet. Replacement may filter on "can't check", never on
    "didn't like the grade": ties and all-fail rows stay, flagged as
    their own stratum — "no strategy works" is routing information (d26
    doctrine, feedback-loop edition).
    (d) *Dark matter (of data).* Checkable-natural rows are visible
    matter — curated, well-formed, judged-by-pooling. Two blind spots:
    unjudged queries (the messy tail never enters graded benchmarks) and
    qrel holes inside judged rows (a strategy retrieving a relevant-but-
    unjudged doc scores zero, biasing labels toward pool-contributor-era
    systems). Counterweights: generation doubles as the dark-matter probe
    (grounded rows carry complete answer sheets and can target exactly
    the ugly signatures no benchmark judges); the minimum natural share
    keeps real texture; the two provenances' biases point in opposite
    directions by design. Cells where strategies retrieve many unjudged
    docs get a low-trust flag; LLM-as-judge over unjudged retrievals is
    the deferred mitigation (MEMERAG's lane). — *No source is
    representative and none can be; every bias gets named and paired
    against an opposite one.*
    Register/box definitions (coarse workload cells for eval weighting)
    deferred until a real log can inform them.
    arch-validator 2026-07-21 (assumption audit): stack KEEP (MEDIUM until
    the hole pilot); labeling stage gains three gates before labels are
    trusted. R1 — qrel-hole asymmetry is documented, not hypothetical
    (BEIR Hole@10: BM25 ≈6.4% vs dense 14.4–31.8%; post-hoc judging lifts
    dense nDCG most), so the d30d low-trust flag becomes a quantitative
    gate: per-strategy Hole@10 per cell, gap over threshold blocks labels
    until post-hoc judging (LLM-as-judge, MEMERAG-calibrated). R2 —
    labels are stack-relative: pin the label schema tuple (dense_model,
    sparse_model, fusion, k, depth) in the artifact; explicit tie margin
    ε; two-dense-model kappa pilot on ~500 rows. R3 — cell scores are not
    corpus-invariant: corpus id is a labeling covariate (not leakage —
    the runtime router knows its corpus); high cross-corpus cell variance
    un-defers corpus-relative features. R4 — floor sizing gets a design-
    effect correction from cluster-robust SEs in the pilot. R5 — dark-
    matter generation imposes mess programmatically (corruption operators
    post-generation + d26 rejection sampling), never by prompting for
    messiness.

31. **First profile-at-scale readout: bimodal sources, equal-weight
    percentile scale, harvest priorities** (grill-me 2026-07-22, over the
    21-dataset catalog, 225K rows). The fingerprint run shows bimodal
    sources: register-realistic but feature-poor (msmarco/orcas/trec-dl/
    dbpedia/limit: 3–7 words, near-zero identifier rates) vs feature-rich
    but register-alien (BRIGHT/CRUMB/RAR-b: 26–253-word problem
    statements, entirely outside the production router's 0–9-token rule
    band). Read as CONFIRMATION of d30d ("no source is representative"),
    NOT as a source ranking — dropping either cluster re-creates the
    ORCAS-anchor trap from one side or the other. Rates are not counts:
    orcas at 0.05 identifier share × 10.4M ≈ 500K natural feature-bearing
    rows, more than all specialized sources combined (~15K queries) —
    ORCAS-tail harvest is the named top source for short × feature-rich
    cells. That region is EMPTY in every public source; the coverage
    chart's empty cells are the generation lane's first concrete order
    sheet. Cluster-B strategy labels lean on R3's corpus covariate
    (register far outside production traffic) — recorded, no new
    machinery. Presentation scale: cross-dataset comparisons use the
    **equal-weight percentile scale** (see CONTEXT.md): reference
    distribution weights every dataset equally (raw catalog pools are
    ~90% msmarco+orcas), dataset value = median percentile — fixes both
    outlier squash (one 252-word source flattening the length axis) and
    pool dominance. New *coverage view* (third fingerprint chart):
    per-query cell counts over the d29 catalog (length × NL-share bins,
    equal-weight), dominant dataset annotated per cell. Legal-domain
    finding: zero legal-bank fires across all 225K rows — legal REGISTER
    (crumb-legal-qa, NL-share 0.48) carries no citation-shaped
    identifiers; all-zero domain columns must render as honest zeros, not
    NaN stripes (chart guard, not data fix). Whether composition cell
    boundaries live on the percentile scale or raw scalars is deferred
    with the recipe values.

32. **Target composition: 50K, macro-split 60/20/20, audit lenses**
    (dataset-audit session 2026-07-22, ratified over
    `src/dataset_audit.ipynb`). Sets the deferred macro recipe values;
    per-cell floor sizes stay deferred (d30a precision rule).
    (a) *The split.* 50K total. 60% (30K) span-evidence rows — queries
    carrying ≥1 span of ANY group (identifiers, markers, logical).
    Inside that slice, 80/20: 80% (24K) allocated against span-type
    targets (per certified identifier domain, per marker type, per
    logical type — the audit's floor-supply columns); 20% (6K)
    entity-carrying rows drawn with NO preference (no mass bias, no type
    targeting) so natural single-span queries keep their share against
    archetypes. 20% (10K) statistical strata: zero-span rows spread over
    the scalar-signal bands (length, NL-share, nesting depth — the
    coverage-grid x-axes), extremes included. 20% (10K) dark forest:
    feature-BLIND uniform draws from ≥3 generalist champions, no
    champion >50% of the slice (candidates: orcas, msmarco, one long-NL
    source) — insurance against the taxonomy's own blind spots,
    deliberately not conditioned on any extractor output.
    (b) *Feasibility facts (audit 2026-07-22).* The catalog holds ~32K
    span-carrying rows against the 30K entity slice — no selection
    slack: the slice REQUIRES the ORCAS full-cache regex harvest
    (~16 min at regex rates; observed supply scales ×104, d31) plus the
    generation order sheet for thin cells. The statistical slice is thin
    at the extremes (60+ words × 0 spans = 288 rows catalog-wide).
    (c) *Certified-only domain floors; shape-guesses pool.* Span-type
    targets count certified banks only. `*_like` spans are wrong about
    the domain, right about identifier-ness (the SciFact gene-symbol
    audit) → they pool into one domain-agnostic shape_guess target that
    supplies sparse-affinity needs but never domain floors. Weighting
    them down was rejected: the error is in the label, not the
    magnitude.
    (d) *Mass ≠ supply.* Floors count queries (≥1 qualifying span),
    never span mass — 115 of crumb-stack-exchange's 117 datetime spans
    sit in ONE query. Mass preference inside the entity slice is
    explicitly rejected: span-dense rows are the easy, low-information
    case for the router (obvious BM25).
    (e) *Dataset value is target-relative; three lenses.* Winner-take-all
    equal-weight dominance (d31 coverage cells) structurally zeroes
    generalists — msmarco dominates 0 cells yet is runner-up supply
    nearly everywhere and the conversational-marker champion (politeness
    933). Keep/leave verdicts (recorded in the audit notebook,
    notebook-only for now) read three lenses: contribution to the 50K
    fill, sole-supplier criticality, span-type supply. Narrow ≠ leave;
    redundant across all three lenses = leave.
    — *Size against the target, not the catalog: debias by dataset size
    when characterizing, allocate by the 50K when selecting.*

33. **Fill recipe: evidence floors, raw bands, B-first order** (grill-me
    2026-07-22, over the d32 fill plan; d29/d30 greedy mechanics and their
    arch-validator verdicts stand unchanged).
    (a) *Floors are evidence amounts in weight currency.* A selected row
    credits a span-type floor at the ambiguity discount of its most-rigid
    qualifying bank (RIGID 1.0, MODERATE 0.75, AMBIGUOUS/shape-guess 0.5 —
    provisional recipe values) while still costing 1 row of budget.
    Floors are sized BELOW budget (19 floors × 1,000 weight under the
    24K-row slice) so discount inflation is absorbed by slack — an
    order-sheet shortfall therefore always means supply ran out (reason:
    exhausted vs capped), never that the arithmetic was infeasible.
    Effective-sample-size logic: guesses are worth less evidence per row.
    (b) *Composition cells live on raw scalar bands.* Resolves d31's
    deferred question: floor boundaries are raw units (3-6 words,
    NL-share 0.4+, nesting 2-3 — the coverage-grid bands) because
    composition needs stable, generation-targetable coordinates that do
    not move when the catalog grows. The equal-weight percentile scale
    stays presentation-only.
    (c) *Fill order B → A → C → D.* The no-preference sub-slice draws
    FIRST (one seeded uniform sample over the whole span pool): drawn
    after A it degenerates into orcas leftovers (A's qrels lane consumes
    all ~19.3K checkable span rows), defeating d32(a)'s "natural queries
    keep their share" rationale. A fills its floors from the remainder;
    C (zero-span pool at 6× its budget) and D (feature-blind) are
    order-insensitive and run last with global dedup.
    (d) *Two-tier label lanes.* Every slice fills checkable rows first;
    QC rows top up flagged `label_lane="deferred"` (clicks / LLM-as-judge
    later) vs `"qrels"`. Dark-forest champions: orcas/msmarco/
    crumb-legal-qa ≈ 50/30/20 within the ≤50% cap. `checkable` stays the
    card-level proxy until d30's per-row column lands.
    Artifact: `src/composition/` package (recipe / floors / fill /
    slices / compose) + `src/data/composition/{selection,order_sheet}`
    parquets + summary; selection schema (dataset, query_id, query,
    slice, floors, checkable, label_lane) feeds the labeling stage's
    FusionRow builders. — *The order sheet is a purchase order, not an
    error log.*

34. **taxonomy-generators: the generation twin package** (grill-me
    2026-07-23). New package `taxonomy-generators` (module
    `taxonomy_generators`, starts as `src/taxonomy_generators/`, extracted
    to a sibling repo when stable — the query-taxonomy precedent). Both
    packages implement the same taxonomy: query-taxonomy detects,
    taxonomy-generators produces.
    (a) *Boundary: dumb surfaces + verify + tool layer; no orchestrator.*
    The package emits grounding-blind feature surfaces (a valid UUID, a
    politeness phrase), wraps the extractor as `verify(text, targets)`
    (d10's library-first verb), and exposes both to LLMs; the calling
    LLM's own agentic loop does the enrichment (weaving surfaces into
    queries). d5's free-floating ban is enforced upstream in the parent
    repo's generation lane, never in this API; doc consistency for
    augmented rows is a separate deferred layer.
    (b) *Mechanism: auto-sample bank patterns; overrides for realism.*
    Default generator per regex span bank = reverse-regex sampling over
    the bank's guard-stripped compiled pattern, so ~90 features get
    generators for free and a bank pattern change flows into its
    generator automatically. Hand-written override classes shadow the
    default under the same feature name where gibberish hurts (ticket
    prefixes, plausible years). Lock-step is enforced by the round-trip
    test: every registered generator, sampled seeded N times, must have
    each surface claimed by its twin bank under the same emitted name.
    (c) *Dependency direction.* taxonomy-generators depends on
    query-taxonomy (versioned dependency); detection stays unaware
    generation exists; no taxonomy-core third package — one taxonomy.py.
    (d) *Coverage.* All regex span banks: certified identifiers, -Like
    banks (shape-guess surfaces supply d32c's shape_guess pool), markers
    (closed phrase lists), logical cue tokens. The five signals are
    verify-only acceptance filters (d26 doctrine) — never generated.
    Corruption operators deferred.
    (e) *Tool surface: parameterized trio.* `list_features()` /
    `generate_surface(feature, n)` / `verify(text, targets)` as a
    framework-agnostic registry (name + description + JSON schema +
    callable) plus an MCP server entry point behind an optional
    dependency group — closes d10's deferred MCP wrapper. One tool per
    feature (~90) rejected: blows agent tool budgets; the catalog lives
    in `list_features()`.
    (f) *Shape.* `SurfaceGenerator` ABC mirroring the bank family
    (feature name, group, `sample(rng, n)`); `PatternGenerator` defaults
    auto-built by iterating FEATURE_BANKS at import; group-keyed registry
    like FEATURE_BANKS; seeded `random.Random` injection end-to-end.
    — *Same taxonomy, two directions: detection certifies what text is;
    generation supplies text that detection will certify.*
    arch-validator 2026-07-23: rstr KEEP (HIGH) — BSD, stdlib-only,
    injectable seeded Random, walks the same `re._parser` tree the banks
    compile to; exrex disqualified outright (AGPL); hypothesis is a test
    framework misused at runtime; DIY re-implements rstr. Revisit if
    bank patterns outgrow rstr's construct support (its crude
    \b/lookaround handling is absorbed by guard-stripping + the
    round-trip test).
    sanity-check 2026-07-23: KEEP — the only custom machinery
    (auto-registry + round-trip test) is exactly what the lock-step and
    tool-surface requirements demand; sampling is delegated to rstr,
    verification to query-taxonomy. Flips if interactive LLM enrichment
    is abandoned for a one-off batch script.

35. **R1 reframed: LLM-judge calibration pilot** (grill-me 2026-07-27,
    supersedes d30's R1 hole-diagnostic framing). **SUPERSEDED by d37,
    2026-07-28 — full record in docs/adr/0001.** 4-level TREC-DL grading
    (500 stratified pairs, Haiku 4.5 vs Sonnet 4.5) against a kappa ≥ 0.6
    + per-level-accuracy ≥ 70% gate: failed on measured evidence (kappa
    0.120–0.139, bootstrap CI upper bound 0.181 — out of reach at any
    sample size). Load-bearing premise error: orcas was called "no
    qrels", but ORCAS ships 18.8M click pairs mapped onto msmarco-document
    doc_ids — positive-only qrels, not unlabelable. `judge.py` and the
    pilot notebook deleted (cae28fc); `data/r1_pilot/` is the audit trail.

36. **R1 prompt-iteration escalation: hand-crafted few-shot** (grill-me
    2026-07-27, follows d35's escalation clause). **SUPERSEDED by d37,
    2026-07-28 — full record in docs/adr/0001.** Ran at n=500: 12
    hand-crafted exemplars (3 source-style queries × 4 grades) moved
    kappa by ±0.02 with fully overlapping bootstrap CIs (Haiku 0.348 →
    0.328, Sonnet 0.320 → 0.340 at grade≥2), so exemplar quality was
    never the binding constraint — the defect was measuring per-document
    grade agreement when the pipeline consumes a per-query route
    decision. The real-exemplar ablation is dropped, not deferred.

37. **Golden set: route labels from retrieval outcomes, not from an LLM's
    opinion** (grill-me 2026-07-28, supersedes d35 and d36; amends d30's
    "which strategy wins NDCG" label rule). Two orthogonal tasks are now
    named: *dataset improvement* (more features) and *golden set* (target
    classes). This decision covers the golden set.
    (a) *The objective is `0.7·HitRate@1 + 0.3·NDCG@10`*, with a
    per-dataset `min_relevance` binarizing graded qrels (2 for TREC-DL's
    0–3 scale, where grade 1 is "related but does not answer"; 1 for
    qrels already binary). Bare NDCG@10 was wrong not because it is
    uncomputable on thin qrels but because it has no top-1 primacy, and
    production cares about the top hit.
    (b) *The weights make it lexicographic, not a blend.* While
    hit_weight > ndcg_weight the two score ranges are disjoint (rank-1 hit
    ⇒ ≥0.700, miss ⇒ ≤0.300), so the secondary term can only discriminate
    *within* each group. Per-query values with one relevant doc: rank 1 →
    1.000, rank 2 → 0.189, rank 3 → 0.150, rank 10 → 0.087.
    (c) *NDCG@10 is the tie-breaker because the cheaper candidates go
    blind on data we hold.* MRR@10 is 1.0 for every route with a relevant
    rank-1 doc, so it cannot separate a route that surfaced 1 of 4
    relevant docs from one that surfaced 4 of 4 — the label would fall
    through to tie-break order. Recall@10 collapses to two values when a
    query has a single relevant doc (rank 2 and rank 10 score alike),
    which is the majority of the corpus (msmarco-dev ~1.06 judged/query,
    rarb, most crumb, orcas). Verified: with exactly one relevant doc all
    three variants are strictly decreasing in its rank, so they agree on
    the route — divergence only exists where a query has ≥2 relevant docs.
    (d) *Judgments leave the result row.* `gold_qrel` as a per-row dict
    became a parquet struct with one field per distinct doc_id in the
    file — measured at 30,000 fields for 76 rows, so it does not survive
    composition scale. Replaced by `QrelStore`: long-format
    `(dataset, query_id, doc_id, relevance, source)`. `source` ∈
    {`human`, `click`, `llm`} with conflict priority human > click > llm,
    so scoring a dataset against either lane is a filter, not a second
    pipeline.
    (e) *Per-route results are persisted.* `GoldenRoutingDataset` carries
    `route_scores` and `route_rankings` keyed by strategy name (three
    stable keys — parquet encodes a fixed struct; keying by doc_id would
    not). Any cutoff-≤10 metric, latency margin, or tie rule is therefore
    re-derivable with zero retrieval. This is what makes (a) and (c)
    reversible decisions rather than one-way doors.
    (f) *An LLM cannot be asked which route wins.* Dense-vs-sparse is a
    property of the (query, corpus, index) triple, not the query: for
    nfcorpus's `"DHA"`, sparse wins if the relevant docs say "DHA" and
    dense wins if they say "docosahexaenoic acid", and the model sees the
    same three characters either way. IDF is a corpus statistic. This is
    an information gap, not a capacity gap — a stronger model shares the
    blind spot. The LLM's only legitimate job is
    `(query, doc) → relevant`, a function of its actual inputs.
    (g) *Measured: the production classifier underperforms a constant.*
    On `data/golden_router` vs `data/router_llm` (trec-dl-2022, 76
    queries, NDCG@10): route agreement 43% (33/76, measured) against 82%
    (62/76) for always-picking `dense_only` — the latter *derived* as the
    majority class of the oracle's own labels, not a measured system.
    Regret +0.077 mean, +0.252 p90; the classifier picks `pure_rrf` 61%
    of the time where the oracle wants 14%, which explains hybrid's poor
    top-1. **Does not generalize**: n=76, one dataset, and that corpus is
    `TrecDL2022(30000).materialize()` — 30K docs that are almost all
    judged answers, which inflates dense and starves sparse. Motivates
    re-measuring on a realistic index; proves nothing on its own.
    (h) *The bar is the best constant route, not random.* Constant-dense
    and constant-sparse baselines must appear in every comparison, or a
    router can look respectable on regret while losing to one line of
    code. Only constant-hybrid (`data/pure_rrf`) exists today.
    (i) *Three outcome shapes; two are usable.* All routes tied above
    zero ⇒ equivalent, send to the cheapest (signal for the speed
    requirement). Routes differ ⇒ the quality signal. All routes 0.0 ⇒
    unanswerable, and **no valid label exists** — currently the argmax
    falls through to whichever route is first in the list, fabricating a
    `dense_only` label (1 of 76 on trec-dl; the rate scales with corpus
    realism and with judging orcas only to depth 10). Unanswerable
    queries need an explicit outcome. Ties anywhere must resolve by a
    deliberate rule, noting `dense_only` is likely *not* the cheapest
    route since BM25 needs no query-side transformer pass.
    (j) *The 50K is a candidate pool, not a training set.* Usable yield
    is unknown until retrieval and scoring run, so it is measured on an
    anchor before committing: nfcorpus, 323 queries over 3,633 docs,
    already indexed as collection `nf`.
    (k) *Build labels with zero LLM involvement first.* Existing qrels
    plus ORCAS clicks cover every dataset in the composition, so no
    manufactured qrels are needed to produce a first golden set. The
    judge's real job shrinks to **hole-filling** — correcting the
    documented asymmetry where dense retrievers hit 14–32% qrel holes
    against BM25's ~6%, which biases labels against dense. That is a
    correction to labels we can already compute, and it is self-
    validating: if hole-filling helps, dense gains in the predicted
    direction. Deferred until the raw hole rate per route is measured.
    (l) *Corpus-relative features are now a blocker, not a deferral.*
    Per (f), a query-only router inherits the LLM's exact ceiling. The
    only features available *pre*-retrieval — and therefore compatible
    with the speed requirement — are query-term IDF in the index and
    out-of-vocabulary rate. Score margin and dense/sparse candidate
    overlap require retrieving first, so they can inform a fusion
    decision but never the choice of which retrieval to run.
    — *The LLM judges documents; arithmetic picks the route. Everything
    needed to re-pick it later is on disk.*

38. **msmarco-passage-dev anchor: the composition's largest lane gets
    labelled on a realistic index** (grill-me 2026-07-28, executes d37j/g
    for the 31% lane; nfcorpus anchor measured same day).
    (a) *Queries are the composition's, full stop.* `TargetComposition().
    build()` → `RouteLabels.rows_for("msmarco-passage-dev")` — 15,678
    rows. Local dev qrels (`~/.ir_datasets/msmarco-passage/dev/qrels`,
    59,273 judgments) cover 7,697 of them (49.1%); those get labelled,
    the other 7,981 stay `unlabelled` in coverage. Source datasets supply
    qrels and passage text for selected query_ids, never queries.
    (b) *Corpus text via ir_datasets*, the same tool that fetched the
    queries/qrels on 07-15: one user-initiated ~1.06GB fetch of the v1
    collection, whose doc_ids are the ones the qrels name. After that,
    fully offline-re-runnable.
    (c) *Corpus recipe: all 8,219 judged-relevant passages force-included
    + uniform-random distractors (fixed seed) to 100K total.* Uniform
    sampling preserves the collection's vocabulary/IDF profile — the
    d37(g) fix: trec-dl's judged-docs-only 30K was near-all answers,
    inflating dense and starving sparse. Full 8.8M rejected for now
    (~day-scale embedding); reversible, the recipe is a parameter.
    (d) *Artifacts follow the existing owners.* `data/msmarco-passage-dev/
    {corpus,qrels}.parquet` in `SnapshotDataset` layout; Qdrant collection
    `msmarco_routes` with the same `dense_base`/`sparse_base` configs
    (bge-small-en-v1.5 384 cosine, Qdrant/bm25); `min_relevance=1`
    (grades are binary). Labels merge into `data/route_labels/
    labels.parquet` via `RouteLabels.label`.
    (e) *Rule parity with the nfcorpus anchor.* The shipped argmax
    labels this run too, so the two datasets differ by exactly one
    variable. The deliberate tie/all_zero rule (d37i) stays a separate
    change, re-derivable from persisted route_scores/route_rankings
    (d37e) with zero retrieval.
    (f) *Notebook: append per-dataset sections* to `route_labels.ipynb`
    following the §4–§7 pattern, closing with coverage over both
    datasets. A dataset loop is premature at 2; revisit at 5+.
    (g) *Why this dataset second:* median 1 relevant/query (vs nfcorpus
    16) is the regime where two different top-10 lists cannot both be
    right — measured on nfcorpus that dense-vs-sparse top-10 Jaccard is
    0.184 yet 66% of rows sit within 0.06 of a tie, so rich judging, not
    retrieval agreement, was eating the signal.

39. **Qrels acquisition: every human-judged lane, snapshot-first**
    (grill-me 2026-07-29, d38 follow-on; scales d37j's anchor to the
    composition). Scope: the 18 QQ-grounded lanes, ~18.3K selected rows.
    Motivation: the labelled distribution (84% dense over decisive rows)
    is an artifact of *which* lanes are labelled — two natural-language
    lanes where the dense encoder is at home. The sparse/hybrid signal
    lives in the unlabelled technical and entity lanes.
    (a) *Registry bypass, settled.* Doc-side acquisition lives in
    per-lane `RetrievalDataset` classes in
    `hybrid_search_rrf_dataset/retrieval.py`; the registry stays
    queries-only — its own documented doctrine. Proof the registry was
    never on the labeling path: zero `dataset_registry` imports in the
    labeling package, and 8,020 rows labelled while the cache stayed
    `[query_id, text]`. The TODOS "lift queries-only" item is re-scoped
    to what actually needs it — corpus-relative features (d37l) — and
    per-lane Qdrant indexes may serve query-term IDF before any refill.
    (b) *Corpus-pending snapshots.* Pass 1 writes
    `data/<lane>/{queries,qrels}.parquet` for all 18 lanes; `corpus.
    parquet` arrives in pass 2. `SnapshotDataset` init tolerates the
    pending state, `corpus()` fails loud. `coverage()` gains a
    `qrels_ready` state: unlabelled → qrels_ready → labelled. Network
    hit once per lane ever.
    (c) *Declarative lane table.* `LANES` in
    `hybrid_search_rrf_dataset` maps composition key → (source class,
    min_relevance, quirks — BRIGHT's `excluded_ids` honored). The
    notebook iterates the table; amends d38(f): the per-lane-sections
    pattern was right at 2 lanes, the loop is earned at 18.
    nfcorpus/msmarco sections stay as shipped; done lanes sit in the
    table and skip via `label()`'s existing idempotency.
    (d) *min_relevance defaults:* 1 for the binary lanes (rarb, bright,
    crumb, quest, limit, miracl); dbpedia-entity 1 with the §9
    zero-retrieval sensitivity cell as the escape hatch; trec-dl 2 per
    d37(a) (parked anyway).
    (e) *One threshold corpus rule.* Corpus ≤ 100K docs ⇒ index in
    full; > 100K ⇒ d38(c) recipe verbatim (judged force-included +
    uniform-random distractors to 100K, seed=0). No per-lane hand
    decisions; the table records measured size and which branch fired.
    Expected: rarb/bright/crumb full; quest, dbpedia-entity,
    miracl-en-dev capped. Embedding budget ≈ 500–800K docs total,
    cached. **Amended 2026-07-29 (i)**: the rule presumes relevant ≪ cap,
    and crumb-code-retrieval breaks it (108,782 judged-relevant of a
    232,444 corpus — 47% answer-dense intrinsically). `Lane.corpus_cap`
    is the deliberate per-lane override for exactly this case; set to
    250K there, so the lane indexes its full real corpus. A global
    raise was rejected: it would drag five ~1M-passage crumb corpora
    to the new cap for ~15h of embedding nobody asked for.
    **Amended 2026-07-29 (ii): flat 100K default replaced by the 20/80
    recipe, and the recipe is computed, not hand-written** —
    `CorpusRecipe(answer_share=0.2, floor=10_000, ceiling=100_000)` in
    retrieval.py; `target = clamp(relevant / answer_share, floor,
    ceiling)` evaluated from each lane's own qrels at materialize time.
    `LANES` carries only pinned exceptions with reasons
    (`Lane.corpus_target`): crumb-code 120K (its 108,782 relevant — a
    median of 23 relevant docs per query, the benchmark's design —
    exceed the ceiling; floor-plus-pad chosen over full-232K for the
    embedding budget, the hard confusables being in the forced set
    either way), limit 50K (the recipe would compute the floor and
    delete its stress-test design), rarb-code kept at its
    already-embedded 100K. Rationale
    (user-argued, measurement-backed): the label is an argmax over
    three routes on the same index, and the corpus-room diagnostic
    showed 9.3× corpus growth flips only 13.6% of labels while
    `all_zero` only grows — distractor mass buys difficulty, not route
    signal. The three recipe parameters are the tunable surface; floor
    keeps every corpus above the strategies' fetch depth (trivial
    ranking otherwise), ceiling stops answer-heavy lanes ballooning
    (clinical computes to 198K unbounded). Wave-1 embedding ~25h →
    ~11h. Caveat carried: lanes differ in corpus size, so cross-lane
    margin comparisons pick up a size term; per-lane labels stay
    internally valid.
    (f) *Pass-2 order: sparse signal first.* Wave 1, the ≤100K
    technical lanes — rarb-code, crumb-code-retrieval, bright-leetcode/
    aops/theoremqa, crumb-theorem/legal/clinical/stack-exchange/paper,
    rarb-math, crumb-set-operation, crumb-tip-of-the-tongue, limit
    (~16K rows). Wave 2, capped lanes by entity signal: quest →
    dbpedia-entity → miracl-en-dev. Row-count-first ordering rejected:
    it answers the sparse question last.
    (g) *Parked, with reopen triggers.* ORCAS click lane (15,744 rows;
    own grill — needs the 18.8M click pairs and a trust model for
    clicks). LLM judgment source for never-judged rows (msmarco's
    7,981 + pass-1 findings; trigger: d37k hole rate measured).
    trec-dl-2022 rebuild (16 of 79 selected rows judged; hours of v2
    scanning; trigger: hole-filling reopens it).
    (h) *Standing rules carried:* collections named `<lane>_routes`;
    shared `./.embedding_cache`; seed=0; missing qrels ⇒ `unlabelled`,
    never query substitution; argmax rule parity until the tie-rule
    decision lands.
    — *The registry catalogs queries; lanes own their judgments. The
    distribution question is answered when the technical lanes land.*

40. **Generated and augmented rows: golden only with an answer key valid
    by construction** (grill-me 2026-07-29; resolves the golden-set half
    of d34a; gates the generation lane before its first row exists —
    measured: the current 50K is 100% natural, label_lane
    {qrels: 34,256, deferred: 15,744 = ORCAS}).
    (a) *The admission rule.* A row enters `labels.parquet` only when
    its answer key follows from how the query was made: natural →
    source qrels/clicks; doc_grounded → the grounding doc (a 1-known-
    answer key — msmarco's regime, holes acknowledged per d37k);
    augmented → only meaning-preserving operators (parent qrels
    inherit) or doc-consistent injection (surface copied from the
    parent's gold doc) that also passes the row-level coherence test;
    synthetic → own closed-world lane with generator-emitted complete
    qrels (LIMIT is the shipped precedent — 46 invented docs, 1,000
    queries, enumerated key). Everything else is **feature-stock**:
    composition-diversity value, never argmax-labelled. Fabricating a
    label was rejected for the same reason `unlabelled` exists.
    (b) *`constructed` is a first-class judgment source.*
    QrelStore.source grows to {human, constructed, click, llm},
    priority in that order — construction is definitional, clicks are
    noisy behavior, the llm lane is unvalidated (ADR 0001). Filed
    under llm it could never be filtered apart again.
    (c) *Creation-time metadata contract.* Every non-natural row is
    born carrying (provenance, home lane, grounding doc_id, parent
    query_id where applicable); its qrels are minted mechanically from
    that. A row without the metadata has nothing to mint from —
    feature-stock by definition, no post-hoc debates.
    (d) *Operators declare meaning preservation.* Every augmentation
    operator states explicitly whether it preserves the original
    sentence meaning (typos/case/word-order/politeness: yes;
    identifier/date/constraint injection: no — those need (a)'s
    doc-consistent path). Undeclared operator ⇒ its rows are
    feature-stock. Default-deny, auditable in code review — same
    spirit as the banks' ambiguity tiers.
    (e) *Coherence test for injected rows, mechanism deferred to a
    pilot.* Doc-sourced surfaces keep the answer valid but not the
    query readable; user requirement: semantic sense must be tested,
    not assumed. An LLM meaning-gate is d37(f)-compatible here — need-
    identity between parent and augmented query is a function of the
    two texts, no information gap — but ships only after validation
    against a human-audited sample (d34b audit pattern). Until then,
    injected rows stay feature-stock.
    (f) *Label validity ≠ query realism.* Routes score the (query,
    corpus, index) triple regardless of who wrote the query; what
    generation risks is distributional realism, which is d34(b)'s
    concern, not the label's.
    (g) *Construction-order rule: if the sentence must adapt to the
    surface, invert — the surface's document becomes the grounding doc
    and the row is doc_grounded.* Injection is only safe when the
    surface comes from the parent's own gold doc, because a document's
    vocabulary fits its own topic (a MAC address from a networking doc
    reads naturally in the networking query it grounds); a foreign
    surface (a DrugBank id into an internet sentence) is simply
    unavailable to inject, and "adapting the sentence until it fits"
    is generation wearing augmentation's clothes — parent qrels void.
    Splice-then-adapt is banned. Grounding docs are drawn **from the
    existing lane corpora** (`<lane>_routes`), so the generated row's
    answer sits among its natural distractors, pre-indexed and
    pre-embedded, and the row declares that `home_lane` per (c). For
    logical-structure features, prefer harvesting natural supply first
    (quest, crumb-set-op own those cells); generate only for cells no
    dataset fills.
    (h) *Construction success is read from retrieval outcomes, never
    from an embedding metric.* Cosine similarity as a (query, doc)
    validity check is rejected twice over: circular (the dense encoder
    validating data that will judge the dense route) and directionally
    biased (it would pre-filter the dataset toward dense-friendly
    rows). The non-assumptive gauge already exists: run the three
    routes and read the shape — a mis-constructed row comes back
    `all_zero`, which d37(i) already declares label-less, and
    rejection by all_zero is symmetric across routes (a row no route
    can answer carries no routing signal). The per-batch `all_zero`
    rate is the generation lane's construction-quality metric, free
    and model-free.
    — *No answer key by construction, no route label. The golden set
    stays a measurement, not a guess.*

41. **Label form: the score vector is the record; the route column is a
    derived serving decision** (grill-me 2026-07-29; resolves d37i's
    tie-break + all_zero halves and d38's label-form deferral. Measured
    over the 15,413 rows on disk mid-wave-1: 73% of `route` values were
    decided by Python list position — 3,769 exact top-two ties inside
    routes_differ, 5,688 all_tied, 1,745 all_zero — not by retrieval).
    (a) *Canonical label = the three per-route objective scores*,
    already stored on every row. A tie is honest by construction — two
    equal numbers assert no winner. Training leans regression over the
    vector; any classifier target is derived from it by (b)'s rule,
    never stored as a separate truth.
    (b) *Route derivation — quality first, cost second:* `route` = the
    cheapest route among those achieving the maximal score; null when
    the max is 0 (all_zero: no route worked ⇒ no label — ends the 1,745
    fabricated `dense_only`). Cost never overrides quality: sparse at
    0.0 cannot take a row whose tied-best is {dense 1.0, rrf 1.0}. A
    tied row's route is the operationally correct decision, not a
    tie-break hack — when quality is equal, serve the cheapest — so the
    column encodes production's actual job.
    (c) *Cost order `sparse_only < dense_only < pure_rrf`.* rrf costlier
    than each component is structural (it runs both plus fusion) and
    settles 3,744 of the 3,769 pair ties assumption-free; sparse <
    dense (no query-side transformer pass) is the single assumption,
    pending the latency benchmark (TODOS), load-bearing only for
    all_tied rows + 25 dense+sparse ties, reversible by re-derivation
    in seconds.
    (d) *Decisive redefined parameter-free:* winner hit rank-1 AND
    runner-up missed rank-1 — under the lexicographic objective exactly
    margin ≥ 0.4. Replaces the hand 0.06 band, which was unknowingly
    approximating it (1,673 vs 1,668 on the rows on disk; the 5 lost
    rows are both-hit tail differences, not route separations).
    (e) *Ties are exact* (`outcome_shape`'s 1e-9 tolerance). Near-ties
    stay routes_differ with a thin margin; margin and winner-set are
    read-time derivations, never stored columns. Resolves d33/R2's "set
    tie margin ε": ε = 0.
    (f) *One rule, one place:* `derive_route` + the cost order live
    where `StrategyName` lives (fusion.py), importable by golden.py and
    labels.py without cycles. `GoldenRoutingBuilder.strategy_name` = the
    serving choice (non-null; all-zero degenerates to cheapest overall
    — it names the ranking the row carries, not a label);
    labels.parquet `route` = the label (null on all_zero). The
    list-order `max()` dies at the source.
    (g) *Migration is a re-derivation, not a re-run:* one pandas pass
    over labels.parquet after wave 1 lands rewrites `route` from the
    stored scores; notebook readouts switch to (d)'s decisive. Reading
    rule: quality-dominance headlines are read over decisive rows only
    — the raw route column now mixes quality winners with cost policy
    on tied rows (sparse inherits the all_tied mass by design).
    — *The scores are the measurement; the route is a decision computed
    from them. Nothing in the golden set is a coin flip anymore.*

42. **The augmentation loop: order-sheet floors filled by declared
    operators over the composition's own rows** (grill-me 2026-07-29;
    opens the d40-gated generation lane; resolves d34a's generation half
    and d34d; makes d33's minimum-natural-share binding. Canonical term:
    augmentation — enrichment stays an Avoid word; Enricher → Augmenter,
    enrichment_supply.ipynb → augmentation_supply.ipynb).
    (a) *One loop, one demand source.* While the order sheet has missing
    credit: take the hungriest floor (stat bands drawn with probability
    ∝ missing), dispatch to its operator, produce → verify → write.
    Demand is never re-estimated from a fitted distribution — the
    sheet's floors and bands (floors.py) are the computed truth.
    (b) *Operator registry in a new `src/augmentation` package.*
    Operators are grounding-aware (they mint qrels and read the order
    sheet), so they cannot live in grounding-blind taxonomy_generators
    (d34a); that package stays surfaces + verify. Corrupt operators
    live in this registry too — d34d resolved. Sanity note: the new
    machinery is flat — the registry is small classes; the supply index
    is one parquet per lane plus a join; the mini-fill is one mode on
    the existing fill.
    (c) *Declaration schema, default-deny (d40d extended).* Every
    operator declares: floors served, selection rule, grounding
    requirement, answer-key path, meaning preservation, and
    verifiable-by-what. Undeclared or unverifiable ⇒ feature-stock —
    enforced for free by credit accounting, since floor credit is
    computed from re-measured features only.
    (d) *The families.* **Decorate** (markers; any parent; inherits
    parent qrels). **OperatorSyntaxRewrite** — meaning-preserving
    RESTRICTED: may restructure existing conjuncts and drop function
    words, may not add content words; verify = operator span present +
    content tokens unchanged; inherits. **StatRewrite(axis, band)** —
    one generic operator for every stat floor, any axis, any direction;
    near-parents preferred (smallest move = least meaning risk,
    computed from the stats table); a per-(axis, direction) declaration
    TABLE, each entry human-audit-piloted before earning credit — today
    only (length, up) has demand; (length, down) or (nl_shape, down)
    become declaration rows + pilots the day a band demands them, zero
    new code (Compress is not a concept). **Inject** (identifiers; pair
    from the supply index where the parent's gold doc contains the
    surface; ONE surface per row; answer minted from the grounding doc,
    source='constructed'). **Corrupt** (programmatic damage, no LLM —
    R5).
    (e) *Selection is deterministic — tables, never an LLM.* Span side:
    the **supply index**, a one-time bank profile of each lane corpus
    (doc_id, floor key, surface span), floor keys via the
    composition/floors.py mapping so demand and supply share units.
    Stat side: the feature table's scalars (eligibility rules like
    widest_list_size ≥ 2 are stat rules). Side effect: the d34b realism
    problem (garbage datetime/uri samples) never touches Inject —
    doc-copied surfaces are real by construction; generated surfaces
    survive only in the synthetic rung.
    (f) *Inject's supply ladder*: parent's own gold doc → d40g
    inversion (a lane doc carrying the surface becomes the grounding
    doc; row doc_grounded) → synthetic closed-world rung. The rung is
    read per floor from the supply index at run time — never assumed;
    augmentation_supply.ipynb is the standing readout.
    (g) *Inside every LLM operator: an agentic tool loop.* The LLM gets
    the operator's measuring tools, the metric explained, and the
    target as a RANGE (bands natively), iterating while the goal is
    unreached under a bounded retry budget (exhaustion drops the row —
    parents are plentiful). In-loop measurements steer and are never
    the record: acceptance is a fresh LOCAL verify() on the returned
    text — closes the Augmenter prototype's confirmed d2 gap (the
    final instructor pass can reword after the last in-loop verify;
    features_used was LLM self-report).
    (h) *Admission sequencing (d40 applied).* Decorate earns credit
    immediately (politeness-class declaration). OperatorSyntaxRewrite
    and each StatRewrite entry: after their one-time human-audited
    declaration pilots (d34b pattern). Inject: after the d40e per-row
    coherence gate ships its pilot. Until their gate, rows bank as
    feature-stock.
    (i) *Integration: frozen base + deficit-only mini-fill.* The
    existing 50K rows stay byte-identical — their labels are paid.
    WeakestFirstFill gains a start-from-base mode: only the generated
    pool, only the hungry floors, current credits as the opening
    balance; caps and the minimum natural share enforced by the fill,
    never by the loop's own accounting. The natural-share number lives
    in the recipe (set at recipe review — d33 binding now). The
    composition grows past 50K; slice-proportion drift is an eval-time
    weighting concern (d30).
    (j) *Lineage on every row*: `generated_from` — null for natural
    rows, the parent query_id for constructed rows (d40c's
    parent_query_id in the selection schema). The parent-side
    "superseded" filter is a derived view: one parent may have many
    children, and frozen rows are never mutated.
    (k) *Constructed docs are a separate store, never an in-place index
    write*: `constructed_docs` (doc_id, source_dataset, for_query,
    text). Invariant: adding a doc to an existing lane collection
    silently falsifies that lane's already-computed labels (a new doc
    can steal rank-1). The synthetic rung therefore materializes its
    OWN collection — constructed docs forced, distractors borrowed by
    value from source_dataset's corpus (seeded, CorpusRecipe reused).
    Only the synthetic rung writes documents; every other operator is
    query-side only.
    (l) *Batch gauge*: the per-batch all_zero rate at labeling time is
    the construction-quality metric (d40h) — no embedding metric
    anywhere in the loop.
    — *Demand from the sheet, supply from the tables, meaning by
    declaration, truth by re-measurement. The LLM only weaves.*

## Deferred questions

- Recipe values remaining after d32's macro-split (50K; 60/20/20; entity
  slice 80/20; dark forest ≥3 champions, ≤50% each): per-cell floor sizes
  (d30a precision rule), per-span-type target amounts, minimum natural
  share, harvest-target N, per-quota per-dataset source cap (d29 default
  ≤50%).
- Register/box definitions for eval-time weighting + page-search log
  acquisition (d30; boxes wait for a real log).
- Judgment-shaped MODEL features (word-order sensitivity, syntactic depth,
  corruption degree...) — GLiNER2 classification head is the default
  candidate, but behavioral/perturbation designs may fit better; own
  session.
- Multilingual MODEL tier: GLiNER v1 `gliner_multi` vs future mDeBERTa
  GLiNER2 — revisit when non-English profiling matters.
- Corpus-relative features (IDF profile, vocabulary mismatch, ambiguity,
  specificity, answerability): requires explicitly reopening the registry's
  queries-only decision. **Promoted from deferral to blocker by d37(l)** —
  the router cannot beat a query-only LLM without them. The registry cache
  is `[query_id, text]` for all 21 datasets, so ORCAS clicks and every
  dataset's doc_ids are currently dropped on ingest.
- Strategy labeling stage: empirical dense/sparse/hybrid labels in
  `src/hybrid_search_rrf_dataset`, scored by d37(a)'s objective (was "via
  NDCG"). Remaining: pool extraction; the unanswerable-query outcome
  (d37i) resolved by d41 (route = null); corpus sizing settled
  composition-wide by d39(e)'s threshold rule. Anchor yield (d37j)
  measured 2026-07-28, msmarco lane 2026-07-29:
  `data/route_labels/labels.parquet` (8,020 rows).
- Pass-1 reality check (d39): per-lane qrels_ready counts are unknown
  until the fetches run — msmarco's 49.1% says declared QQ grounding
  does not guarantee per-query coverage. Re-price wave order if a lane
  comes back thin. Qrels dialects (RAR-b tsv, BRIGHT gold_ids +
  excluded_ids, crumb/quest/limit lists) verified at implementation.
- msmarco corpus scale-up past 100K if margins look corpus-limited —
  recipe is a parameter (d38c), embedding cache amortizes the retry.
- ORCAS click-lane labeling (the other 31% of the composition): clicks are
  weak relevance of a different kind (`source='click'` in QrelStore, d37d)
  — own decision, not lumped into qrels-lane work.
- ILP escalation for quota conflicts (solver choice, formulation).
- Next acquisitions: CLERC, the 9 unregistered BRIGHT splits, further BEIR
  subsets (ORCAS with `recommended_sample=100K` + 3 BRIGHT splits
  registered 2026-07-20, d21/d31).
- Enrichment grounding layer (d34a) — resolved: d40 (admission rules) +
  d42 (generation-side design: supply index, Inject ladder, operator
  declarations). Implementation tracked in TODOS d42.
- StatRewrite entries beyond (length, up) — reopen when a band goes
  hungry; a "telegram queries feel underrepresented" instinct is a
  recipe/band question first, demand second (d42d).
- Augmenter LLM model + cost envelope; retry-budget size — set at d42
  implementation, informed by the Decorate pilot batches.
- Hungry-floor rung assignment (d42f) — pending the supply-scan readout
  (augmentation_supply.ipynb, three cells left to run).
- Corruption operators' home (R5 programmatic damage):
  taxonomy-generators later wave vs parent-repo lane (d34d).
- Orchestrator-LLM batch lane (d10/d34e) — superseded by d42's Augmenter
  (the agentic tool loop with local-verify acceptance IS that lane).
- Realism overrides over the d34b defaults: which features need them is
  discovered empirically from seeded round-trip samples, not decided up
  front.
- Demo (b) infrastructure: corpus indexing + local Qdrant
  (docker-compose.yml exists) for the disagreement measurement.
- Model backstop for CODE_FRAGMENT/MATH_EXPRESSION recall (symbol-light
  formal content: "x squared plus y squared", prose pseudo-code) — layered
  bank; needs a code/math detection model choice (d20).
- Attested search-syntax extensions to OPERATOR_SYNTAX (quoted phrases,
  minus-exclusion, `site:`) — attested in query logs but precision-dangerous;
  own decision (d20).

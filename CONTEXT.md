# Query Taxonomy & Dataset Registry

Vocabulary for the Strategy Router dataset work: profiling IR datasets with a
query-feature taxonomy, then building a diversified query dataset from them.

## Language

### Taxonomy

**Feature**:
One row of `query_taxonomy/query-taxonomy.csv` (in the sibling package) — a
measurable query property (e.g. Structured Identifiers, stopword ratio,
length). Each feature has a Method (REGEX, ALGO, MODEL) and is either
query-only or corpus-relative.
_Avoid_: quantifier, signal, dimension

**Structured Identifiers**:
The one feature already implemented — 79 regex banks detecting identifier
types (UUID, CVE, ticker, ...). A single feature among ~30, not the taxonomy
itself.

**Corpus-relative feature**:
A feature computable only against a corpus (IDF profile, vocabulary mismatch,
ambiguity, specificity, answerability). Marked `Corpus Relative: Yes` in the
CSV.

**Natural-language signal**:
The canonical "how natural-language-shaped is this query?" measure —
`natural_language_share`, the fraction of tokens that are function words
(UD closed-class POS; pinned spaCy tagger, ALGO tier). Keyword telegrams
sit near 0.0, proper sentences near 0.4–0.5. The stopword-ratio feature is
its REGEX fallback. Replaced the POS-profile histogram (SPEC d26): only
this scalar answered a router question.
_Avoid_: POS profile (dead), closed_class_share (renamed by d26), POS as
GLiNER entity labels

**The four signals**:
The statistical-metrics group read as a panel: NL-shape
(`natural_language_share`), word variation (`word_variation_share`),
structure (`nesting_depth`, `statement_count`), size (`length_words`,
`length_chars`). Names say what a scalar means, not how it's computed —
the mechanism lives in bank docstrings (SPEC d26 sanity-check). Each
answers one router question; together they are coordinates and acceptance
filters for dataset work — strata are boxes in signal space, synthesis is
rejection-sampled against them — and **never label sources**: strategy
labels come from retrieval outcomes. Read jointly: `nesting_depth` is
meaningful only where `natural_language_share` indicates natural language
(the parser hallucinates structure on non-sentences).
_Avoid_: reading a signal in isolation, labeling by signal

**Operator syntax**:
Word-form boolean operators in uppercase (AND/OR/NOT) — the user deliberately
speaking keyword-search dialect (attested at ~1-10% of queries, higher among
developer audiences). Symbolic forms (`!=`, `<>`, `!x`) are NOT operator
syntax: no query-log study attests them as search dialect; in a real query
they are evidence of embedded formal content.
_Avoid_: boolean operators (ambiguous with lowercase conjunctions)

**Embedded formal content**:
Formal-language material appearing inside a query as its subject — exactly
two kinds: a code fragment (programming-language grammar) or a math
expression (equation grammar). A grammar, not a token format: the same
rationale that excluded SMILES from the identifier group routes this to
Logical Structures. Spreadsheet formulas are code; physics equations are
math; chemical identifiers are ChemicalIdBank's; bare chemical formulas
(H2SO4) deliberately excluded.
_Avoid_: formula (dissolves into code/math), programmatic sentence

**Bank**:
Any feature extractor: `GeneralBank[EngineT, OutT]` — `define()` builds its
engine, `compute(text)` returns either spans or stats (OutT is constrained
to exactly `FeatureSpan | FeatureStat`). Span banks declare an ambiguity
tier; stat banks inherit RIGID ("stats are solid numbers"). Non-RIGID span
banks are Assumptive — see below.
_Avoid_: ScalarExtractor (dead), extractor (reserved for the facade)

**Assumptive bank**:
A span bank whose emitted label is a shape guess, not a certified match —
any non-RIGID bank (MODERATE, AMBIGUOUS, or MODEL-tier). Its class name and
emitted string carry the `-Like` suffix (`StockTickerLikeBank` emitting
`stock_ticker_like`, `PersonLikeBank` emitting `person_like`) so consumers
can distinguish shape guesses from certified matches by name alone.
_Avoid_: assumptive claim (say "shape guess"), fuzzy bank

**FeatureStat**:
A named scalar `(name, value)` emitted by a stat bank — one bank may emit
more than one (syntactic depth emits two). The stats counterpart of
`FeatureSpan`.

**Stat suffix convention**:
Every FeatureStat name carries its scale in the identifier: `<name>_share`
and `<name>_ratio` for ratios in [0,1]; counts name their unit
(`length_words`, `length_chars`, `statement_count`, `nesting_depth`). A
bare ambiguous name (`pos_noun`, or a suffix-less `natural_language_signal`
as a stat) is disallowed.

**Report**:
`CorpusReport` (query_taxonomy `reporting.py`) — the presentation class
consuming `CorpusFeatures`: `.text()` human-readable rendering plus
chart-ready rollup data, stdlib only. Plotting belongs to the consumer
(parent repo owns matplotlib). Replaces `CorpusFeatures.summary()`.
_Avoid_: summary() (dead), mixing rendering into the pydantic models

**Domain rollup**:
The presentation-level view of structured-identifier results at `Domain`
granularity — 8 field domains, with `general` exploded into its top-3
member banks + one `general·other` slice (grab-bag slices explain
nothing; total slice budget ≤10, stacked-bar fallback past it). Charts
split each
domain's mass into certified vs `-like` (the Assumptive-bank doctrine
survives into the viz). Profiles JSON and audit surfaces stay bank-level;
rollup is a view, never the stored data.
_Avoid_: rolling up in profiles JSON, domain slices that hide shape guesses

**Engine**:
The detection machinery a bank is built on — RegEx (edify), spaCy, GLiNER2,
or lang-id model. A feature's engine assignment follows the doctrine: closed
surface form → regex; grammatical → spaCy; contextual-semantic and audited →
GLiNER2; language identity → lang-id model (pending library decision).

**Layered banks**:
Two or more banks serving one feature name with different engines; the
within-group claim registry plus tier ordering arbitrate — deterministic
engine claims first, model engine backstops what it missed. Same-token
spans emitted by banks in *different* FeatureGroups (e.g. `acronym` from
sentence_markers and `stock_ticker_like` from structured_identifiers
claiming `DNA`) are parallel independent layers by design — not double
counts. The Assumptive-bank naming rule (`-Like` suffix) makes the
interpretation self-honest at the emit boundary.
_Avoid_: fallback bank (the model layer adds recall, it does not replace)

**Language parameter**:
The declared primary language(s) passed by the caller at extraction time —
`resolve(text, *, languages=["en"])`. Multiple values are valid (mixed
corpus slice). Banks declare `supported_languages: ClassVar[frozenset[str]
| None]` (None = language-invariant, e.g. UUIDs); the extractor skips
non-supporting banks, leaving those features nullable per-row. Distinct from
the LANGUAGE_SET feature, which is *measured* on the query text.
_Avoid_: lang-id (that is the engine name, not this parameter)

**Code-switching**:
A query whose tokens span two or more languages as a natural whole — not
segmentable, not an edge case. A culturally embedded register: Moldavian
Romanian with Russian jargon, Egyptian Arabic with English terms as class
markers, Tolstoy's French in War and Peace. The phenomenon is holistic — the
utterance IS the mixed language; no part is "foreign". A first-class taxonomy
feature under SEMANTICAL: its presence carries routing signal and cultural
information. Detection asks "is mixing present and which languages?" — not
"where are the boundaries?".

**Coordination**:
The breadth axis of query structure — conjunct sets formed by lowercase
and/or or commas ("one, two or three"), measured as parser scalars with a
clausal (verb-verb) vs nominal (noun-noun, enumeration) split. Orthogonal to
syntactic depth, which measures nesting.
_Avoid_: enumeration (say nominal coordination), complexity (overloaded)

**Smoke eval**:
The hand-audited acceptance gate for a MODEL-tier extractor: a small
stratified query sample from the cached datasets, measured for precision,
offset integrity, and lowercase parity before the extractor is committed.

### Subgoals

**Generation**:
Creating brand-new queries exhibiting chosen feature values (e.g. UUID-bearing
queries), with extracted metrics telling the generator how much of each
feature to include.

**Classification**:
Profiling an existing dataset's queries with the feature extractors to decide
what to take from it (e.g. "only short queries → not RAG-compatible → take 20").
_Avoid_: labeling (labels = dense/sparse/hybrid ground truth, a later stage)

**Augmentation**:
Deriving new queries from existing ones by injecting or altering feature
values (e.g. inserting identifiers to strengthen sparse signal).

### Datasets

**Profiling**:
Running feature extractors over a query sample from a registered dataset
(`DatasetRegistry.profile`) to estimate its feature distribution.

**Diversified dataset**:
The end product — a query set composed across datasets to cover the feature
space, as opposed to any single source dataset's natural (skewed)
distribution.

**Fingerprint**:
The per-dataset profile view, two charts over `data/profiles/*.json` (d9
seeded sampling — never a new sampling pass): the *catalog view*, a heatmap
of datasets × (8 domain query-shares + 7 stat means), color normalized per
column with raw values printed in cells; and the *comparison view*, a
spider overlay for 2–4 hand-picked datasets (the only regime where radar
is readable). Cross-dataset axes use the equal-weight percentile scale.
_Avoid_: hotmap (say heatmap), spider charts past 4 overlays

**Equal-weight percentile scale**:
The canonical scale for comparing datasets on a scalar: build the reference
distribution from all queries with each dataset given equal total weight
(one crumb-theorem query ≈ 1450 msmarco queries), then report a dataset as
its queries' median percentile in that reference. Size-independent (the
100K msmarco/orcas samples are ~90% of raw catalog rows and would dominate
any raw pool) and outlier-proof (one 252-word dataset can't flatten the
length axis for everyone else). Applies to fingerprints and pooled charts;
presentation-only — composition cell boundaries live on raw scalar bands
(d33), which stay stable and generation-targetable as the catalog grows.
_Avoid_: raw min-max across dataset means, unweighted pooled percentiles

**Order sheet**:
Per-floor shortfall amounts (with reason: exhausted vs capped) emitted by a
fill run — the generation lane's purchase order. After d33 a shortfall
always means supply ran out, never that the accounting was infeasible.
_Avoid_: gap report, error log

**Label lane**:
Per-row labeling route recorded at selection time: `qrels` (gradeable
today) vs `deferred` (QC rows awaiting clicks or LLM-as-judge).
_Avoid_: checkable (that is the input proxy, not the route), tier

**Dark forest**:
The feature-blind 20% of the target dataset: uniform draws from ≥3
generalist champions, deliberately unconditioned on any extractor output —
insurance against the taxonomy's own blind spots.
_Avoid_: unknown universe, random slice

**Provenance**:
Per-row origin of a query in the diversified dataset: `natural` (taken as-is
from a source dataset), `doc_grounded` (generated/augmented against a real
corpus document), or `synthetic` (query + document generated together).
_Avoid_: source (already means acquisition backend, `SourceKind`)

**Verification loop**:
The generator-side check: LLM produces a query with feature targets, the
regex banks/extractors verify the targets are actually present, PASS/FAIL,
retry on FAIL. Guarantees feature fidelity, not answerability.

**Answerability (grounding)**:
Whether a query has a home corpus containing at least one document it is
about. Guaranteed by construction for `doc_grounded` and `synthetic` rows.

**Representative**:
A claim about proportions-vs-reality — the dataset's mix matches some real
workload's mix. Never about amounts: "UUID ≥ 50" is a floor, not
representativeness. Untestable in-house (nobody holds the true traffic
distribution), so it lives at eval time as a swappable weighting over
cells — score cells separately, weight by a workload's proportions — not
at selection time. Per-cell facts are workload-invariant; only the
headline aggregate needs proportions (d30).
_Avoid_: "representative targets" (contradiction), anchoring the recipe
on ORCAS

**Checkable**:
A row whose strategy label can actually be computed: it has ≥1 judged
relevant doc (qrels), or is `doc_grounded`/`synthetic` — answerable by
construction. Selection counts only checkable rows toward floors.
Replacement filters on "can't check", never on "didn't like the grade":
ties and all-fail rows stay, flagged (d30c). Extends [[answerability]] —
grounded means a home document exists; checkable means the grade is
computable.
_Avoid_: conflating with answerability, discarding graded-but-ugly rows

**Dark matter (of data)**:
What graded data systematically cannot show: unjudged queries (the messy
tail never enters qrels-bearing benchmarks) and qrel holes (relevant docs
no pooled system retrieved — our strategy finds one, scores zero, labels
skew toward pool-contributor-era systems). Cannot be eliminated, only
mapped and counterbalanced: generation probes it (grounded rows with
complete answer sheets aimed at ugly signatures); hole-rich cells carry a
low-trust flag (d30d).
_Avoid_: treating qrels-bearing rows as unbiased samples of anything

**Strategy label**:
The ground-truth retrieval strategy for a query — `dense`, `sparse`, or
`hybrid` — determined empirically (which strategy wins NDCG). The Strategy
Router's target variable; deferred stage, not computed by the taxonomy.
_Avoid_: class, category (overloaded with identifier types)

**Feature target**:
Per-feature quantity a generator is asked to hit (counts, ratios, ranges).
Targets are quantities; measured features are spans + scalars. The
verification loop checks measured-vs-target.

### Composition

**Feature table**:
The materialized composition substrate: one parquet of (dataset, query_id,
per-bank span counts, stat scalars) produced by a single full extraction
pass per dataset. Greedy quota-fill selects *from* the table, so no prune
phase exists; recipe tweaks re-run selection, never re-pay extraction. Also
the future labeling-stage substrate and the audit trail.
_Avoid_: 2D table (say feature table), re-extracting per recipe change

**Recipe**:
The global target distribution of the diversified dataset: quotas over
features and their within-feature strata, plus provenance mix. A reviewable
artifact, not an emergent property.
_Avoid_: target distribution (say "recipe"), objective

**Harvest target**:
A registered dataset's comparative advantage — the top-N features it is
mined for, ranked by prevalence relative to the cross-dataset average
(CLERC → legal identifiers, citations, ...). Proposed by profiling,
human-ratified (the FP smell-test moment), committed as a declared object.
A dataset rich in nothing serves the background stratum — plain queries the
recipe needs too.

**Natural core**:
Step A of composition — the dataset assembled from natural rows only, driven
by harvest targets, before any generation.

**Order sheet**:
Step B input — the deficit between natural core and recipe; exactly what
Generation must produce (doc_grounded/synthetic), under the provenance mix.

**Within-feature strata**:
Sub-quotas inside one feature: density (feature mass vs query length),
surface diversity (distinct forms per type), repetition (same form repeated),
clustering (feature spans concentrated into a centroid of mass vs scattered
across the query). Computed as views over spans, not separate extractors.

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

**POS profile**:
The UD-17 part-of-speech histogram of a query (pinned spaCy tagger, ALGO
tier) with derived scalar views: open/closed-class share, noun share, verb
presence, PROPN share. The canonical function-word measure — the
stopword-ratio feature is its REGEX fallback.
_Avoid_: POS as GLiNER entity labels

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
many (a histogram is many stats). The stats counterpart of `FeatureSpan`.

**Stat suffix convention**:
Every FeatureStat name carries its scale in the identifier: `pos_count_<tag>`
for integer counts, `<name>_share` for ratios in [0,1], `<name>_presence`
for binaries in {0,1}. Bare `pos_<tag>` is disallowed — a count is always
`pos_count_<tag>`.

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

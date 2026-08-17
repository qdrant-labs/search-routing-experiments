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
(UD closed-class POS **minus NUM**; pinned spaCy tagger, ALGO tier). Keyword
telegrams sit near 0.0, proper sentences near 0.4–0.5. The stopword-ratio
feature is its REGEX fallback. Replaced the POS-profile histogram (SPEC d26):
only this scalar answered a router question. NUM left the set at d56: UD files
numerals as closed-class, but this measure asks whether a query is grammatical
glue or a keyword telegram, and a numeral is content — spaCy also tags bare
identifiers NUM, so counting them scored `v1.2.3 nginx.conf 502` as one-third
function words. Membership is gated by the [[word-shape-guard]] since d62: the
tag alone is a guess on any unseen token, and a 32-char hex digest tagged `AUX`
scored 1.0 — a stronger signal than real prose.
_Avoid_: POS profile (dead), closed_class_share (renamed by d26), POS as
GLiNER entity labels, reading it as literal UD closed-class share, trusting the
tag on a token the pinned model never saw

**The five signals**:
The statistical-metrics group read as a panel: NL-shape
(`natural_language_share`), word variation (`word_variation_share`),
structure (`nesting_depth`, `statement_count`), size (`length_words`,
`length_chars`), coordination breadth (`widest_list_size`, d26
amendment). Names say what a scalar means, not how it's computed —
the mechanism lives in bank docstrings (SPEC d26 sanity-check). Each
answers one router question; together they are coordinates and acceptance
filters for dataset work — strata are boxes in signal space, synthesis is
rejection-sampled against them — and **never label sources**: strategy
labels come from retrieval outcomes. Read jointly: `nesting_depth` is
meaningful only where `natural_language_share` indicates natural language
(the parser hallucinates structure on non-sentences).
_Avoid_: reading a signal in isolation, labeling by signal, "the four
signals" (superseded count)

**Word-shape guard**:
The rule that a closed-class token must be word-shaped — no digits — before it
counts toward [[natural-language-signal]] (d62c). The POS tag is a *guess* for
any token the tagger has not seen, and an unpredictable one: a 32-char hex
digest tags `AUX` and scored as pure glue, while `deadbeefcafe1234` tags
`NOUN`. Stated about token shape rather than about `AUX`, so a sibling tag on
a future unseen token is covered without a second repair.
_Avoid_: excluding one more tag (that is d56's shape, and it did not
generalize), `is_alpha` (drops `'s`/`n't`), `is_oov` (true for every token
under the pinned model)

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
The detection machinery a bank is built on — RegEx (edify) or spaCy since
the 2026-07-21 GLiNER drop (SPEC d13 amendment). Doctrine: closed surface
form → regex; grammatical → spaCy; contextual-semantic → future MODEL
tier (deferred); language identity → lang-id model (pending library
decision).

**Layered banks**:
Two or more banks serving one feature name with different engines; the
within-group claim registry plus tier ordering arbitrate — deterministic
engine claims first, model engine backstops what it missed (no model
layer currently ships — the GLiNER temporal backstop left with the
2026-07-21 drop; the mechanism stands, d16). Same-token
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
and/or or commas ("one, two or three"), measured as ONE parser scalar,
`widest_list_size` (the clausal/nominal split was pruned by the d26
amendment: one scalar per router question). Orthogonal to syntactic
depth, which measures nesting.
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
values (e.g. inserting identifiers to strengthen sparse signal). Since
d42 the act is owned by the **augmentation loop**: order-sheet floors
dispatch to declared [[operator-family]] classes (`src/augmentation`);
the LLM only weaves, inside a bounded tool loop with the measuring tools
and a range target; acceptance is always a fresh local re-measure of the
returned text.
_Avoid_: enrichment (same act — say augmentation), Enricher (the class
is Augmenter since d42)

**Operator family**:
One augmentation operator class plus its declaration: floors served,
selection rule (tables, never an LLM), grounding requirement, answer-key
path (inherit parent qrels vs minted from the grounding doc), meaning
preservation, and what re-measurement verifies it. Families: Decorate
(markers), OperatorSyntaxRewrite, StatRewrite(axis, band) — one generic
operator with per-(axis, direction) declaration entries, each piloted
before earning credit — Inject (identifiers), Corrupt (programmatic, no
LLM). Operators also declare parent-relative structural checks
(`structural(parent, text, targets)`: no-new-spans, content tokens unchanged,
literal surface containment), run after local accept (d43b) — `targets` is
what the request authorised, so composed mints do not veto each other (d52d).
Each `instruction()` states its positive move only; the exclusion is the
composed request's, emitted once. Undeclared or unverifiable ⇒ feature-stock.
_Avoid_: Expand/Compress as operators (they are StatRewrite entries),
one generic enrich() endpoint

**Rung**:
How a cell's shortfall gets served, chosen per (cell, parent) rather than per
cell (d51c). Rung 1 augments a real parent: the parent satisfies every cell
band except the ones one operator will mint (**predicate − 1**), and where that
operator declares `surface_origin=DOC_COPIED` its own gold document must carry
the surface. The synthetic rung generates query and document together, reached
when no operator mints the band (acronym, negation, comparative — undeclared by
design) or no corpus supplies the surface. d43's rung 2 (inversion — any lane
doc carrying the surface) is deferred: its key points at a document that has
the surface but need not answer the parent.
_Avoid_: branch (say rung), calling the synthetic rung a fallback (it is a
dispatch outcome), reading "no natural supply" as "no parent available"

**Supply index**:
The one-time bank profile of a lane corpus — (doc_id, floor key, surface
span) in composition floor-key units (the floors.py mapping) — so
Inject's find_pair is a parquet join and demand (order sheet) and supply
share units. `augmentation_supply.ipynb` is its readout.
_Avoid_: LLM fit-scoring (selection is deterministic)

**Constructed-docs lane**:
The separate store for generated documents — (doc_id, source_dataset,
for_query, text) — materialized as its own collection: constructed docs
forced, distractors borrowed by value from `source_dataset`'s corpus.
Never indexed into an existing lane collection: one added doc can steal
rank-1 and silently falsify that lane's already-paid labels. Only the
synthetic rung writes documents; every other operator is query-side
only.
_Avoid_: appending docs to an existing collection or corpus

**Lineage (generated_from)**:
Selection column on every row: null for natural rows, the parent
query_id for constructed rows (d40c's parent_query_id landing in the
selection schema). The parent-side "superseded" filter (has offspring →
ignorable in diversity-focused cuts) is a derived view, never a stored
column — one parent may have many children, and frozen rows are not
mutated.
_Avoid_: forward pointers written onto parent rows

**Surface**:
A generated text snippet exhibiting exactly one taxonomy feature — a valid
UUID, a politeness phrase, a `!=` cue token. The unit taxonomy-generators
emits; grounding-blind by design (whether the surface exists in any corpus
is the upstream lane's concern, d34a). LLMs weave surfaces into queries;
the package never touches surrounding text.
_Avoid_: sample (overloaded with dataset sampling), mint (implies placing
into text)

**Round-trip test**:
The lock-step invariant between the twin packages: every registered
generator, sampled seeded N times, must have each surface claimed by its
twin detection bank under the same emitted feature name. A bank pattern
change that breaks generation fails this test at CI time, never a profile
run.
_Avoid_: drift test, mirror test

**Fault** (campaign scheduling):
A parent attempt that does not bank an accepted row — an engine error
(rounds exhausted, no text), a structural rejection, and a measured-but-
failed target all count identically (d59). Unifies three previously distinct
"drop" reasons under one signal for the chances mechanism; does not replace
`ErrorCase` or a structural check's own message, which still explain WHY a
given attempt faulted.
_Avoid_: drop (a fault is a drop, but "drop" alone doesn't carry the
scheduling signal)

**Chance**:
One continuous turn a floor holds at the front of the campaign's queue
(d59). Ends ordinarily (its need is met, or its parents run out) with no
cost, or is cut short by 3 consecutive faults, which spends the chance and
sends the floor to the back of the queue. A floor starts with 3; burning all
3 drops it from the queue for the rest of that campaign run.
_Avoid_: attempt (a chance contains many attempts), retry (implies the same
parent, not a fresh one)

### Datasets

**Profiling**:
Running feature extractors over a query sample from a registered dataset
(`DatasetRegistry.profile`) to estimate its feature distribution.

**Diversified dataset**:
The end product — a query set composed across datasets to cover the feature
space, as opposed to any single source dataset's natural (skewed)
distribution.

**Fingerprint**:
The per-dataset profile view, three charts over `data/profiles/*.json` (d9
seeded sampling — never a new sampling pass): the *catalog view*, a heatmap
of datasets × (8 domain query-shares + 7 stat means), color normalized per
column with raw values printed in cells; the *comparison view*, a
spider overlay for 2–4 hand-picked datasets (the only regime where radar
is readable); and the *coverage view* (d31), per-query cell counts over
the catalog, dominant dataset annotated per cell. Cross-dataset axes use
the equal-weight percentile scale.
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
fill run — the deficit Generation must produce (doc_grounded/synthetic,
under the provenance mix); the generation lane's purchase order. After d33
a shortfall always means supply ran out, never that the accounting was
infeasible.
_Avoid_: gap report, error log

**Label lane**:
Per-row labeling route recorded at selection time, named after the
[[qrel-source]] that will judge it: `qrels` (human) or `click` (behavioral).
Never a claim about judgment *coverage* — a `qrels`-lane row from a
positive-only dataset has ~1 judged doc against an 18–25 doc pool, so ~95%
of what the routes retrieve is unjudged there too.
`deferred` is retired (d49): its only member was orcas, which ships 18.8M
click pairs and is now a lane of its own.
_Avoid_: `deferred` (retired), reading `qrels` as "fully judged", checkable
(that is the input proxy, not the route), tier

**Qrel source**:
Where a judgment came from, ranked by trust: `human` > `constructed` >
`click` > `llm` (`QrelSource` in `qrels.py`; declaration order IS the
precedence when two lanes judge the same pair). A route label inherits the
weakest source that produced it.
_Avoid_: label quality, confidence (both suggest a score, not a provenance)

**Supplement** (d61):
Read-time addition of admitted augmented rows to a lane's own queries/qrels,
never a mutation of the lane's persisted snapshot — `QuerySupplement` mirrors
`QuerySubset`'s shape (wraps a `source: RetrievalDataset`) but adds rows
instead of narrowing them. Chosen over "overlay" specifically because a
supplement never replaces or shadows what's already there, only adds
alongside it — the lane's own snapshot stays rebuildable from scratch without
risk of losing augmented rows that were never written into it in the first
place.
_Avoid_: overlay (wrong connotation — implies covering/replacing what's
underneath), merge (already means the qrels-conflict-precedence step
specifically, a different operation)

**Dark forest**:
The feature-blind ~20% of the target dataset: random draws from queries that
entered no cell, deliberately unconditioned on any extractor output —
insurance against the taxonomy's own blind spots, and the never-trained
control slice. A *selection* concept. Since d49(j) it is the fill's
leftovers, not draws from named champion datasets.
_Avoid_: unknown universe, random slice, and — since 2026-07-28 — using it
for unanswerable queries; that is [[route-outcome-shape]]'s all-zero case,
an *outcome* discovered after retrieval, not a slice chosen up front. The
two are unrelated: a dark-forest row may be perfectly answerable, and an
unanswerable query may sit in any slice.

**Provenance**:
Per-row origin of a query in the diversified dataset: `natural` (taken as-is
from a source dataset), `augmented` (rewritten from a parent under a
meaning-preserving operator, no document consulted — the parent's judgments
carry over), `doc_grounded` (generated/augmented against a real corpus
document), or `synthetic` (query + document generated together). An explicit
selection column since d51(k): the natural-share ceiling tests
`provenance == 'natural'`, and inferring it from `generated_from.isna()`
counts a parentless row as natural.
_Avoid_: source (already means acquisition backend, `SourceKind`), inferring
it from the lineage column

**Grounding**:
The dataset-card field for which retrieval assets a source ships: `QO`
(queries only), `QC` (queries + corpus), `QQ` (queries + corpus + qrels).
A statement about assets, never about whether we can label the rows — that
is [[checkable]], which admits QC sources carrying clicks or passage answers.
_Avoid_: using it for a row's origin (that is [[provenance]]) or for a
generated surface's source (that is [[surface-origin]])

**Surface origin**:
Where an augmentation operator's new text came from: `none` (rewritten from
the parent), `doc_copied` (lifted from the grounding doc), `synthetic`.
Named `Grounding` until 2026-08-04, which collided with the card field.
_Avoid_: grounding

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
computable. The feature table's `checkable` column is a card-level stand-in
for this — True when a dataset ships any relevance signal (qrels, clicks, or
an answer passage doubling as the gold doc), so the same value for every row.
_Avoid_: conflating with answerability, discarding graded-but-ugly rows,
reading the column as "[[grounding]] is QQ" (the old, narrower rule)

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
The per-query record is the three per-route objective scores (d41),
measured by running `dense_only`, `pure_rrf`, `sparse_only` and scoring
each with the [[router-objective]] (d37a). The single-route `route`
column is a *derived serving decision*: the cheapest route among those
achieving the maximal score — quality first, cost only between exact
ties (cost order sparse < dense < rrf) — and null when all scores are
zero. Computed from retrieval outcomes, never asked of an LLM:
dense-vs-sparse depends on corpus vocabulary and IDF, which are not in
the query (d37f). The Strategy Router's target variable; not computed by
the taxonomy.
_Avoid_: treating `route` as the record (it is a view over the scores),
argmax list order (dead, d41), class, category (overloaded with
identifier types), continuous alpha (not a value the router can emit)

**Router objective**:
`0.7·HitRate@1 + 0.3·NDCG@10`, with a per-dataset `min_relevance`
binarizing graded qrels (2 for TREC-DL's 0–3 scale, 1 for already-binary
qrels). Lexicographic, not a blend: while the hit weight exceeds the NDCG
weight the score ranges are disjoint (rank-1 hit ⇒ ≥0.700, miss ⇒ ≤0.300),
so top-1 decides and NDCG@10 only breaks ties within each group. NDCG is
the tie-breaker because MRR@10 is 1.0 for every route with a relevant
rank-1 doc (blind to coverage) and Recall@10 collapses to two values when a
query has one relevant doc — the majority case (d37c).
_Avoid_: "scored by NDCG" (bare NDCG has no top-1 primacy), calling the
weights a blend

**Route outcome shape**:
Which of three situations a query's per-route scores fall into (d37i): all
routes tied above zero (equivalent — send to the cheapest, the signal for
the speed requirement), routes differ (the quality signal), or all routes
zero (unanswerable — **no valid label exists**). Two of the three are
usable; the all-zero case carries `route = null` (d41 — no label exists,
so none is stored).
_Avoid_: treating an all-zero tie as a `dense_only` label

**Qrel hole**:
A document a retriever surfaces but no human ever judged — treated as
irrelevant by NDCG whether it is or isn't. Asymmetric across retrievers:
BM25 was in most historical pooling rounds so its holes are small
(~6%); dense retrievers post-date the pools and hit 14–32% holes on
older-era corpora. Labels computed without hole correction systematically
under-rate dense retrieval.
_Avoid_: unjudged doc (vaguer), missing qrel (that's the qrel's frame)

**LLM-as-judge**:
An LLM asked whether one document is relevant to one query — binary, and
strategy-blind. It never sees which retriever surfaced the document and is
never asked which route wins (d37f). Its scope narrowed sharply on
2026-07-28: since every composition dataset has qrels or ORCAS clicks, the
judge no longer manufactures labels, and its only remaining job is
[[hole-filling]] — deferred until the raw per-route hole rate is measured.
Never trained by the router; always upstream.
_Avoid_: LLM annotator (assessor is the IR term), 4-level grading (the d35
design; failed 2026-07-28), treating it as a source of strategy labels

**Hole-filling**:
Judging documents a retriever surfaced but no human graded, to correct the
[[qrel-hole]] asymmetry that under-rates dense retrieval. A correction to
labels that are already computable, not a prerequisite for having them —
and self-validating: if it works, dense gains in the predicted direction
(d37k).
_Avoid_: conflating with label manufacture (no dataset needs that)

**Cohen's kappa**:
Chance-adjusted agreement between two labelers on the same items. Ranges
[-1, 1]. Used in d35's pilot as a per-document judge gate at ≥0.6 — a
threshold imported from IR assessor-agreement literature without checking
that it matched this pipeline's consumer, which is a per-query route
decision, not a per-document grade. Retained as a diagnostic; **not** a
gate. Report bootstrap CIs, since a point estimate cannot distinguish a
failed gate from an underpowered one.
_Avoid_: using it as a pass/fail gate, accuracy alone (chance-inflated),
comparing point estimates without CIs

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

**Identifier-span aggregate**:
`derived.identifier_spans` — the column summing every
`structured_identifiers.*` span count, so "this query carries no identifier of
any kind" is one band rather than 54 (d62h). Derived on every read by
`floors.with_derived` and applied at all three catalog-shaped producers; never
stored, because a persisted sum can disagree with its parts after a bank
change, and generated children reach cell matching through `mini_catalog`
without touching the parquet at all. The `derived.` prefix is load-bearing:
`SpanCountAxis` sums everything under a group prefix, so a total named
`structured_identifiers.*` would be counted twice.
_Avoid_: writing it into `catalog.parquet`, naming it under a span-group
prefix (double-count), reading it as a boolean (it is a span count; absence is
`below: 1`), enumerating member banks in a predicate

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

**Within-feature strata**:
Sub-quotas inside one feature: density (feature mass vs query length),
surface diversity (distinct forms per type), repetition (same form repeated),
clustering (feature spans concentrated into a centroid of mass vs scattered
across the query). Computed as views over spans, not separate extractors.

### Route labels

**Lane**:
One composition dataset's labeling pipeline: source fetch → snapshot
(`data/<lane>/`) → route collection (`<lane>_routes`) → rows in
`labels.parquet`. Queries always come from the composition; the lane supplies
qrels and documents for them.
_Avoid_: dataset (ambiguous — the composition is *the* dataset), subset

**Judgment source**:
Where a relevance judgment came from: `human`, `constructed`, `click`, or
`llm` (`QrelStore.source`, conflict priority in that order — d40b).
Inherit-path augmented children copy the parent's judgments keeping
`source='human'` — the judgment is still a human's, only the query changed
under a declared meaning-preserving operator; the transfer is recorded in
`inherited_from` (d43d). Orthogonal to Lane — one lane's rows may
eventually carry several sources.
_Avoid_: judgment lane (collides with Lane), label_lane (that column is the
composition's provenance field, not the QrelStore source), 'constructed'
for inherited copies (the human judged the doc, not the construction)

**Corpus-pending snapshot**:
A lane directory holding `queries.parquet` + `qrels.parquet` but no
`corpus.parquet` yet — the state between qrels acquisition (pass 1) and
corpus indexing (pass 2). Coverage reports its rows as `qrels_ready`.

**Decisive label**:
A `routes_differ` row whose winner hit rank 1 AND whose runner-up missed
it — parameter-free, and under the lexicographic objective exactly
margin ≥ 0.4 (d41; replaced the hand 0.06 band, which approximated this
rule on all but 5 of 1,673 rows on disk). The honest trainable count,
and the only rows quality-dominance headlines are read over.
_Avoid_: trainable rows (routes_differ alone overcounts — 47% are exact
top-two ties), margin ≥ 0.06 (dead)

**Acceptability label**:
Per route, `score >= oracle − 0.3` (d60): did this route land in the same
top-1 band as the best route? A derived view over the stored score vector
(like `route` itself), never materialized columns; tolerance 0.3 is the
objective's own `ndcg_weight` — the widest gap that cannot involve a
top-1 flip — not a hand number. All-zero rows stay null (d41 holds).
_Avoid_: multi-label route (says the mechanism, not the meaning), storing
ok_* in labels.parquet (twin structure), tolerance as a dataset fact (it
is the view's parameter; 0.3 is only the canonical default)

**Serve decision**:
The cheapest route whose acceptability is true (cost order sparse <
dense < rrf). At tolerance 0 this is exactly the stored `route` column;
at 0.3 it is the cost-aware oracle readout routers are evaluated
against (d60e). Training targets are the three acceptability heads, not
this column.
_Avoid_: serve as a training label (it hard-codes cost policy into the
label), winner (that is the quality argmax)

**Headroom**:
The measured value of routing: mean per-query oracle (best of the three
route scores) minus the best constant route's mean, same rows. Measured
2026-07-30 (24,338 rows): +15.8% total ceiling, decomposing into the
[[per-collection-constant]] gain (+6.3%) and the per-query residual
(+9.0%). A ceiling — what a perfect router would capture, not what a
trained one will (published QPP-driven selection achieved ≤~4%). Always
quoted with d44(a)'s caveats: arbitrary lane mix, feature-diversity
composition, holes unmeasured, stack-pinned.
_Avoid_: quoting pooled headroom bare, reading decisive share as
headroom (limit: 43% decisive, 0.3% headroom — one-sided decisiveness
is already captured by the constant)

**Per-collection constant**:
The best single route for a whole collection (nfcorpus → rrf, msmarco →
dense). Picking it right — a collection-level decision needing no
per-query intelligence — captures +6.3% over one global constant; the
tier-one product and d44(c)'s side test (16 predictions from collection
statistics alone).
_Avoid_: conflating with per-query routing, "default route" (that is
the production classifier's fallback, not this)

**Collection statistics**:
The six per-(query, lane) scalars of d44(b): avg/max query-term IDF in
the lane corpus, OOV share, collection size N, avgdl, query-vocabulary
overlap share. The router's corpus eyes — computed offline from
`corpus.parquet` after labeling, stored in
`collection_features.parquet` beside the frozen labels. Lineage:
resource selection (CORI/ReDDE/Taily) — stats transfer to unseen
collections; query-only predictors do not.
_Avoid_: corpus features (vague), computing them at labeling time
(router features, never label inputs), editing labels.parquet to hold
them

**Transfer pilot**:
d44(c)'s two-protocol experiment. Protocol (i): hide a random 20% of
every lane's queries, train on the rest — "new queries on a known
collection". Protocol (ii): hide one entire lane, train on the other
15, rotate through all 16 — "a collection never seen". Each runs with
and without collection statistics; the (i)−(ii) gap IS the
corpus-dependence measurement, and the statistics succeed iff they
shrink it. Random splits must be near-duplicate-aware (~5.5% cos>0.95
pairs).
_Avoid_: "leave-one-lane-out" without unpacking it, running only one
protocol, reading protocol-(i) success as transfer

**List-preference judge**:
An LLM shown the query plus each route's retrieved top-10 and asked
which list answers best — pairwise, order swapped between two askings,
ties allowed. Sees corpus evidence, so d37(f)'s information gap does
not apply; still a judge, so it is calibrated against the empirical
spine and never trusted raw. Scope today: the d44(d) spike (500 rows,
3 lanes, agreement vs empirical routes on decisive rows; thresholds
80/60).
_Avoid_: judging bare queries (d37f), graded pointwise scales
(ADR 0001), conflating with [[llm-as-judge]] (that term is per-document
relevance; this one prefers between result lists)

**Feature-stock**:
A composition row without an answer key valid by construction — it serves
feature diversity but is never argmax-labelled (d40a). Includes rows from
undeclared augmentation operators and generations without grounding
metadata.
_Avoid_: unlabelled (that is a coverage state of labelable rows, not a
category)

**Meaning-preserving augmentation**:
An augmentation operator declared to keep the original sentence meaning —
typos, case damage, word-order corruption, politeness filler. Its rows
inherit the parent query's qrels. The declaration is a required, explicit
property on every operator; undeclared ⇒ feature-stock.
_Avoid_: need-preserving (same concept, non-canonical phrasing)

**Doc-consistent injection**:
A need-narrowing augmentation whose injected surface is copied from the
parent query's gold document, so that document still answers by
construction. Golden only after the row-level coherence test passes
(mechanism pending the d40e pilot).
_Avoid_: grounded injection (grounding already means answerability)

**Auto-fusion baseline**:
The LLM query router this project is replacing — a Rust HTTP service
(`POST /v1/classify`) returning a 0–9 score mapped to a route via the
production hard bands (0–2 dense, 3–6 rrf, 7–9 sparse). Corpus-blind
surface classifier that asks "does this query look identifier-shaped?"
Runs as a row in `RouterExperiment.run(autofusion=True)` via
`AutoFusionRouter` + `LLMScoreClient`.
_Avoid_: LLM baseline (ambiguous with the list-preference judge),
auto-classifier

**Privileged corpus features**:
Corpus-side knowledge that enters the model at training only — since d63,
the corpus branch â's three target blocks: per-lane corpus stats (scan +
sampled doc extraction), the per-query [[gold-doc-block]], and fold-local
[[route-outcome-stats]]. Masked as an INPUT at inference (deployment
target unknown at ship time, so `query → route` is a hard constraint); at
serve time the model feeds its own â estimate forward instead (Vapnik
2015 LUPI, feed-forward wiring — see [[privileged-branch]]).
_Avoid_: `collection_stats` at inference (that was d47(b), retired
2026-08-04), aux-A (say corpus branch / â)

**Per-archetype eval**:
Grouping held-out decisive rows by feature signature (`has_uri`,
`has_uuid`, `is_short`, `is_math`, `is_natural_language`) and reporting
headroom captured per group. Surfaces coverage gaps in the composition —
the aggregate metric hides archetype-level failures where a small
fraction of rows carries a big qualitative gap (URL case: 0.1% of eval,
invisible in the mean).

### Encoder router

**Encoder router**:
The d63 route model: frozen input channels (bge-small embedding ⊕
char-3–5-gram SVD fit on training queries), one MLP encoder to a latent,
two feed-forward privileged branches, two acceptability heads (sparse,
dense — rrf is the [[hedge]], never predicted). Serve rule since
2026-08-12: the most probable head among those clearing their tuned
thresholds; none fires → the rrf hedge. Cost never picks between heads —
thresholds are tuned on raw captured score. Lives in `src/encoder_router/`;
serving sees the raw query string ONLY (no extractor, no corpus, no spaCy)
— the model generalizes features via its branches, it never extracts them.
_Avoid_: NN router / MLP router (name the artifact), LUPI option B (dead
— d63's wiring superseded it), cheapest-acceptable serving (dead
2026-08-12 — cost-discounted thresholds went with it), feature extraction
at serve time (the constraint is the design)

**Privileged branch**:
A supervised side output of the encoder router graded against
training-only knowledge, whose PREDICTION feeds the route layers at serve
time: the cell branch ĉ (44 sigmoids graded against evaluation of every
cell predicate — never the fill's single assignment, which is quota
bookkeeping) and the corpus branch â (regression against the three
[[privileged-corpus-features]] blocks). The loss pins the branch's
meaning; the feed-forward puts it to work. A branch is never an inference
input — always an inference estimate.
_Avoid_: aux head (silent about the wiring), dropped head (that variant
lost the grill), hallucination head (mechanism citation, not the concept)

**Gold-doc block**:
Per-query taxonomy stats of the row's qrel documents (at the lane's
min_relevance, mean over several) plus query↔gold lexical overlap — the
per-query third of â's targets, and what breaks the ~15-lane lane-ID
degeneracy of purely per-lane targets. Overlap is the closest measurable
cause of a sparse win.
_Avoid_: golden set (that is the labelled dataset itself), doc features
(ambiguous with the corpus sample)

**Route-outcome stats**:
Per-lane acceptability rates (ok-rate per route + decisive share)
computed from TRAINING rows only, fold-locally — a property of corpus ×
selected queries, so never called corpus stats; the target-aware third of
â's targets.
_Avoid_: corpus win rates (hides the selection dependence), whole-lane
computation (test labels leak into training targets through the average)

### Serving

**Serve-time budget**:
The four conditions a feature must meet to be readable when a query arrives
(settled 2026-08-17, replacing the blanket "raw query string only" ban):
computed once ahead of the query; read with no network call and no model load;
available on every training row; and produced by the same code path at
training and at serving. Admits per-collection statistics, cluster centroids,
and the regex banks; still excludes the spaCy pipeline.
_Avoid_: query-only API (retired), "never extract at inference" (that was a
category ban; this is a cost-and-provenance rule)

**Compress**:
The final serving step — projecting an already-retrieved superset of results
onto the chosen route's view. Named so the zero-cost property is visible in
the signature: nothing is retrieved, so choosing among lists already held is
free.
_Avoid_: select / filter (both imply a retrieval), rerank (no score changes)

**Presearch gate**:
The route model's own confidence on the results-free call, deciding whether to
commit to one leg before searching or to retrieve both and decide afterward.
One model serves both regimes, so results must be dropped out during training
or the results-free call is out of distribution.
_Avoid_: a second classifier (one model, two call sites), CAN_PRESEARCH as a
separate component

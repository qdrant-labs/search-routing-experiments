# Dataset acquisition catalog

Candidate datasets for the diversified-dataset pipeline (SPEC.md), evaluated
along the six `DatasetCard` dimensions and annotated with a **harvest
hypothesis** — the taxonomy features each source is expected to be rich in.
Hypotheses are proposals: per SPEC decision 7, profiling proposes ranked
harvest targets and a human ratifies them at registration.

Card tuple notation (matches `dataset_registry.core.DatasetCard`):
`(grounding, llm_target, scope, non_trivial, multilingual, multimodal)` —
e.g. `(QQ, LLM-0, G, NT-1, i18-0, MM-0)`. Grounding: QO = queries only,
QC = queries+corpus, QQ = queries+corpus+qrels.

Backends: `irds` = `IRDatasetsBacked` subclass (one-liner), `hf` =
HF `load_dataset(..., streaming=True)` fetcher (MiraclDev pattern), `url` =
jsonl over HTTP via the HF `json` loader (still streaming, still cached once).

## Already registered

| dataset | card | backend | notes |
|---|---|---|---|
| msmarco-passage-dev | (QQ, LLM-0, G, NT-0, i18-0, MM-0) | irds | ~55.5K Bing queries, baseline |
| trec-dl-2022 | (QQ, LLM-0, G, NT-0, i18-0, MM-0) | irds | judged DL topics |
| beir-nfcorpus | (QQ, LLM-0, S, NT-0, i18-0, MM-0) | irds | medical; profiled: ~1% true identifier density → dense stratum |
| miracl-en-dev | (QQ, LLM-0, G, NT-0, i18-1, MM-0) | hf | 799 native questions → natural-question stratum |

Open discrepancy: the 2026-07-20 candidate list codes MIRACL as
`(LLM-1, NT-1)`; the registered card says `llm_target=False,
non_trivial=False`. Ratify one reading (proposal: queries are human-written
natives → LLM-0; the RAG/judge orientation belongs to MEMERAG, not MIRACL).

## Wave 1 — register next

High taxonomy value, open access, direct backend fit.

### ORCAS
`(QC, LLM-0, G, NT-0, i18-0, MM-0)` · irds `msmarco-document/orcas` · open
10.4M unique real Bing queries (18.8M click pairs). The only *real query
log* that is openly downloadable (AQL survey, Table 2). Needs
`recommended_sample` (proposal: 100_000).
**Harvest hypothesis:** web-scale natural register; misspellings/corruption;
identifier density at scale; the honest base rate for OPERATOR_SYNTAX
(SPEC d20 research says ≤10%, likely far less).

### BRIGHT
`(QQ, LLM-0, G, NT-1, i18-0, MM-0)` · hf `xlangai/BRIGHT`, config
`examples`, one split per domain · open
Reasoning-intensive retrieval, 12 domains (SoA 59.0 nDCG on BEIR drops to
18.3 here). Parameterized class per split, MiraclDev-style.
**Harvest hypothesis:** long multi-clause queries (SYNTACTIC_DEPTH,
COORDINATION); `leetcode` split → CODE_FRAGMENT, `aops`/`theoremqa_*`
splits → MATH_EXPRESSION (d20 strata, directly).

### QUEST
`(QQ, LLM-0, G, NT-1, i18-0, MM-0)` · hf `cmalaviya/quest` · open
3,357 natural queries with *implicit* set operations (intersection, union,
difference: "shorebirds that are not sandpipers"); 6,307/323/1,727 splits
including augmented.
**Harvest hypothesis:** the natural-language counterpart of d20's operator
discussion — nominal COORDINATION width, negation markers, set-op phrasing
with zero explicit operator syntax. Prime stratum for router disagreement.

### CRUMB
`(QQ, LLM-1, G, NT-1, i18-0, MM-0)` · hf `jfkback/crumb`, config per task ·
open
Eight complex multi-constraint tasks: tip-of-the-tongue movies, multi-aspect
paper search, set-based entity retrieval, state-specific legal, theorem
retrieval from math problems, StackExchange, clinical trials, code retrieval.
**Harvest hypothesis:** multi-constraint queries (COORDINATION,
SYNTACTIC_DEPTH); code task → CODE_FRAGMENT; theorem task →
MATH_EXPRESSION; legal task → legal identifiers.

### RAR-b (math-pooled, code-pooled)
`(QQ, LLM-0, S, NT-1, i18-0, MM-0)` · hf `RAR-b` org, one repo per subset ·
open
Reasoning-as-retrieval: math-pooled = MATH + GSM8K (6,319 test queries),
code-pooled = HumanEvalPack + MBPP (1,484). Register only these two pools
v0; the commonsense pools (αNLI, HellaSwag, TempReason…) are optional later
(TempReason would stress the TEMPORAL feature).
**Harvest hypothesis:** the single densest source for MATH_EXPRESSION and
CODE_FRAGMENT (d20) — word problems with inline equations, code problems
with inline signatures.

### LIMIT
`(QQ, LLM-0, G, NT-1, i18-0, MM-0)` · url — jsonl in
`github.com/google-deepmind/limit` `data/limit/queries.json` (1,000 queries;
MTEB format) via the HF `json` loader · open (Apache-2.0)
Stress set built from embedding-dimension theory: for any dimension d some
top-k document combinations are unreturnable — dense fails by construction.
**Harvest hypothesis:** queries are deliberately trivial ("who likes X?") —
the value is NOT feature harvesting but the strategy-labeling stage: a
guaranteed sparse-wins stratum to calibrate labels and the production-router
baseline against. Register for completeness; exclude from harvest targets.

### DBPedia-entity
`(QQ, LLM-0, G, NT-0, i18-0, MM-0)` · irds `beir/dbpedia-entity/test` · open
400 entity-centric keyword queries over DBPedia. Candidate list codes scope
S; entity lookups over general Wikipedia read as G — ratify at registration.
**Harvest hypothesis:** sparse-friendly bare-entity queries: proper nouns,
acronyms, low stopword ratio — the keyword register MIRACL/NFCorpus lack.

## Wave 2 — after wave 1 lands

Open access; either smaller marginal value or minor fetcher work.

### FreshStack
`(QQ, LLM-0, S, NT-1, i18-0, MM-0)` · hf `freshstack/queries-oct-2024`,
config per topic (langchain, yolo, laravel, angular, godot) · open
Realistic technical queries from StackOverflow over niche doc corpora.
**Harvest:** CODE_FRAGMENT, version/package identifiers, error strings —
the sparse-technical stratum the profiles memory says we lack.

### GooAQ
`(QC, LLM-0, G, NT-0, i18-0, MM-0)` · hf `allenai/gooaq` · open
3M questions mined from Google autocomplete, with answers.
Needs `recommended_sample` (proposal: 50_000).
**Harvest:** natural-question register at scale; question-word/marker
distribution; background stratum filler.

### ANTIQUE
`(QQ, LLM-0, G, NT-1, i18-0, MM-0)` · irds `antique/test` · open
Non-factoid QA benchmark (2.4K train / 200 test queries, Yahoo Answers).
**Harvest:** non-factoid, opinion/how-to phrasing; politeness and
conversational markers.

### LoTTE
`(QQ, LLM-0, S, NT-1, i18-0, MM-0)` · hf `colbertv2/lotte`, config per
domain, `search` + `forum` query sets · open
StackExchange-derived long-tail queries (writing, recreation, science,
technology, lifestyle).
**Harvest:** forum-natural register; `technology` slice for identifiers;
search-vs-forum phrasing contrast within one corpus.

### WebFAQ
`(QQ, LLM-0, G, NT-1, i18-1, MM-0)` · hf `PaDaS-Lab/webfaq`, config per
language · open
96M FAQ-style QA pairs, 75 languages. Needs `recommended_sample` per
language (proposal: 50_000).
**Harvest:** the LANGUAGE_SET / multilingual stratum feeder for d18-19;
FAQ register (imperative + question mix).

### Query wellformedness
`(QO, LLM-0, G, NT-0, i18-0, MM-0)` · hf
`google-research-datasets/google_wellformed_query` · open
25,100 Paralex queries, human-rated wellformedness 0-1. First QO
registration (queries-only grounding, registry handles it natively).
**Harvest:** corruption-group calibration — the rating is external ground
truth for fragment/ungrammatical detection; natural misspellings.

### Natural Questions (via nq_open)
`(QC, LLM-0, G, NT-0, i18-0, MM-0)` · hf `google-research-datasets/nq_open`
· open
91K real Google queries (TACL Q19-1026). `nq_open` sidesteps the 42GB full
NQ artifact — queries + short answers only.
**Harvest:** authentic search-engine question register (vs MSMARCO's Bing);
background stratum.

### XOR-TyDi QA
`(QQ, LLM-0, G, NT-0, i18-1, MM-0)` · url — jsonl from
`nlp.cs.washington.edu/xorqa` (verify exact URLs at registration) · open
Cross-lingual open-retrieval QA, 7 typologically diverse languages;
questions in-language, evidence in English.
**Harvest:** the cross-lingual stratum (d18 code-switching/language-set):
queries whose answers live in another language's corpus — sparse breaks by
construction.

### ScIRGen-Geo
`(QQ, LLM-1, S, NT-1, i18-1, MM-0)` · hf `usail-hkust/ScIRGen-Geo` · open
61K synthetic-but-filtered scientific QA over geoscience datasets/papers
(KDD'25), bilingual EN/ZH.
**Harvest:** scientific register, dataset-seeking queries; also a reference
point for our own `synthetic` provenance quality bar (their
cognitive-taxonomy generation + perplexity filtering ≈ our verification
loop).

### CLERC (carried from existing roadmap)
`(QQ, LLM-0, S, NT-1, i18-0, MM-0)` · hf `jhu-clsp/CLERC` (availability:
verify — may require terms acceptance) ·
Legal case retrieval; carried over from the 2026-07-15 roadmap.
**Harvest:** legal citations and identifiers — top comparative-advantage
source for the legal Domain banks.

### Simulated Wikipedia queries (hotchpotch)
`(QC, LLM-1, G, NT-0, i18-0, MM-0)` · hf
`hotchpotch/wikipedia-english-ir-simulated-search-queries` · open
LLM-simulated search queries over English Wikipedia. Structured and precise
only — no ambiguity (user assessment 2026-07-20).
**Harvest:** limited as a natural source; useful as a *generation
calibration* contrast — what LLM-written queries look like vs real logs
(feature-distribution diff against ORCAS).

## Gated / deferred

### LMSYS-Chat-1M
`(QO, LLM-0, G, NT-1, i18-1, MM-0)` · hf `lmsys/lmsys-chat-1m` · GATED
(terms acceptance)
1M real LLM conversations. Not an IR dataset: needs a first-user-turn
extraction design before it yields "queries" — that design decision plus the
gating defers it. The query-taxonomy CSV's own rationale for operator syntax
("current LLM boom" phrasing diversity) makes this the highest-value
deferred source: real prompts with code blocks, politeness, multi-intent.

### Archive Query Log (AQL)
`(QO, LLM-0, G, NT-0, i18-1, MM-0)` · Webis, data-use agreement · RESTRICTED
356M queries, hundreds of engines, 25-year span. The definitive empirical
base for operator-syntax prevalence (d20's research would become measurable
in-house). Defer until someone signs the agreement.

### NIST BETTER
`ir.nist.gov/better` · access conditions unverified
Cross-lingual analytic retrieval. Park until XOR-TyDi (same axis, verified
open) is exhausted.

### TREC-RAG
`(QC, LLM-1, G, NT-0, i18-0, MM-0)` · topics file over MS MARCO v2.1
segmented · open
~300 RAG topics; marginal query-side value over registered MSMARCO — the
corpus decision (segmented v2.1) is the real cost. Revisit at the
strategy-labeling stage where its judgments help.

### M-BEIR
`(QQ, LLM-0, G, NT-1, i18-0, MM-1)` · hf `TIGER-Lab/M-BEIR` · open
Multimodal retrieval. Out of scope: the taxonomy is text-only v0
(multimodal is exactly the MM axis the card tracks; no banks apply).

### MEMERAG
github `amazon-science/MEMERAG` · open
Meta-evaluation benchmark on MIRACL questions (5 languages) with
LLM-generated answers + expert faithfulness/relevance judgments. **Not a
query source** — its queries are MIRACL's, already registered. Its value
(human judgments for LLM-as-judge) belongs to the deferred strategy-labeling
/ judge stage; revisit there.

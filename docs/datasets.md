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

Acquisition standing is one question, **publication** standing is another:
what we may redistribute is in "Licensing and publication standing" at the
end of this file — the source of truth for that, nowhere else.

## Already registered

| dataset             | card                              | backend | notes                                                          |
| ---------------------| -----------------------------------| ---------| ----------------------------------------------------------------|
| msmarco-passage-dev | (QQ, LLM-0, G, NT-0, i18-0, MM-0) | irds    | ~55.5K Bing queries, baseline                                  |
| trec-dl-2022        | (QQ, LLM-0, G, NT-0, i18-0, MM-0) | irds    | judged DL topics                                               |
| beir-nfcorpus       | (QQ, LLM-0, S, NT-0, i18-0, MM-0) | irds    | medical; profiled: ~1% true identifier density → dense stratum |
| miracl-en-dev       | (QQ, LLM-0, G, NT-0, i18-1, MM-0) | hf      | 799 native questions → natural-question stratum                |

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

## Wave 3 — query-distribution expansion

Wave 3 deliberately adds query distributions that the earlier roadmap
undersampled: product search, finance, conversational context, enterprise
support, argument retrieval, and claim retrieval. Six sources are implemented
in `LANES`; the remaining pilot candidates stay proposals. A pilot is
promoted only after it demonstrates decisive dense/sparse/hybrid rows; qrels
alone do not justify a lane if most sampled queries become ties or all-zero
rows. CAsT / TechQA / Touché were promoted alongside the build-next three to
extend the agentic-flavored (multi-turn, long-form troubleshooting,
comparative argument) query mix — measure them next.

### Implemented

#### Amazon Shopping Queries / ESCI — `amazon-esci-en-hard`
`(QQ, LLM-0, S, NT-1, i18-0, MM-0)` · url, official GitHub LFS media over
`amazon-science/esci-data` · Apache-2.0 repository

The reduced Task 1 English split has 29,844 authentic customer queries and
601,354 query-product judgments. It adds a major missing distribution:
short, attribute-heavy product search with model numbers, units, brands,
comparatives, negation, and substitution intent.

- **Query:** the original customer query.
- **Corpus:** one document per product, concatenating title, brand, color,
  bullet points, description, and other searchable attributes while retaining
  the product ID as `doc_id`.
- **Qrels:** map ESCI labels `E=2`, `S=1`, `C=0`, `I=0`; minimum relevant
  grade is 1. Exact and substitute are grades in one lane, not separate lanes.
- **Harvest:** SKU/BARCODE-like identifiers, VALUE_WITH_UNIT, NUMBER, proper
  nouns, acronyms, comparative and negation markers, coordination, short
  keyword queries, and high-OOV queries.
- **First cut:** English reduced/hard only. Spanish and Japanese stay deferred
  until multilingual labeling is in place.
- **Materialization:** the full hard set has 394,057 unique E/S products, so
  metadata must first narrow to the composition's query IDs; the standard
  corpus recipe then force-includes their relevant products and samples
  distractors. An unscoped full-catalog build is intentionally rejected.

#### WANDS — `wands`
`(QQ, LLM-0, S, NT-1, i18-0, MM-0)` · url fetcher over
`wayfair/WANDS` · MIT

480 genuine historical Wayfair queries over 42,994 products with 233,448
human relevance judgments. Its unusually deep judgments (roughly 486 per
query) make it a calibration lane for distinguishing real strategy ties from
shallow-qrel artifacts.

- **Query:** the original customer query.
- **Corpus:** product name, product class, category hierarchy, description,
  and structured features, keyed by product ID.
- **Qrels:** `Exact=2`, `Partial=1`, `Irrelevant=0`; use the full product
  corpus, including judged and unjudged products.
- **Harvest:** furniture dimensions, materials, styles, colors, product types,
  coordination, and short commercial keyword queries.

#### FinDER — `finder`
`(QQ, LLM-0, S, NT-1, i18-0, MM-0)` · hf
`Linq-AI-Research/FinDER` · CC-BY-NC-4.0

5,703 professional finance queries grounded in evidence from 490 companies'
10-K filings. This adds ticker symbols, filing terminology, fiscal periods,
currency, percentages, and quantitative comparison questions.

- **Query:** the `text` field; retain reasoning/category metadata.
- **Corpus:** deduplicate every passage in `references` by normalized content;
  generate stable content-derived `doc_id` values and preserve source metadata.
- **Qrels:** every referenced passage for a query is relevant. Preserve any
  upstream relevance distinctions if the schema exposes them; otherwise use
  binary relevance 1.
- **Publication:** pointer-only unless the non-commercial license is explicitly
  accepted for the intended artifact.

#### TREC CAsT 2020 — `trec-cast-2020-history`
`(QQ, LLM-0, G, NT-1, i18-0, MM-0)` · irds
`trec-cast/v1/2020/judged` · mixed upstream terms

208 judged conversational turns over the MS MARCO passage and TREC CAR
corpora. The lane is valuable only if the production router receives dialogue
history, so both the registry and the retrieval lane serialize each turn as
`prior_utterances [CURRENT] current_utterance`, grouped by `topic_number` and
ordered by `turn_number`. The manual/automatic rewrites remain upstream and
are not composed into the query text. Use the official passage qrels unchanged.

#### TechQA — `techqa`
`(QQ, LLM-0, S, NT-1, i18-0, MM-0)` · hf `rojagtap/tech-qa` (unofficial
mirror; `IBM/TechQA` is not on the Hub) · repository Apache-2.0; content
terms need verification

Real technical-support forum questions linked to Technote passages: 600
train, 310 validation, 490 test rows in the mirror. Query text is the full
forum question; each row's `document` becomes a corpus entry keyed by
`techqa-<sha256(24)>`, deduplicated across splits; qrels bind the query to
its linked document with binary relevance 1. The mirror ships the linked
Technote text per row rather than the full ~802K IBM Technote catalog, so
this is a pilot substrate, not the full-corpus lane the routing measurement
would ultimately want. Expected harvest includes error strings, product and
version identifiers, log fragments, and long troubleshooting language.

#### Touché 2020 — `beir-touche-2020`
`(QQ, LLM-0, G, NT-1, i18-0, MM-0)` · irds
`beir/webis-touche2020/v2` · verify redistribution terms

49 comparative or controversial decision topics over roughly 383K argument
passages with deep judgments. Use the topic title/text as the query, argument
passages as the corpus, and the official graded qrels. Its small topic count
makes it better suited to evaluation/calibration unless profiling finds a
clear decisive stratum.

### Pilot before promotion

#### SciFact — `beir-scifact`
`(QQ, LLM-0, S, NT-1, i18-0, MM-0)` · irds `beir/scifact/test` (optionally
train after leakage review) · CC-BY-NC-2.0

Expert-written scientific claims retrieve evidence from about 5.2K abstracts.
The declarative-claim query form is distinct from ordinary search and should
therefore earn promotion through measured routing value. Preserve official
qrels and support/contradiction metadata; index the complete abstract corpus,
not only cited abstracts.

#### TREC Fair Ranking 2020 — `trec-fair-2020`
`(QQ, LLM-0, S, NT-1, i18-0, MM-0)` · irds/custom · verify corpus access

About 200 production academic-search query sequences over a multi-million
document Semantic Scholar corpus. It adds real paper-seeking shorthand that
differs from CRUMB's generated academic tasks. Use the production query text,
paper title+abstract documents, and official relevance judgments; ignore the
fair-exposure objective for routing labels. Pilot because the corpus is large
relative to the number of unique topics.

## Gated / deferred

### TripClick
`(QQ, LLM-0, S, NT-0, i18-0, MM-0)` · irds `tripclick/val` · RESTRICTED
(source dataset must be requested)
Real health-search logs: the validation collection alone has about 3.5K
queries, 82K click-derived qrels, and 1.52M documents. This is a valuable
consumer-health distribution, but access terms, user-data/privacy review,
medical-domain overlap, and noisy behavioral qrels gate it.

### QReCC
`(QQ, LLM-0, G, NT-1, i18-0, MM-0)` · custom/HF · CC-BY-SA-3.0
14K conversations, 81K question-answer pairs, and a roughly 54M-passage web
corpus. It offers rich conversational rewriting, but is expensive and
share-alike; run the smaller CAsT history pilot before taking on QReCC. If
admitted, use serialized history as the query and the official retrieval
labels rather than treating answer strings as qrels.

### KuaiSearch
`(QQ, LLM-0, S, NT-1, i18-1, MM-0)` · hf
`benchen4395/KuaiSearch` · MIT
Large Chinese e-commerce search data: 2.57M authentic queries and 18.6M
products in the full release; the Lite variant still has about 556K queries
and 6.63M items. It includes recall, ranking, and relevance signals and would
be a strong commerce lane after multilingual dense labeling is supported.
Running it through the current English labeler would manufacture sparse wins.

### KuaiSAR
`(QQ, LLM-0, S, NT-1, i18-1, MM-0)` · custom · CC-BY-NC-SA-4.0
Chinese short-video search/recommendation logs with about 454K queries, 3.0M
search items, and 5.1M search actions. It adds authentic behavioral queries,
but non-commercial/share-alike terms, Chinese labeling support, and the design
of interaction-derived qrels must be resolved first.

### TREC Product Search 2023
`(QQ, LLM-1, S, NT-1, i18-0, MM-0)` · custom/NIST + ESCI corpus · mixed terms
1.66M products, about 30.7K train/dev queries, and 926 test queries, only 182
of which are judged. Because the topics were generated with GPT-4 or extracted
from product title/description spans, this is an evaluation stress set rather
than a new authentic commerce distribution; ESCI and WANDS come first. Use
official qrels and text fields only—images remain out of scope for text v0.

### JDsearch
`(QQ, LLM-0, S, NT-1, i18-1, MM-0)` · custom · CC-BY-NC-SA-4.0 data
171,728 test queries, 12.87M items, and 26.7M interactions from Chinese
e-commerce. Query and product text is anonymized into token IDs, which defeats
taxonomy feature harvesting; keep it out unless a future experiment targets
retrieval behavior independent of readable query features.

### Amazon Reviews / MAVE
Not lane-ready. Amazon Reviews supplies product/review text but no authentic
query-to-corpus qrels. MAVE (`google-research-datasets/MAVE`) supplies
attribute labels over 2.2M Amazon product profiles, but likewise has no search
queries or relevance judgments. Treat either as an augmentation source for
commerce documents/features, never as a standalone lane without an explicit
query-and-qrels provenance design.

### TREC CrisisFACTS
`(QQ, LLM-0, S, NT-1, i18-0, MM-0)` candidate · custom/NIST · mixed social
media terms
Disaster questions over time-ordered Twitter, Reddit, news, and Facebook
streams. The task is temporal fact extraction/summarization rather than plain
ad-hoc retrieval, and source-post redistribution is problematic. Defer until
there is a principled document unit, timestamp-aware query representation,
and qrels conversion.

### TREC Health Misinformation 2020/2021
`(QQ, LLM-0, S, NT-1, i18-0, MM-0)` candidate · custom/NIST +
Common Crawl/C4 · mixed terms
Consumer-health questions and keyword queries judged for relevance,
correctness, and credibility. It could add decision and credibility language,
but has few topics and a multi-aspect objective. A future pilot must define
whether routing utility uses relevance alone or a correctness/credibility
filter; those choices must not be silently collapsed into ordinary qrels.

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

## Licensing and publication standing

Verified 2026-08-20 against primary sources (HF dataset cards, upstream
repos, the program's own terms page) for the 42 pre-Wave-3 lanes, which
collapse to 18 upstream source families. `LANES` now contains 45 lanes; the
three implemented Wave-3 families retain the initial standing below until a
release-focused verification. Re-verify before any release: upstream cards
get retagged.

Wave 3 and the newly deferred candidates above are **not** part of that
verified 42-lane matrix. Their initial standing, to be reverified before
publication, is:

| candidate family | initial license/access standing | ids | query | docs |
| -----------------| --------------------------------| -----| -------| ------|
| Amazon ESCI | repository Apache-2.0; product-content provenance needs review | ok | check | check |
| WANDS | MIT | ok | ok | ok |
| FinDER | CC-BY-NC-4.0 | ok | no | no |
| TREC CAsT 2020 | NIST topics/qrels over MS MARCO + TREC CAR | ok | check | no |
| TechQA | repository Apache-2.0; forum/Technote content needs review | ok | check | check |
| Touché 2020 | redistribution terms not yet verified | check | check | check |
| SciFact | CC-BY-NC-2.0 | ok | no | no |
| TREC Fair Ranking 2020 | topic and Semantic Scholar corpus terms need review | check | check | check |
| TripClick | request-only source dataset and user-log terms | terms | no | no |
| QReCC | CC-BY-SA-3.0 compilation; underlying web corpus needs review | ok | SA | check |
| KuaiSearch | MIT | ok | ok | ok |
| KuaiSAR | CC-BY-NC-SA-4.0 | ok | no | no |
| TREC Product Search 2023 | NIST topics/qrels + Amazon product corpus | ok | check | check |
| JDsearch | CC-BY-NC-SA-4.0 data | ok | no | no |
| Amazon Reviews / MAVE | product-profile/review provenance needs review | check | no | no |
| TREC CrisisFACTS | NIST task + mixed social/news source terms | ok | check | no |
| TREC Health Misinformation | NIST topics/qrels + Common Crawl/C4 corpus | ok | check | no |

**What decides everything is the shape of the artifact, not the lane.**
`select_v3_prototype.py` writes `(dataset, query_id, route_class, certified)`
— pointers plus our own measurement. IDs are facts and `route_class` is ours,
so that shape is the TREC-qrels / `ir_datasets` pattern and carries almost no
upstream copyright. The `query` column (already in `catalog_v3.parquet`)
redistributes their text and per-source licenses bind. Document text binds
share-alike and the scraped-content problems on top.

Columns below: standing for the pointer artifact / + query text / + doc text.
`ok` = permitted, `attrib` = attribution required, `SA` = share-alike
propagates to whatever file it ships in, `terms` = permitted but the upstream
terms restrict *our* use, `no` = do not ship, `check` = unresolved.

| lanes | family | license (verified) | ids | query | docs |
| ------| --------| --------------------| -----| -------| ------|
| 8 | CRUMB `jfkback/crumb` | **CC-BY-NC-4.0** | ok | no | no |
| 1 | msmarco-passage-dev | **non-commercial research only** | terms | no | no |
| 1 | trec-dl-2022 | MS MARCO v2 terms; NIST qrels terms unstated | terms | no | no |
| 1 | orcas | MS MARCO terms; also a real Bing click log | terms | no | no |
| 1 | antique | **none stated**; from Yahoo Webscope L6, whose DUA forbids reposting the data on the web | terms | no | no |
| 2 | RAR-b pools | **no license, no dataset card** on either pooled repo | terms | no | no |
| 12 | BRIGHT `xlangai/BRIGHT` | CC-BY-4.0 (HF tag + repo badge) | ok | attrib | check |
| 5 | FreshStack | **CC-BY-SA-4.0** (Stack Overflow derived) | ok | SA | SA |
| 2 | LoTTE | repo apache-2.0; content is StackExchange CC-BY-SA | ok | attrib | SA |
| 1 | beir-nfcorpus | **CC-BY-SA-4.0** | ok | SA | SA |
| 1 | dbpedia-entity | **CC-BY-SA-4.0** | ok | SA | SA |
| 1 | quest | apache-2.0 | ok | ok | ok |
| 1 | miracl-en-dev | apache-2.0 (topics/qrels repo); corpus is Wikipedia, CC-BY-SA | ok | ok | SA |
| 1 | limit | data CC-BY-4.0, code apache-2.0 | ok | ok | ok |
| 1 | webfaq-eng | CC-BY-4.0; Common Crawl derived, card defers to source-site ToS | ok | ok | check |
| 1 | scirgen-geo-en | CC-BY-4.0 | ok | ok | ok |
| 1 | clerc | underlying Caselaw Access Project is CC0; the HF repo shows **no license tag** | ok | ok | check |
| 1 | gooaq | apache-2.0 on the compilation; the answers are scraped Google answer-box text Allen AI never owned | ok | ok | no |

### The three constraints that bite

**Non-commercial contaminates 11 of 42 lanes.** CRUMB (8) is CC-BY-NC.
MS MARCO, TREC-DL and ORCAS (3) are "non-commercial research purposes only
… without extending any license or other intellectual property rights."
A router dataset published by Qdrant is not obviously non-commercial
research. This is contract/terms, not copyright — it restricts our use, not
only redistribution, so it reaches the derived artifact.

**Share-alike is viral into our own file.** 9 lanes (beir-nfcorpus,
dbpedia-entity, freshstack x5, lotte x2) are CC-BY-SA-4.0. Their *text* in
the same file as CC-BY material forces the whole file to CC-BY-SA, including
our labels. Separate parquet, separate license — that is the only reason to
partition rather than concatenate.

**Two unknowns need an email, not an assumption.** The RAR-b pooled repos
state no license at all (the upstream MATH / GSM8K / HumanEvalPack / MBPP
components are individually permissive, unverified here — permissive parents
do not license a silent repackaging). ANTIQUE states none while sitting on a
Yahoo agreement that forbids reposting; its authors distribute it publicly
anyway, which is their risk position, not a license to us.

Also non-copyright: ORCAS and msmarco-passage-dev query text is real user
search traffic. Republishing raw user-log text is a privacy question
independent of licensing — a second reason those two stay pointer-only.

### Proposed release shape — NOT ratified

A proposal in the sense of the header above: a human ratifies before release.

- **Pointer artifact, the 42 verified pre-Wave-3 lanes** — `selected.parquet`,
  `eval_reserve.parquet`, numeric feature columns, our annotations under
  CC-BY-4.0, plus a per-lane attribution table and a loader that rebuilds
  text from upstream. Precedent: BEIR, MTEB, `ir_datasets`.
- **`queries_permissive.parquet`** — query text for the 18 CC-BY / apache
  lanes (BRIGHT 12, quest, miracl-en-dev, limit, webfaq-eng, scirgen-geo-en,
  clerc, gooaq).
- **`queries_sharealike.parquet`** — the 9 SA lanes, marked CC-BY-SA-4.0.
- **Pointer-only, no text** — the 13 NC / unlicensed lanes (CRUMB 8,
  MS MARCO family 3, antique, RAR-b 2).

Cheap because the artifact is already ID-keyed: the partition is a filter on
`dataset` at write time. The eval reserve stays 42-lane complete either way
— reserves are IDs — so ablations and later versions remain comparable even
for lanes whose text we cannot ship.

### Open items

- `jhu-clsp/CLERC` shows no license tag; CC0 is inferred from CAP upstream.
- Whether NIST asserts terms over the TREC DL-2022 qrels.
- BRIGHT is CC-BY-4.0 as a whole, but its leetcode and aops splits carry
  third-party problem statements the authors cannot license; document text
  from those two splits is the one BRIGHT exposure.

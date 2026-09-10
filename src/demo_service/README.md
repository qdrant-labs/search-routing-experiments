# demo-service — from a collection to inspectable search tests

The 3-minute live demo (design:
`docs/superpowers/specs/2026-09-10-collection-to-search-tests-demo-design.md`).
One pinned Home Depot document, two audience-chosen query behaviors, one
budget-capped live generation with at most one repair, a dense/sparse/hybrid
comparison, and a retained test. **Internal-only** — the corpus is Kaggle
competition data; nothing containing its text may be committed or published.

## Run

```bash
# one-time frontend build (Vite + React -> static dist/, served by FastAPI)
cd src/demo_service/frontend && npm install && npm run build && cd -

poetry run uvicorn pipeline_service.api:app --port 8000   # terminal 1
poetry run uvicorn demo_service.api:app --port 8100       # terminal 2
open http://127.0.0.1:8100
```

The banner tells the truth about the run: **LIVE** (green) when preflight
passed — pipeline-service up, Qdrant collection reachable, `ANTHROPIC_API_KEY`
set, model pricing verified — and **PREPARED REPLAY** (amber) with the reason
otherwise. `.env` at the repo root is loaded automatically.

## Ceilings (all enforced, not advisory)

- **$1.00** total LLM spend per session, refused *before* the call
  (`Budget.reserve`), shared by generate + repair.
- **2 LLM calls** hard ceiling per session (`/run/reset` starts a new one).
- **20 s** generation deadline, **15 s** retrieval deadline; a late result is
  discarded, never rendered.

## Rehearsal

Run each behavior end to end while LIVE, then freeze it as the fallback:

```bash
curl -X POST localhost:8100/run/record-replay   # after retrieve succeeded
```

That writes `replays/<behavior>.json` (gitignored — contains corpus text).
Every fallback trigger (preflight failure, deadline, budget, verify-fail with
no repair room) swaps in that bundle whole: query, checks and rankings travel
together, so prepared rankings can never attach to a live query.

Accepted tests export to `src/data/home-depot/demo_tests/` with full
provenance (behavior, targets, measured checks, answerability verdict and
method, live/replay flag, spend, retrieval config and ranks).

## The three behaviors are three acts

1. **Exact identifier** — "find me this exact part": the query carries the
   document's own detected identifier, never an injected one.
2. **Conversational request** — how people type to AI assistants today: chat
   register, courtesy padding, the need buried mid-sentence. (Folklore says
   padding hurts BM25; measured here, sparse still won — say the measured
   thing, not the folk claim.)
3. **Real traffic** — the head of the actual query distribution: terse,
   damaged, sometimes not in English. The model writes a terse Spanish query,
   a seeded QwertyTypo applies deterministic damage (free, reproducible), and
   the gate demands the mess be *measurable*: a typo span or non-word rate on
   the final text, and the ordered language (es/de) measured by langid on the
   pre-damage text. Language is checked pre-damage on purpose — a typo can
   fool langid on a short string (measured: 'xup' pushed an English query to
   'sv'), and the gate must not pass on an artifact of its own damage.

The runner measures typo/language itself with an all-engines extractor
(spaCy + wordfreq + langid, lazy-loaded), so pipeline-service needs no special
flags.

## Verified rehearsal results (2026-09-10, post-query_embed-fix)

- identifier: `"concrete planer wheel PC5000C"` — all checks pass, source
  ranked **#1 on all three strategies**. Spend $0.0003.
- conversational: first candidate **failed live** (`stopword_ratio` 0.0 vs
  ≥0.4 — a keyword pile with "hello," stapled on); after wiring the cell's
  `looks_like` into the prompt, `"hi there, could you help me find a makita
  cup wheel for concrete planer"` passes (ratio 0.429) — source **dense #2,
  sparse #1, rrf #2**. Spend $0.0006.
- real_traffic: `"rueda de clpa diamante hormigón Makita"` (Spanish, seeded
  typo `clpa`) — all gates green on honest measurements, and **no strategy
  finds the source in top 20**; sparse's top hit lexically matched a product
  named "Diamante Brick". The demo's measured blind spot: this collection has
  no answer for bilingual traffic, and now there is a saved failing test that
  proves it.
- typo folklore, measured: `"makita cup whel"` (one typo) — **dense: not in
  top 20; sparse: #1**. On this collection the typo kills dense, and BM25's
  surviving exact tokens carry it. The opposite of the usual claim.

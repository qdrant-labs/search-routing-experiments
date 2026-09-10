# pipeline-service

Query generation, augmentation, corpus examination, and qrels expansion over HTTP.
Everything deterministic runs here; anything needing a model hands you the prompt
and the postcondition instead of calling one.

## Run it

```bash
poetry run uvicorn pipeline_service.api:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000/docs** — FastAPI generates an interactive page
from the schemas, and it is the fastest way to explore. Everything below is that
page in copy-pasteable form.

In Docker (BuildKit required — see `Dockerfile.pipeline`):

```bash
git submodule update --init src/query-taxonomy
DOCKER_BUILDKIT=1 docker build -f Dockerfile.pipeline -t pipeline-service .
docker run --rm -p 8000:8000 pipeline-service
```

Set `PIPELINE_ALL_ENGINES=1` to enable the spaCy-backed stat banks. Off by
default: they cost a model load per worker and only matter when a verify target
names a syntax signal.

## The workflow, end to end

The endpoints are a loop: **find what your query set lacks → get the orders to
fill it → verify what your model produced → deepen the answer key that made the
measurement ambiguous.**

### 1 · What can this thing generate?

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/generate/list_features -d '{}' -H 'content-type: application/json'
```

`list_features` returns the full catalog — 92 features with their group and
ambiguity tier. Those names are the valid `feature` values everywhere else.

```bash
curl -s -X POST localhost:8000/generate/generate_surface \
  -H 'content-type: application/json' \
  -d '{"feature": "structured_identifiers:sku", "n": 2, "seed": 3}'
```

```json
{"feature": "structured_identifiers:sku",
 "surfaces": ["SRE-919079O", "WPR-23286"]}
```

Surfaces are **grounding-blind**: they are snippets exhibiting one feature, not
queries. You weave them in yourself and then verify the result. `seed` makes them
reproducible.

That last part is not a formality — try it on `WPR-23286`:

```bash
curl -s -X POST localhost:8000/generate/verify -H 'content-type: application/json' -d '{
  "text": "cordless drill WPR-23286 20v",
  "targets": {"spans": [{"feature": "sku", "min_count": 1}], "stats": []}
}'
# {"passed": false, "checks": [{"target": "sku: at least 1 span(s)", "measured": 0.0, "passed": false}]}
```

It fails. `sku` is `AMBIGUOUS` tier; `ticket_like` (PREFIX-digits refs like
`TCK-84721`, `banks/finance.py`) is `MODERATE` and shares its group, so it claims
`WPR-23286` first in resolution — the query reads as a ticket reference, not a
SKU. The generator wasn't wrong, and neither is the resolver: a generated
surface's feature identity is only decided once it's re-measured in the group's
claim order, which is exactly why this loop ends at `verify`, not at
`generate_surface`.

### 2 · What is my query set missing?

```bash
curl -s -X POST localhost:8000/examine -H 'content-type: application/json' -d '{
  "queries": ["cordless drill 20v", "kitchen faucet brushed nickel",
              "how do I install a dishwasher", "DEWALT DCD771C2",
              "paint roller 9 inch", "led shop light 4ft"],
  "floor": 3
}'
```

```
6 queries -> 3/63 cells covered, 0 unclaimed
  damage_free_query               rows=6  predicts=['dense']
  bare_acronym                    rows=1  predicts=['dense_only','sparse_only']
  short_number_lookup             rows=1  predicts=['dense_only','sparse_only']

order sheet: 62 lines, 11 of them mintable
  conversational_courtesy_wrapper deficit=3 operator=decorate
  boolean_operator_query          deficit=3 operator=operator_syntax_rewrite
  datetime_token_present          deficit=3 operator=inject
```

`floor` is the per-cell target you want. Omit it (or pass 0) for coverage only,
with no order sheet.

Read `operator` on each order line. **`operator: null` means no operator family
mints that cell** — it fills from natural supply or not at all. In the run above
only 11 of 62 deficits are mintable, and that is the honest state of the
generator coverage, not a bug in the report.

For a whole lane, upload the archive instead. Any `queries.parquet` or
`queries.csv` inside it works, so the contract is the `LanePaths` layout the repo
already uses:

```bash
curl -s -X POST "localhost:8000/examine/archive?floor=3" -F file=@mylane.zip
```

### 3 · Get the orders for a thin cell

Take a `cell` from the order sheet:

```bash
curl -s -X POST localhost:8000/augment/cell -H 'content-type: application/json' -d '{
  "cell": "conversational_courtesy_wrapper",
  "text": "cordless drill 20v",
  "parent": {"query_id":"q1","dataset":"home-depot","query":"cordless drill 20v",
             "floors":[],"surfaces":[],"bank":"politeness"}
}'
```

```
cell=conversational_courtesy_wrapper  predicts=['dense_only']  actionable=2/3

[select     ] stopword_ratio>=0.4                            operator=None
[query_only ] length_words>=10.0                             operator=stat_rewrite
    prompt: "Rewrite the user's query so that its length_words lands at least 10..."
    must satisfy: {"stats":[{"stat":"length_words","min_value":10.0}]}
[query_only ] greeting>=1, politeness>=1, interjection>=1     operator=decorate
    prompt: "Rewrite the user's search query by weaving in a natural greeting..."
    must satisfy: {"spans":[{"feature":"greeting","min_count":1}]}
```

One cell is several requirements, and each gets its own stage:

| stage | meaning |
|---|---|
| `query_only` | an operator rewrites the text — actionable, has a prompt |
| `corpus` | needs a document surface, so the answer key is minted from it |
| `constrain` | the requirement wants *less* of something; nothing can be added |
| `select` | nothing mints it — must be found in existing supply |
| `rebuild` | every alternative would change what the query asks for |

`parent` is the source row. Operators read named columns off it, so an incomplete
one gets you a `422` naming the column rather than a 500.

### 4 · Verify what your model wrote

The service never calls a model. Run your own against the `prompt`, then check the
result against the `targets` you were handed:

```bash
curl -s -X POST localhost:8000/generate/verify -H 'content-type: application/json' -d '{
  "text": "cordless drill 20v",
  "targets": {"spans": [{"feature": "greeting", "min_count": 1}], "stats": []}
}'
# {"passed": false, "checks": [{"target": "greeting: at least 1 span(s)", "measured": 0.0, "passed": false}]}

curl -s -X POST localhost:8000/generate/verify -H 'content-type: application/json' -d '{
  "text": "hi there, could you please help me find a cordless drill 20v for my garage",
  "targets": {"spans": [{"feature": "greeting", "min_count": 1}],
              "stats": [{"stat": "length_words", "min_value": 10}]}
}'
# {"passed": true, "checks": [{"measured": 1.0, "passed": true}, {"measured": 15.0, "passed": true}]}
```

Iterate until it passes. This is the whole quality gate: a candidate that fails
re-measurement never earned the shape it was ordered to have.

### 5 · Deepen the answer key

The measurement bottleneck in this project was never the objective weights — it
was qrels depth. Judge pairs the answer key never covered:

```bash
curl -s -X POST localhost:8000/qrels/expand -H 'content-type: application/json' -d '{
  "pairs": [{"dataset":"home-depot","query_id":"q1","doc_id":"d1",
             "query":"cordless drill 20v","doc_text":"20V MAX cordless drill, 2 batteries"}],
  "run_id": "my-run",
  "max_spend_usd": 0.50,
  "dry_run": true
}'
```

```json
{"submitted": 1, "already_judged": 0, "to_judge": 1,
 "estimated_usd": 0.0003, "within_ceiling": true,
 "operating_point": {"precision_relevant": 0.987, "recall_relevant": 0.308,
                     "agreement_overall": 0.523, "measured_on_pairs": 3600}}
```

Drop `dry_run` to actually spend. Four things to know:

- **`max_spend_usd` is required.** No default, because this is the only endpoint
  that costs money. Omitting it is a `422`.
- **Always dry-run first.** The estimate uses the same generous 3-chars-per-token
  rule as the enforcing budget, so it over-estimates rather than under.
- **Read `operating_point` before acting on a negative.** At 98.7% precision and
  30.8% recall, `relevant: false` overwhelmingly means *the judge found no
  support*, not *the judge checked and rejected*. Never delete a qrel on a
  negative.
- **Batches cap at 500 and the cache does the rest.** Verdicts key on
  `(dataset, query_id, doc_id)`, so looping a large work list re-pays nothing for
  pairs an earlier batch banked. A budget stop is a partial success:
  already-paid verdicts are banked and reported with `stopped_on_budget: true`.

## Endpoint reference

| Method | Path | Spends? |
|---|---|---|
| GET | `/health` | no |
| GET | `/generate/tools` | no |
| POST | `/generate/{tool}` — `list_features`, `generate_surface`, `verify` | no |
| POST | `/examine` | no |
| POST | `/examine/archive` | no |
| GET | `/augment/operators` | no |
| GET | `/augment/dispatch?floor=…` | no |
| POST | `/augment/cell` | no |
| POST | `/augment/plan` | no |
| POST | `/augment/apply` | no |
| POST | `/augment/check` | no |
| POST | `/qrels/expand` | **yes** |

Two ways in to augmentation, for two different callers. **`/augment/cell`** takes
a cell name and is what `/examine`'s order sheet feeds — use this one.
**`/augment/plan` / `/augment/apply` / `/augment/check`** take a single floor key
(`id:uuid`, `marker:politeness`, `logical:operator_syntax`) for callers already
working in floor terms; `/augment/dispatch` tells you which operator serves one.

`/augment/apply` returns the rewritten text when the operator needs no model, and
`needs_model: true` with `text: null` when it does.

## Notes

- Bound to `127.0.0.1` with no auth. It writes to a shared corpus and can spend
  money, so put a token in front of it before binding it to a public interface.
- `/examine` caps at 50,000 queries; archives cap at 256 MB uncompressed and are
  read in memory, never extracted to disk.
- The judge is constructed on first `/qrels/expand` call — it reads a verdict bank
  off disk, and every other endpoint works without one.

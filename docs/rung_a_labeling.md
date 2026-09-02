# Labeling a frozen Rung A plan

`src/scripts/label_rung_plan.py` labels exactly the identities selected in one
immutable `planned_set.parquet`. It is version-neutral: it does not read a
historical composition, a historical route-label artifact, or a release-specific
pool.

The current 100K-ceiling plan contains 91,093 selected rows. The directory name
describes the composition run; the labeler uses the parquet row count as the
exact completion target and does not pad it to 100,000.

## Inspect without spending

```bash
poetry run python src/scripts/label_rung_plan.py \
  --plan src/data/rungs/100k-v2/planned_set.parquet \
  --plan-only
```

This prints planned, checkpointed, and remaining rows by corpus regime and lane.
It does not connect to Qdrant and does not create or modify files.

## Run labeling

Start Qdrant and configure `QDRANT_URL` and, when required,
`QDRANT_API_KEY`. Then run:

```bash
poetry run python src/scripts/label_rung_plan.py \
  --plan src/data/rungs/100k-v2/planned_set.parquet
```

## Advisory spend check

The spend check is independent of labeling and can be run before, during, or
after an interrupted labeling process. Supply the actual initial novelty
threshold used to compose this plan; it is deliberately not inferred from a
candidate default:

```bash
poetry run python src/scripts/check_rung_spend.py \
  --plan-dir src/data/rungs/100k-v2 \
  --theta0 <actual-composition-theta0>
```

It reads only the frozen Rung A plan, trace, coverage, catalog, and provenance.
It never reads label checkpoints or contacts Qdrant. The deterministic report is
written to `src/data/rungs/100k-v2/spend_check.json`.

Both `ADVISORY PASS` and `ADVISORY FAIL` exit successfully. A FAIL records
structural reasons for review but does not stop or alter labeling. A non-zero
exit is reserved for missing, malformed, or mutually inconsistent frozen
inputs.

The current check covers reachable floors, unreachable generation debt, answer
coverage, novelty, family/operator concentration, and admitted-versus-selected
augmentation supply. Predicted downstream label yield remains unactivated, and
concentration/novelty have no calibrated warning thresholds, so this is a
structural spend sanity check rather than a claim of downstream utility.

The default run directory is:

```text
src/data/rungs/100k-v2/labeling/
```

The frozen plan is partitioned by answer regime:

- `natural`: native queries evaluated against the immutable lane corpus;
- `supplemented`: admitted generated or augmented queries evaluated with their
  declared answer judgments against the lane corpus;
- `constructed`: `synthesize` queries evaluated against an isolated collection
  containing their constructed answer documents and lane distractors.

`lane_synthesize` belongs to `supplemented`, not `constructed`: its queries are
grounded in documents already owned by the lane. Only the `synthesize` operator
uses constructed answer documents.

Each regime checkpoints independently below its own directory. Re-running the
same command resumes only those run-local checkpoints. A directory without
`run.json`, or one owned by a different plan fingerprint, is rejected rather
than treated as reusable label supply.

The combined artifact is published atomically at:

```text
src/data/rungs/100k-v2/labeling/labels.parquet
```

It appears only when there is exactly one valid label row for every planned
identity. All-zero retrieval is a completed measurement; a missing row is not.

To process selected lanes while testing operational setup, use `--only`:

```bash
poetry run python src/scripts/label_rung_plan.py \
  --plan src/data/rungs/100k-v2/planned_set.parquet \
  --only antique beir-nfcorpus
```

That command checkpoints those lanes but does not publish the combined artifact
until every planned lane is complete.

The labeling command does not automatically run or enforce the advisory spend
check, and it does not run or optimize Rung B. Rung A has already fixed
membership; labeling only measures the planned rows. Rung B remains a separate
post-label integrity and reporting operation.

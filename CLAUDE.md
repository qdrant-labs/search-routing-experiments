# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Python project managed with Poetry, targeting Python >=3.11. The venv is stored in-project at `.venv/`.

**Purpose:** Building a dataset for hybrid search using Reciprocal Rank Fusion (RRF), likely for Qdrant evaluation or benchmarking.

## Commands

```bash
# Install dependencies
poetry install

# Add a dependency
poetry add <package>

# Run a script
poetry run python <script.py>

# Activate the venv
poetry shell

# Tests and lint (scope ruff to these paths — notebooks have unrelated warnings)
poetry run pytest
poetry run ruff check src/query_taxonomy tests
```

## Query taxonomy (`src/query_taxonomy/`)

Feature extractors over queries/documents for corpus profiling and Strategy
Router dataset work, organized by the parent groups of
`query-taxonomy.csv` (`FeatureGroup`). Span-emitting extractors are "banks";
statistical metrics are scalar extractors. Claim resolution is within-group
only: banks of different groups never compete for the same char ranges.

### Layout

- `taxonomy.py` — the taxonomy itself, code twin of `query-taxonomy.csv`:
  `FeatureGroup` plus every group's member vocabulary (`StructuralIdentifier`
  79-member enum, `Domain`, `SentenceMarker`, future enums). No machinery —
  imports nothing but the stdlib.
- `core.py` — extraction mechanics: `FeatureSpan`, `FeatureStat`,
  `AmbiguityTier`, and one bank family `GeneralBank[EngineT, OutT]` —
  `EngineT` is the definition artifact (`RegexBuilder` for regex banks, a
  tokenizer/model for stat banks), `OutT` is a constrained TypeVar over
  exactly `FeatureSpan | FeatureStat`, and `compute(text)` is the single
  output verb. `RegexBank` and `StatBank` fix one side each. `ambiguity`
  defaults to RIGID (stats are solid numbers); span-group bases re-abstract
  it.
- `__init__.py` — `FEATURE_BANKS: dict[FeatureGroup, tuple[type[GeneralBank],
  ...]]`, the group registry. Must stay defined before any `features`
  re-export (import-order rule); group↔key consistency is validated in
  `FeatureExtractor.__init__`.
- `banks/` — Structured Identifiers group. `banks/core.py` holds
  `IdentifierBank` (`RegexBank` + `domain`, group fixed) and shared
  char-class subexpressions. `banks/<domain>.py` — one module per `Domain`;
  a bank MUST live in the module matching its `domain` property (tested
  invariant). `banks/__init__.py` — the `BANKS` tuple. Resolution sorts
  stably by `AmbiguityTier`, so tuple order = claim priority within a tier.
  Ordering constraints are load-bearing and commented inline (e.g. CIDR
  before IP, URI before Email, Booking before SKU).
- `markers/` — Sentence Markers group: `MarkerBank`, `phrase_alternation`
  (boundary-guarded, case-insensitive closed-list helper), `MARKER_BANKS`
  tuple. New groups (`logical/`, `corruption/`, `metrics/`) follow this
  package pattern: vocabulary enum in `taxonomy.py`, machinery in the
  package.
- `features.py` — pydantic models (`QueryFeatures`, `SpanProfile`,
  `StatProfile`, `CorpusFeatures`; spans and stats sections nested by
  group) and `FeatureExtractor`: `resolve(text, *, groups=None)` /
  `extract(queries, *, groups=None)` run banks group by group in tier order
  with a per-group, per-text registry of claimed char ranges, so a
  lower-priority bank never re-claims overlapping text (NUMBER can't steal
  `1.0` from a claimed `v1.0.0`) while groups stay independent layers.
  `summary()` sections by group; Domain sub-grouping only inside
  structured_identifiers.

### Writing banks

- edify `RegexBuilder` is IMMUTABLE — every call returns a clone. `define()`
  must return the completed chain; mutating in place silently yields an empty
  pattern.
- Quantifiers come BEFORE their element: `.exactly(4).digit()` -> `\d{4}`.
- Alternation (`any_of`) is first-match, not longest-match — order branches
  so the longer/consuming branch comes first.
- Identifier banks implement `name`, `ambiguity`, `domain`, `define`; banks
  of other groups implement `name`, `ambiguity`, `define` (group is fixed by
  their base class). Keyword-gated or high-entropy formats -> RIGID; known
  collisions -> MODERATE; bare caps/digit shapes or common words ->
  AMBIGUOUS.
- Deliberate exclusions (SMILES, SEDOL, IMSI, decimal odds, bare Dewey/ISO)
  have their rationale in the bank/enum docstring — don't "fix" them back in
  without reading it.
- Every bank needs ~2 positive / 2 negative cases in `tests/test_banks.py`
  (`test_every_bank_has_cases` enforces presence). Keep negatives blatant,
  not exhaustive.


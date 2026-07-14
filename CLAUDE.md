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

Structured-identifier detection: 79 regex banks that tag queries/documents with
identifier types (UUID, version, ticker, ICD code, ...) for corpus profiling
and Strategy Router dataset work.

### Layout

- `banks/core.py` — `StructuralIdentifier` (79-member enum), `Domain`,
  `AmbiguityTier`, `IdentifierMatch`, `RegexBank` base, shared char-class
  subexpressions.
- `banks/<domain>.py` — one module per `Domain` (`general`, `network`, `tech`,
  `finance`, `legal`, `medical`, `logistics`, `media`). A bank MUST live in the
  module matching its `domain` property (tested invariant).
- `banks/__init__.py` — the `BANKS` tuple. Resolution sorts stably by
  `AmbiguityTier`, so tuple order = claim priority within a tier. Ordering
  constraints are load-bearing and commented inline (e.g. CIDR before IP,
  URI before Email, Booking before SKU).
- `features.py` — pydantic models (`DocumentIdentifier`, `QueryIdentifiers`,
  `CorpusIdentifiers`) and `CorpusIdentifierExtractor`: runs banks in tier
  order with a per-text registry of claimed char ranges, so a lower-priority
  bank never re-claims overlapping text (NUMBER can't steal `1.0` from a
  claimed `v1.0.0`).

### Writing banks

- edify `RegexBuilder` is IMMUTABLE — every call returns a clone. `define()`
  must return the completed chain; mutating in place silently yields an empty
  pattern.
- Quantifiers come BEFORE their element: `.exactly(4).digit()` -> `\d{4}`.
- Alternation (`any_of`) is first-match, not longest-match — order branches
  so the longer/consuming branch comes first.
- Each bank implements `name`, `ambiguity`, `domain`, `define`. Keyword-gated
  or high-entropy formats -> RIGID; known collisions -> MODERATE; bare
  caps/digit shapes -> AMBIGUOUS.
- Deliberate exclusions (SMILES, SEDOL, IMSI, decimal odds, bare Dewey/ISO)
  have their rationale in the bank/enum docstring — don't "fix" them back in
  without reading it.
- Every bank needs ~2 positive / 2 negative cases in `tests/test_banks.py`
  (`test_every_bank_has_cases` enforces presence). Keep negatives blatant,
  not exhaustive.


"""Anchored ordinal buckets over `query_corpus_stats.parquet` for the LLM-A
judge prompt. Quartiles are computed per feature across per-collection means,
so a bucket says "top quartile across profiled collections" rather than
"top quartile of this collection's queries"."""
from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from hybrid_search_rrf_dataset.router import DATA_DIR

STATS_PATH: Final[Path] = DATA_DIR / "route_labels" / "query_corpus_stats.parquet"
QUARTILES_PATH: Final[Path] = DATA_DIR / "route_labels" / "stats_quartiles.parquet"

MIN_PMI_SENTINEL: Final[float] = -1.0

FIELD_NAMES: Final[dict[str, str]] = {
    "avg_idf":        "Query term rarity (avg IDF)",
    "max_idf":        "Rarest query term (max IDF)",
    "oov_share":      "Out-of-vocabulary share",
    "collection_size": "Collection size",
    "avg_doc_length": "Average document length",
    "vocab_overlap":  "Query-collection vocabulary overlap",
    "mean_pmi":       "Query term co-occurrence (mean PMI)",
    "min_pmi":        "Rarest term co-occurrence (min PMI)",
}

# Features that get quartile buckets. Others get raw/absolute rendering below.
QUARTILE_FEATURES: Final[tuple[str, ...]] = (
    "avg_idf", "max_idf", "vocab_overlap", "avg_doc_length", "mean_pmi",
)

# Domain-appropriate bucket names per feature. Order: (bottom, below-median, above-median, top).
BUCKET_LABELS: Final[dict[str, tuple[str, str, str, str]]] = {
    "avg_idf":        ("low", "moderate", "high", "very high"),
    "max_idf":        ("low", "moderate", "high", "very high"),
    "vocab_overlap":  ("low", "moderate", "high", "very high"),
    "avg_doc_length": ("short", "medium", "long", "very long"),
    "mean_pmi":       ("weak", "moderate", "strong", "very strong"),
}

QUALIFIERS: Final[tuple[str, str, str, str]] = (
    "bottom quartile", "below median", "above median", "top quartile",
)


def compute_quartiles(stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate to per-collection means first, then quartile per feature."""
    if stats is None:
        stats = pd.read_parquet(STATS_PATH)
    per_collection = stats.groupby("dataset")[list(QUARTILE_FEATURES)].mean()
    q = per_collection.quantile([0.25, 0.5, 0.75]).T
    q.columns = ["q25", "q50", "q75"]
    return q


def cached_quartiles(force: bool = False) -> pd.DataFrame:
    """Load or compute-and-cache the quartile table."""
    if QUARTILES_PATH.exists() and not force:
        return pd.read_parquet(QUARTILES_PATH)
    q = compute_quartiles()
    QUARTILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    q.to_parquet(QUARTILES_PATH)
    return q


def _bucket_index(value: float, q: pd.Series) -> int:
    """0=bottom, 1=below-median, 2=above-median, 3=top."""
    if value < q["q25"]:
        return 0
    if value < q["q50"]:
        return 1
    if value < q["q75"]:
        return 2
    return 3


def _bucket_line(feat: str, value: float, q: pd.DataFrame) -> str:
    i = _bucket_index(value, q.loc[feat])
    return f"{BUCKET_LABELS[feat][i]} ({QUALIFIERS[i]} across profiled collections)"


def _fmt_docs(n: float) -> str:
    # thresholds slightly under the pretty boundary so the log10(N+1) un-log doesn't
    # push a clean 1e6 to "999K": 10**6 - 1 == 999999 rounds to 1.0M for display.
    if n >= 999_500:
        return f"~{n / 1e6:.1f}M"
    if n >= 950:
        return f"~{n / 1e3:.0f}K"
    return f"~{int(round(n))}"


def render_stats_block(row: pd.Series, quartiles: pd.DataFrame | None = None) -> str:
    """Render the bulleted stats block for one query's row. Returns plain text —
    the caller adds any leading header. Raw values stay out of the prompt except
    where they carry their own meaning (OOV %, collection doc count)."""
    q = quartiles if quartiles is not None else cached_quartiles()
    lines: list[str] = []

    lines.append(f"- {FIELD_NAMES['avg_idf']}: {_bucket_line('avg_idf', float(row['avg_idf']), q)}")
    lines.append(f"- {FIELD_NAMES['max_idf']}: {_bucket_line('max_idf', float(row['max_idf']), q)}")

    oov_pct = float(row["oov_share"]) * 100
    if oov_pct <= 0:
        oov_desc = "no query terms absent from collection vocabulary"
    else:
        oov_desc = f"{oov_pct:.0f}% of query terms absent from collection vocabulary"
    lines.append(f"- {FIELD_NAMES['oov_share']}: {oov_desc}")

    n_docs = 10 ** float(row["collection_size"]) - 1
    lines.append(f"- {FIELD_NAMES['collection_size']}: {_fmt_docs(n_docs)} documents")

    n_tokens = 10 ** float(row["avg_doc_length"]) - 1
    doc_len_bucket = _bucket_line("avg_doc_length", float(row["avg_doc_length"]), q)
    lines.append(f"- {FIELD_NAMES['avg_doc_length']}: {doc_len_bucket} (~{int(n_tokens)} tokens)")

    lines.append(f"- {FIELD_NAMES['vocab_overlap']}: {_bucket_line('vocab_overlap', float(row['vocab_overlap']), q)}")
    lines.append(f"- {FIELD_NAMES['mean_pmi']}: {_bucket_line('mean_pmi', float(row['mean_pmi']), q)}")

    min_pmi = float(row["min_pmi"])
    if min_pmi <= MIN_PMI_SENTINEL + 1e-6:
        lines.append(f"- {FIELD_NAMES['min_pmi']}: contains a never-co-occurring term pair")
    else:
        lines.append(f"- {FIELD_NAMES['min_pmi']}: {min_pmi:+.2f} (raw PMI)")

    return "\n".join(lines)

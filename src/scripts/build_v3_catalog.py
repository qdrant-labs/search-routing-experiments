"""v3 catalog — ADDITIVE; the v2 catalog.parquet is frozen and untouched.

Over the labelled pool only (the v3 selector's universe), reuse every v2
feature column by JOIN and add the NEW taxonomy columns computed here:
corruption.* / unknown_token_rate.* (CORRUPTION), term_rarity.* /
subword_fragmentation.* (STATISTICAL_METRICS), plus per-query corpus stats
(query_corpus.*) when query_corpus_stats.parquet exists. No spaCy re-run, no
440K rebuild, no v2 mutation — the spaCy-derived v2 columns come from the join.

    poetry run python src/scripts/build_v3_catalog.py           # build data/v3/catalog_v3.parquet
    poetry run python src/scripts/build_v3_catalog.py --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from composition.mini_catalog import feature_columns
from query_taxonomy.core import Engine
from query_taxonomy.features import FeatureExtractor
from query_taxonomy.taxonomy import FeatureGroup

DATA = Path(__file__).resolve().parent.parent / "data"
V2_CATALOG = DATA / "feature_table" / "catalog.parquet"
LABELS = DATA / "route_labels" / "labels.parquet"
QUERY_CORPUS_STATS = DATA / "route_labels" / "query_corpus_stats.parquet"
OUT = DATA / "v3" / "catalog_v3.parquet"

# groups holding the NEW columns; the spaCy stat banks in STATISTICAL_METRICS
# are filtered out by the engine set, so only the new stats are computed.
NEW_GROUPS = [FeatureGroup.CORRUPTION, FeatureGroup.STATISTICAL_METRICS]
NEW_PREFIXES = (
    "corruption.", "unknown_token_rate.", "term_rarity.", "subword_fragmentation.",
)


def _new_columns(labels: pd.DataFrame) -> pd.DataFrame:
    """The new taxonomy columns per labelled query (v2 catalog lacks them)."""
    extractor = FeatureExtractor(
        engines=(Engine.REGEX, Engine.WORDFREQ, Engine.TOKENIZER)
    )
    rows = []
    for text in tqdm(labels["query"].fillna("").astype(str), unit="query"):
        cols = feature_columns(extractor.resolve(text, groups=NEW_GROUPS))
        rows.append({k: v for k, v in cols.items() if k.startswith(NEW_PREFIXES)})
    return pd.DataFrame(rows).fillna(0.0)


def build_v3_catalog(*, force: bool = False) -> pd.DataFrame:
    if OUT.exists() and not force:
        return pd.read_parquet(OUT)
    labels = pd.read_parquet(
        LABELS, columns=["dataset", "query_id", "query"]
    ).astype({"query_id": str})
    v2 = pd.read_parquet(V2_CATALOG).astype({"query_id": str})

    # reuse v2 feature columns for the labelled rows (left join keeps every row)
    base = labels[["dataset", "query_id"]].merge(
        v2, on=["dataset", "query_id"], how="left"
    )
    new = _new_columns(labels)
    v3 = pd.concat(
        [base.reset_index(drop=True), new.reset_index(drop=True)], axis=1
    )

    if QUERY_CORPUS_STATS.exists():
        qcs = pd.read_parquet(QUERY_CORPUS_STATS).astype({"query_id": str})
        stat_cols = [c for c in qcs.columns if c not in ("dataset", "query_id")]
        qcs = qcs.rename(columns={c: f"query_corpus.{c}" for c in stat_cols})
        v3 = v3.merge(qcs, on=["dataset", "query_id"], how="left")

    # span/stat absence = 0; corpus stats stay NaN where a lane has no index
    # (a corpus band over a NaN row is correctly excluded, not read as 0).
    taxonomy_cols = [c for c in v3.columns if c.startswith(NEW_PREFIXES)]
    v3[taxonomy_cols] = v3[taxonomy_cols].fillna(0.0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    v3.to_parquet(OUT, index=False)
    return v3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    v3 = build_v3_catalog(force=args.force)
    new_cols = [c for c in v3.columns if c.startswith(NEW_PREFIXES)]
    corpus_cols = [c for c in v3.columns if c.startswith("query_corpus.")]
    print(f"v3 catalog: {len(v3):,} rows x {v3.shape[1]} cols -> {OUT}")
    print(f"  new taxonomy cols ({len(new_cols)}): {new_cols}")
    print(f"  corpus cols ({len(corpus_cols)}): {corpus_cols or '— (run collection_features --per-query)'}")


if __name__ == "__main__":
    main()

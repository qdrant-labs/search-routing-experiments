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
from query_taxonomy.banks import BANKS
from query_taxonomy.core import Engine
from query_taxonomy.features import FeatureExtractor
from query_taxonomy.taxonomy import FeatureGroup

DATA = Path(__file__).resolve().parent.parent / "data"
V2_CATALOG = DATA / "feature_table" / "catalog.parquet"
QUERY_CORPUS_STATS = DATA / "route_labels" / "query_corpus_stats.parquet"
OUT = DATA / "v3" / "catalog_v3.parquet"

# groups holding the NEW columns; the spaCy stat banks in STATISTICAL_METRICS
# are filtered out by the engine set, so only the new stats are computed.
# STRUCTURED_IDENTIFIERS rides along for the banks that shipped after the v2
# catalog was built (adversarial review, Tier 1 #11): the whole group is
# resolved, never a subset, because claim priority within a group decides which
# bank wins a char range.
NEW_GROUPS = [
    FeatureGroup.CORRUPTION,
    FeatureGroup.STATISTICAL_METRICS,
    FeatureGroup.STRUCTURED_IDENTIFIERS,
]
NEW_PREFIXES = (
    "corruption.", "unknown_token_rate.", "term_rarity.", "subword_fragmentation.",
)
IDENTIFIER_PREFIX = f"{FeatureGroup.STRUCTURED_IDENTIFIERS.value}."


def _absent_identifier_columns(present: set[str]) -> list[str]:
    """Identifier banks the v2 join carries no column for. Zero-filled rather
    than left out: a cell banding on an absent column raises instead of reading
    the zero it means."""
    shipped = {f"{IDENTIFIER_PREFIX}{bank().name.value}" for bank in BANKS}
    return sorted(shipped - present)


def _new_columns(labels: pd.DataFrame, absent_ids: list[str]) -> pd.DataFrame:
    """The new taxonomy columns per labelled query (v2 catalog lacks them)."""
    extractor = FeatureExtractor(
        engines=(Engine.REGEX, Engine.WORDFREQ, Engine.TOKENIZER)
    )
    keep = (*NEW_PREFIXES, *absent_ids)
    rows = []
    for text in tqdm(labels["query"].fillna("").astype(str), unit="query"):
        cols = feature_columns(extractor.resolve(text, groups=NEW_GROUPS))
        rows.append({k: v for k, v in cols.items() if k.startswith(keep)})
    new = pd.DataFrame(rows)
    missing = [c for c in absent_ids if c not in new.columns]
    return new.reindex(columns=[*new.columns, *missing]).fillna(0.0)


def _pool_labels() -> pd.DataFrame:
    """The selector's own union (re-derived v2 pool + additive v3 labels), so a
    v3-labelled query gets catalog rows and therefore cell membership — the 342
    in-loop rows previously fell through to zero cells silently."""
    from composition.pool_v3 import LabelledPool

    return LabelledPool().labels()[["dataset", "query_id", "query"]]


def build_v3_catalog(*, force: bool = False) -> pd.DataFrame:
    if OUT.exists() and not force:
        return pd.read_parquet(OUT)
    labels = _pool_labels()
    v2 = pd.read_parquet(V2_CATALOG).astype({"query_id": str})

    # reuse v2 feature columns for the labelled rows (left join keeps every row)
    base = labels[["dataset", "query_id"]].merge(
        v2, on=["dataset", "query_id"], how="left"
    )
    absent_ids = _absent_identifier_columns(set(base.columns))
    new = _new_columns(labels, absent_ids)
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
    taxonomy_cols = [
        c for c in v3.columns if c.startswith((*NEW_PREFIXES, *absent_ids))
    ]
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
    still_absent = _absent_identifier_columns(set(v3.columns))
    print(f"v3 catalog: {len(v3):,} rows x {v3.shape[1]} cols -> {OUT}")
    print(f"  new taxonomy cols ({len(new_cols)}): {new_cols}")
    print(f"  identifier banks without a column: {still_absent or '— (all 79 present)'}")
    print(f"  corpus cols ({len(corpus_cols)}): {corpus_cols or '— (run collection_features --per-query)'}")


if __name__ == "__main__":
    main()

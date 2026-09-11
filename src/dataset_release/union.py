"""Runtime `union_233k` and `91k`: cascade + v2 + v3 + v3_aug, deduped on (dataset, query_id)."""

from pathlib import Path

import pandas as pd

KEY = ("dataset", "query_id")
# Spec §4 D1 schema — provenance and scored_against are load-bearing.
SCHEMA = (
    "dataset", "query_id", "route", "query", "score",
    "score_dense_only", "score_pure_rrf", "score_sparse_only",
    "shape", "metric_name", "min_relevance", "provenance", "scored_against",
)


def _cascade_union(
    cascade: pd.DataFrame,
    v2: pd.DataFrame,
    v3: pd.DataFrame,
    v3_aug: pd.DataFrame,
) -> pd.DataFrame:
    """Concat [v2, v3, v3_aug, cascade] on the SCHEMA cols, drop dupes keep=last on (dataset, query_id)."""
    sources = (v2, v3, v3_aug, cascade)
    frames = [s[[c for c in SCHEMA if c in s.columns]] for s in sources]
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .drop_duplicates(subset=list(KEY), keep="last")
        .reset_index(drop=True)
    )


def _text_bearing(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["query"].notna() & df["query"].astype(str).str.strip().ne("")].copy()


def build_91k(data_dir: Path) -> pd.DataFrame:
    """The text-bearing cascade rung on its own — the current 91K target, and the union's collision winner."""
    return _text_bearing(
        pd.read_parquet(Path(data_dir) / "rungs" / "100k-v2" / "labeling" / "labels.parquet")
    )


def build_union_233k(data_dir: Path) -> pd.DataFrame:
    """Load the four D1 sources under data_dir and construct the text-bearing union_233k."""
    data_dir = Path(data_dir)
    v2 = _text_bearing(pd.read_parquet(data_dir / "route_labels" / "labels.parquet"))
    v3 = _text_bearing(pd.read_parquet(data_dir / "v3" / "labels.parquet"))
    v3_aug = _text_bearing(pd.read_parquet(data_dir / "v3" / "augmented" / "labels.parquet"))
    return _cascade_union(build_91k(data_dir), v2, v3, v3_aug)
    
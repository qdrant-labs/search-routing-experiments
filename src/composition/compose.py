"""Artifact paths shared by the composition builders, and the query-text
join every selection artifact ends with."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from dataset_registry.core import RegistryDataset

DEFAULT_CATALOG = (
    Path(__file__).resolve().parent.parent
    / "data" / "feature_table" / "catalog.parquet"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "composition"


def join_text(
    frame: pd.DataFrame, datasets: Mapping[str, RegistryDataset],
) -> pd.Series:
    """Query text for (dataset, query_id) rows, read from the registry caches."""
    texts = pd.Series(pd.NA, index=frame.index, dtype="object")
    for name, group in frame.groupby("dataset"):
        ids = group["query_id"].astype(str)
        cache = pd.read_parquet(
            datasets[name].cache_path,
            columns=["query_id", "text"],
            filters=[("query_id", "in", ids.tolist())],
        )
        lookup = cache.set_index(cache["query_id"].astype(str))["text"]
        texts.loc[group.index] = ids.map(lookup).to_numpy()
    missing = texts.isna()
    assert not missing.any(), (
        f"text join missed {int(missing.sum())} rows — cache drift"
    )
    return texts

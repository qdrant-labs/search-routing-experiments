"""The generated pool — accepted candidates awaiting the mini-fill.

The mini-fill (d42i, next pass) is the only door into the composition;
this artifact is the queue in front of it. Append is idempotent on
query_id, so re-running a batch never duplicates rows or re-spends LLM
calls on parents that already have a child for the floor.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from augmentation.core import AugmentedCandidate

DEFAULT_POOL_DIR = Path("data") / "augmentation"

_COLUMNS = list(AugmentedCandidate.model_fields)


class GeneratedPool:
    """Owns `<dir>/pool.parquet`."""

    def __init__(self, out_dir: Path | str = DEFAULT_POOL_DIR) -> None:
        self._out_dir = Path(out_dir)

    @property
    def path(self) -> Path:
        return self._out_dir / "pool.parquet"

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=_COLUMNS)
        return pd.read_parquet(self.path)

    def parents_used(self, floor: str) -> set[str]:
        """Parents that already have a child for this floor — excluded from
        selection so reruns spend nothing twice."""
        pool = self.load()
        return set(pool.loc[pool["floor"] == floor, "generated_from"].astype(str))

    def append(self, candidates: list[AugmentedCandidate]) -> pd.DataFrame:
        existing = self.load()
        fresh = pd.DataFrame([c.model_dump() for c in candidates], columns=_COLUMNS)
        fresh = fresh[~fresh["query_id"].isin(set(existing["query_id"]))]
        merged = pd.concat([existing, fresh], ignore_index=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self.path, index=False)
        return merged

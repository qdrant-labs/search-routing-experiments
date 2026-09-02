"""Cost planner — partitions the frozen Rung A plan into cache hits and
misses. Never mutates the plan: if misses exceed the paid-label budget the
run fails BEFORE spending (spec:156-159). Changing the plan size or budget
is a new, explicitly fingerprinted run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rungs.cache import CacheKey, LabelCache


class BudgetExceeded(RuntimeError):
    """Misses exceed the paid-label budget. Never silently truncate."""


@dataclass(frozen=True)
class CostPartition:
    """The frozen partition. `plan_size == len(hits) + len(misses)` always."""

    hits: pd.DataFrame
    misses: pd.DataFrame
    plan_size: int
    budget: int

    @property
    def spend(self) -> int:
        return len(self.misses)


class CostPlanner:
    """Given a frozen planned set and a fingerprint resolver, partition into
    (hits, misses). Fails hard if misses > budget."""

    def __init__(self, cache: LabelCache) -> None:
        self._cache = cache

    def partition(
        self,
        planned_set: pd.DataFrame,
        *,
        fingerprints: dict[tuple[str, str], tuple[str, str, str, str]],
        budget: int,
    ) -> CostPartition:
        """`fingerprints[(dataset, query_id)] = (query_fp, corpus_fp, qrels_fp,
        retrieval_stack_fp)`. Missing/blank fingerprints (spec:151-153) go to
        misses."""
        hit_rows: list[int] = []
        miss_rows: list[int] = []
        for i, row in planned_set.iterrows():
            key = self._key(row, fingerprints)
            if key is not None and self._cache.has(key):
                hit_rows.append(int(i))
            else:
                miss_rows.append(int(i))
        misses = len(miss_rows)
        if misses > budget:
            raise BudgetExceeded(
                f"paid-label budget exceeded: {misses} misses > {budget} budget. "
                "Never silently truncate — change plan size or budget in a new run."
            )
        return CostPartition(
            hits=planned_set.iloc[hit_rows].reset_index(drop=True),
            misses=planned_set.iloc[miss_rows].reset_index(drop=True),
            plan_size=len(planned_set),
            budget=budget,
        )

    @staticmethod
    def _key(
        row: pd.Series,
        fingerprints: dict[tuple[str, str], tuple[str, str, str, str]],
    ) -> CacheKey | None:
        fps = fingerprints.get((str(row["dataset"]), str(row["query_id"])))
        if fps is None or not all(fps):
            return None
        return CacheKey(
            dataset=str(row["dataset"]),
            query_id=str(row["query_id"]),
            query_fp=fps[0],
            corpus_fp=fps[1],
            qrels_fp=fps[2],
            retrieval_stack_fp=fps[3],
        )

    def write_report(self, partition: CostPartition, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        report = pd.DataFrame([{
            "plan_size": partition.plan_size,
            "hits": len(partition.hits),
            "misses": len(partition.misses),
            "budget": partition.budget,
            "spend": partition.spend,
        }])
        report.to_parquet(tmp, index=False)
        tmp.replace(path)

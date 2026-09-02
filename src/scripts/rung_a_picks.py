"""Pick a proportional, deterministic batch of unlabelled registry queries.
Answer coverage is NOT gated here: materialize_rung_a runs next and lands qrels
for these picks, so gating on today's coverage would make that stage a no-op."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from composition.pool_v3 import LabelledPool
from dataset_registry import DATASETS, Query, RegistryDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 42
COLUMNS = ["query_id", "dataset", "operator", "query", "home_lane"]


def _log(t0: float, message: str) -> None:
    print(
        f"[+{time.perf_counter() - t0:6.1f}s] {message}",
        file=sys.stderr,
        flush=True,
    )


def _sample_lane(
    dataset: RegistryDataset,
    excluded: set[tuple[str, str]],
    cap: int,
    rng: np.random.Generator,
) -> list[Query]:
    reservoir: list[Query] = []
    seen = 0
    for query in dataset.load_queries():
        query = Query(str(query.query_id), query.text)
        if (dataset.name, query.query_id) in excluded:
            continue
        seen += 1
        if len(reservoir) < cap:
            reservoir.append(query)
            continue
        replacement = int(rng.integers(seen))
        if replacement < cap:
            reservoir[replacement] = query
    return reservoir


def _proportional_quotas(sizes: list[int], target: int) -> list[int]:
    total = sum(sizes)
    if total <= target:
        return sizes
    exact = np.asarray(sizes, dtype=float) * target / total
    quotas = np.floor(exact).astype(int)
    remainder = target - int(quotas.sum())
    order = np.argsort(-(exact - quotas), kind="stable")
    quotas[order[:remainder]] += 1
    return quotas.tolist()


def _build_picks(
    target: int,
    candidates_per_lane: int,
    t0: float,
) -> pd.DataFrame:
    _log(t0, "loading labelled-pool exclusions")
    # every rung's labels, v4's included: a pick already labelled last night is
    # a wasted retrieval tonight. This asks what is LABELLED, not what is v3
    # supply, so the pool must not be opened native-only.
    labelled = LabelledPool(native_only=False).labels()
    excluded = set(
        zip(labelled["dataset"].astype(str), labelled["query_id"].astype(str))
    )
    _log(t0, f"loaded {len(excluded):,} labelled keys")

    rng = np.random.default_rng(SEED)
    _log(t0, f"sampling candidates from {len(DATASETS)} lanes")
    candidates = [
        _sample_lane(dataset, excluded, candidates_per_lane, rng)
        for dataset in DATASETS
    ]
    quotas = _proportional_quotas([len(rows) for rows in candidates], target)

    records: list[dict[str, str]] = []
    for dataset, rows, quota in zip(DATASETS, candidates, quotas):
        if quota < len(rows):
            indexes = np.sort(rng.choice(len(rows), size=quota, replace=False))
            rows = [rows[int(index)] for index in indexes]
        records.extend(
            {
                "query_id": query.query_id,
                "dataset": dataset.name,
                "operator": "natural",
                "query": query.text,
                "home_lane": dataset.name,
            }
            for query in rows
        )
    return pd.DataFrame.from_records(records, columns=COLUMNS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=100_000)
    parser.add_argument("--candidates-per-lane", type=int, default=5_000)
    parser.add_argument(
        "--out", type=Path, default=Path("src/data/v4/rung_a_picks.parquet")
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.target < 1 or args.candidates_per_lane < 1:
        parser.error("--target and --candidates-per-lane must be positive")

    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out = out.resolve()
    if out.exists() and not args.force:
        print(f"error: {out} already exists; pass --force to overwrite", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    picks = _build_picks(args.target, args.candidates_per_lane, t0)
    _log(
        t0,
        f"selected {len(picks):,} queries across {picks['dataset'].nunique()} lanes",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    try:
        picks.to_parquet(tmp, index=False)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    _log(t0, f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

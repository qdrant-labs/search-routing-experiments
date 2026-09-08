"""Append Rung A picks to per-dataset queries.parquet + qrels.parquet snapshots.

Labelling reads from those snapshots; picks whose (dataset, query_id) isn't in
the snapshot are silently dropped downstream. This script closes that gap by
pulling text from registry cache and qrels from ir_datasets for each dataset
in the picks, appending only rows that (a) have a qrel and (b) whose gold
doc_ids are already in the local corpus.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from dataset_registry import DATASETS

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "src/data"


def _log(t0: float, message: str) -> None:
    print(
        f"[+{time.perf_counter() - t0:6.1f}s] {message}",
        file=sys.stderr,
        flush=True,
    )


def _dataset_lookup() -> dict[str, object]:
    return {str(d.name): d for d in DATASETS}


def _atomic_append(path: Path, extra: pd.DataFrame, dedup_cols: list[str]) -> int:
    """Append rows to a parquet, dedup on `dedup_cols`, write atomically.
    Returns number of NEW rows landed."""
    if extra.empty:
        return 0
    if path.exists():
        current = pd.read_parquet(path)
    else:
        current = pd.DataFrame(columns=extra.columns)
        path.parent.mkdir(parents=True, exist_ok=True)
    merged = pd.concat([current, extra], ignore_index=True)
    merged = merged.drop_duplicates(dedup_cols, keep="first")
    new_count = len(merged) - len(current)
    if new_count == 0:
        return 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        merged.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return new_count


def _corpus_doc_ids(dataset_dir: Path) -> set[str] | None:
    corpus = dataset_dir / "corpus.parquet"
    if not corpus.exists():
        return None
    return set(pd.read_parquet(corpus, columns=["doc_id"])["doc_id"].astype(str))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--picks",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    if not args.picks.exists():
        print(f"error: {args.picks} does not exist", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    picks = pd.read_parquet(args.picks).astype({"query_id": str, "dataset": str})
    _log(t0, f"loaded {len(picks):,} picks across {picks['dataset'].nunique()} datasets")

    lookup = _dataset_lookup()
    stats: dict[str, dict[str, int]] = defaultdict(dict)

    for dataset_name, group in picks.groupby("dataset"):
        dataset_dir = DATA_DIR / dataset_name
        if not dataset_dir.exists():
            _log(t0, f"{dataset_name}: no snapshot dir — skipped")
            stats[dataset_name] = {"picks": len(group), "landed": 0}
            continue
        registry_dataset = lookup.get(dataset_name)
        if registry_dataset is None or not getattr(registry_dataset, "irds_id", None):
            _log(t0, f"{dataset_name}: no irds_id in registry — skipped")
            stats[dataset_name] = {"picks": len(group), "landed": 0}
            continue

        import ir_datasets

        try:
            irds = ir_datasets.load(registry_dataset.irds_id)
        except Exception as error:
            _log(t0, f"{dataset_name}: ir_datasets load failed ({error}) — skipped")
            stats[dataset_name] = {"picks": len(group), "landed": 0}
            continue
        if not irds.has_qrels():
            _log(t0, f"{dataset_name}: ir_datasets has no qrels — skipped")
            stats[dataset_name] = {"picks": len(group), "landed": 0}
            continue

        wanted_qids = set(group["query_id"].astype(str))
        qrels_rows = [
            {"query_id": str(q.query_id), "doc_id": str(q.doc_id), "relevance": int(q.relevance)}
            for q in irds.qrels_iter()
            if str(q.query_id) in wanted_qids
        ]
        if not qrels_rows:
            _log(t0, f"{dataset_name}: no qrels for {len(wanted_qids):,} picks — skipped")
            stats[dataset_name] = {"picks": len(group), "landed": 0}
            continue

        qrels_df = pd.DataFrame(qrels_rows)
        corpus_docs = _corpus_doc_ids(dataset_dir)
        if corpus_docs is not None:
            in_corpus = qrels_df["doc_id"].isin(corpus_docs)
            missing_docs = int((~in_corpus).sum())
            if missing_docs:
                _log(
                    t0,
                    f"{dataset_name}: dropping {missing_docs:,} qrels with docs not in local corpus",
                )
            qrels_df = qrels_df[in_corpus]

        labelable_qids = set(qrels_df["query_id"].astype(str))
        if not labelable_qids:
            _log(t0, f"{dataset_name}: all picks' qrels reference missing docs — skipped")
            stats[dataset_name] = {"picks": len(group), "landed": 0}
            continue

        queries_df = group[group["query_id"].astype(str).isin(labelable_qids)][
            ["query_id", "query"]
        ].rename(columns={"query": "text"})
        queries_df = queries_df.astype({"query_id": str, "text": str}).drop_duplicates(
            "query_id"
        )

        added_queries = _atomic_append(
            dataset_dir / "queries.parquet", queries_df, ["query_id"]
        )
        added_qrels = _atomic_append(
            dataset_dir / "qrels.parquet", qrels_df, ["query_id", "doc_id"]
        )
        _log(
            t0,
            f"{dataset_name}: +{added_queries:,} queries, +{added_qrels:,} qrels",
        )
        stats[dataset_name] = {"picks": len(group), "landed": added_queries}

    total_picks = sum(s["picks"] for s in stats.values())
    total_landed = sum(s["landed"] for s in stats.values())
    _log(t0, f"done: {total_landed:,}/{total_picks:,} picks materialized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

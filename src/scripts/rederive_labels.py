"""Rebuild the route-score pool from the oracle caches, no retrieval.

Each cached row stores the top-k doc ids per route, so scoring them against the
lane's on-disk qrels at its CURRENT `min_relevance` reproduces the label without
Qdrant. This supersedes v2's `labels.parquet`, which predates the fetch-depth
change, the nfcorpus relevance raise and the NDCG round-trip fix.

    poetry run python src/scripts/rederive_labels.py --dry-run
    poetry run python src/scripts/rederive_labels.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from hybrid_search_rrf_dataset.paths import LanePaths
from hybrid_search_rrf_dataset.labels import outcome_shape, route_label
from hybrid_search_rrf_dataset.lanes import LANES
from hybrid_search_rrf_dataset.objective import RouterObjective

DATA = Path(__file__).resolve().parent.parent / "data"
V2_LABELS = DATA / "route_labels" / "labels.parquet"
OUT = DATA / "v3" / "labels_rederived.parquet"

CARRIED = ["query", "checkable", "cell", "stage", "route_selected"]


def _cache_path(lane: str) -> Path:
    return LanePaths(data_dir=DATA).oracle_rows(lane)


def _qrels(lane: str) -> dict[str, dict[str, int]]:
    """Judged grades per query, as `assess` wants them."""
    path = DATA / lane / "qrels.parquet"
    if not path.exists():
        return {}
    frame = pd.read_parquet(path).astype({"query_id": str, "doc_id": str})
    return {
        query_id: dict(zip(group["doc_id"], group["relevance"].astype(int)))
        for query_id, group in frame.groupby("query_id")
    }


def _score(objective: RouterObjective, top: list[str], gold: dict[str, int]) -> float:
    """One route's score from its stored order.

    `assess` takes a {doc: score} ranking and sorts it, so the stored order is
    handed back as descending synthetic scores: `ordered()` is a stable sort, so
    it returns the same list and the objective's own arithmetic stays the single
    definition of a score.
    """
    ranking = {doc: float(len(top) - i) for i, doc in enumerate(top)}
    return objective.assess(ranking, gold)[0]


def rederive() -> tuple[pd.DataFrame, list[str]]:
    """Every v2-labelled row rescored from its cached rankings."""
    pool = pd.read_parquet(V2_LABELS).astype({"query_id": str})
    carried = [c for c in CARRIED if c in pool.columns]
    rows: list[dict[str, object]] = []
    uncached: list[str] = []

    for lane, group in tqdm(pool.groupby("dataset", sort=True), unit="lane"):
        lane = str(lane)
        if not _cache_path(lane).exists():
            uncached.append(lane)
            continue
        cache = (
            pd.read_parquet(_cache_path(lane))
            .astype({"query_id": str})
            .drop_duplicates("query_id", keep="last")
            .set_index("query_id")
        )
        min_relevance = LANES[lane].min_relevance if lane in LANES else 1
        objective = RouterObjective(min_relevance=min_relevance)
        gold = _qrels(lane)

        for row in group.itertuples(index=False):
            query_id = str(row.query_id)
            if query_id not in cache.index:
                uncached.append(f"{lane}:{query_id}")
                continue
            rankings = cache.at[query_id, "route_rankings"]
            judged = gold.get(query_id, {})
            scores = {
                name: _score(objective, list(order), judged)
                for name, order in rankings.items()
            }
            served = route_label(scores)
            rows.append({
                "dataset": lane,
                "query_id": query_id,
                "route": served,
                "score": scores[served] if served else max(scores.values()),
                **{f"score_{name}": value for name, value in scores.items()},
                "shape": outcome_shape(scores),
                "metric_name": objective.name,
                "min_relevance": min_relevance,
                "provenance": "natural",
                **{c: getattr(row, c) for c in carried},
            })
    return pd.DataFrame(rows), uncached


def _sidecar(frame: pd.DataFrame) -> dict[str, object]:
    """What this artifact was built from — the record no v3 artifact had."""
    def rev(spec: str) -> str:
        return subprocess.run(
            ["git", "rev-parse", spec], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent.parent,
        ).stdout.strip()

    inputs = {}
    for lane in sorted(frame["dataset"].unique()):
        for path in (_cache_path(str(lane)), DATA / str(lane) / "qrels.parquet"):
            if path.exists():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                inputs[str(path.relative_to(DATA))] = digest[:16]
    return {
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_head": rev("HEAD"),
        "query_taxonomy": rev("HEAD:src/query-taxonomy"),
        "objective": RouterObjective().name,
        "min_relevance": {
            lane: LANES[lane].min_relevance if lane in LANES else 1
            for lane in sorted(frame["dataset"].unique())
        },
        "rows": len(frame),
        "source": "oracle route_rankings rescored against on-disk qrels",
        "inputs_sha256_16": inputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    frame, uncached = rederive()
    old = pd.read_parquet(V2_LABELS).astype({"query_id": str})
    key = ["dataset", "query_id"]
    joined = old.merge(frame, on=key, how="inner", suffixes=("_v2", ""))

    print(f"\nre-derived {len(frame):,} rows over {frame['dataset'].nunique()} lanes "
          f"(v2 pool {len(old):,})")
    if uncached:
        print(f"UNCACHED, dropped: {len(uncached)} -> {uncached[:5]}")
    print("\nshape shift:")
    print(pd.DataFrame({
        "v2": joined["shape_v2"].value_counts(),
        "rederived": joined["shape"].value_counts(),
    }).fillna(0).astype(int).to_string())
    moved = joined[joined["shape_v2"] != joined["shape"]]
    print(f"\nrows whose shape changed: {len(moved):,} ({len(moved) / len(joined):.2%})")
    for name in ("dense_only", "pure_rrf", "sparse_only"):
        delta = (joined[f"score_{name}_v2"] - joined[f"score_{name}"]).abs()
        n = int((delta > 1e-9).sum())
        print(f"  score_{name:<12} changed on {n:>6,} rows ({n / len(joined):6.2%})")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    args.out.with_suffix(".provenance.json").write_text(
        json.dumps(_sidecar(frame), indent=2) + "\n"
    )
    print(f"\n-> {args.out}\n-> {args.out.with_suffix('.provenance.json')}")


if __name__ == "__main__":
    main()

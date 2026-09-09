"""Score Home Depot route rankings into the standard labels frame, no retrieval.

Rankings come from retrieve_home_depot_routes.py; qrels from the materialized
lane snapshot (float Kaggle grades — `relevant()` compares >= min_relevance,
so nothing is truncated). Output matches the 100k-v2 labels schema, so the
frame feeds `AcceptabilityLabels` unchanged.

    poetry run python src/scripts/derive_home_depot_labels.py
    poetry run python src/scripts/derive_home_depot_labels.py \
        --rankings src/data/home-depot/route_rankings_minted.parquet \
        --qrels src/data/home-depot/minted_qrels.parquet \
        --queries-parquet src/data/home-depot/minted_queries.parquet \
        --out src/data/home-depot/route_labels_minted.parquet \
        --provenance minted
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

from hybrid_search_rrf_dataset.labels import outcome_shape, route_label
from hybrid_search_rrf_dataset.lanes import LANES
from hybrid_search_rrf_dataset.objective import RouterObjective

LANE = "home-depot"
DATA = Path(__file__).resolve().parent.parent / "data"
DEFAULT_RANKINGS = DATA / LANE / "route_rankings_full.parquet"
DEFAULT_QUERIES = DATA / LANE / "queries.parquet"
DEFAULT_QRELS = DATA / LANE / "qrels.parquet"
DEFAULT_OUT = DATA / LANE / "route_labels_real.parquet"


def derive(
    rankings: pd.DataFrame,
    queries: dict[str, str],
    gold: dict[str, dict[str, float]],
    provenance: str,
) -> pd.DataFrame:
    objective = RouterObjective(min_relevance=LANES[LANE].min_relevance)
    rows: list[dict[str, object]] = []
    for query_id, group in tqdm(rankings.groupby("query_id"), unit="query"):
        query_id = str(query_id)
        judged = gold.get(query_id, {})
        scores = {
            str(route): objective.assess_order(
                list(orders.sort_values("rank")["doc_id"]), judged
            )[0]
            for route, orders in group.groupby("route")
        }
        served = route_label(scores)
        rows.append({
            "dataset": LANE,
            "query_id": query_id,
            "route": served,
            "query": queries.get(query_id, ""),
            "score": scores[served] if served else max(scores.values()),
            **{f"score_{name}": value for name, value in scores.items()},
            "shape": outcome_shape(scores),
            "metric_name": objective.name,
            "min_relevance": float(objective.min_relevance),
            "provenance": provenance,
            "scored_against": provenance,
        })
    return pd.DataFrame(rows)


def _sidecar(args: argparse.Namespace, frame: pd.DataFrame) -> dict:
    def rev(spec: str) -> str:
        return subprocess.run(
            ["git", "rev-parse", spec], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent.parent,
        ).stdout.strip()

    inputs = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        for path in (args.rankings, args.qrels, args.queries_parquet)
    }
    return {
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_head": rev("HEAD"),
        "objective": RouterObjective().name,
        "min_relevance": LANES[LANE].min_relevance,
        "provenance": args.provenance,
        "rows": len(frame),
        "inputs_sha256_16": inputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, default=DEFAULT_RANKINGS)
    parser.add_argument("--queries-parquet", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--provenance", default="natural")
    args = parser.parse_args()

    if not args.rankings.exists():
        parser.error(
            f"{args.rankings} missing — run retrieve_home_depot_routes.py first"
        )
    for path in (args.queries_parquet, args.qrels):
        if not path.exists():
            parser.error(
                f"{path} missing — run materialize_corpora.py --only home-depot first"
            )

    rankings = pd.read_parquet(args.rankings).astype({"query_id": str})
    queries_df = pd.read_parquet(args.queries_parquet).astype({"query_id": str})
    unranked = set(queries_df["query_id"]) - set(rankings["query_id"])
    if unranked:
        parser.error(
            f"{len(unranked)} queries have no rankings — retrieval is still "
            "running or incomplete; re-run it (resumable), then derive"
        )
    text_col = "text" if "text" in queries_df.columns else "query"
    queries = dict(zip(queries_df["query_id"], queries_df[text_col]))
    qrels = pd.read_parquet(args.qrels).astype({"query_id": str, "doc_id": str})
    gold = {
        query_id: dict(zip(group["doc_id"], group["relevance"].astype(float)))
        for query_id, group in qrels.groupby("query_id")
    }

    frame = derive(rankings, queries, gold, args.provenance)

    print(f"\n{len(frame):,} labelled queries ({args.provenance})")
    print("\nshape:")
    print(frame["shape"].value_counts().to_string())
    print("\nserved route (decided rows):")
    print(frame["route"].value_counts(dropna=True).to_string())
    print("\nconstant-route means (the bar the experiment compares against):")
    for name in sorted(c for c in frame.columns if c.startswith("score_")):
        print(f"  {name:<18} {frame[name].mean():.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    args.out.with_suffix(".provenance.json").write_text(
        json.dumps(_sidecar(args, frame), indent=2) + "\n"
    )
    print(f"\n-> {args.out}\n-> {args.out.with_suffix('.provenance.json')}")


if __name__ == "__main__":
    main()

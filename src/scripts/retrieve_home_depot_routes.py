"""Retrieve all three routes' top-k for Home Depot queries against Qdrant Cloud.

The twice-calibration retrieval pass: one long-form parquet (query_id, route,
rank, doc_id), resumable by query_id, with a provenance sidecar. Run only after
the fusion query_embed fix — pre-fix rankings are not comparable.

    poetry run python src/scripts/retrieve_home_depot_routes.py
    poetry run python src/scripts/retrieve_home_depot_routes.py \
        --queries-parquet src/data/home-depot/minted_queries.parquet \
        --out src/data/home-depot/route_rankings_minted.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from tqdm.auto import tqdm

from hybrid_search_rrf_dataset.fusion import (
    DenseOnlyStrategy,
    FusionStrategy,
    PureRRFStrategy,
    SparseOnlyStrategy,
)
from scripts.index_home_depot import DENSE, SPARSE

DATA = Path(__file__).resolve().parent.parent / "data"
DEFAULT_QUERIES = DATA / "home-depot" / "queries.parquet"
DEFAULT_OUT = DATA / "home-depot" / "route_rankings_full.parquet"


def _strategies(
    client: QdrantClient, collection: str, fetch_limit: int
) -> dict[str, FusionStrategy]:
    kinds = (DenseOnlyStrategy, SparseOnlyStrategy, PureRRFStrategy)
    return {
        str(cls.name): cls(client, collection, DENSE, SPARSE, fetch_limit)
        for cls in kinds
    }


def _sidecar(args: argparse.Namespace, queries: Path, rows: int, n: int) -> dict:
    def rev(spec: str) -> str:
        return subprocess.run(
            ["git", "rev-parse", spec], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent.parent,
        ).stdout.strip()

    return {
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_head": rev("HEAD"),
        "collection": args.collection,
        "fetch_limit": args.fetch_limit,
        "dense_model": DENSE.model_id,
        "sparse_model": SPARSE.model_id,
        "queries_file": str(queries),
        "queries_sha256_16": hashlib.sha256(
            queries.read_bytes()
        ).hexdigest()[:16],
        "queries": n,
        "rows": rows,
    }


def main() -> None:
    load_dotenv(".env") or load_dotenv("../.env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default="home-depot")
    parser.add_argument(
        "--qdrant-url",
        # cloud names first: the home-depot collection lives on the cloud
        # cluster the bakeoff notebook targeted with these exact vars
        default=os.environ.get("QDRANT_CLOUD_URL") or os.environ.get("QDRANT_URL"),
    )
    parser.add_argument("--queries-parquet", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fetch-limit", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--flush-every", type=int, default=500)
    args = parser.parse_args()

    api_key = (
        os.environ.get("QDRANT_CLOUD_API_KEY") or os.environ.get("QDRANT_API_KEY")
    )
    if not args.qdrant_url or not api_key:
        parser.error(
            "set --qdrant-url or $QDRANT_CLOUD_URL/$QDRANT_URL and "
            "$QDRANT_CLOUD_API_KEY/$QDRANT_API_KEY (a repo-root .env is loaded)"
        )
    if not args.queries_parquet.exists():
        parser.error(
            f"{args.queries_parquet} missing — materialize the lane first "
            "(materialize_corpora.py --only home-depot)"
        )

    queries = pd.read_parquet(args.queries_parquet).astype({"query_id": str})
    text_col = "text" if "text" in queries.columns else "query"
    done_frames: list[pd.DataFrame] = []
    if args.out.exists():
        existing = pd.read_parquet(args.out).astype({"query_id": str})
        done_frames.append(existing)
        queries = queries[~queries["query_id"].isin(set(existing["query_id"]))]
        print(f"resume: {len(existing['query_id'].unique()):,} queries already done")

    client = QdrantClient(
        url=args.qdrant_url, api_key=api_key, cloud_inference=True, timeout=120
    )
    strategies = _strategies(client, args.collection, args.fetch_limit)
    for strategy in strategies.values():  # build bm25 once before the pool
        strategy._sparse("warm up the tokenizer")

    def fetch(row: tuple[str, str]) -> list[dict[str, object]]:
        query_id, text = row
        out = []
        for name, strategy in strategies.items():
            # _ranking preserves hit order, so dict order IS rank order
            for rank, doc_id in enumerate(strategy.rank(text)):
                out.append({
                    "query_id": query_id, "route": name,
                    "rank": rank, "doc_id": str(doc_id),
                })
        return out

    pending = list(zip(queries["query_id"], queries[text_col]))
    fresh: list[dict[str, object]] = []
    since_flush = 0

    def flush() -> None:
        nonlocal since_flush
        if not fresh and done_frames:
            return
        frame = pd.concat(
            [*done_frames, pd.DataFrame(fresh)], ignore_index=True
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.out, index=False)
        done_frames[:] = [frame]
        fresh.clear()
        since_flush = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for rows in tqdm(
            pool.map(fetch, pending), total=len(pending), unit="query"
        ):
            fresh.extend(rows)
            since_flush += 1
            if since_flush >= args.flush_every:
                flush()
    flush()

    total = done_frames[0] if done_frames else pd.DataFrame()
    args.out.with_suffix(".provenance.json").write_text(
        json.dumps(
            _sidecar(args, args.queries_parquet, len(total),
                     total["query_id"].nunique() if len(total) else 0),
            indent=2,
        ) + "\n"
    )
    print(f"-> {args.out} ({len(total):,} rows, "
          f"{total['query_id'].nunique() if len(total) else 0:,} queries)")


if __name__ == "__main__":
    main()

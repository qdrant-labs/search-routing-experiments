"""Give every 100k-v2 lane the bake-off's winning sparse leg (os_distill) beside
its gemini dense vectors. Qdrant fixes a collection's vector schema at creation,
so os_distill cannot be added in place — each lane gets its own
`..._opensearch-...-distill_routes` collection. Where a gemini collection exists
its dense vectors are COPIED by point id (no re-embed, no dense spend); the 40
lanes without one are built fresh and pay for their dense side.

    poetry run python src/scripts/apply_os_distill_sparse.py --plan
    ~/.claude/bin/memguard poetry run python src/scripts/apply_os_distill_sparse.py \
        --allow-paid-dense --sparse-batch-size 8 2>&1 | tee legb_index.log
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from tqdm.auto import tqdm

from sentence_transformers.util import get_device_name

from hybrid_search_rrf_dataset.fusion import SparseOnlyStrategy
from hybrid_search_rrf_dataset.retrieval import SnapshotDataset
from relevance_judge.config import RelevanceJudgeConfig
from scripts.label_routes import DATA_DIR, _source_name
from scripts.legb import (
    LegBPilot,
    gemini_dense_cfg,
    indexable,
    opensearch_distill_sparse_cfg,
)

UNREACHED = ("all_zero", "all_tied")
"""The rows a later l2 relabel exists to reach — this run makes their lanes
carry the strong sparse leg, ordered so an abort banks the most of them."""


def _pilot(client: QdrantClient, lanes: tuple[str, ...], *, sparse_batch_size: int = 16) -> LegBPilot:
    return LegBPilot(client, gemini_dense_cfg(), lanes=lanes,
                     sparse_cfg=opensearch_distill_sparse_cfg(batch_size=sparse_batch_size))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _plan(cfg: RelevanceJudgeConfig, pilot: LegBPilot) -> pd.DataFrame:
    """Every 100k-v2 lane, its unreached-row count, and whether paid gemini
    vectors exist to copy — free lanes first, then the paid tail by rows."""
    labels = pd.read_parquet(cfg.labels)
    unreached = labels[labels["shape"].isin(UNREACHED)].groupby("dataset").size()
    rows = [{"lane": lane,
             "unreached": int(unreached.get(lane, 0)),
             "reuses_dense": pilot.dense_source(lane) is not None}
            for lane in sorted(labels["dataset"].unique())]
    plan = pd.DataFrame(rows)
    return plan.sort_values(["reuses_dense", "unreached"], ascending=[False, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="read-only; print the table and exit")
    parser.add_argument("--allow-paid-dense", action="store_true",
                        help="permit lanes with no gemini vectors (paid dense embedding)")
    parser.add_argument("--only", nargs="+", metavar="LANE", help="restrict to these lanes")
    parser.add_argument("--sparse-batch-size", type=_positive_int, default=16, metavar="N",
                        help="documents per learned-sparse encoder call (default: 16)")
    args = parser.parse_args()

    load_dotenv()
    cfg = RelevanceJudgeConfig()
    client = QdrantClient(url=os.environ["QDRANT_CLOUD_URL"],
                          api_key=os.environ["QDRANT_CLOUD_API_KEY"],
                          timeout=120, cloud_inference=True)

    lanes = tuple(pd.read_parquet(cfg.labels)["dataset"].unique())
    pilot = _pilot(client, lanes, sparse_batch_size=args.sparse_batch_size)
    plan = _plan(cfg, pilot)
    if args.only:
        plan = plan[plan["lane"].isin(args.only)]

    # The sparse pass is the run's wall-clock. `cpu` here means a fallback on
    # this machine and a ~3x slower run than the measured MPS number — catch it
    # before an overnight job, not after.
    device = get_device_name()
    print(f"os_distill sparse device: {device}"
          + ("   ⚠ NOT the GPU — expect a ~3x slower run" if device == "cpu" else ""))

    if args.plan:
        print(plan.to_string(index=False))
        free, paid = plan["reuses_dense"].sum(), (~plan["reuses_dense"]).sum()
        print(f"\n{free} reuse paid dense (free), {paid} to build (paid)")
        return

    smoke_done = False
    results: list[tuple[str, float | None, str]] = []
    for _, row in plan.iterrows():
        lane, reuses = row["lane"], row["reuses_dense"]
        if not reuses and not args.allow_paid_dense:
            tqdm.write(f"[{lane}] SKIPPED — no gemini vectors to copy and --allow-paid-dense not set")
            results.append((lane, None, "skipped"))
            continue
        t0 = time.monotonic()
        coll = pilot.index(lane, allow_paid_dense=args.allow_paid_dense)
        dt = time.monotonic() - t0
        got = client.count(coll, exact=True).count
        want = indexable(SnapshotDataset(_source_name(lane), path=str(DATA_DIR)).corpus())
        status = f"{got:,}/{want:,} {'OK' if got >= want else 'INCOMPLETE'}"
        results.append((lane, dt, status))
        tqdm.write(f"[{lane}] {dt/60:.1f} min — {status}")
        if not smoke_done:  # prove the query side on the first lane, before any abort
            hits = SparseOnlyStrategy(client, coll, gemini_dense_cfg(),
                                      opensearch_distill_sparse_cfg()).rank("test query")
            tqdm.write(f"[{lane}] sparse smoke query returned {len(hits)} hits")
            smoke_done = True

    print("\n=== done ===")
    for lane, dt, status in results:
        print(f"{lane:24s} {'--' if dt is None else f'{dt/60:6.1f} min'}  {status}")


if __name__ == "__main__":
    main()

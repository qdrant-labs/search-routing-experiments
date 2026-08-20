"""Run the augmentation campaign from the terminal (2026-08) — a long,
LLM-spending batch belongs in a script that survives a closed laptop lid,
not a notebook cell that dies with the kernel and loses everything past a
Jupyter output-truncation cutoff.

Prints plainly to stdout/stderr; redirect with the shell, don't build
logging into the script — `tee` gives you the live view AND a saved file,
tracebacks included, which a `redirect_stdout` context manager cannot.

    poetry run python src/scripts/run_augmentation.py --plan
    poetry run python src/scripts/run_augmentation.py 2>&1 | tee campaign.log
    poetry run python src/scripts/run_augmentation.py --floor version_pinned_technical --n 5
"""

import argparse

import pandas as pd

from augmentation.campaign import AugmentationCampaign
from augmentation.config import AugmentationPaths
from augmentation.judge import CoherenceJudge
from augmentation.loop import AugmentationLoop
from augmentation.parents import ParentPool
from composition.cellfill import CellFill
from composition.composer import V3Composition
from dataset_registry import DATASETS


def _loop(v3: bool) -> AugmentationLoop:
    if v3:
        composer = V3Composition()
        catalog_path = composer.catalog_path
        selection = pd.read_parquet(composer.dataset_path).astype({"query_id": str})
        sheet_path = composer.order_sheet_path
    else:
        fill = CellFill()
        catalog_path = AugmentationPaths().catalog
        selection = pd.read_parquet(fill.selection_path).astype({"query_id": str})
        sheet_path = fill.order_sheet_path
    catalog = pd.read_parquet(catalog_path).astype({"query_id": str})
    parents = ParentPool(catalog, selection, {d.name: d for d in DATASETS})
    return AugmentationLoop(selection, sheet_path=sheet_path, parents=parents)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the augmentation campaign, or one floor, against the real pool."
    )
    parser.add_argument(
        "--floor", metavar="CELL",
        help="run just this floor instead of the whole campaign",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="row count for --floor (default: the floor's own missing credit)",
    )
    parser.add_argument(
        "--plan", action="store_true",
        help="print the plan and stop — no LLM spend",
    )
    parser.add_argument(
        "--pilot-n", type=int, default=None,
        help="override the audit-sample size for gated floors (default: config.pilot_n)",
    )
    parser.add_argument(
        "--llm-coherence", action="store_true",
        help="open coherence gates from the judge's verdicts "
             "(coherence_audit.parquet); run_v3_generation writes them",
    )
    parser.add_argument(
        "--v3", action="store_true",
        help="run against V3Composition's selection + order sheet instead of CellFill's",
    )
    args = parser.parse_args()

    if args.n is not None and not args.floor:
        parser.error("--n only applies with --floor")
    if args.plan and args.floor:
        parser.error("--plan and --floor are mutually exclusive")

    loop = _loop(args.v3)

    if args.floor:
        produced = loop.run(args.floor, n=args.n)
        print(f"{len(produced)} rows -> {loop.pool.path}")
        return

    campaign = AugmentationCampaign(
        loop,
        pilot_n=args.pilot_n,
        judge=CoherenceJudge(config=loop.config) if args.llm_coherence else None,
    )
    if args.plan:
        campaign.plan()
        return
    campaign.run()


if __name__ == "__main__":
    main()

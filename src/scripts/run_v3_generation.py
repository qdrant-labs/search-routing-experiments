"""One explicit driver for a v3 generation round: every stage banners itself
and prints its inputs, outputs and cost before the next begins. LLM spend
happens ONLY in stages 4-5, plus 5b under --llm-coherence; --plan stops before
any of it.

    poetry run python src/scripts/run_v3_generation.py --plan
    poetry run python src/scripts/run_v3_generation.py 2>&1 | tee generation.log
    poetry run python src/scripts/run_v3_generation.py --synthetic-cap 10
    poetry run python src/scripts/run_v3_generation.py --limit 500
    poetry run python src/scripts/run_v3_generation.py --llm-coherence
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from augmentation.campaign import AugmentationCampaign, RowBudget
from augmentation.config import AugmentationConfig
from augmentation.judge import CoherenceJudge
from augmentation.loop import AugmentationLoop
from augmentation.parents import ParentPool
from augmentation.pool import GeneratedPool
from composition.composer import V3Composition
from composition.floors import read_catalog
from dataset_registry import DATASETS

_T0 = time.time()
AUDIT_PASSED = (
    AugmentationConfig().paths.augmentation_dir / "declaration_audit_passed.txt"
)


def stage(n: int | str, title: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'=' * 72}\n== STAGE {n}: {title}   [{now}  +{time.time() - _T0:,.0f}s]\n{'=' * 72}")


def _loop(composer: V3Composition) -> AugmentationLoop:
    catalog = read_catalog(composer.catalog_path)
    selection = pd.read_parquet(composer.dataset_path).astype({"query_id": str})
    parents = ParentPool(catalog, selection, {d.name: d for d in DATASETS})
    return AugmentationLoop(
        selection, sheet_path=composer.order_sheet_path, parents=parents
    )


def _sheet_state(composer: V3Composition) -> pd.DataFrame:
    sheet = pd.read_parquet(composer.order_sheet_path)
    return sheet[sheet["missing"] > 0]


def _tiers(composer: V3Composition) -> dict[str, int]:
    selected = pd.read_parquet(composer.dataset_path)
    cert = selected[selected["certified"]]
    return {
        "rows": len(selected),
        "certified": len(cert),
        **{f"certified_{c}": int(n) for c, n in cert["route_class"].value_counts().items()},
    }


def preflight(composer: V3Composition) -> None:
    stage(0, "PREFLIGHT — nothing is spent here")
    import subprocess

    from qdrant_client import QdrantClient

    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    config = AugmentationConfig()
    print(f"git HEAD          : {head}")
    print(f"engine            : {config.engine.model} "
          f"(max_attempts={config.engine.max_attempts})")
    print(f"recipe            : {composer._recipe.model_dump()}")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"), timeout=10,
    )
    n = len(client.get_collections().collections)
    print(f"qdrant            : UP, {n} collections")
    print(f"anthropic key set : {bool(os.getenv('ANTHROPIC_API_KEY'))}")


def compose(composer: V3Composition) -> None:
    stage(1, "COMPOSE — select from the labelled pool (no spend)")
    selected = composer.build(force=True)
    t = _tiers(composer)
    print(f"pool -> selected  : {t['rows']:,} rows "
          f"(certified {t['certified']:,}: "
          f"{t.get('certified_dense', 0)}/{t.get('certified_sparse', 0)}/"
          f"{t.get('certified_hybrid', 0)} dense/sparse/hybrid, "
          f"waste {int((selected['route_class'] == 'waste').sum())})")
    print(f"artifacts         : {composer.dataset_path.parent}")


def sheet_readout(composer: V3Composition) -> None:
    stage(2, "ORDER SHEET — what generation still owes (no spend)")
    hungry = _sheet_state(composer)
    print(f"{len(hungry)} hungry lines, {hungry['missing'].sum():,.0f} rows owed "
          f"by GENERATION (the class shortfall toward the target is "
          f"LABELLING's — stage 2b):")
    print(hungry.to_string(index=False))


def campaign_plan(
    campaign: AugmentationCampaign, budget: RowBudget
) -> pd.DataFrame:
    stage(3, "CAMPAIGN PLAN — the spend, priced before any call (no spend)")
    plan = campaign.plan()
    print(plan.to_string(index=False))
    parents_n = int(plan["target_rows"].sum())
    synth_n = int(plan["synthetic_rows"].sum())
    print(f"\n-> {parents_n} rows from real parents (operators; gated floors "
          f"stage audit pilots first)\n-> {synth_n} rows need the synthetic "
          f"rung (3 LLM calls each: 1 query + 2 answer docs)")
    if budget.limit is not None:
        held = max(0, parents_n + synth_n - budget.limit)
        print(f"-> --limit {budget.limit:,}: the round stops there "
              f"({held:,} of those rows wait for the next one)")
    return plan


def parent_generation(campaign: AugmentationCampaign, budget: RowBudget) -> None:
    stage(4, "PARENT GENERATION — operators over real queries (LLM SPEND for "
             "non-deterministic operators; corrupt is free)")
    campaign.run(budget)


def synthetic_rung(
    loop: AugmentationLoop, plan: pd.DataFrame, cap: int | None, budget: RowBudget
) -> None:
    stage(5, "SYNTHETIC RUNG — mint query + 2 answer docs per row (LLM SPEND). "
             "The ONLY synthesizer: stage 4's campaign produces from parents "
             "and hands the whole remainder here")
    # every line owing synthetic rows, not just the wholly unreachable ones: a
    # floor whose parents cover part of its need still leaves a remainder, and
    # its action reads `produce`
    todo = plan[plan["synthetic_rows"] > 0]
    if todo.empty:
        print("nothing routed to the synthetic rung this round")
        return
    if budget.limit is not None:
        print(f"budget left       : {budget.left():,} rows "
              f"({budget.spent:,} spent by stage 4)")
    for line in todo.itertuples(index=False):
        if budget.exhausted():
            print(f"row budget spent — stopping before [{line.floor}]")
            return
        if not line.source_dataset:
            print(f"[{line.floor}] SKIPPED: no source lane to borrow "
                  f"distractors from — name one by hand")
            continue
        want = int(line.synthetic_rows) if cap is None else min(cap, int(line.synthetic_rows))
        n = budget.take(want)
        print(f"\n[{line.floor}] minting {n} rows against "
              f"{line.source_dataset} (grade bar = that lane's min_relevance)")
        produced = loop.synthesize(
            line.floor, n, source_dataset=str(line.source_dataset)
        )
        budget.add(len(produced))
        print(f"[{line.floor}] minted {len(produced)}/{n}")


def coherence(judge: CoherenceJudge) -> None:
    stage("5b", "COHERENCE JUDGE — one verdict per coherence-gated row: is "
                "the query one a real user could issue, and does its document "
                "answer it? (LLM SPEND, 1 call per unjudged row)")
    counts = judge.run(GeneratedPool().load())
    print(" ".join(f"{key}={value}" for key, value in counts.items()))
    print(f"verdicts          : {judge.path}")


def audit_passed() -> set[str] | None:
    """The human's declaration-audit verdict: one cleared query_id per line in
    `declaration_audit_passed.txt`, absent file = nobody has answered yet."""
    if not AUDIT_PASSED.exists():
        return None
    return {
        line.strip() for line in AUDIT_PASSED.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def admit(composer: V3Composition, judge: CoherenceJudge | None) -> None:
    stage(6, "ADMIT — credit generated rows against the sheet (no spend). "
             "GATED rows (d42h) are SKIPPED until their gate is answered: "
             "coherence rows enter iff stage 5b's judge passed them "
             "(--llm-coherence), declaration_audit rows iff the HUMAN listed "
             "their query_id in declaration_audit_passed.txt")
    pool = GeneratedPool().load()
    print(f"generated pool    : {len(pool):,} rows, gates: "
          f"{pool['credit_gate'].fillna('none').value_counts().to_dict()}")
    cleared = audit_passed()
    if cleared is None:
        print(f"declaration audit : NO verdict file — every audit-gated row "
              f"waits on the human. Write cleared query_ids, one per line, to "
              f"{AUDIT_PASSED}")
    else:
        print(f"declaration audit : {len(cleared):,} query_ids cleared by hand "
              f"({AUDIT_PASSED})")
    composer.admit(
        pool,
        coherence_passed=judge.passed() if judge else None,
        audit_passed=cleared,
    )


def selection_readout(composer: V3Composition) -> None:
    stage("2b", "SELECT NEW TARGETS — the labelling order: real dataset "
                "queries allocated by measured class yield, answer-coverage "
                "gated (no spend)")
    summary = pd.read_parquet(composer.selection_summary_path)
    print(summary.round(4).to_string())
    order = pd.read_parquet(composer.selection_order_path)
    print(f"\n{len(order):,} queries named -> {composer.selection_order_path}")


def label_targets(composer: V3Composition, cap: int | None) -> None:
    stage("2c", "LABEL NEW TARGETS — all 3 routes against the REAL lane "
                "collections (qdrant retrieval + local embeddings, no LLM; "
                "these labels certify immediately)")
    from qdrant_client import QdrantClient

    from scripts.label_routes_v3 import V3LabelSweep, order_selection

    selection = order_selection(cap=cap)
    if selection.empty:
        print("nothing to label — the order is empty or already labelled")
        return
    print(f"to label          : {len(selection):,} queries across "
          f"{selection['dataset'].nunique()} lane(s)"
          + (f" (batch cap {cap:,})" if cap else " (FULL order)"))
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"), timeout=60,
    )
    sweep = V3LabelSweep(client, selection)
    print(sweep.plan().to_string(index=False))
    failed = sweep.run()
    if failed:
        print(f"skipped lanes     : {failed}")

    stage("2d", "RECOMPOSE — the new labels enter the pool and the dataset is "
                "re-drawn BEFORE any generation is priced (no spend)")
    composer.build(force=True)
    t = _tiers(composer)
    print(f"pool -> selected  : {t['rows']:,} rows (certified {t['certified']:,}: "
          f"{t.get('certified_dense', 0)}/{t.get('certified_sparse', 0)}/"
          f"{t.get('certified_hybrid', 0)} dense/sparse/hybrid)")
    hungry = _sheet_state(composer)
    print(f"generation residual: {len(hungry)} lines, "
          f"{hungry['missing'].sum():,.0f} rows still owed")


def label_synthetic() -> None:
    stage(7, "LABEL SYNTHETIC — run all 3 routes against ISOLATED collections "
             "(qdrant retrieval, no LLM). Paid lane collections are never touched")
    from qdrant_client import QdrantClient

    from scripts.label_routes_synthetic import OUT_DIR, SyntheticLabelSweep, synthetic_rows

    rows = synthetic_rows(GeneratedPool().load())
    if rows.empty:
        print("no synthetic rows in the pool — nothing to label")
        return
    selection = pd.DataFrame({
        "dataset": rows["home_lane"].astype(str),
        "query_id": rows["query_id"].astype(str),
        "query": rows["query"].astype(str),
    })
    print(f"to label          : {len(selection):,} rows across "
          f"{selection['dataset'].nunique()} lane(s)")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"), timeout=60,
    )
    sweep = SyntheticLabelSweep(client, selection)
    print(sweep.plan().to_string(index=False))
    failed = sweep.run()
    if failed:
        print(f"FAILED lanes      : {failed}")
    print(f"labels            : {OUT_DIR / 'labels.parquet'}")


def catalog_refresh() -> None:
    stage(8, "CATALOG — extend catalog_v3 with the newly labelled queries "
             "(local feature extraction over the whole pool, minutes of CPU, "
             "no LLM). Skipping this leaves new rows with ZERO cell "
             "membership, so their floors never get credited")
    from scripts.build_v3_catalog import OUT, build_v3_catalog

    catalog = build_v3_catalog(force=True)
    print(f"catalog           : {len(catalog):,} rows x {catalog.shape[1]} cols "
          f"-> {OUT}")


def rebuild(composer: V3Composition, before: dict[str, int]) -> None:
    stage(9, "REBUILD — fold the new labels into the pool and re-select "
             "(no spend). This recompute is authoritative; stage 6's credits "
             "were the mid-cycle view")
    composer.build(force=True)
    after = _tiers(composer)
    for key in sorted(set(before) | set(after)):
        b, a = before.get(key, 0), after.get(key, 0)
        marker = f"  ({a - b:+d})" if a != b else ""
        print(f"{key:<20}: {b:,} -> {a:,}{marker}")
    hungry = _sheet_state(composer)
    print(f"\nhungry lines left : {len(hungry)} "
          f"({hungry['missing'].sum():,.0f} rows still owed)")
    print(f"report            : {composer.report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true",
                        help="stop after stage 3 — zero spend")
    parser.add_argument("--synthetic-cap", type=int, default=None,
                        help="max synthetic rows per floor this round "
                             "(default: the plan's own numbers)")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop the round once this many rows have been "
                             "produced in total — stages 4 and 5 draw from one "
                             "budget, so the round ends mid-queue rather than "
                             "serving every floor (default: the plan's own "
                             "numbers)")
    parser.add_argument("--skip-parents", action="store_true",
                        help="skip stage 4 (parent-based operators)")
    parser.add_argument("--skip-synthetic", action="store_true",
                        help="skip stage 5 (the synthetic rung)")
    parser.add_argument("--llm-coherence", action="store_true",
                        help="judge the coherence gate with an LLM (stage 5b) "
                             "and admit the rows it passes; without it the "
                             "gate waits on the human audit")
    parser.add_argument("--label-cap", type=int, default=None,
                        help="max labelling-order queries this round "
                             "(default: the WHOLE order — labelling precedes "
                             "generation by design)")
    parser.add_argument("--skip-labelling", action="store_true",
                        help="skip stages 2c-2d (the labelling rung)")
    args = parser.parse_args()

    load_dotenv()
    composer = V3Composition()
    preflight(composer)
    compose(composer)
    sheet_readout(composer)
    selection_readout(composer)
    if not args.plan and not args.skip_labelling:
        label_targets(composer, args.label_cap)
    before = _tiers(composer)
    loop = _loop(composer)
    judge = CoherenceJudge(
        loop.engine, config=loop.config, docs=loop.docs, parents=loop.parents
    ) if args.llm_coherence else None
    campaign = AugmentationCampaign(loop, judge=judge)
    # one budget, both spending stages: what stage 4 produces is what stage 5
    # no longer may
    budget = RowBudget(args.limit)
    plan = campaign_plan(campaign, budget)
    if args.plan:
        print("\n--plan: stopping before any spend.")
        return
    if not args.skip_parents:
        parent_generation(campaign, budget)
    if not args.skip_synthetic:
        synthetic_rung(loop, plan, args.synthetic_cap, budget)
    if judge is not None:
        coherence(judge)
    admit(composer, judge)
    label_synthetic()
    catalog_refresh()
    rebuild(composer, before)
    print(f"\nDONE in {time.time() - _T0:,.0f}s")


if __name__ == "__main__":
    main()

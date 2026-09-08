"""Drive the relevance-atom judge, one gate at a time.

    # §1.0 free precondition audit — how many residual rows are rescore-able
    poetry run python src/scripts/run_relevance_judge.py --audit

    # §1.1 accuracy gate (spends ~$1) — MUST pass before any judging
    poetry run python src/scripts/run_relevance_judge.py --validate

    # §1.2 sub-1.0-tie pilot (spends ~$0.5) — needs a passed validation run
    poetry run python src/scripts/run_relevance_judge.py --judge-ties
    poetry run python src/scripts/run_relevance_judge.py --judge-ties --dry-run

    # read what the atoms bought (no spend)
    poetry run python src/scripts/run_relevance_judge.py --score
"""

from __future__ import annotations

import argparse
import json

from augmentation.engine import Budget
from relevance_judge import (
    JudgeQueue,
    JudgeRunLog,
    PilotScorer,
    RelevanceJudge,
    RelevanceJudgeConfig,
    Sources,
    ValidationHarness,
)


def _budget(config: RelevanceJudgeConfig, max_usd: float) -> Budget:
    return Budget(
        max_usd,
        usd_per_mtok_in=config.engine.usd_per_mtok_in,
        usd_per_mtok_out=config.engine.usd_per_mtok_out,
    )


def _latest_passed_run(config: RelevanceJudgeConfig) -> str | None:
    runs = JudgeRunLog(config).load()
    passed = runs[runs["passed"]] if not runs.empty else runs
    return str(passed.iloc[-1]["judge_run_id"]) if not passed.empty else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit", action="store_true", help="§1.0 precondition audit (free)")
    parser.add_argument("--validate", action="store_true", help="§1.1 accuracy gate")
    parser.add_argument("--rescore", action="store_true",
                        help="re-score banked predictions under current gate logic — NO LLM spend")
    parser.add_argument("--judge-ties", action="store_true",
                        help="judge every tied row's tail docs (qrels depth)")
    parser.add_argument("--score", action="store_true", help="report what the atoms bought")
    parser.add_argument("--per-lane", type=int, default=200, help="validation pairs per lane")
    parser.add_argument("--limit", type=int, default=None, help="cap pilot rows")
    parser.add_argument("--max-spend-usd", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true", help="build the work list, spend nothing")
    args = parser.parse_args()

    config = RelevanceJudgeConfig()

    if args.audit:
        print(json.dumps(JudgeQueue(config).precondition_audit(), indent=2))

    if args.rescore:
        import pandas as pd
        harness = ValidationHarness(config)
        preds = pd.read_parquet(config.validation_predictions)
        valid = set(Sources(config).lanes_with_negatives()) | set(config.anchor_lanes)
        # keep valid referees + anchors + genuinely positive-only lanes; drop lanes
        # that only LOOK negative under a stale threshold (e.g. nfcorpus grade-1).
        keep = preds.groupby("dataset").filter(
            lambda g: g.name in valid or not bool((~g["human_relevant"]).any())
        )
        report = harness.finalize(harness.score(keep, sorted(keep["dataset"].unique())))
        print("GATE (rescored, no spend):", "PASS" if report["passed"] else "FAIL")
        print(json.dumps({k: v for k, v in report.items() if k != "agreement_by_lane"}, indent=2))

    if args.validate:
        # residual lanes give recall coverage; negative-bearing human lanes are the
        # only valid precision referees (excludes positive-only + synthetic lanes).
        residual = sorted(JudgeQueue(config).residual()["dataset"].unique())
        referees = Sources(config).lanes_with_negatives()
        lanes = sorted(set(residual) | set(referees))
        report = ValidationHarness(config).run(
            lanes=lanes, per_lane=args.per_lane,
            budget=_budget(config, args.max_spend_usd),
            deploy_lanes=residual,
        )
        print(json.dumps({k: v for k, v in report.items() if k != "agreement_by_lane"}, indent=2))
        print("GATE:", "PASS" if report.get("passed") else "FAIL", "->", config.validation_report)

    if args.judge_ties:
        queue = JudgeQueue(config)
        pairs = queue.tie_pairs(limit=args.limit)
        if args.dry_run:
            print(f"--dry-run: {len(pairs)} pairs, nothing spent")
            return
        run_id = _latest_passed_run(config)
        if run_id is None:
            parser.error("no passed validation run — run --validate first")
        counts = RelevanceJudge(config).judge_pairs(
            pairs, run_id=run_id, budget=_budget(config, args.max_spend_usd)
        )
        print(json.dumps(counts, indent=2))

    if args.score:
        print(json.dumps(PilotScorer(config).audit(), indent=2, default=str))


if __name__ == "__main__":
    main()

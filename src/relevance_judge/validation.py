"""The accuracy gate (doc §3): run the judge on already-human-judged pairs and
refuse the program if it is not accurate enough — BEFORE any labelling spend.

Primary metric is positive precision, not overall agreement: a judged atom
becomes gold, so a false positive injects wrong truth that corrupts every
downstream score. A false negative only leaves a hole. The gate reflects that
asymmetry.

`run()` is three inspectable stages — `sample()` -> `judge_rows()` -> `score()` —
so a notebook can look at the eval frame, the per-pair predictions (persisted to
`validation_predictions.parquet`, with the judge's own reason), and the confusion
breakdown, rather than trusting one opaque call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from augmentation.engine import Budget, windowed_map
from hybrid_search_rrf_dataset.lanes import LANES
from relevance_judge.config import RelevanceJudgeConfig
from relevance_judge.judge import JudgeRunLog, RelevanceJudge
from relevance_judge.sources import Sources


def _sample_lane(
    qrels: pd.DataFrame, min_relevance: int, per_lane: int, seed: int
) -> pd.DataFrame:
    """Up to `per_lane` CLEAN judged pairs, balanced toward negatives when present.
    Positive = grade >= min_relevance; hard negative = grade 0 (human-judged
    irrelevant). The ambiguous middle (0 < grade < min_relevance — relevant but
    below a strict bar) is EXCLUDED: it is neither a clean positive a strict judge
    must catch, nor a clean negative it must reject."""
    pos = qrels[qrels["relevance"] >= min_relevance].assign(human_relevant=True)
    neg = qrels[qrels["relevance"] < 1].assign(human_relevant=False)
    half = per_lane // 2
    take_neg = min(len(neg), half)
    take_pos = min(len(pos), per_lane - take_neg)
    parts = [pos.sample(take_pos, random_state=seed)]
    if take_neg:
        parts.append(neg.sample(take_neg, random_state=seed))
    return pd.concat(parts, ignore_index=True)


class ValidationHarness:
    """Owns `validation_report.json` + `validation_predictions.parquet`; opens a
    `judge_runs.parquet` row on pass."""

    def __init__(
        self,
        config: RelevanceJudgeConfig | None = None,
        judge: RelevanceJudge | None = None,
        sources: Sources | None = None,
    ) -> None:
        self.config = config or RelevanceJudgeConfig()
        self.judge = judge or RelevanceJudge(self.config)
        self.sources = sources or Sources(self.config)

    def sample(
        self, lanes: list[str], *, per_lane: int = 200, seed: int = 0, verbose: bool = True
    ) -> pd.DataFrame:
        """The eval frame: [dataset, query_id, doc_id, human_relevant, query,
        doc_text]. Prints a per-lane pos/neg/dropped breakdown so the sample is
        never a black box (a lane with 0 negatives cannot test precision)."""
        frames, trace = [], []
        for lane in lanes:
            qrels = self.sources.base_qrels(lane)
            if qrels.empty:
                trace.append({"lane": lane, "pos": 0, "neg": 0, "kept": 0, "dropped_no_text": 0})
                continue
            # per-lane bar, not a global 1 — trec-dl-2022 grades 0-3 and needs >=2,
            # else its grade-1 "marginally related" docs count as relevant and a
            # correctly-strict judge reads as false negatives (rederive_labels rule).
            min_relevance = LANES[lane].min_relevance if lane in LANES else self.config.min_relevance
            sample = _sample_lane(qrels, min_relevance, per_lane, seed)
            texts = self.sources.corpus_text(lane, set(sample["doc_id"]))
            queries = self.sources.query_text(lane, set(sample["query_id"]))
            sample = sample.assign(
                dataset=lane,
                query=sample["query_id"].map(queries),
                doc_text=sample["doc_id"].map(texts),
            )
            kept = sample[sample["doc_text"].astype(bool) & sample["query"].astype(bool)]
            frames.append(kept)
            trace.append({
                "lane": lane,
                "pos": int(sample["human_relevant"].sum()),
                "neg": int((~sample["human_relevant"]).sum()),
                "kept": len(kept),
                "dropped_no_text": len(sample) - len(kept),
            })
        if verbose:
            print("sampling per lane:")
            print(pd.DataFrame(trace).to_string(index=False))
            no_neg = [t["lane"] for t in trace if t["neg"] == 0 and t["kept"] > 0]
            if no_neg:
                print(f"  lanes with NO negatives (cannot test precision): {no_neg}")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def judge_rows(
        self, rows: pd.DataFrame, *, budget: Budget | None = None, bank_every: int = 50
    ) -> pd.DataFrame:
        """Judge every eval row, with a progress bar, and persist the per-pair
        verdicts (human vs predicted, plus the judge's reason) for inspection.
        A cumulative snapshot is banked every `bank_every` readable rows, so a
        crash or budget stop keeps everything already paid for (a partial run is
        rescored with `score(pd.read_parquet(config.validation_predictions), ...)`).
        Returns the readable predictions; unreadable replies are dropped and
        counted, never silently passed."""
        records: list[dict] = []
        unreadable = 0

        def attempt(row):
            relevant, reason, _hash, _spend = self.judge.judge_one(
                str(row.query), str(row.doc_text), budget=budget
            )
            return relevant, reason

        bar = tqdm(total=len(rows), desc="validate", unit="pair")
        for row, (relevant, reason) in windowed_map(
            attempt, rows.itertuples(index=False), self.config.llm_workers
        ):
            bar.update(1)
            if relevant is None:
                unreadable += 1
                continue
            records.append({
                "dataset": row.dataset,
                "query_id": str(row.query_id),
                "doc_id": str(row.doc_id),
                "human_relevant": bool(row.human_relevant),
                "pred_relevant": bool(relevant),
                "reason": reason,
            })
            if len(records) % bank_every == 0:
                self._persist_predictions(pd.DataFrame(records))
        bar.close()
        preds = pd.DataFrame(records)
        self._persist_predictions(preds)
        if unreadable:
            print(f"  {unreadable} unreadable replies dropped (not counted as agree/disagree)")
        return preds

    def run(
        self,
        *,
        lanes: list[str],
        per_lane: int = 200,
        seed: int = 0,
        budget: Budget | None = None,
    ) -> dict:
        """One-shot for the CLI: sample -> judge -> score -> finalize. A notebook
        should call the four stages itself (each returns an inspectable object)."""
        rows = self.sample(lanes, per_lane=per_lane, seed=seed)
        preds = self.judge_rows(rows, budget=budget)
        return self.finalize(self.score(preds, lanes))

    def finalize(self, report: dict) -> dict:
        """Write the report and, on pass, open a `judge_runs.parquet` row the
        pilot references — the side-effecting tail of `run()`, split out so the
        staged notebook path reaches the same end state."""
        self._write(report)
        if report.get("passed"):
            report["judge_run_id"] = JudgeRunLog(self.config).open(self.judge.model, report)
            self._write(report)
        return report

    def score(self, preds: pd.DataFrame, lanes: list[str]) -> dict:
        """Metrics + a confusion breakdown. False positives are called out
        explicitly — they are the errors the precision gate exists to catch."""
        if preds.empty:
            return {"passed": False, "reason": "no readable predictions"}
        pred_rel, human_rel = preds["pred_relevant"], preds["human_relevant"]
        tp = int((pred_rel & human_rel).sum())
        fp = int((pred_rel & ~human_rel).sum())
        fn = int((~pred_rel & human_rel).sum())
        tn = int((~pred_rel & ~human_rel).sum())

        has_negatives = bool((fp + tn) > 0)
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        by_lane = {
            lane: float((group["pred_relevant"] == group["human_relevant"]).mean())
            for lane, group in preds.groupby("dataset")
        }
        hardest = min(by_lane.items(), key=lambda kv: kv[1]) if by_lane else (None, float("nan"))
        cfg = self.config

        # Precision only where a false positive is POSSIBLE: negative-bearing lanes.
        # Positive-only lanes cannot mis-fire, so pooling them inflates precision.
        neg_lanes = [
            lane for lane, group in preds.groupby("dataset")
            if bool((~group["human_relevant"]).any())
        ]
        neg = preds[preds["dataset"].isin(neg_lanes)]
        ntp = int((neg["pred_relevant"] & neg["human_relevant"]).sum())
        nfp = int((neg["pred_relevant"] & ~neg["human_relevant"]).sum())
        precision = ntp / (ntp + nfp) if (ntp + nfp) else float("nan")

        # Recall (non-degeneracy) only where the judge shares the dataset's
        # definition: the answer-oriented anchor lanes.
        anchor = preds[preds["dataset"].isin(cfg.anchor_lanes)]
        atp = int((anchor["pred_relevant"] & anchor["human_relevant"]).sum())
        afn = int((~anchor["pred_relevant"] & anchor["human_relevant"]).sum())
        anchor_recall = atp / (atp + afn) if (atp + afn) else float("nan")

        passed = bool(
            has_negatives
            and precision == precision
            and anchor_recall == anchor_recall
            and precision >= cfg.min_precision_relevant
            and anchor_recall >= cfg.min_anchor_recall
        )
        return {
            "passed": passed,
            "gate_enforced": ["has_negatives", "precision_relevant(neg-lanes)", "anchor_recall"],
            "precision_relevant": precision,
            "anchor_recall": anchor_recall,
            "anchor_lanes_present": sorted(set(anchor["dataset"].unique())),
            "recall_relevant": recall,
            "precision_pooled": tp / (tp + fp) if (tp + fp) else float("nan"),
            "has_negatives": has_negatives,
            "agreement_overall": (tp + tn) / (tp + fp + fn + tn),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "false_positives": fp,
            "agreement_by_lane": by_lane,
            "hardest_lane": {"lane": hardest[0], "agreement": hardest[1]},
            "n_validation": int(len(preds)),
            "lanes_without_negatives": [
                lane for lane in lanes
                if lane in by_lane
                and not bool((~preds[preds["dataset"] == lane]["human_relevant"]).any())
            ],
            "predictions_path": str(self.config.validation_predictions),
            "thresholds": {
                "min_precision_relevant": cfg.min_precision_relevant,
                "min_anchor_recall": cfg.min_anchor_recall,
                "min_agreement_overall_DIAGNOSTIC": cfg.min_agreement_overall,
                "min_agreement_lane_DIAGNOSTIC": cfg.min_agreement_lane,
            },
        }

    def audit_false_positives(
        self, preds: pd.DataFrame | None = None, *, per_lane: int | None = None, chars: int = 600
    ) -> pd.DataFrame:
        """Enrich the judge's false positives with the human grade + query/doc text
        so each can be hand-checked: a real judge error, or a qrels-hole the program
        exists to find? Persisted to `false_positives_audit.parquet` for review."""
        if preds is None:
            preds = pd.read_parquet(self.config.validation_predictions)
        preds = preds.astype({"query_id": str, "doc_id": str})
        false_pos = preds[preds["pred_relevant"] & ~preds["human_relevant"]]
        rows: list[dict] = []
        for lane, group in false_pos.groupby("dataset"):
            group = group.head(per_lane) if per_lane else group
            queries = self.sources.query_text(lane, set(group["query_id"]))
            docs = self.sources.corpus_text(lane, set(group["doc_id"]))
            qrels = self.sources.base_qrels(lane)
            grade = {(r.query_id, r.doc_id): int(r.relevance) for r in qrels.itertuples(index=False)}
            for row in group.itertuples(index=False):
                rows.append({
                    "dataset": lane, "query_id": row.query_id, "doc_id": row.doc_id,
                    "human_grade": grade.get((row.query_id, row.doc_id), -1),
                    "judge_reason": row.reason,
                    "query": queries.get(row.query_id, "")[:300],
                    "doc": docs.get(row.doc_id, "")[:chars],
                })
        audit = pd.DataFrame(rows)
        self.config.false_positives_audit.parent.mkdir(parents=True, exist_ok=True)
        audit.to_parquet(self.config.false_positives_audit, index=False)
        return audit

    def _persist_predictions(self, preds: pd.DataFrame) -> None:
        path = self.config.validation_predictions
        path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(path, index=False)

    def _write(self, report: dict) -> None:
        path: Path = self.config.validation_report
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")


def _row(lane, i, human, pred):
    return {"dataset": lane, "query_id": f"{lane}{i}", "doc_id": f"{lane}{i}",
            "human_relevant": human, "pred_relevant": pred}


def _self_check() -> None:
    harness = ValidationHarness.__new__(ValidationHarness)
    harness.config = RelevanceJudgeConfig()
    anchor = harness.config.anchor_lanes[0]  # an answer-oriented lane the floor uses

    # no negatives anywhere => precision unmeasurable => fail
    only_pos = pd.DataFrame([_row("x", i, True, True) for i in range(3)])
    assert harness.score(only_pos, ["x"])["passed"] is False

    # precision measured on negative-bearing lanes: one fp drops it to 0.5 => fail
    mixed = pd.DataFrame([_row("wands", 0, True, True), _row("wands", 1, False, True)])
    r2 = harness.score(mixed, ["wands"])
    assert abs(r2["precision_relevant"] - 0.5) < 1e-9 and r2["passed"] is False

    # degenerate always-'no' judge: zero recall on the anchor lane => fail the floor
    all_no = pd.DataFrame([_row(anchor, 0, True, False), _row(anchor, 1, False, False)])
    r3 = harness.score(all_no, [anchor])
    assert r3["anchor_recall"] == 0.0 and r3["passed"] is False

    # precise negatives lane + adequate anchor recall => PASS
    good = pd.DataFrame(
        [_row("wands", i, True, True) for i in range(10)]
        + [_row("wands", 100 + i, False, False) for i in range(5)]      # negatives, no fp
        + [_row(anchor, i, True, True) for i in range(8)]                # anchor recall 0.8
        + [_row(anchor, 100 + i, True, False) for i in range(2)]
    )
    r4 = harness.score(good, ["wands", anchor])
    assert r4["precision_relevant"] == 1.0 and r4["anchor_recall"] == 0.8
    assert r4["passed"] is True
    print("validation self-check ok")


if __name__ == "__main__":
    _self_check()

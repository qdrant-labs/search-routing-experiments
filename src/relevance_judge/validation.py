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
from math import isnan
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from tqdm.auto import tqdm

from augmentation.engine import Budget, Spend, windowed_map
from hybrid_search_rrf_dataset.lanes import LANES
from relevance_judge.config import RelevanceJudgeConfig
from relevance_judge.judge import RATIONALE_FIELDS, JudgeRunLog, RelevanceJudge
from relevance_judge.sources import Sources


class Confusion(NamedTuple):
    """One 2x2 table and the rates read off it — nan when a rate has no
    denominator, so an unmeasurable metric never reads as a passing 0."""

    tp: int
    fp: int
    fn: int
    tn: int

    @classmethod
    def of(cls, preds: pd.DataFrame) -> Confusion:
        pred, human = preds["pred_relevant"], preds["human_relevant"]
        return cls(
            tp=int((pred & human).sum()), fp=int((pred & ~human).sum()),
            fn=int((~pred & human).sum()), tn=int((~pred & ~human).sum()),
        )

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else float("nan")

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else float("nan")

    @property
    def agreement(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total else float("nan")

    @property
    def has_negatives(self) -> bool:
        """Whether a false positive was even possible — precision is
        unmeasurable without a human-judged irrelevant doc."""
        return (self.fp + self.tn) > 0


class Gate:
    """The pass decision: positive precision on the referees, plus a
    non-degeneracy recall floor on the anchors. Nothing else is enforced."""

    ENFORCED = ("has_negatives", "precision_relevant(neg-lanes)", "anchor_recall")

    def __init__(self, config: RelevanceJudgeConfig) -> None:
        self.config = config

    def passed(self, referees: Confusion, anchors: Confusion) -> bool:
        precision, recall = referees.precision, anchors.recall
        return bool(
            referees.has_negatives
            and not isnan(precision)
            and not isnan(recall)
            and precision >= self.config.min_precision_relevant
            and recall >= self.config.min_anchor_recall
        )


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
        self.last_spend_usd: float | None = None
        self.last_spend: Spend | None = None

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
        spend = Spend()

        def attempt(row):
            return self.judge.judge_one(str(row.query), str(row.doc_text),
                                        dataset=str(row.dataset), budget=budget)

        bar = tqdm(total=len(rows), desc="validate", unit="pair")
        # `finally`: a budget stop arrives through windowed_map's future.result(),
        # and the verdicts since the last bank are already PAID for.
        try:
            for row, verdict in windowed_map(
                attempt, rows.itertuples(index=False), self.config.llm_workers
            ):
                bar.update(1)
                relevant, reason = verdict.relevant, verdict.reason
                spend.add(verdict.spend)
                dropped = sum(self.judge.dropped.values()) if hasattr(self.judge, "dropped") else 0
                cost = f"${budget.spent_usd:.2f}" if budget is not None else ""
                bar.set_postfix_str(f"{cost} drop={dropped}" if dropped else cost)
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
                    **{f: verdict.fields.get(f, "") for f in RATIONALE_FIELDS},
                    # random-corpus negative, not a human judgment: kept out of
                    # every gate metric so easy negatives cannot inflate precision
                    "pseudo": bool(getattr(row, "pseudo", False)),
                })
                if len(records) % bank_every == 0:
                    self._persist_predictions(pd.DataFrame(records))
        finally:
            bar.close()
            self._persist_predictions(pd.DataFrame(records))
            # validation is the larger of the two spends and reported nothing
            self.last_spend_usd = budget.spent_usd if budget is not None else None
            self.last_spend = spend
            paid = f" | ${budget.spent_usd:.2f}" if budget is not None else ""
            print(f"  {spend.summary(len(records))}{paid}")
        preds = pd.DataFrame(records)
        if unreadable:
            print(f"  {unreadable} unreadable replies dropped (not counted as agree/disagree)")
        if getattr(self.judge, "dropped", None):
            print(f"  dropped: {dict(self.judge.dropped)}")
            for name, message in getattr(self.judge, "dropped_detail", {}).items():
                print(f"    {name}: {message}")
        for reply in getattr(self.judge, "unreadable", []):
            print(f"    unparsed reply: {reply!r}")
        return preds

    def run(
        self,
        *,
        lanes: list[str],
        per_lane: int = 200,
        seed: int = 0,
        budget: Budget | None = None,
        deploy_lanes: list[str] | None = None,
        deploy_negatives_per_lane: int = 25,
    ) -> dict:
        """One-shot for the CLI: sample -> judge -> score -> finalize. A notebook
        should call the four stages itself (each returns an inspectable object)."""
        rows = self.sample(lanes, per_lane=per_lane, seed=seed)
        # Unrefereed deploy lanes get random-corpus pseudo-negatives, so the
        # report carries a false-positive signal from where the judge runs.
        unrefereed = sorted(set(deploy_lanes or []) - set(self.sources.lanes_with_negatives()))
        if unrefereed:
            rows = pd.concat(
                [rows, self.deploy_negatives(
                    unrefereed, per_lane=deploy_negatives_per_lane, seed=seed)],
                ignore_index=True,
            )
        preds = self.judge_rows(rows, budget=budget)
        return self.finalize(self.score(preds, lanes, deploy_lanes=deploy_lanes))

    def finalize(self, report: dict) -> dict:
        """Write the report and, on pass, open a `judge_runs.parquet` row the
        pilot references — the side-effecting tail of `run()`, split out so the
        staged notebook path reaches the same end state."""
        self._write(report)
        if report.get("passed"):
            report["judge_run_id"] = JudgeRunLog(self.config).open(self.judge.model, report)
            self._write(report)
        return report

    def score(
        self, preds: pd.DataFrame, lanes: list[str], deploy_lanes: list[str] | None = None
    ) -> dict:
        """Metrics + a confusion breakdown. False positives are called out
        explicitly — they are the errors the precision gate exists to catch."""
        if preds.empty:
            return {"passed": False, "reason": "no readable predictions"}
        cfg = self.config
        # Pseudo-negatives are a deploy-lane diagnostic, never gate input: random
        # docs are far easier to reject than human-judged near-misses, so pooling
        # them would inflate precision exactly where it is least earned.
        pseudo = (
            preds[preds["pseudo"].fillna(False).astype(bool)]
            if "pseudo" in preds.columns else preds.iloc[0:0]
        )
        preds = preds.drop(index=pseudo.index)
        if preds.empty:
            return {"passed": False, "reason": "no human-judged predictions"}
        overall = Confusion.of(preds)

        # Precision only where a false positive is POSSIBLE: negative-bearing
        # lanes. Positive-only lanes cannot mis-fire, so pooling them inflates it.
        by_lane_negatives = {
            lane: bool((~group["human_relevant"]).any())
            for lane, group in preds.groupby("dataset")
        }
        neg_lanes = [lane for lane, has_neg in by_lane_negatives.items() if has_neg]
        referees = Confusion.of(preds[preds["dataset"].isin(neg_lanes)])

        # Recall (non-degeneracy) only where the judge shares the dataset's
        # definition: the answer-oriented anchor lanes.
        anchor = preds[preds["dataset"].isin(cfg.anchor_lanes)]
        anchors = Confusion.of(anchor)

        by_lane = {
            lane: float((group["pred_relevant"] == group["human_relevant"]).mean())
            for lane, group in preds.groupby("dataset")
        }
        hardest = min(by_lane.items(), key=lambda kv: kv[1]) if by_lane else (None, float("nan"))

        # Precision is measured on referee lanes; the judge is applied to the
        # residual ones. When those sets are disjoint the number transfers, it
        # does not measure — say so in the artifact rather than in a comment.
        refereed_deploy = sorted(set(deploy_lanes or []) & set(neg_lanes))
        unrefereed_deploy = sorted(set(deploy_lanes or []) - set(neg_lanes))

        # Recall needs only POSITIVES, so unlike precision it IS measurable on the
        # positive-only deploy lanes. Reported, never gated: it measures agreement
        # with each corpus's OWN notion of relevant, and those differ wildly
        # (rarb-math is answer-oriented like the judge; crumb-legal-qa counts
        # topical precedent). A low number is the strict-vs-topical gap made
        # visible, not a verdict on the judge.
        on_deploy = preds[preds["dataset"].isin(deploy_lanes or [])]
        deploy_recall = Confusion.of(on_deploy).recall if len(on_deploy) else float("nan")
        deploy_recall_by_lane = {
            lane: Confusion.of(group).recall
            for lane, group in on_deploy.groupby("dataset")
        }

        return {
            "passed": Gate(cfg).passed(referees, anchors),
            "gate_enforced": list(Gate.ENFORCED),
            "precision_relevant": referees.precision,
            "precision_is_transfer_estimate": bool(deploy_lanes) and not refereed_deploy,
            "referee_lanes": sorted(neg_lanes),
            "deploy_lanes_refereed": refereed_deploy,
            "deploy_lanes_unrefereed": unrefereed_deploy,
            # DIAGNOSTIC, not gated — see the comment above `on_deploy`
            "deploy_recall": deploy_recall,
            "deploy_recall_by_lane": deploy_recall_by_lane,
            "deploy_lanes_measured": int(on_deploy["dataset"].nunique()) if len(on_deploy) else 0,
            # DIAGNOSTIC, not gated: share of random corpus docs the judge called
            # relevant, on the lanes it is actually applied to. Bounds gross
            # over-calling; a near-miss boundary error is invisible to it.
            "deploy_pseudo_negatives": int(len(pseudo)),
            "deploy_false_positive_rate": (
                float(pseudo["pred_relevant"].mean()) if len(pseudo) else float("nan")
            ),
            "deploy_fp_by_lane": {
                lane: float(group["pred_relevant"].mean())
                for lane, group in pseudo.groupby("dataset")
            } if len(pseudo) else {},
            "anchor_recall": anchors.recall,
            "anchor_lanes_present": sorted(set(anchor["dataset"].unique())),
            "recall_relevant": overall.recall,
            "precision_pooled": overall.precision,
            "has_negatives": referees.has_negatives,
            "agreement_overall": overall.agreement,
            "confusion": overall._asdict(),
            "false_positives": overall.fp,
            "agreement_by_lane": by_lane,
            "hardest_lane": {"lane": hardest[0], "agreement": hardest[1]},
            "n_validation": int(len(preds)),
            "lanes_without_negatives": [
                lane for lane in lanes if by_lane_negatives.get(lane) is False
            ],
            "predictions_path": str(cfg.validation_predictions),
            # what this gate cost: the run priced it and nothing recorded it
            "spend_usd": self.last_spend_usd,
            "llm_calls": self.last_spend.hops if self.last_spend else None,
            "llm_tokens": self.last_spend.tokens if self.last_spend else None,
            "thresholds": {
                "min_precision_relevant": cfg.min_precision_relevant,
                "min_anchor_recall": cfg.min_anchor_recall,
                "min_agreement_overall_DIAGNOSTIC": cfg.min_agreement_overall,
                "min_agreement_lane_DIAGNOSTIC": cfg.min_agreement_lane,
            },
        }

    def deploy_negatives(
        self, lanes: list[str], *, per_lane: int = 25, seed: int = 0
    ) -> pd.DataFrame:
        """Random-corpus pseudo-negatives for lanes that ship no human negatives.

        The referee lanes and the lanes the judge is APPLIED to are disjoint, so
        `precision_relevant` is a transfer estimate, never a measurement on the
        deployment population. A doc drawn uniformly from a 10K+ corpus is
        irrelevant to a given query with near-certainty, so calling one relevant
        is a visible false positive on the deploy lane itself. This catches gross
        over-calling only — random docs sit far from the query, where the real
        boundary errors are near-misses — so it bounds the error, not measures it.
        """
        frames = []
        for lane in lanes:
            gold = self.sources.manifest_gold(lane)
            if not gold:
                print(f"  {lane}: no manifest gold — no pseudo-negatives")
                continue
            # Intersect against the ids the lane ships text for: queries.parquet
            # is natural-only, and a lane whose gold is mostly synthetic (quest:
            # 15,256 `lane-*` of 17,377) yields nothing if sampled blind.
            candidates = sorted(set(gold) & self.sources.query_ids(lane))[:per_lane]
            if not candidates:
                print(f"  {lane}: none of {len(gold)} gold ids ship query text"
                      f" — no pseudo-negatives")
                continue
            queries = self.sources.query_text(lane, set(candidates))
            usable = [q for q in candidates if queries.get(q, "").strip()]
            doc_ids = self.sources.sample_doc_ids(lane, len(usable) * 2, seed=seed)
            texts = self.sources.corpus_text(lane, set(doc_ids))
            pool = [(d, t) for d, t in texts.items() if t]
            for i, query_id in enumerate(usable):
                if i >= len(pool):
                    break
                doc_id, doc_text = pool[i]
                if doc_id in gold.get(query_id, set()):
                    continue     # a real gold doc is not a pseudo-negative
                frames.append({
                    "dataset": lane, "query_id": query_id, "doc_id": doc_id,
                    "human_relevant": False, "query": queries[query_id],
                    "doc_text": doc_text, "pseudo": True,
                })
        return pd.DataFrame(frames, columns=[
            "dataset", "query_id", "doc_id", "human_relevant", "query", "doc_text",
            "pseudo",
        ])

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

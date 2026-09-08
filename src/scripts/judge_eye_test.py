"""Print real (query, document, verdict) triples from both judge stages, so the
labels can be eyeballed rather than trusted.

Stage 1 (`judge_rows`) shows the judge against a HUMAN verdict that shipped with
the lane's qrels; stage 2 (`judge_pairs`) shows the gold it invents where no human
verdict exists. Reads only what is on disk — no LLM calls, no spend.

    poetry run python src/scripts/judge_eye_test.py
    poetry run python src/scripts/judge_eye_test.py --lane clerc --per-case 3
"""

from __future__ import annotations

import argparse
import textwrap

import pandas as pd

from relevance_judge import RelevanceJudge, RelevanceJudgeConfig, Sources
from relevance_judge.judge import RATIONALE_FIELDS

WRAP = 100


def _show(label: str, text: str, limit: int = 300) -> None:
    body = " ".join(str(text).split())[:limit] or "(empty)"
    wrapped = textwrap.fill(body, WRAP, initial_indent="", subsequent_indent=" " * 12)
    print(f"  {label:<10}{wrapped}")


def _texts(sources: Sources, frame: pd.DataFrame) -> tuple[dict, dict]:
    queries, docs = {}, {}
    for lane, group in frame.groupby("dataset"):
        queries |= {
            (lane, q): t
            for q, t in sources.query_text(lane, set(group.query_id)).items()
        }
        docs |= {
            (lane, d): t
            for d, t in sources.corpus_text(lane, set(group.doc_id)).items()
        }
    return queries, docs


def stage_one(config: RelevanceJudgeConfig, sources: Sources, per_case: int) -> None:
    """The judge scored against a human verdict — the accuracy measurement."""
    print("=" * WRAP)
    print("STAGE 1  judge_rows()  ->  validation_predictions.parquet")
    print("=" * WRAP)
    print("The HUMAN column is not ours: it is the relevance grade that SHIPPED with")
    print("the lane's qrels.parquet (BEIR / MIRACL / freshstack annotators). We only")
    print("sample it. That is what makes accuracy measurable here — and why it can")
    print("only be measured on lanes whose annotators also marked docs IRRELEVANT.\n")

    preds = pd.read_parquet(config.validation_predictions)
    if "pseudo" in preds.columns:
        preds = preds[~preds["pseudo"].fillna(False).astype(bool)]
    queries, docs = _texts(sources, preds)

    cases = {
        "AGREE, relevant     (true positive)":  (preds.pred_relevant & preds.human_relevant),
        "JUDGE OVER-CALLED   (false positive)": (preds.pred_relevant & ~preds.human_relevant),
        "JUDGE MISSED IT     (false negative)": (~preds.pred_relevant & preds.human_relevant),
        "AGREE, irrelevant   (true negative)":  (~preds.pred_relevant & ~preds.human_relevant),
    }
    for title, mask in cases.items():
        subset = preds[mask]
        print(f"\n--- {title}   [{len(subset):,} rows] ---")
        for row in subset.head(per_case).itertuples(index=False):
            grade = _human_grade(sources, row)
            print(f"\n  [{row.dataset}  q={row.query_id}  doc={row.doc_id}]")
            _show("QUERY", queries.get((row.dataset, row.query_id), ""))
            _show("DOC", docs.get((row.dataset, row.doc_id), ""))
            print(f"  {'HUMAN':<10}{'RELEVANT' if row.human_relevant else 'not relevant'}"
                  f"   (qrels grade {grade})")
            print(f"  {'JUDGE':<10}{'RELEVANT' if row.pred_relevant else 'not relevant'}"
                  f"   — {row.reason}")


def _human_grade(sources: Sources, row) -> str:
    qrels = sources.base_qrels(row.dataset)
    if qrels.empty:
        return "?"
    hit = qrels[(qrels.query_id.astype(str) == str(row.query_id))
                & (qrels.doc_id.astype(str) == str(row.doc_id))]
    return str(hit.relevance.iloc[0]) if len(hit) else "unjudged"


def stage_two(config: RelevanceJudgeConfig, sources: Sources, per_case: int) -> None:
    """The gold the judge invents where no human verdict exists."""
    print("\n" + "=" * WRAP)
    print("STAGE 2  judge_pairs()  ->  judged_qrels.parquet")
    print("=" * WRAP)
    print("No HUMAN column exists here, by construction: these are the pairs nobody")
    print("judged. The judge's verdict BECOMES the gold, which is why stage 1 has to")
    print("pass a precision gate before this is allowed to run at all.\n")

    atoms = RelevanceJudge(config).load()
    if atoms.empty:
        print("  no atoms banked yet")
        return
    queries, docs = _texts(sources, atoms)

    cases = {
        "QRELS HOLE FILLED  (judged relevant -> new gold)": atoms.relevance == 1,
        "REJECTED           (judged irrelevant)":           atoms.relevance == 0,
    }
    for title, mask in cases.items():
        subset = atoms[mask]
        print(f"\n--- {title}   [{len(subset):,} atoms] ---")
        for row in subset.head(per_case).itertuples(index=False):
            print(f"\n  [{row.dataset}  q={row.query_id}  doc={row.doc_id}]")
            _show("QUERY", queries.get((row.dataset, row.query_id), ""))
            _show("DOC", docs.get((row.dataset, row.doc_id), ""))
            print(f"  {'HUMAN':<10}(never judged this pair)")
            print(f"  {'JUDGE':<10}{'RELEVANT' if row.relevance else 'not relevant'}"
                  f"   — {row.reason}")
            for field in RATIONALE_FIELDS:
                value = getattr(row, field, "")
                if isinstance(value, str) and value:
                    _show(f"  {field}", value, limit=200)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-case", type=int, default=2, help="examples per case")
    parser.add_argument("--lane", default=None, help="restrict to one lane")
    args = parser.parse_args()

    config = RelevanceJudgeConfig()
    sources = Sources(config)
    stage_one(config, sources, args.per_case)
    stage_two(config, sources, args.per_case)

    print("\n" + "=" * WRAP)
    print("HOW THEY RELATE")
    print("=" * WRAP)
    print("stage 1 measures the instrument  ->  gate passes  ->  stage 2 spends it")
    print("Every atom carries the judge_run_id of the gate that authorized it, so any")
    print("gold label can be traced back to a measured precision. The catch: the lanes")
    print("stage 1 can measure and the lanes stage 2 runs on are DISJOINT, so that")
    print("precision is a transfer estimate (see validation_report.json ->")
    print("precision_is_transfer_estimate).")


if __name__ == "__main__":
    main()

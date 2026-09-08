"""Six-lane paired A/B test: does adding a per-lane card preserve precision and
improve recall on lanes that have BOTH human positives and negatives?

Same 200 pos + 200 neg pairs per lane are judged twice: arm A with the universal
INSTRUCTION only, arm B with INSTRUCTION + a supplied lane card. Per-lane Δp/Δr
with bootstrap CIs decide whether the card mechanism generalizes.

Not a validator of any specific deploy card. It measures whether the CLASS of
"add a card" is safe-and-useful on lanes where safety CAN be measured. If yes,
the same mechanism has empirical footing on deploy lanes (which cannot measure
their own safety); if no, deploy cards are riding on hope.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from augmentation.engine import Budget  # noqa: E402
from relevance_judge import RelevanceJudgeConfig, RelevanceJudge, Sources, ValidationHarness  # noqa: E402
from relevance_judge.card_generator import CardGenerator  # noqa: E402
from relevance_judge.lane_context import LaneContext  # noqa: E402
import relevance_judge.judge as judge_mod  # noqa: E402

# The six well-powered lanes (>=1,000 positives AND >=1,000 negatives).
# beir-touche-2020 is dropped as marginal (932/1282).
SIX_LANES: tuple[str, ...] = (
    "crumb-clinical-trial",
    "wands",
    "dbpedia-entity",
    "freshstack-langchain",
    "freshstack-laravel",
    "miracl-en-dev",
)

# Seed cards: hand-written from the dataset_registry docstrings. These are the
# INPUT to the test — swap them for generator output to validate the generator
# itself. Every card names the operative relevance boundary its lane uses; none
# loosens the universal standard of evidence.
SEED_CARDS: dict[str, LaneContext] = {
    "crumb-clinical-trial": LaneContext(
        task="clinical-trial retrieval: find a trial matching a patient description "
             "under multiple eligibility constraints simultaneously",
        queries="a paragraph describing a real patient (age, sex, condition, prior "
                "treatments, comorbidities). Every stated constraint is a filter, "
                "not a preference.",
        gold="the clinical-trial record whose eligibility criteria the patient "
             "satisfies on every named axis",
        judging="a trial is relevant only when its visible criteria admit the patient "
                "on ALL stated axes — condition, stage, prior treatment, key "
                "exclusions. A trial matching the condition but violating any other "
                "named constraint is NOT relevant. Do not infer eligibility the "
                "criteria do not state.",
    ),
    "wands": LaneContext(
        task="product search over a home-goods catalog",
        queries="short product-search queries from real Wayfair sessions — often a "
                "category plus attributes (material, color, dimension, style)",
        gold="a catalog listing for a product that IS the queried item with the "
             "requested attributes",
        judging="a product is relevant only when the listing's own text names the "
                "queried item AND every attribute the query specifies (color, "
                "material, size, style, room). A related product in the same "
                "category without a queried attribute is NOT relevant.",
    ),
    "dbpedia-entity": LaneContext(
        task="entity retrieval: return the Wikipedia entity a keyword query names",
        queries="a bare noun phrase or short keyword string naming ONE target entity "
                "(person, place, work, organization)",
        gold="the DBpedia article ABOUT that specific entity",
        judging="an article is relevant only when it IS the article about the named "
                "entity — same identity, not a related, containing, or same-category "
                "entity. An article mentioning the entity in passing is NOT relevant.",
    ),
    "freshstack-langchain": LaneContext(
        task="developer-QA retrieval over the langchain documentation and issue tracker",
        queries="a StackOverflow-style question about using langchain — an API, a "
                "configuration, an error, an integration",
        gold="a doc page, issue, or discussion that directly addresses the asked "
             "usage or fix",
        judging="a page is relevant when its visible text ANSWERS the specific "
                "usage question — names the API/config/fix the question asks about. "
                "A page about a nearby feature or the same subsystem generally is "
                "NOT relevant.",
    ),
    "freshstack-laravel": LaneContext(
        task="developer-QA retrieval over the laravel documentation and issue tracker",
        queries="a StackOverflow-style question about using laravel — a class, a "
                "config, an error, an integration",
        gold="a doc page, issue, or discussion that directly addresses the asked "
             "usage or fix",
        judging="a page is relevant when its visible text ANSWERS the specific "
                "usage question — names the API/config/fix the question asks about. "
                "A page about a nearby feature or the same subsystem generally is "
                "NOT relevant.",
    ),
    "miracl-en-dev": LaneContext(
        task="multilingual passage retrieval (English split): find a Wikipedia "
             "passage answering a natural-language question",
        queries="a natural-language factoid or explanation question in English",
        gold="a Wikipedia passage that states the answer",
        judging="a passage is relevant only when its visible words state the "
                "answer to the question. A passage about the same topic that does "
                "not state the answer is NOT relevant.",
    ),
}


@dataclass(frozen=True)
class LaneResult:
    lane: str
    n_pos: int
    n_neg: int
    p_baseline: float
    p_carded: float
    r_baseline: float
    r_carded: float
    dp: float
    dp_ci: tuple[float, float]
    dr: float
    dr_ci: tuple[float, float]

    def verdict(self, eps: float, delta: float) -> str:
        safe = self.dp_ci[0] >= -eps
        useful = self.dr_ci[0] > 0
        strong = self.dr >= delta
        if safe and strong:
            return "PASS"
        if safe and useful:
            return "SAFE_MARGINAL"
        if safe:
            return "SAFE_NO_LIFT"
        return "UNSAFE"


def _bootstrap_diff(a: np.ndarray, b: np.ndarray, *, iters: int = 2000, seed: int = 0
                    ) -> tuple[float, tuple[float, float]]:
    """Paired-bootstrap 95% CI on the difference of means. a and b are 0/1
    predictions on the SAME set of pairs (one per arm)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    if n == 0:
        return float("nan"), (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(iters, n))
    diffs = (b[idx].mean(1) - a[idx].mean(1))
    return float(b.mean() - a.mean()), (float(np.quantile(diffs, 0.025)),
                                        float(np.quantile(diffs, 0.975)))


def _score_arm(preds: pd.DataFrame, lane: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Per-pair 0/1 arrays for precision (over predicted-yes) and recall (over
    human-yes), aligned so paired-bootstrap sees the same pair-ids across arms."""
    g = preds[preds["dataset"] == lane].sort_values(["query_id", "doc_id"])
    return (
        g["pred_relevant"].astype(int).values, g["human_relevant"].astype(int).values,
        int(g["human_relevant"].sum()), int((~g["human_relevant"]).sum()),
    )


def _summarize(base: pd.DataFrame, carded: pd.DataFrame, lane: str, seed: int) -> LaneResult:
    """Fold two arms into one lane result. Bootstraps on the shared pair set."""
    key = ["dataset", "query_id", "doc_id", "human_relevant"]
    m = base.merge(carded, on=key, suffixes=("_base", "_card"))
    g = m[m["dataset"] == lane].sort_values(["query_id", "doc_id"])
    if g.empty:
        return LaneResult(lane, 0, 0, *([float("nan")] * 4),
                          float("nan"), (float("nan"),) * 2,
                          float("nan"), (float("nan"),) * 2)
    hum = g["human_relevant"].astype(bool).values
    pb, pc = g["pred_relevant_base"].astype(bool).values, g["pred_relevant_card"].astype(bool).values

    def _prec(pred, hum):
        return float((hum & pred).sum() / pred.sum()) if pred.sum() else float("nan")
    pb_p = _prec(pb, hum)
    pc_p = _prec(pc, hum)

    # For CI: bootstrap over the shared pair set, precision computed per arm
    # per resample. When an arm predicts no positives in a resample, that resample
    # is skipped for the diff (nan-safe).
    rng = np.random.default_rng(seed)
    n = len(g)
    diffs_p, diffs_r = [], []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        pb_s, pc_s, h_s = pb[idx], pc[idx], hum[idx]
        db, dc = pb_s.sum(), pc_s.sum()
        if db and dc:
            diffs_p.append((pc_s & h_s).sum() / dc - (pb_s & h_s).sum() / db)
        hb = h_s.sum()
        if hb:
            diffs_r.append((pc_s & h_s).sum() / hb - (pb_s & h_s).sum() / hb)
    dp = pc_p - pb_p
    dp_ci = (float(np.quantile(diffs_p, 0.025)), float(np.quantile(diffs_p, 0.975))) if diffs_p else (float("nan"), float("nan"))

    hb = hum.sum()
    rb_p = float((pb & hum).sum() / hb) if hb else float("nan")
    rc_p = float((pc & hum).sum() / hb) if hb else float("nan")
    dr = rc_p - rb_p
    dr_ci = (float(np.quantile(diffs_r, 0.025)), float(np.quantile(diffs_r, 0.975))) if diffs_r else (float("nan"), float("nan"))

    return LaneResult(
        lane=lane, n_pos=int(hum.sum()), n_neg=int((~hum).sum()),
        p_baseline=pb_p, p_carded=pc_p, r_baseline=rb_p, r_carded=rc_p,
        dp=dp, dp_ci=dp_ci, dr=dr, dr_ci=dr_ci,
    )


def _install_cards(cards: dict[str, LaneContext]) -> None:
    """Monkey-patch judge.context_for to serve THIS harness's cards. Any lane
    not in the dict resolves to "" — a clean baseline arm."""
    def _ctx(dataset: str) -> str:
        card = cards.get(dataset)
        return card.render() if card else ""
    judge_mod.context_for = _ctx


def run(lanes: tuple[str, ...] = SIX_LANES, *, cards: dict[str, LaneContext] | None = None,
        use_generator: bool = False, regenerate: bool = False,
        per_lane: int = 200, seed: int = 0, max_spend_usd: float = 3.0,
        out_dir: Path | None = None) -> pd.DataFrame:
    """Run both arms, persist per-pair predictions, return a per-lane result frame."""
    out_dir = out_dir or (Path(__file__).resolve().parent.parent / "data" / "relevance_judge" / "card_transfer")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = RelevanceJudgeConfig()
    src = Sources(config)
    lanes = tuple(name for name in lanes if name in src.lanes_with_negatives())
    if use_generator:
        print(f"generating cards for {len(lanes)} lane(s) via CardGenerator...")
        cards = CardGenerator(config, src).generate_all(list(lanes), regenerate=regenerate)
    else:
        cards = cards or SEED_CARDS

    print(f"card_transfer: {len(lanes)} lane(s) with negatives; per_lane={per_lane}; seed={seed}")
    print(f"  spend ceiling: ${max_spend_usd:.2f} per arm ($ {2 * max_spend_usd:.2f} total)")

    def _arm(name: str, arm_cards: dict[str, LaneContext]) -> pd.DataFrame:
        _install_cards(arm_cards)
        judge = RelevanceJudge(config)
        harness = ValidationHarness(config, judge=judge, sources=src)
        rows = harness.sample(list(lanes), per_lane=per_lane, seed=seed, verbose=(name == "baseline"))
        budget = Budget(
            max_spend_usd,
            usd_per_mtok_in=config.engine.usd_per_mtok_in,
            usd_per_mtok_out=config.engine.usd_per_mtok_out,
        )
        preds = harness.judge_rows(rows, budget=budget)
        path = out_dir / f"predictions_{name}.parquet"
        preds.to_parquet(path, index=False)
        print(f"  arm={name}: {len(preds):,} predictions -> {path.name}  "
              f"spend=${budget.spent_usd:.3f}")
        return preds

    base = _arm("baseline", {})
    carded = _arm("carded",   cards)

    rows = [_summarize(base, carded, lane, seed) for lane in lanes]
    frame = pd.DataFrame([r.__dict__ for r in rows])
    frame["verdict"] = [r.verdict(eps=0.02, delta=0.05) for r in rows]
    frame.to_parquet(out_dir / "summary.parquet", index=False)
    (out_dir / "summary.json").write_text(json.dumps(
        {r.lane: {**r.__dict__, "verdict": r.verdict(0.02, 0.05)} for r in rows},
        indent=2, default=float,
    ))
    return frame


def _print_report(frame: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("per-lane Δprecision / Δrecall (95% paired-bootstrap CI). eps=0.02, delta=0.05.")
    print("=" * 90)
    for r in frame.itertuples(index=False):
        pci = f"[{r.dp_ci[0]:+.3f}, {r.dp_ci[1]:+.3f}]"
        rci = f"[{r.dr_ci[0]:+.3f}, {r.dr_ci[1]:+.3f}]"
        print(f"  {r.lane:24s} n={r.n_pos:>3}+/{r.n_neg:<3}-  "
              f"Δp={r.dp:+.3f} {pci}   Δr={r.dr:+.3f} {rci}   {r.verdict}")
    verdicts = frame["verdict"].value_counts().to_dict()
    print("\nverdict counts:", verdicts)
    unsafe = frame[frame["verdict"] == "UNSAFE"]
    if unsafe.empty:
        print("SAFETY: no lane's Δp CI lower bound falls below -0.02 — cards do no harm.")
    else:
        print("SAFETY: FAILED on", ", ".join(unsafe["lane"]))


def _self_check() -> None:
    # arm summary: identical arms => Δp≈0, Δr≈0.
    idx = pd.MultiIndex.from_product([["wands"], ["q0", "q1", "q2", "q3"], ["d"]],
                                     names=["dataset", "query_id", "doc_id"]).to_frame(index=False)
    idx["human_relevant"] = [True, True, False, False]
    idx["pred_relevant"] = [True, False, False, True]
    r = _summarize(idx.copy(), idx.copy(), "wands", seed=0)
    assert abs(r.dp) < 1e-9 and abs(r.dr) < 1e-9
    # Perfect improvement: baseline all no; carded matches human.
    a = idx.assign(pred_relevant=[False] * 4)
    b = idx.assign(pred_relevant=[True, True, False, False])
    r2 = _summarize(a, b, "wands", seed=0)
    assert r2.dr == 1.0
    print("card_transfer self-check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="run self-check and exit")
    ap.add_argument("--per-lane", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=float, default=3.0, help="USD ceiling per arm")
    ap.add_argument("--use-generator", action="store_true",
                    help="build arm B from CardGenerator output (cached)")
    ap.add_argument("--regenerate", action="store_true",
                    help="ignore the generator cache and re-issue every card")
    args = ap.parse_args()
    if args.check:
        _self_check()
        sys.exit(0)
    frame = run(per_lane=args.per_lane, seed=args.seed, max_spend_usd=args.cap,
                use_generator=args.use_generator, regenerate=args.regenerate)
    _print_report(frame)

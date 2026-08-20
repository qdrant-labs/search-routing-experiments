"""Phase 5.3: the unweighted pooled metric is the sole headline, and every lane
weighting is a robustness CHECK, not an anchor — the pool's own mix is our
acquisition caps, never a traffic prior. Pre-registered rule: one sign across
every candidate is weighting-robust, anything else is UNDECIDED and is never
averaged into a single number.

    poetry run python src/scripts/weighting_sensitivity.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from scripts.run_ablation import ABLATION_OUT
from scripts.select_v3_prototype import OUT, SelectorRecipe, _load_labels

SENSITIVITY_OUT = OUT / "weighting_sensitivity"
PER_LANE = ABLATION_OUT / "per_lane.parquet"


def candidate_weightings(
    per_lane: pd.DataFrame, pool_counts: pd.Series, lane_share_cap: float
) -> dict[str, pd.Series]:
    """Every weighting the sensitivity check sweeps, keyed by name. `unweighted`
    is the headline (weights proportional to each lane's own n, i.e. the pooled
    row mean — no reweighting applied); `pool_share` and `pool_share_capped` are
    CANDIDATES, not anchors: they encode which lanes we happened to acquire and
    how hard the selector caps that, and neither is evidence about traffic."""
    lanes = per_lane.index
    counts = pool_counts.reindex(lanes).fillna(0.0)
    capped = counts / counts.sum() if counts.sum() else counts
    capped = capped.clip(upper=lane_share_cap)
    return {
        "unweighted": per_lane["n"].astype(float),
        "lane_uniform": pd.Series(1.0, index=lanes),
        "pool_share": counts,
        "pool_share_capped": capped,
    }


def sign_robustness(
    values: dict[str, float], headline: str, tol: float = 0.0
) -> dict[str, object]:
    """The pre-registered pass/fail: a sign that holds strictly across every
    candidate is `weighting-robust`; a candidate of the opposite sign is a
    `sign_flip` and one within `tol` of zero is `at_zero` — both UNDECIDED, and
    neither is resolved by averaging the candidates together."""
    signs = {name: int(np.sign(v)) if abs(v) > tol else 0 for name, v in values.items()}
    positive = {n for n, s in signs.items() if s > 0}
    negative = {n for n, s in signs.items() if s < 0}
    at_zero = {n for n, s in signs.items() if s == 0}
    if positive and negative:
        verdict, reason = "UNDECIDED", "sign_flip"
    elif at_zero:
        verdict, reason = "UNDECIDED", "at_zero"
    else:
        verdict, reason = "weighting-robust", ""
    return {
        "headline": headline,
        "headline_value": values[headline],
        "values": values,
        "signs": signs,
        "verdict": verdict,
        "reason": reason,
        "flipping_candidates": sorted(
            n for n, s in signs.items() if s and s != signs[headline]
        ),
        "at_zero_candidates": sorted(at_zero),
        "tol": tol,
    }


def check(
    per_lane: pd.DataFrame,
    value_col: str,
    pool_counts: pd.Series,
    lane_share_cap: float = SelectorRecipe().lane_share_cap,
    tol: float = 0.0,
) -> dict[str, object]:
    """Reweight one per-lane metric under every candidate and apply the rule."""
    weights = candidate_weightings(per_lane, pool_counts, lane_share_cap)
    values = {
        name: float(np.average(per_lane[value_col], weights=w))
        for name, w in weights.items()
        if float(np.asarray(w, dtype=float).sum()) > 0
    }
    share = pool_counts.reindex(per_lane.index).fillna(0.0)
    return sign_robustness(values, "unweighted", tol) | {
        "metric": value_col,
        "lanes": len(per_lane),
        "rows": int(per_lane["n"].sum()),
        "lane_share_cap": lane_share_cap,
        "cap_binds": bool((share / share.sum() > lane_share_cap).any()),
    }


def report(result: dict[str, object]) -> None:
    print(f"metric: {result['metric']} over {result['lanes']} lanes "
          f"({result['rows']:,} rows)")
    print(f"\nHEADLINE (unweighted, the only non-circular number): "
          f"{result['headline_value']:+.6f}\n")
    print("candidate weightings (checks, NOT anchors — the pool's mix is our "
          "own acquisition caps):")
    for name, value in result["values"].items():
        mark = "  <- headline" if name == result["headline"] else ""
        print(f"  {name:<20} {value:+.6f}  sign {result['signs'][name]:+d}{mark}")
    if not result["cap_binds"]:
        print(f"  (no lane exceeds {result['lane_share_cap']:.0%} of the pool, so "
              f"pool_share_capped is not a distinct candidate today)")
    if result["verdict"] == "weighting-robust":
        print("\nVERDICT: weighting-robust — one sign under every candidate.")
        return
    print(f"\n*** VERDICT: UNDECIDED ({result['reason']}) ***")
    if result["reason"] == "sign_flip":
        print(f"    the sign flips under: {', '.join(result['flipping_candidates'])}")
    else:
        print(f"    indistinguishable from zero under: "
              f"{', '.join(result['at_zero_candidates'])}")
    print("    Do NOT average these into one number — the pre-registered rule "
          "reports UNDECIDED and stops.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-lane", default=str(PER_LANE))
    parser.add_argument("--column", default="mean_diff")
    parser.add_argument("--tol", type=float, default=0.0)
    args = parser.parse_args()

    per_lane = pd.read_parquet(args.per_lane)
    pool_counts = _load_labels().groupby("dataset").size()
    result = check(per_lane, args.column, pool_counts, tol=args.tol)
    SENSITIVITY_OUT.mkdir(parents=True, exist_ok=True)
    (SENSITIVITY_OUT / f"{args.column}.json").write_text(json.dumps(result, indent=2))
    report(result)
    print(f"\n-> {SENSITIVITY_OUT}")


if __name__ == "__main__":
    main()

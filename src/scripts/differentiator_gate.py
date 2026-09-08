"""Differentiator gate: is an alternative stack a net-positive labeller vs the baseline?
Signed and trust-aware — raw movement is never a win by itself. Consumed by
leg2_encoder_pilot (trust=verified) and r1_reranker_pilot (trust=opinion)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from composition.pool_v3 import CEILING, TOL

# Thresholds live here, once. all_zero carries the 0.7 Hit@1 weight (P0.4) so its
# bar is high; a tie AT ceiling cannot improve Hit@1 (P0.3) so breaking it is
# regression/over-accept, gated as damage rather than rewarded.
# ponytail: these five bars are hand-set placeholders, NOT derived from pipeline
# data — the honest weak point of this gate. Calibrate against a real quantity
# (e.g. the baseline's own certifiable-decisive rate, or an explicit corruption
# budget) before treating a PASS/FAIL near the line as load-bearing.
ALL_ZERO_MIN = 0.25       # >= this share of all_zero must move off zero
LOW_TIE_MIN = 0.10        # >= this share of below-ceiling ties must break UPWARD
CEILING_BREAK_MAX = 0.05  # <= this share of at-ceiling ties may break at all
NET_RATIO_MIN = 3.0       # benefit must outweigh damage at least this much
DAMAGE_RATE_MAX = 0.05    # <= this share of baseline-decisive rows may be damaged
DEFAULT_MARGIN = 0.1      # top - runner needed to call a row certifiably decisive
CONSENSUS_DECISIVE_MIN = 0.60  # combiner target: share that must be CERTIFIABLY (margin) decisive

ROUTES = ("dense_only", "sparse_only", "pure_rrf")


def _view(paired: pd.DataFrame, suffix: str, routes) -> pd.DataFrame:
    """Per-row top/runner/low/margin/bucket for one score suffix (e.g. '1', '2', 'r1')."""
    m = np.sort(paired[[f"score_{r}_{suffix}" for r in routes]].to_numpy(dtype=float), axis=1)
    low, runner, top = m[:, 0], m[:, -2], m[:, -1]
    bucket = np.where(top <= TOL, "all_zero",
             np.where(top - low <= TOL, "all_tied", "routes_differ"))
    return pd.DataFrame({"top": top, "runner": runner, "low": low,
                         "margin": top - runner, "bucket": bucket}, index=paired.index)


def _rate(mask: pd.Series, pop: pd.Series) -> float:
    n = int(pop.sum())
    return float(mask.sum()) / n if n else float("nan")


@dataclass
class Scorecard:
    trust: str
    n: int
    checks: dict = field(default_factory=dict)   # name -> (value, threshold, passed|None)
    counts: dict = field(default_factory=dict)   # raw numbers for the report
    verdict: str = "FAIL"

    def report(self) -> "Scorecard":
        c = self.counts
        print(f"differentiator gate  ({self.trust}, n={self.n})")
        print(f"  (1) all_zero rescued : {c['rescued']}/{c['all_zero_n']}"
              f"  low-tie broke UP : {c['broke_up']}/{c['low_tie_n']}"
              f"  ceiling-tie broke : {c['ceil_broke']}/{c['ceiling_tie_n']}")
        print(f"  (2) benefit {c['benefit']}  (rescued {c['rescued']}, "
              f"tie-broke-up {c['broke_up']}, strengthened {c['decisive_strengthened']})")
        print(f"      damage  {c['damage']}  (destroyed {c['destroyed']}, "
              f"collapsed {c['collapsed']}, thinned {c['decisive_thinned']}, "
              f"ceiling-regress {c['ceil_broke']}, tie-regress {c['broke_down']}, "
              f"tie-lost {c['low_tie_lost']})")
        print(f"  (3) certifiable-decisive (margin>= {self.counts['class_margin']:.2f}): "
              f"baseline {self.counts['base_margin_dec']:.0%} -> candidate {self.counts['margin_dec']:.0%}"
              f"   (argmax {self.counts['argmax_dec']:.0%}, "
              f"inflation {self.counts['argmax_dec'] - self.counts['margin_dec']:+.0%})")
        for name, (val, thr, ok) in self.checks.items():
            flag = "n/a " if ok is None else ("PASS" if ok else "FAIL")
            print(f"      [{flag}] {name:18s} {val:.3f}  (bar {thr})")
        print(f"  => {self.verdict}"
              + ("  (opinion leg: upper bound until A11 human pairs)" if self.trust == "opinion" else ""))
        return self


def differentiator_gate(paired: pd.DataFrame, *, base: str = "1", cand: str = "2",
                        routes=ROUTES, trust: str = "verified",
                        class_margin: float = DEFAULT_MARGIN,
                        require_decisive: float | None = None,
                        weights: pd.Series | None = None) -> Scorecard:
    """Score a candidate labelling (`score_{route}_{cand}`) against the baseline
    (`score_{route}_{base}`). trust='verified' when candidate scores are against
    real qrels (leg2); 'opinion' when they ride on unvalidated relevance (R1) —
    an opinion leg tops out at PROVISIONAL. A candidate must ALWAYS raise the
    certifiable-decisive share above the baseline (else it could pass 1 & 2 by
    parking all_zero rows in ties); `require_decisive` adds the combiner's
    absolute CERTIFIABLE-decisive target (on the margin share, not argmax, so it
    cannot be met with non-certifiable labels). `weights` (per row) reweights the
    decisiveness shares to pool proportions; None = raw (stratified) sample."""
    b, c = _view(paired, base, routes), _view(paired, cand, routes)

    is_zero = b.bucket == "all_zero"
    is_tie = b.bucket == "all_tied"
    ceiling_tie = is_tie & (b.top >= CEILING)
    low_tie = is_tie & (b.top < CEILING)
    is_decisive = b.bucket == "routes_differ"

    # (1) coverage, split by headroom
    rescued = is_zero & (c.top > TOL)
    broke_up = low_tie & (c.bucket == "routes_differ") & (c.top > b.top + TOL)
    ceil_broke = ceiling_tie & (c.bucket != "all_tied")

    # (2) net utility, signed. A baseline-with-a-hit row that regresses is damage:
    # a decisive row lost (destroyed), flattened (collapsed), or reassigned to a
    # DIFFERENT route below the certifiable margin (decisive_thinned); a low tie broken
    # by regression (broke_down) or lost to zero (low_tie_lost); a ceiling tie moved at
    # all (ceil_broke). Mirror benefit: a thin decisive row strengthened past the margin
    # (decisive_strengthened), plus rescues and upward tie breaks.
    #   `collapsed` and `decisive_thinned` require a CERTIFIABLE baseline (margin >=
    #   class_margin) — losing a weak, barely-decided label is not losing a strong one.
    #   `decisive_thinned` also requires the winning ROUTE to change: a same-winner
    #   margin drop is the same routing decision, not damage.
    b_cert = is_decisive & (b.margin >= class_margin)
    c_differ = c.bucket == "routes_differ"
    b_win = paired[[f"score_{r}_{base}" for r in routes]].to_numpy(dtype=float).argmax(1)
    c_win = paired[[f"score_{r}_{cand}" for r in routes]].to_numpy(dtype=float).argmax(1)
    winner_flip = pd.Series(b_win != c_win, index=paired.index)
    destroyed = is_decisive & (c.bucket == "all_zero")
    collapsed = b_cert & (c.bucket == "all_tied")
    decisive_thinned = b_cert & c_differ & (c.margin < class_margin) & winner_flip
    decisive_strengthened = is_decisive & (b.margin < class_margin) & c_differ & (c.margin >= class_margin)
    broke_down = low_tie & c_differ & (c.top <= b.top + TOL)
    low_tie_lost = low_tie & (c.bucket == "all_zero")
    benefit = int(rescued.sum() + broke_up.sum() + decisive_strengthened.sum())
    damage = int(destroyed.sum() + collapsed.sum() + ceil_broke.sum()
                 + broke_down.sum() + low_tie_lost.sum() + decisive_thinned.sum())

    # (3) consensus decisiveness — a candidate must add CERTIFIABLE decisive labels
    # over the baseline. Without this it can clear 1 & 2 by moving every all_zero
    # into a tie: off zero (passes 1), positive benefit (passes 2), zero new
    # decisive labels. `require_decisive` adds the combiner's absolute target.
    def _share(mask):
        return float(mask.mean()) if weights is None else float((mask * (weights / weights.sum())).sum())

    c_decisive = c.bucket == "routes_differ"
    argmax_dec = _share(c_decisive)
    margin_dec = _share(c_decisive & (c.margin >= class_margin))
    base_margin_dec = _share(is_decisive & (b.margin >= class_margin))

    net_ratio = benefit / max(damage, 1)
    damage_rate = damage / max(int(is_decisive.sum()), 1)
    checks = {
        "all_zero_rescue": (_rate(rescued, is_zero), f">={ALL_ZERO_MIN}",
                            None if not is_zero.any() else _rate(rescued, is_zero) >= ALL_ZERO_MIN),
        "low_tie_break_up": (_rate(broke_up, low_tie), f">={LOW_TIE_MIN}",
                             None if not low_tie.any() else _rate(broke_up, low_tie) >= LOW_TIE_MIN),
        "ceiling_break": (_rate(ceil_broke, ceiling_tie), f"<={CEILING_BREAK_MAX}",
                          None if not ceiling_tie.any() else _rate(ceil_broke, ceiling_tie) <= CEILING_BREAK_MAX),
        "net_ratio": (net_ratio, f">={NET_RATIO_MIN}", net_ratio >= NET_RATIO_MIN),
        "damage_rate": (damage_rate, f"<={DAMAGE_RATE_MAX}", damage_rate <= DAMAGE_RATE_MAX),
        "decisive_gain": (margin_dec, f">{base_margin_dec:.2f} baseline", margin_dec > base_margin_dec),
    }
    if require_decisive is not None:
        # gate the CERTIFIABLE (margin) share, not argmax — else the target is
        # reachable with non-certifiable labels (e.g. a ceiling tie broken by regression)
        checks["decisive_target"] = (margin_dec, f">={require_decisive}", margin_dec >= require_decisive)
    passed = all(ok for _, _, ok in checks.values() if ok is not None)
    verdict = "PASS" if passed and trust == "verified" else "PROVISIONAL" if passed else "FAIL"

    counts = dict(all_zero_n=int(is_zero.sum()), rescued=int(rescued.sum()),
                  low_tie_n=int(low_tie.sum()), broke_up=int(broke_up.sum()),
                  ceiling_tie_n=int(ceiling_tie.sum()), ceil_broke=int(ceil_broke.sum()),
                  destroyed=int(destroyed.sum()), collapsed=int(collapsed.sum()),
                  broke_down=int(broke_down.sum()), low_tie_lost=int(low_tie_lost.sum()),
                  decisive_thinned=int(decisive_thinned.sum()),
                  decisive_strengthened=int(decisive_strengthened.sum()),
                  benefit=benefit, damage=damage,
                  argmax_dec=argmax_dec, margin_dec=margin_dec, base_margin_dec=base_margin_dec,
                  class_margin=class_margin)
    return Scorecard(trust=trust, n=len(paired), checks=checks, counts=counts, verdict=verdict)


# --- combiners (dormant until a shared 3-leg draw with score_{route}_{1,2,r1}) ---
# each maps the per-route arrays (leg1, leg2, r1) -> a combined per-route score.
COMBINERS = {
    "leg1":               lambda s1, s2, r1: s1,
    "leg2":               lambda s1, s2, r1: s2,
    "r1":                 lambda s1, s2, r1: r1,
    "sum_all":            lambda s1, s2, r1: s1 + s2 + r1,
    "encoder_max":        lambda s1, s2, r1: np.maximum(s1, s2),
    "encoder_max_plus_r1": lambda s1, s2, r1: np.maximum(s1, s2) + r1,  # the hypothesised strongest
}


def combine(paired3: pd.DataFrame, name: str, routes=ROUTES) -> pd.DataFrame:
    """Return `paired3` with `score_{route}_{name}` columns for one combiner."""
    fn = COMBINERS[name]
    out = paired3.copy()
    for r in routes:
        out[f"score_{r}_{name}"] = fn(paired3[f"score_{r}_1"].to_numpy(float),
                                      paired3[f"score_{r}_2"].to_numpy(float),
                                      paired3[f"score_{r}_r1"].to_numpy(float))
    return out


def bake_off(paired3: pd.DataFrame, names=tuple(COMBINERS), routes=ROUTES,
             trust: str = "opinion", require_decisive: float = CONSENSUS_DECISIVE_MIN,
             weights: pd.Series | None = None) -> pd.DataFrame:
    """Run the gate on each named combiner over a shared 3-leg frame; one row per
    candidate for plotting argmax vs margin decisiveness and the verdict. The
    consensus target is enforced here (criterion 3), so a combiner that does not
    reach it cannot PASS."""
    rows = []
    for name in names:
        frame = combine(paired3, name, routes)
        sc = differentiator_gate(frame, base="1", cand=name, routes=routes,
                                 trust="verified" if name in ("leg1", "leg2") else trust,
                                 require_decisive=require_decisive, weights=weights)
        rows.append({"candidate": name, "verdict": sc.verdict,
                     "argmax_decisive": sc.counts["argmax_dec"],
                     "margin_decisive": sc.counts["margin_dec"],
                     "benefit": sc.counts["benefit"], "damage": sc.counts["damage"]})
    return pd.DataFrame(rows).set_index("candidate")

"""Signed-accounting checks for the differentiator gate: a clean stack passes, and
each way a stack can look good while doing harm (collapse, rescue-into-ties, a target
met with non-certifiable labels, low ties zeroed out) is caught."""

import pandas as pd

from scripts.differentiator_gate import differentiator_gate


def _frame(rows):
    """Rows of {route: {suffix: score}} -> a paired frame with score_{route}_{suffix}."""
    return pd.DataFrame([{f"score_{r}_{s}": v for r, vs in row.items() for s, v in vs.items()}
                         for row in rows])


def test_clean_differentiator_passes():
    # 20 all_zero rescued to a real dense win + 6 decisive kept.
    good = _frame(
        [{"dense_only": {"1": 0.0, "2": 1.0}, "sparse_only": {"1": 0.0, "2": 0.0},
          "pure_rrf": {"1": 0.0, "2": 0.19}} for _ in range(20)]
        + [{"dense_only": {"1": 1.0, "2": 1.0}, "sparse_only": {"1": 0.1, "2": 0.1},
            "pure_rrf": {"1": 0.2, "2": 0.2}} for _ in range(6)])
    g = differentiator_gate(good, cand="2", trust="verified")
    assert g.verdict == "PASS"
    assert g.counts["rescued"] == 20 and g.counts["damage"] == 0


def test_decisive_rows_collapsed_to_ties_fail():
    bad = _frame(
        [{"dense_only": {"1": 1.0, "2": 0.5}, "sparse_only": {"1": 0.2, "2": 0.5},
          "pure_rrf": {"1": 0.3, "2": 0.5}} for _ in range(20)])
    b = differentiator_gate(bad, cand="2", trust="verified")
    assert b.verdict == "FAIL" and b.counts["collapsed"] == 20


def test_rescue_into_ties_adds_no_decisive_label_and_fails():
    # off zero (passes coverage) but only into a tie -> zero decisive gain.
    tie_only = _frame(
        [{"dense_only": {"1": 0.0, "2": 0.5}, "sparse_only": {"1": 0.0, "2": 0.5},
          "pure_rrf": {"1": 0.0, "2": 0.5}} for _ in range(20)])
    t = differentiator_gate(tie_only, cand="2", trust="verified")
    assert t.counts["rescued"] == 20
    assert t.checks["decisive_gain"][2] is False and t.verdict == "FAIL"


def test_consensus_target_is_certifiable_not_argmax():
    # 100% argmax-decisive but only 40% certifiable must miss a 0.6 target.
    mixed = _frame(
        [{"dense_only": {"1": 0.0, "2": 1.0}, "sparse_only": {"1": 0.0, "2": 0.0},
          "pure_rrf": {"1": 0.0, "2": 0.0}} for _ in range(4)]
        + [{"dense_only": {"1": 0.0, "2": 0.55}, "sparse_only": {"1": 0.0, "2": 0.50},
            "pure_rrf": {"1": 0.0, "2": 0.50}} for _ in range(6)])
    m = differentiator_gate(mixed, cand="2", trust="verified", require_decisive=0.6)
    assert m.counts["argmax_dec"] == 1.0 and abs(m.counts["margin_dec"] - 0.4) < 1e-9
    assert m.checks["decisive_target"][2] is False and m.verdict == "FAIL"


def test_low_tie_regressing_to_all_zero_counts_as_damage():
    # 30 clean rescues + 8 low ties broken up clears coverage & ratio, but 2 low
    # ties zeroed against 10 decisive rows -> damage_rate 0.2 > 0.05 FAILs.
    regress = _frame(
        [{"dense_only": {"1": 0.0, "2": 1.0}, "sparse_only": {"1": 0.0, "2": 0.0},
          "pure_rrf": {"1": 0.0, "2": 0.19}} for _ in range(30)]
        + [{"dense_only": {"1": 1.0, "2": 1.0}, "sparse_only": {"1": 0.1, "2": 0.1},
            "pure_rrf": {"1": 0.2, "2": 0.2}} for _ in range(10)]
        + [{"dense_only": {"1": 0.5, "2": 1.0}, "sparse_only": {"1": 0.5, "2": 0.0},
            "pure_rrf": {"1": 0.5, "2": 0.19}} for _ in range(8)]
        + [{"dense_only": {"1": 0.5, "2": 0.0}, "sparse_only": {"1": 0.5, "2": 0.0},
            "pure_rrf": {"1": 0.5, "2": 0.0}} for _ in range(2)])
    rg = differentiator_gate(regress, cand="2", trust="verified")
    assert rg.counts["low_tie_lost"] == 2 and rg.counts["damage"] == 2
    assert rg.checks["damage_rate"][2] is False and rg.verdict == "FAIL"


def test_winner_flipping_thin_counts_as_damage():
    # 30 clean rescues + 18 decisive kept, + 2 certifiable rows that thin AND flip the
    # winning route (dense -> sparse) -> damage_rate 2/20 = 0.1 > 0.05 FAILs. Thinning
    # only counts when the route label actually changes (a same-winner thin does not).
    flipped = _frame(
        [{"dense_only": {"1": 0.0, "2": 1.0}, "sparse_only": {"1": 0.0, "2": 0.0},
          "pure_rrf": {"1": 0.0, "2": 0.19}} for _ in range(30)]
        + [{"dense_only": {"1": 1.0, "2": 1.0}, "sparse_only": {"1": 0.1, "2": 0.1},
            "pure_rrf": {"1": 0.2, "2": 0.2}} for _ in range(18)]
        + [{"dense_only": {"1": 1.0, "2": 0.10}, "sparse_only": {"1": 0.1, "2": 0.15},
            "pure_rrf": {"1": 0.2, "2": 0.05}} for _ in range(2)])
    th = differentiator_gate(flipped, cand="2", trust="verified")
    assert th.counts["decisive_thinned"] == 2 and th.counts["damage"] == 2
    assert th.checks["damage_rate"][2] is False and th.verdict == "FAIL"


def test_same_winner_thinning_is_not_damage():
    # a certifiable decisive row whose margin drops below the bar but keeps the SAME
    # winning route is the same routing decision -> not counted as damage (Layer 2).
    same = _frame(
        [{"dense_only": {"1": 0.0, "2": 1.0}, "sparse_only": {"1": 0.0, "2": 0.0},
          "pure_rrf": {"1": 0.0, "2": 0.19}} for _ in range(10)]
        + [{"dense_only": {"1": 1.0, "2": 0.15}, "sparse_only": {"1": 0.1, "2": 0.10},
            "pure_rrf": {"1": 0.2, "2": 0.05}} for _ in range(4)])
    s = differentiator_gate(same, cand="2", trust="verified")
    assert s.counts["decisive_thinned"] == 0 and s.counts["damage"] == 0


def test_weak_decisive_collapse_is_not_damage_but_strong_is():
    # a WEAKLY decisive baseline (margin < class_margin) collapsing to a tie is not
    # losing a strong label -> not damage; a certifiable one collapsing still is (Layer 1).
    weak = _frame(
        [{"dense_only": {"1": 0.15, "2": 0.5}, "sparse_only": {"1": 0.10, "2": 0.5},
          "pure_rrf": {"1": 0.05, "2": 0.5}} for _ in range(10)])
    assert differentiator_gate(weak, cand="2", trust="verified").counts["collapsed"] == 0
    strong = _frame(
        [{"dense_only": {"1": 1.0, "2": 0.5}, "sparse_only": {"1": 0.1, "2": 0.5},
          "pure_rrf": {"1": 0.2, "2": 0.5}} for _ in range(10)])
    assert differentiator_gate(strong, cand="2", trust="verified").counts["collapsed"] == 10


def test_thin_decisive_row_strengthened_counts_as_benefit():
    # 10 all_zero rescued + 10 thin decisive rows strengthened past the margin;
    # no damage, and the strengthening is credited as benefit -> PASS.
    strong = _frame(
        [{"dense_only": {"1": 0.0, "2": 1.0}, "sparse_only": {"1": 0.0, "2": 0.0},
          "pure_rrf": {"1": 0.0, "2": 0.19}} for _ in range(10)]
        + [{"dense_only": {"1": 0.15, "2": 1.0}, "sparse_only": {"1": 0.10, "2": 0.1},
            "pure_rrf": {"1": 0.05, "2": 0.2}} for _ in range(10)])
    s = differentiator_gate(strong, cand="2", trust="verified")
    assert s.counts["decisive_strengthened"] == 10 and s.counts["damage"] == 0
    assert s.counts["benefit"] == 20 and s.verdict == "PASS"


def test_opinion_leg_tops_out_at_provisional():
    good = _frame(
        [{"dense_only": {"1": 0.0, "2": 1.0}, "sparse_only": {"1": 0.0, "2": 0.0},
          "pure_rrf": {"1": 0.0, "2": 0.19}} for _ in range(20)]
        + [{"dense_only": {"1": 1.0, "2": 1.0}, "sparse_only": {"1": 0.1, "2": 0.1},
            "pure_rrf": {"1": 0.2, "2": 0.2}} for _ in range(6)])
    p = differentiator_gate(good, cand="2", trust="opinion")
    assert p.verdict == "PROVISIONAL"

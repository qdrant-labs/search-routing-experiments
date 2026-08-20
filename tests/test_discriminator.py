"""Statistical core of the Phase 5 discriminator on synthetic fixtures with
known answers: the permutation null, Westfall-Young step-down max-T, the
sibling near-duplicate share, and 5.3's weighting sign rule. No real
augmentation pool needed, so these stay meaningful before any generated rows
exist."""

import numpy as np
import pandas as pd
import pytest

from scripts.run_discriminator import (
    cv_auc,
    permutation_null,
    step_down_max_t,
    vectorizer,
    within_group_dup_share,
)
from scripts.weighting_sensitivity import candidate_weightings, check, sign_robustness


# ------------------------------------------------------- step-down max-T ---
def test_max_t_matches_the_hand_computed_adjusted_p():
    # family of 2, 3 resamples. Largest observed is 2.0; the family max per
    # resample is [1, 3, 4], two of which reach 2.0 -> (1+2)/(1+3) = 0.75.
    # Step 2 alone would give (1+1)/4 = 0.5, but monotonicity holds it at 0.75.
    observed = np.array([2.0, 0.5])
    null = np.array([[1.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    assert step_down_max_t(observed, null) == pytest.approx([0.75, 0.75])


def test_max_t_floors_at_one_over_b_plus_one_when_no_resample_reaches():
    observed = np.array([10.0, 9.0])
    null = np.zeros((2, 99))
    assert step_down_max_t(observed, null) == pytest.approx([0.01, 0.01])


def test_max_t_enforces_monotonicity_down_the_step():
    # hypothesis 1 has a null that swamps it, hypothesis 2's own null is empty:
    # unadjusted it would read 0.2, but no hypothesis may be more significant
    # than one ahead of it in the step-down order.
    observed = np.array([3.0, 2.9])
    null = np.array([[5.0] * 4, [0.0] * 4])
    adjusted = step_down_max_t(observed, null)
    assert adjusted[0] == pytest.approx(1.0)
    assert adjusted[1] == pytest.approx(1.0)


def test_max_t_is_order_invariant():
    observed = np.array([2.0, 0.5])
    null = np.array([[1.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    flipped = step_down_max_t(observed[::-1], null[::-1])
    assert flipped[::-1] == pytest.approx(step_down_max_t(observed, null))


# ---------------------------------------------------- permutation null ---
def _texts(n: int, marker: str, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
    return [
        " ".join([marker, *rng.choice(words, size=6)]).strip() for _ in range(n)
    ]


def _fit(positives, negatives, seed=0):
    texts = positives + negatives
    y = np.r_[np.ones(len(positives)), np.zeros(len(negatives))].astype(int)
    features = vectorizer().fit_transform(texts)
    return cv_auc(features, y, seed), features, y


def test_a_marker_the_generator_always_adds_beats_its_own_null():
    auc, features, y = _fit(_texts(60, "GREETING", 1), _texts(60, "", 2))
    null = permutation_null(features, y, n_perm=40, seed=0)
    assert auc > np.percentile(null, 95)
    assert auc > 0.9


def test_indistinguishable_classes_sit_inside_their_own_null():
    # both classes drawn from one process: the observed AUC must NOT clear the
    # cell's own 95th percentile, which is the miscalibration a fixed 0.75 hides
    auc, features, y = _fit(_texts(60, "", 3), _texts(60, "", 4))
    null = permutation_null(features, y, n_perm=40, seed=0)
    assert auc <= np.percentile(null, 95)


def test_the_null_is_centred_on_chance():
    _, features, y = _fit(_texts(50, "", 5), _texts(50, "", 6))
    null = permutation_null(features, y, n_perm=60, seed=0)
    assert abs(null.mean() - 0.5) < 0.08


# ----------------------------------------------------- sibling similarity ---
def test_dup_share_counts_only_rows_inside_groups_of_two_plus():
    texts = ["the same query text here", "the same query text here",
             "an entirely different sentence", "solitary row with no group"]
    groups = np.array([0, 0, 1, 1])
    space = vectorizer().fit(texts * 2)
    twinned, considered = within_group_dup_share(texts, groups, space)
    assert (twinned, considered) == (2, 4)


def test_a_singleton_group_contributes_nothing():
    texts = ["one two three four", "five six seven eight"]
    space = vectorizer().fit(texts * 3)
    assert within_group_dup_share(texts, np.array([0, 1]), space) == (0, 0)


def test_manufactured_minimal_pairs_read_far_above_a_varied_group():
    stamped = [f"please find the record for item {i}" for i in range(6)]
    varied = ["quantum tunnelling in semiconductors",
              "recipe for sourdough starter",
              "how tall is the eiffel tower",
              "python asyncio event loop internals",
              "treatment options for type 2 diabetes",
              "history of the hanseatic league"]
    space = vectorizer().fit(stamped + varied)
    groups = np.zeros(6, dtype=int)
    manufactured, _ = within_group_dup_share(stamped, groups, space)
    organic, _ = within_group_dup_share(varied, groups, space)
    assert manufactured == 6
    assert organic == 0


# ------------------------------------------------------ weighting rule ---
def _per_lane(**lanes: tuple[int, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"n": n, "mean_diff": v} for n, v in lanes.values()],
        index=pd.Index(lanes, name="dataset"),
    )


def test_one_sign_everywhere_is_weighting_robust():
    per_lane = _per_lane(a=(100, 0.2), b=(10, 0.4))
    result = check(per_lane, "mean_diff", pd.Series({"a": 5000, "b": 50}))
    assert result["verdict"] == "weighting-robust"
    assert result["reason"] == ""
    assert result["headline_value"] == pytest.approx((100 * 0.2 + 10 * 0.4) / 110)


def test_a_big_negative_lane_with_few_rows_flips_the_sign_under_uniform():
    # pooled by n the small lane barely counts (+0.086); one lane one vote it
    # dominates (-0.15). That is the flip the rule must refuse to average away.
    per_lane = _per_lane(big=(1000, 0.1), tiny=(10, -0.4))
    result = check(per_lane, "mean_diff", pd.Series({"big": 9000, "tiny": 90}))
    assert result["verdict"] == "UNDECIDED"
    assert result["reason"] == "sign_flip"
    assert result["signs"]["unweighted"] == 1
    assert result["signs"]["lane_uniform"] == -1


def test_exactly_zero_under_one_candidate_is_undecided_not_robust():
    per_lane = _per_lane(a=(10, 0.5), b=(10, -0.5))
    result = check(per_lane, "mean_diff", pd.Series({"a": 100, "b": 100}))
    assert result["verdict"] == "UNDECIDED"
    assert result["reason"] == "at_zero"
    assert "unweighted" in result["at_zero_candidates"]


def test_the_cap_pulls_a_dominant_lane_back_to_the_policy_share():
    per_lane = _per_lane(hog=(10, 1.0), small=(10, 1.0))
    weights = candidate_weightings(
        per_lane, pd.Series({"hog": 9900, "small": 100}), lane_share_cap=0.2
    )
    capped = weights["pool_share_capped"]
    assert capped["hog"] == pytest.approx(0.2)
    assert capped["small"] == pytest.approx(0.01)
    assert weights["pool_share"]["hog"] == 9900


def test_tolerance_widens_the_at_zero_band():
    values = {"unweighted": 0.001, "lane_uniform": 0.5}
    assert sign_robustness(values, "unweighted")["verdict"] == "weighting-robust"
    strict = sign_robustness(values, "unweighted", tol=0.01)
    assert strict["verdict"] == "UNDECIDED"
    assert strict["reason"] == "at_zero"

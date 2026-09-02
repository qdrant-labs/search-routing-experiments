"""Regression coverage for composition stability under nearby dial changes."""

from statistics import mean

from composition.composer_v4 import ComposerV4
from tests.test_composer_v4 import _broad

# enough lanes that the kappa cap is not the sole constraint at target=300
_LANES = tuple(f"lane{i}" for i in range(6))


def _identities(frame):
    if "row_id" in frame:
        return set(frame["row_id"])
    return set(frame["dataset"].astype(str) + ":" + frame["query_id"].astype(str))


def test_nearby_dial_changes_preserve_most_rows():
    """Nearby (rho, kappa) changes retain a stable core of selected rows."""
    catalog = _broad(600, lanes=_LANES)
    baseline, _ = ComposerV4().compose(catalog, 300, kappa=0.20, rho=0.5)
    baseline_ids = _identities(baseline)
    jaccards = {}
    for rho, kappa in ((0.4, 0.20), (0.3, 0.20), (0.5, 0.15), (0.5, 0.25)):
        perturbed, _ = ComposerV4().compose(catalog, 300, kappa=kappa, rho=rho)
        perturbed_ids = _identities(perturbed)
        jaccards[f"rho={rho}, kappa={kappa}"] = len(
            baseline_ids & perturbed_ids
        ) / len(baseline_ids | perturbed_ids)

    details = ", ".join(f"{dial}: {score:.3f}" for dial, score in jaccards.items())
    assert mean(jaccards.values()) >= 0.80, details
    assert min(jaccards.values()) >= 0.70, details

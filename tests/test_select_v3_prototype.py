"""Core-logic checks for the v3 selector prototype: the pure class-assignment
policy and the closed-form feasibility bound."""

import numpy as np

from scripts.select_v3_prototype import (
    CLASSES,
    SelectorRecipe,
    assign_classes,
    feasible_total,
)

R = SelectorRecipe()


def _classes(scores, depth):
    """assign_classes over a list of 3-score rows (dense, rrf, sparse order)."""
    arr = np.array(scores, dtype=float)
    ordered = np.sort(arr, axis=1)
    winner = np.array(["dense_only", "pure_rrf", "sparse_only"])[arr.argmax(axis=1)]
    return assign_classes(
        ordered[:, -1], ordered[:, -2], ordered[:, 0],
        winner, np.array(depth, dtype=float), R,
    )


def test_decisive_routes_map_to_classes():
    # dense clear win, sparse clear win, rrf clear win (margin >= 0.4)
    kind, cls = _classes(
        [[1.0, 0.3, 0.2], [0.2, 0.3, 1.0], [0.3, 1.0, 0.2]],
        depth=[5, 5, 5],
    )
    assert list(kind) == ["decisive", "decisive", "decisive"]
    assert list(cls) == ["dense", "sparse", "hybrid"]


def test_genuine_tie_is_hybrid_fake_tie_is_waste():
    # both tie at ceiling; deep qrels -> genuine (hybrid), single doc -> fake (waste)
    kind, cls = _classes([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], depth=[3, 1])
    assert list(kind) == ["genuine_tie", "fake_tie"]
    assert list(cls) == ["hybrid", ""]


def test_all_zero_and_undecisive():
    # nothing retrieved; and a routes_differ below the decisive margin
    kind, cls = _classes([[0.0, 0.0, 0.0], [0.5, 0.3, 0.4]], depth=[0, 5])
    assert kind[0] == "all_zero" and cls[0] == ""
    assert kind[1] == "undecisive" and cls[1] == ""


def test_feasible_total_is_the_binding_class():
    # sparse is scarcest relative to its share -> it binds
    supply = {"dense": 3149, "sparse": 1731, "hybrid": 1702}
    total = feasible_total(supply, (0.45, 0.45, 0.10))
    assert round(total) == round(1731 / 0.45)  # sparse binds
    assert all(round(total * s) <= supply[c]
               for c, s in zip(CLASSES, (0.45, 0.45, 0.10)))

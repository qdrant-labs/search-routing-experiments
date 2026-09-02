import pandas as pd
import pytest
import warnings

from hybrid_search_rrf_dataset.fusion import StrategyName
from hybrid_search_rrf_dataset.router import (
    AcceptabilityRouter,
    Representation,
    _derive_engineered,
)


@pytest.fixture
def train_frame():
    """Four answerable rows + one all_zero; every head sees both classes.

    ok at tolerance 0.3 —   dense: T F T T | rrf: F F T F | sparse: T T T F
    """
    return pd.DataFrame(
        {
            "query": ["a b", "ERR_X", "c d", "how to e", "unanswerable"],
            "score_dense_only": [1.0, 0.2, 1.0, 1.0, 0.0],
            "score_pure_rrf": [0.6, 0.1, 1.0, 0.4, 0.0],
            "score_sparse_only": [1.0, 0.9, 1.0, 0.5, 0.0],
            "length.length_chars": [3.0, 5.0, 3.0, 8.0, 12.0],
            "length.length_words": [2.0, 1.0, 2.0, 3.0, 1.0],
        }
    )


def test_fit_predict_smoke(train_frame):
    router = AcceptabilityRouter(Representation.ENGINEERED).fit(train_frame)
    routes = router.predict_routes(train_frame)
    assert len(routes) == len(train_frame)
    assert all(isinstance(r, StrategyName) for r in routes)
    assert router.tolerance == pytest.approx(0.3)  # hit-parity default resolved


def test_everything_fires_serves_cheapest(train_frame):
    router = AcceptabilityRouter(
        Representation.ENGINEERED,
        threshold=0.0,
        priority=AcceptabilityRouter.COST_ORDER,
    )
    routes = router.fit(train_frame).predict_routes(train_frame)
    assert routes == [StrategyName.SPARSE_ONLY] * len(train_frame)


def test_no_priority_serves_most_probable(train_frame):
    """The default rule carries no cost policy — highest P(ok) wins."""
    router = AcceptabilityRouter(Representation.ENGINEERED).fit(train_frame)
    probs = router.probabilities(train_frame)
    for i, route in enumerate(router.predict_routes(train_frame)):
        assert probs[route.value][i] == max(p[i] for p in probs.values())


def test_probabilities_shape(train_frame):
    router = AcceptabilityRouter(Representation.ENGINEERED).fit(train_frame)
    probs = router.probabilities(train_frame.head(2))
    assert set(probs) == {"dense_only", "pure_rrf", "sparse_only"}
    assert all(len(p) == 2 for p in probs.values())


def test_derived_features_do_not_fragment_wide_frames():
    frame = pd.DataFrame(
        {
            "query": ["reset ERR_X"],
            "length.length_words": [2.0],
            "length.length_chars": [10.0],
            "structured_identifiers.error_code": [1.0],
            "derived.identifier_density": [99.0],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        for i in range(120):
            frame.insert(len(frame.columns), f"catalog_{i}", float(i))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _derive_engineered(frame)

    assert not any(isinstance(w.message, pd.errors.PerformanceWarning) for w in caught)
    assert list(result.columns).count("derived.identifier_density") == 1
    assert result.loc[0, "derived.identifier_density"] == 0.5
    assert result.loc[0, "derived.short_id_query"] == 1.0

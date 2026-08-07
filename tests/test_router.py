import pandas as pd
import pytest

from hybrid_search_rrf_dataset.fusion import StrategyName
from hybrid_search_rrf_dataset.router import AcceptabilityRouter, Representation


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
    router = AcceptabilityRouter(Representation.ENGINEERED, threshold=0.0)
    routes = router.fit(train_frame).predict_routes(train_frame)
    assert routes == [StrategyName.SPARSE_ONLY] * len(train_frame)


def test_probabilities_shape(train_frame):
    router = AcceptabilityRouter(Representation.ENGINEERED).fit(train_frame)
    probs = router.probabilities(train_frame.head(2))
    assert set(probs) == {"dense_only", "pure_rrf", "sparse_only"}
    assert all(len(p) == 2 for p in probs.values())

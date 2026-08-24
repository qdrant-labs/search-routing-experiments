"""The fill run's two failure contracts: what it owes the network, and what a
query the endpoint refuses costs the rest of the batch."""

import pandas as pd
import pytest

from hybrid_search_rrf_dataset.router import AutoFusionRouter
from scripts.autofusion_fill import GuardedScoreClient, _fill, _permanent, _remaining


class Response:
    def __init__(self, status_code: int, text: str = "text too long") -> None:
        self.status_code = status_code
        self.text = text


class HTTPError(Exception):
    """Stands in for `requests.HTTPError`, which `_permanent` reads by duck type."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"{status_code} Client Error")
        self.response = Response(status_code)


class Client:
    """Scores by length, and refuses anything longer than `limit` with a 400."""

    SCORE_MAX = 9

    def __init__(self, limit: int = 100) -> None:
        self.limit = limit
        self.calls = 0

    def score(self, query: str) -> int:
        self.calls += 1
        if len(query) > self.limit:
            raise HTTPError(400)
        return len(query) % 10


def frame(pairs: list[tuple[str, object]], query: str = "q") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": [d for d, _ in pairs],
            "query_id": [q for _, q in pairs],
            "query": [query] * len(pairs),
        }
    )


def batch(long_at: int) -> pd.DataFrame:
    """Four rows, one of them too long for the fake endpoint."""
    rows = frame([("d", str(i)) for i in range(4)])
    rows.loc[long_at, "query"] = "x" * 500
    return rows


def test_cached_rows_are_dropped_across_query_id_dtypes(tmp_path):
    """The cache round-trips query_id through parquet; labels hold strings."""
    cache = tmp_path / "cache.parquet"
    frame([("gooaq", 1)]).assign(score=3).to_parquet(cache, index=False)

    todo = _remaining(frame([("gooaq", "1"), ("gooaq", "2"), ("limit", "1")]), cache)

    assert list(zip(todo["dataset"], todo["query_id"])) == [("gooaq", "2"), ("limit", "1")]


def test_no_cache_owes_every_row(tmp_path):
    labels = frame([("gooaq", "1"), ("limit", "2")])

    assert len(_remaining(labels, tmp_path / "absent.parquet")) == len(labels)


def test_nothing_left_to_score_keeps_its_columns(tmp_path):
    """An empty mask list would select COLUMNS, handing the caller a frame with
    no `dataset` to score by."""
    cache = tmp_path / "cache.parquet"
    frame([("gooaq", 1)]).assign(score=3).to_parquet(cache, index=False)

    todo = _remaining(frame([]), cache)

    assert todo.empty
    assert "dataset" in todo.columns and "query_id" in todo.columns


def test_permanent_only_for_4xx():
    assert _permanent(HTTPError(400))
    assert _permanent(HTTPError(422))
    assert not _permanent(HTTPError(500))
    assert not _permanent(TimeoutError("read timed out"))


def test_scores_already_paid_for_survive_a_refusal(tmp_path):
    """`route_batch` saves in a `finally`, or the run re-buys the whole chunk."""
    cache = tmp_path / "cache.parquet"
    router = AutoFusionRouter(client=Client(), cache_path=cache)

    with pytest.raises(HTTPError):
        router.route_batch(batch(long_at=2))

    assert list(pd.read_parquet(cache)["query_id"]) == ["0", "1"]


def test_fill_steps_over_the_refused_row(tmp_path):
    cache = tmp_path / "cache.parquet"
    client = GuardedScoreClient(Client(), attempts=1)
    router = AutoFusionRouter(client=client, cache_path=cache)

    refused = _fill(router, batch(long_at=2), desc="t", cache_path=cache)

    assert [row["query_id"] for row in refused] == ["2"]
    assert list(pd.read_parquet(cache)["query_id"]) == ["0", "1", "3"]


def test_a_transient_failure_still_stops_the_run(tmp_path):
    """Only 4xx is stepped over — a 500 means the endpoint, not the query."""

    class Flaky:
        def score(self, query: str) -> int:
            raise HTTPError(503)

    cache = tmp_path / "cache.parquet"
    router = AutoFusionRouter(client=Flaky(), cache_path=cache)

    with pytest.raises(HTTPError):
        _fill(router, frame([("d", "0")]), desc="t", cache_path=cache)


def test_max_chars_truncates_below_the_endpoint_limit():
    client = Client(limit=100)
    guarded = GuardedScoreClient(client, attempts=1, max_chars=50)

    assert guarded.score("x" * 500) == 0  # 50 chars through, 50 % 10

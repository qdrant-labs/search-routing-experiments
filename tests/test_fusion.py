from qdrant_client.models import Document

from hybrid_search_rrf_dataset.fusion import DenseOnlyStrategy
from hybrid_search_rrf_dataset.indexer import EmbeddingConfig

CLOUD_DENSE = EmbeddingConfig(
    name="dense_legb", model_id="openrouter/qwen/qwen3-embedding-8b",
    kind="dense", cloud=True, provider_options={"openrouter-api-key": "k"},
    query_prompt="Instruct: ...\nQuery: ",
)
SPARSE = EmbeddingConfig(name="sparse_base", model_id="Qdrant/bm25", kind="sparse")


def test_a_cloud_dense_slot_embeds_the_query_as_a_document():
    """The regression this exists for: fusion.py's query side never checked
    cloud at all — an OpenRouter model_id has no local ONNX build, so calling
    TextEmbedding(model_id) on it would crash rather than route server-side."""
    strategy = DenseOnlyStrategy(None, "collection", CLOUD_DENSE, SPARSE)
    result = strategy._dense("how do lakes freeze")
    assert isinstance(result, Document)
    assert result.text == "Instruct: ...\nQuery: how do lakes freeze"
    assert result.model == "openrouter/qwen/qwen3-embedding-8b"
    assert result.options == {"openrouter-api-key": "k"}


class _FlakyClient:
    """query_points fails with the given exceptions, then succeeds."""

    def __init__(self, failures):
        self._failures = list(failures)
        self.calls = 0

    def query_points(self, **_):
        self.calls += 1
        if self._failures:
            raise self._failures.pop(0)

        class _R:
            points = []
        return _R()


def _server_error(status: int) -> Exception:
    from qdrant_client.http.exceptions import UnexpectedResponse

    return UnexpectedResponse(status_code=status, reason_phrase="err",
                              content=b"", headers=None)


def test_search_retries_transient_500(monkeypatch):
    """The regression this exists for: a saturated cluster answers 'Operation
    Search timed out' as HTTP 500, which killed a 6-hour labelling run at the
    first busy moment instead of backing off and continuing."""
    import hybrid_search_rrf_dataset.fusion as fusion

    monkeypatch.setattr(fusion.time, "sleep", lambda _: None)
    client = _FlakyClient([_server_error(500), _server_error(503)])
    strategy = DenseOnlyStrategy(client, "collection", CLOUD_DENSE, SPARSE)
    assert strategy.rank("q") == {}
    assert client.calls == 3  # two transient failures, then success


def test_search_fails_fast_on_non_transient(monkeypatch):
    import pytest

    import hybrid_search_rrf_dataset.fusion as fusion
    from qdrant_client.http.exceptions import UnexpectedResponse

    monkeypatch.setattr(fusion.time, "sleep", lambda _: None)
    client = _FlakyClient([_server_error(403)])
    strategy = DenseOnlyStrategy(client, "collection", CLOUD_DENSE, SPARSE)
    with pytest.raises(UnexpectedResponse):
        strategy.rank("q")
    assert client.calls == 1  # a 403 never recovers; no retries burned

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

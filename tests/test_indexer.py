import uuid
from types import SimpleNamespace

import numpy as np
from qdrant_client.models import Document, Modifier

from hybrid_search_rrf_dataset.indexer import (
    CorpusDocument,
    CorpusIndexer,
    EmbeddingCache,
    EmbeddingConfig,
)

DENSE = EmbeddingConfig(name="dense", model_id="unused", kind="dense", size=4)
SPARSE = EmbeddingConfig(name="sparse", model_id="unused", kind="sparse")


class _Client:
    """Answers `retrieve` from a fixed id set; no other call is exercised."""

    def __init__(self, present: set[str]) -> None:
        self.present = present
        self.asked: list[list[str]] = []

    def retrieve(self, collection_name, ids, with_payload, with_vectors):
        del collection_name, with_payload, with_vectors
        self.asked.append(list(ids))
        return [SimpleNamespace(id=i) for i in ids if str(i) in self.present]


def _docs(*doc_ids: str) -> list[CorpusDocument]:
    return [CorpusDocument(doc_id=d, text=f"text {d}") for d in doc_ids]


def _point_id(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))


def _indexer(present: set[str]) -> tuple[CorpusIndexer, _Client]:
    client = _Client(present)
    return CorpusIndexer(client, "lane", embeddings=[DENSE]), client


def test_missing_returns_only_unindexed_documents():
    indexer, _ = _indexer({_point_id("a"), _point_id("c")})
    assert [d.doc_id for d in indexer.missing(_docs("a", "b", "c"))] == ["b"]


def test_a_second_pass_over_an_indexed_corpus_uploads_nothing():
    """The regression this exists for: a corpus short by one document used to
    re-embed every document in the lane."""
    indexer, _ = _indexer({_point_id("a"), _point_id("b")})
    assert indexer.missing(_docs("a", "b")) == []


def test_missing_chunks_its_lookups():
    indexer, client = _indexer(set())
    assert len(indexer.missing(_docs("0", "1", "2", "3", "4"), batch_size=2)) == 5
    assert [len(chunk) for chunk in client.asked] == [2, 2, 1]


class _SchemaClient:
    """Captures the one `create_collection` call."""

    cloud_inference = False

    def __init__(self) -> None:
        self.created: dict = {}

    def collection_exists(self, collection_name):
        del collection_name
        return False

    def create_collection(self, **kwargs):
        self.created = kwargs


def test_a_sparse_slot_carries_idf_without_being_told():
    """The regression this exists for: `Modifier.IDF` was passed per call site,
    so each new sweep could omit it and quietly build a TF-only collection that
    every later `ensure_collection` then skips."""
    client = _SchemaClient()
    CorpusIndexer(client, "lane", embeddings=[DENSE, SPARSE]).ensure_collection()
    assert client.created["sparse_vectors_config"]["sparse"].modifier == Modifier.IDF


def test_a_learned_sparse_model_can_still_opt_out():
    client = _SchemaClient()
    splade = EmbeddingConfig(
        name="sparse", model_id="unused", kind="sparse", modifier=None
    )
    CorpusIndexer(client, "lane", embeddings=[splade]).ensure_collection()
    assert client.created["sparse_vectors_config"]["sparse"].modifier is None


class _TextEmbedder(CorpusIndexer):
    """Embeds from the text, so a vector served for the wrong document is
    visible as a value rather than only as a missing call."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedded: list[str] = []

    def _embed(self, cfg, texts, batch_size):
        del cfg, batch_size
        self.embedded.extend(texts)
        return [
            np.asarray([len(t), sum(t.encode())], dtype=np.float32) for t in texts
        ]


def _vector_for(cache: EmbeddingCache, doc: CorpusDocument) -> tuple:
    indexer = _TextEmbedder(_SchemaClient(), "lane", embeddings=[DENSE], cache=cache)
    vectors = indexer._vectors_for(DENSE, [indexer.item_id(doc)], [doc.text], 8)
    return tuple(vectors[0]), indexer.embedded


def test_the_cache_never_crosses_lanes(tmp_path):
    """Two lanes ship the same bare doc_id under different text — measured, 413
    of them between msmarco-passage-dev and gooaq. `item_id` hashes the doc_id
    alone, so an unnamespaced cache hands the first lane's vector to the second.
    """
    shared_id = "1024984"
    one = CorpusDocument(doc_id=shared_id, text="books by tracey hecht")
    two = CorpusDocument(doc_id=shared_id, text="your item is moving")

    vec_one, embedded_one = _vector_for(
        EmbeddingCache(tmp_path, namespace="msmarco-passage-dev"), one
    )
    vec_two, embedded_two = _vector_for(
        EmbeddingCache(tmp_path, namespace="gooaq"), two
    )

    assert embedded_one == [one.text] and embedded_two == [two.text]
    assert vec_one != vec_two


def test_one_lane_reuses_its_own_cached_vector(tmp_path):
    doc = CorpusDocument(doc_id="1024984", text="books by tracey hecht")
    first, embedded_first = _vector_for(EmbeddingCache(tmp_path, namespace="a"), doc)
    second, embedded_second = _vector_for(EmbeddingCache(tmp_path, namespace="a"), doc)

    assert embedded_first == [doc.text]
    assert embedded_second == []  # served from disk, never re-embedded
    assert first == second


def test_a_cloud_slot_embeds_as_a_document_bypassing_the_cache(tmp_path):
    """cfg.cloud is per-slot, not per-client: a dense slot with no local ONNX
    build (an OpenRouter model) must embed via Document while a sibling local
    slot keeps its cache — mixing them on one indexer is the whole point."""
    cloud_cfg = EmbeddingConfig(
        name="dense_legb", model_id="openrouter/qwen/qwen3-embedding-8b",
        kind="dense", cloud=True, provider_options={"openrouter-api-key": "k"},
    )
    indexer = CorpusIndexer(
        _SchemaClient(), "lane", embeddings=[cloud_cfg],
        cache=EmbeddingCache(tmp_path),
    )
    vecs = indexer._vectors_for(cloud_cfg, ["id-1"], ["hello"], 8)
    assert len(vecs) == 1 and isinstance(vecs[0], Document)
    assert vecs[0].model == "openrouter/qwen/qwen3-embedding-8b"
    assert vecs[0].options == {"openrouter-api-key": "k"}
    assert EmbeddingCache(tmp_path).load(cloud_cfg.model_id, "dense") == {}


def test_a_cloud_doc_prompt_is_applied_before_wrapping():
    cfg = EmbeddingConfig(
        name="d", model_id="m", kind="dense", cloud=True, doc_prompt="passage: ",
    )
    indexer = CorpusIndexer(_SchemaClient(), "lane", embeddings=[cfg])
    [doc] = indexer._embed(cfg, ["hello"], 8)
    assert doc.text == "passage: hello"


def test_an_unnamespaced_cache_keeps_its_filename(tmp_path):
    """Existing pickles are named without a namespace; the default must still
    find them rather than silently starting a cold cache."""
    plain = EmbeddingCache(tmp_path)
    assert plain._path("BAAI/bge-small-en-v1.5", "dense").name == (
        "BAAI__bge-small-en-v1.5.dense.pkl"
    )

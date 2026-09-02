import uuid
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Document, Modifier

import hybrid_search_rrf_dataset.indexer as indexer_module
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


class _FlakyClient:
    """Raises UnexpectedResponse(status) on the first `fails` upsert calls."""

    def __init__(self, status: int, fails: int) -> None:
        self.status, self.fails, self.calls = status, fails, 0

    def upsert(self, **kwargs):
        del kwargs
        self.calls += 1
        if self.calls <= self.fails:
            from qdrant_client.http.exceptions import UnexpectedResponse
            raise UnexpectedResponse(self.status, "err", b"boom", {})


def test_upsert_fails_fast_on_non_retryable_status(monkeypatch):
    """A 403 (exhausted key) must raise on the first attempt — retrying can't fix
    it and would burn 4x the wasted calls."""
    from qdrant_client.http.exceptions import UnexpectedResponse

    monkeypatch.setattr("hybrid_search_rrf_dataset.indexer.time.sleep", lambda *_: None)
    client = _FlakyClient(status=403, fails=99)
    idx = CorpusIndexer(client, "lane", embeddings=[DENSE])
    try:
        idx._upsert_with_retry([])
        raise AssertionError("expected UnexpectedResponse")
    except UnexpectedResponse:
        pass
    assert client.calls == 1, "403 must not be retried"


def test_upsert_retries_transient_status(monkeypatch):
    """A 429 recovers on retry — one failure then success, not a raise."""
    monkeypatch.setattr("hybrid_search_rrf_dataset.indexer.time.sleep", lambda *_: None)
    client = _FlakyClient(status=429, fails=1)
    CorpusIndexer(client, "lane", embeddings=[DENSE])._upsert_with_retry([])
    assert client.calls == 2, "429 should retry once then succeed"


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


def test_a_crash_mid_embed_keeps_earlier_chunks_on_disk(tmp_path, monkeypatch):
    """The regression this exists for: a hard kill (memguard, OOM) used to lose
    the whole missing set because the cache only flushed after every vector in
    it was already embedded in memory — a `finally` never gets to run against a
    SIGKILL, so only a save that already hit disk before the crash survives."""
    monkeypatch.setattr(indexer_module, "_EMBED_CHECKPOINT_SIZE", 2)

    class _CrashesOnThirdChunk(CorpusIndexer):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.calls = 0

        def _embed(self, cfg, texts, batch_size):
            del cfg, batch_size
            self.calls += 1
            if self.calls == 3:
                raise MemoryError("simulated OOM")
            return [np.asarray([len(t)], dtype=np.float32) for t in texts]

    ids = [str(i) for i in range(5)]
    texts = [f"doc {i}" for i in range(5)]
    indexer = _CrashesOnThirdChunk(
        _SchemaClient(), "lane", embeddings=[DENSE],
        cache=EmbeddingCache(tmp_path, namespace="lane"),
    )

    with pytest.raises(MemoryError):
        indexer._vectors_for(DENSE, ids, texts, 8)

    # a fresh cache instance, as a rerun would open — only what already hit
    # disk before the crash counts
    survived = EmbeddingCache(tmp_path, namespace="lane").load(DENSE.model_id, "dense")
    assert set(survived) == {"0", "1", "2", "3"}


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


class _RecordingSparse:
    """Stands in for a downloaded fastembed model, keeping the texts it was handed."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed(self, texts, **_):
        self.seen.extend(texts)
        return [
            SimpleNamespace(indices=np.array([1]), values=np.array([1.0]))
            for _ in texts
        ]


def test_max_input_chars_truncates_local_embeds_too():
    """The cap used to live inside the `cloud` branch, so local models never saw it —
    and miniCOIL pads a batch to its longest member, so one 66k-char doc took 11.45GB
    on its own and OOM-killed a batch of 64."""
    cfg = EmbeddingConfig(name="s", model_id="m", kind="sparse", max_input_chars=10)
    indexer = CorpusIndexer(_SchemaClient(), "lane", embeddings=[cfg])
    recorder = _RecordingSparse()
    indexer._sparse = lambda _cfg: recorder

    indexer._embed(cfg, ["x" * 500], 8)

    assert recorder.seen == ["x" * 10]


def test_model_options_reach_the_model_and_split_its_cache_entry():
    """miniCOIL's `avg_len` is corpus-specific, so two lanes share a model_id while
    needing different vectors — one cache slot for both would serve the wrong model."""
    built: list[dict] = []

    class _Indexer(CorpusIndexer):
        pass

    cfg_a = EmbeddingConfig(
        name="s", model_id="m", kind="sparse", model_options={"avg_len": 272.0}
    )
    cfg_b = cfg_a.model_copy(update={"model_options": {"avg_len": 150.0}})
    indexer = _Indexer(_SchemaClient(), "lane", embeddings=[cfg_a])

    import hybrid_search_rrf_dataset.indexer as module

    original = module.SparseTextEmbedding
    module.SparseTextEmbedding = lambda model_id, **kw: built.append(kw) or object()
    try:
        indexer._sparse(cfg_a)
        indexer._sparse(cfg_a)  # cached — must not rebuild
        indexer._sparse(cfg_b)
    finally:
        module.SparseTextEmbedding = original

    assert [kw["avg_len"] for kw in built] == [272.0, 150.0]


def test_an_unnamespaced_cache_keeps_its_filename(tmp_path):
    """Existing pickles are named without a namespace; the default must still
    find them rather than silently starting a cold cache."""
    plain = EmbeddingCache(tmp_path)
    assert plain._path("BAAI/bge-small-en-v1.5", "dense").name == (
        "BAAI__bge-small-en-v1.5.dense.pkl"
    )


class _FlakyUpsertClient:
    """Fails `upsert` with `UnexpectedResponse` a fixed number of times, mimicking
    a cloud-inference slot's transient 30s server-side timeout, then succeeds."""

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def upsert(self, **_):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise UnexpectedResponse(408, "Request Timeout", b"{}", httpx.Headers())


def test_upsert_retries_past_a_transient_failure(monkeypatch):
    """The regression this exists for: a single 30s Qdrant Cloud Inference timeout
    used to kill the whole batch upload with no retry."""
    monkeypatch.setattr(indexer_module.time, "sleep", lambda _: None)
    client = _FlakyUpsertClient(fail_times=2)
    indexer = CorpusIndexer(client, "lane", embeddings=[DENSE])

    indexer._upsert_with_retry([])

    assert client.calls == 3


def test_upsert_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(indexer_module.time, "sleep", lambda _: None)
    client = _FlakyUpsertClient(fail_times=99)
    indexer = CorpusIndexer(client, "lane", embeddings=[DENSE])

    with pytest.raises(UnexpectedResponse):
        indexer._upsert_with_retry([])

    assert client.calls == indexer_module._UPSERT_MAX_ATTEMPTS


CLOUD_DENSE = EmbeddingConfig(
    name="dense", model_id="openrouter/fake/model", kind="dense", size=4, cloud=True
)


class _SourceClient(_Client):
    """`retrieve` also serves stored named vectors for the ids it knows."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        super().__init__(set(vectors))
        self.vectors = vectors

    def retrieve(self, collection_name, ids, with_payload, with_vectors):
        del collection_name, with_payload
        self.asked.append(list(ids))
        return [
            SimpleNamespace(id=i, vector={with_vectors[0]: self.vectors[str(i)]})
            for i in ids
            if str(i) in self.vectors
        ]


def test_reuse_cloud_from_copies_vectors_instead_of_embedding():
    stored = {_point_id("a"): [1.0, 0, 0, 0], _point_id("b"): [0, 1.0, 0, 0]}
    client = _SourceClient(stored)
    indexer = CorpusIndexer(
        client, "lane_new", embeddings=[CLOUD_DENSE], reuse_cloud_from="lane_base"
    )
    ids = [_point_id("a"), _point_id("b")]
    vectors = indexer._vectors_for(CLOUD_DENSE, ids, ["ta", "tb"], batch_size=8)
    assert vectors == [stored[ids[0]], stored[ids[1]]]  # copied, not Documents


def test_reuse_falls_back_to_paid_embed_only_for_absent_ids():
    stored = {_point_id("a"): [1.0, 0, 0, 0]}
    client = _SourceClient(stored)
    indexer = CorpusIndexer(
        client, "lane_new", embeddings=[CLOUD_DENSE], reuse_cloud_from="lane_base"
    )
    ids = [_point_id("a"), _point_id("b")]
    vectors = indexer._vectors_for(CLOUD_DENSE, ids, ["ta", "tb"], batch_size=8)
    assert vectors[0] == stored[ids[0]]
    assert isinstance(vectors[1], Document)  # only the absent id pays


def test_without_reuse_a_cloud_slot_still_embeds_as_documents():
    indexer = CorpusIndexer(_SourceClient({}), "lane_new", embeddings=[CLOUD_DENSE])
    vectors = indexer._vectors_for(
        CLOUD_DENSE, [_point_id("a")], ["ta"], batch_size=8
    )
    assert all(isinstance(v, Document) for v in vectors)

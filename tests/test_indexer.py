import uuid
from types import SimpleNamespace

from hybrid_search_rrf_dataset.indexer import (
    CorpusDocument,
    CorpusIndexer,
    EmbeddingConfig,
)

DENSE = EmbeddingConfig(name="dense", model_id="unused", kind="dense", size=4)


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

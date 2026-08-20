"""The synthetic labelling sweep's pure parts: collection isolation, row
discovery, corpus composition, and the deliberate gate override — everything
that can break without a Qdrant in the loop."""

import pandas as pd
import pytest

from augmentation.config import AugmentationPaths
from augmentation.constructed import ConstructedDocs
from hybrid_search_rrf_dataset.labels import _augmented_rows
from scripts import label_routes_synthetic as sweep
from scripts.label_routes import _collection


def test_synthetic_collection_never_reuses_the_paid_one():
    for lane in ("dbpedia-entity", "beir-nfcorpus", "msmarco-passage-dev"):
        assert sweep.synthetic_collection(lane) != _collection(lane)


def test_synthetic_rows_filters_on_the_operator():
    pool = pd.DataFrame({
        "query_id": ["a", "b", "c"],
        "operator": ["synthesize", "inject", "decorate"],
    })
    assert list(sweep.synthetic_rows(pool)["query_id"]) == ["a"]
    assert sweep.synthetic_rows(pool.iloc[:0]).empty


def test_eval_corpus_appends_constructed_docs_for_the_lane_only(tmp_path, monkeypatch):
    paths = AugmentationPaths(data_dir=tmp_path)
    docs = ConstructedDocs(paths)
    docs.add(query_id="syn-1", source_dataset="lane-a", text="answer a")
    docs.add(query_id="syn-2", source_dataset="lane-b", text="answer b")

    source = pd.DataFrame({
        "doc_id": ["d1", "d2"], "title": ["t1", "t2"], "text": ["x", "y"],
    })

    class _Stub:
        def __init__(self, *a, **k): ...
        def corpus(self):
            return source

    monkeypatch.setattr(sweep, "SnapshotDataset", _Stub)
    merged = sweep.eval_corpus("lane-a", docs)
    assert list(merged["doc_id"]) == ["d1", "d2", "constructed-syn-1-1"]
    assert list(merged.columns) == list(source.columns)
    assert merged["text"].iloc[-1] == "answer a"


def test_eval_corpus_never_duplicates_a_doc_id(tmp_path, monkeypatch):
    paths = AugmentationPaths(data_dir=tmp_path)
    docs = ConstructedDocs(paths)
    docs.add(query_id="syn-1", source_dataset="lane-a", text="constructed twin")
    source = pd.DataFrame({
        "doc_id": ["constructed-syn-1-1"], "title": ["real"], "text": ["kept"],
    })

    class _Stub:
        def __init__(self, *a, **k): ...
        def corpus(self):
            return source

    monkeypatch.setattr(sweep, "SnapshotDataset", _Stub)
    merged = sweep.eval_corpus("lane-a", docs)
    assert len(merged) == 1
    assert merged["text"].iloc[0] == "kept"  # the source wins, first kept


@pytest.fixture
def gated_pool():
    return pd.DataFrame({
        "query_id": ["g", "u"],
        "home_lane": ["lane", "lane"],
        "credit_gate": ["coherence_gate", "none"],
        "floor": ["some_cell", "some_cell"],
    })


def test_gated_rows_stay_excluded_by_default(gated_pool):
    wanted = pd.DataFrame({"query_id": ["g", "u"]})
    got = _augmented_rows(gated_pool, wanted, "lane", "cell_based")
    assert list(got["query_id"]) == ["u"]


def test_include_gated_is_the_deliberate_override(gated_pool):
    wanted = pd.DataFrame({"query_id": ["g", "u"]})
    got = _augmented_rows(
        gated_pool, wanted, "lane", "cell_based", include_gated=True
    )
    assert list(got["query_id"]) == ["g", "u"]

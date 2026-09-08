"""The generated-row labelling sweeps' pure parts: collection isolation, row
discovery, corpus composition, the deliberate gate override, the admitted
selection and the corpus-stat artifact's append bookkeeping — everything that
can break without a Qdrant in the loop."""

import pandas as pd
import pytest

from augmentation.config import AugmentationPaths
from augmentation.constructed import ConstructedDocs
from hybrid_search_rrf_dataset.labels import _augmented_rows
from scripts import label_routes_synthetic as sweep
from scripts import label_routes_v3 as v3
from scripts.collection_features import QueryCorpusStats
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


@pytest.fixture
def admitted_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(v3, "V3_DIR", tmp_path)
    monkeypatch.setattr(v3, "AUGMENTED_DIR", tmp_path / "augmented")
    monkeypatch.setattr(v3, "V4_DIR", tmp_path / "v4")
    monkeypatch.setattr(v3, "V2_LABELS", tmp_path / "v2_labels.parquet")
    pd.DataFrame({"dataset": ["lane-a"], "query_id": ["natural-1"]}).to_parquet(
        tmp_path / "v2_labels.parquet", index=False
    )
    return tmp_path


def test_picks_selection_drops_labelled_and_synthesized_rows(admitted_dirs):
    pd.DataFrame({
        "query_id": ["aug-1", "aug-2", "syn-1", "aug-1"],
        "query": ["q1", "q2", "q3", "q1"],
        "home_lane": ["lane-a", "lane-b", "lane-a", "lane-a"],
        "operator": ["inject", "decorate", "synthesize", "inject"],
    }).to_parquet(admitted_dirs / "admitted.parquet", index=False)
    pd.DataFrame({"dataset": ["lane-b"], "query_id": ["aug-2"]}).to_parquet(
        admitted_dirs / "labels.parquet", index=False
    )

    got = v3.picks_selection(admitted_dirs / "admitted.parquet")
    assert list(got.columns) == ["dataset", "query_id", "query", "home_lane"]
    assert list(got["query_id"]) == ["aug-1"]  # aug-2 labelled, syn-1 isolated
    assert list(got["dataset"]) == ["lane-a"] == list(got["home_lane"])


def test_picks_selection_survives_a_missing_or_empty_file(admitted_dirs):
    columns = ["dataset", "query_id", "query", "home_lane"]
    assert list(v3.picks_selection(admitted_dirs / "admitted.parquet").columns) == columns
    assert v3.picks_selection(admitted_dirs / "admitted.parquet").empty
    pd.DataFrame(columns=["query_id", "query", "home_lane", "operator"]).to_parquet(
        admitted_dirs / "admitted.parquet", index=False
    )
    assert v3.picks_selection(admitted_dirs / "admitted.parquet").empty


class _NoCorpora:
    """Stands in for CollectionIndexStore where no corpus index exists, so
    build() can only exercise its append bookkeeping."""

    def indexable(self) -> dict[str, str]:
        return {}


def test_query_corpus_stats_keeps_measured_rows_and_skips_them(tmp_path):
    out = tmp_path / "query_corpus_stats.parquet"
    pd.DataFrame({
        "dataset": ["lane-a"], "query_id": ["1"], "avg_idf": [0.5],
    }).to_parquet(out, index=False)
    labels = tmp_path / "labels.parquet"
    pd.DataFrame({
        "dataset": ["lane-a", "lane-a"], "query_id": ["1", "2"],
        "query": ["measured", "fresh"],
    }).to_parquet(labels, index=False)

    stats = QueryCorpusStats(_NoCorpora(), (labels, tmp_path / "absent.parquet"), out)
    assert list(stats._labels()["query_id"]) == ["1", "2"]  # missing file skipped
    # query_id 2 has no corpus index, so it is skipped, never dropped from disk
    frame = stats.build()
    assert frame.to_dict("records") == [
        {"dataset": "lane-a", "query_id": "1", "avg_idf": 0.5}
    ]
    assert stats.build(force=True)["query_id"].tolist() == ["1"]

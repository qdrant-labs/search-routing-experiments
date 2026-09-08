"""Offline invariants for the lane table, mirroring test_registry.py for the
corpus side. `DatasetName` is the single source of truth for names; a lane's
key is authoritative and its artifacts live under it.
"""

import pandas as pd

from dataset_registry import DatasetName
from hybrid_search_rrf_dataset.labels import oracle_dir
from hybrid_search_rrf_dataset.lanes import LANELESS, LANES
from hybrid_search_rrf_dataset.paths import LanePaths
from hybrid_search_rrf_dataset.retrieval import QREL_COLUMNS, QUERY_COLUMNS


def test_lane_keys_are_registered_datasets():
    assert set(LANES) <= {member.value for member in DatasetName}


def test_lane_name_matches_its_key():
    assert {key for key, lane in LANES.items() if lane.source.name != key} == set()


def test_every_registered_dataset_has_a_lane_or_is_declared_laneless():
    covered = set(LANES) | {member.value for member in LANELESS}
    assert {member.value for member in DatasetName} - covered == set()


def test_beir_subset_slug_is_separate_from_the_key():
    """NFCorpus is `beir-nfcorpus` to us and `nfcorpus` upstream."""
    nfcorpus = LANES["beir-nfcorpus"].source
    assert (nfcorpus.name, nfcorpus.subset) == ("beir-nfcorpus", "nfcorpus")


def test_restrict_narrows_queries_and_qrels_together():
    """A lane cut to a subset of its queries must size its corpus from only
    those queries' qrels."""
    lane = LANES["gooaq"].source
    lane._queries_df = pd.DataFrame(
        [{"query_id": "a", "text": "one"}, {"query_id": "b", "text": "two"}],
        columns=QUERY_COLUMNS,
    )
    lane._qrels_df = pd.DataFrame(
        [
            {"query_id": "a", "doc_id": "d1", "relevance": 1},
            {"query_id": "b", "doc_id": "d2", "relevance": 1},
        ],
        columns=QREL_COLUMNS,
    )
    lane.restrict(["a"])
    assert lane.queries()["query_id"].tolist() == ["a"]
    assert lane.qrels()["doc_id"].tolist() == ["d1"]


def test_every_lane_artifact_lives_under_the_lane_dir(tmp_path):
    """`LanePaths` is the repo-wide owner of the layout — every artifact of a
    lane sits in that lane's own directory, on whatever root it is given."""
    paths = LanePaths(data_dir=tmp_path)
    artifacts = [
        paths.lane_queries("a"), paths.lane_qrels("a"), paths.lane_corpus("a"),
        paths.lane_corpus_index("a"), paths.lane_surfaces("a"), paths.lane_excluded("a"),
    ]
    assert all(p.parent == paths.lane_dir("a") == tmp_path / "a" for p in artifacts)
    assert len({p.name for p in artifacts}) == len(artifacts)   # no two share a file


def test_lanes_with_reuses_the_artifact_filename(tmp_path):
    """The listing and the path must agree by construction, not by a literal
    repeated in two places."""
    paths = LanePaths(data_dir=tmp_path)
    for lane in ("b", "a"):
        paths.lane_dir(lane).mkdir()
        paths.lane_qrels(lane).write_bytes(b"")
    paths.lane_dir("c").mkdir()                                  # no qrels
    assert paths.lanes_with_qrels() == ["a", "b"]                # sorted, c absent
    assert paths.lanes_with_corpus() == []


def test_oracle_results_have_one_suffix_spelling(tmp_path):
    """`labels.py` writes the oracle dirs and `LanePaths` reads them; a second
    spelling of the suffix is how those two silently drift apart."""
    paths = LanePaths(data_dir=tmp_path)
    rung = tmp_path / "rungs" / "labeling"
    assert oracle_dir(paths.oracle_root(), "a") == paths.oracle_lane_dir("a")
    assert oracle_dir(rung, "a") == paths.oracle_lane_dir("a", under=rung)
    assert paths.oracle_rows("a").parent == paths.oracle_lane_dir("a")

    paths.oracle_lane_dir("a").mkdir(parents=True)
    paths.oracle_rows("a").write_bytes(b"")
    paths.oracle_lane_dir("b").mkdir()                            # dir, no rows
    assert paths.oracle_lanes() == ["a"]

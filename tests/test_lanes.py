"""Offline invariants for the lane table, mirroring test_registry.py for the
corpus side. `DatasetName` is the single source of truth for names; a lane's
key is authoritative and its artifacts live under it.
"""

import pandas as pd

from dataset_registry import DatasetName
from hybrid_search_rrf_dataset.lanes import LANELESS, LANES
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

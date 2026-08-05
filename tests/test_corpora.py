"""Offline checks for the lane-corpus script: the directory a lane owns, and
the plan's arithmetic over synthetic qrels in tmp_path — no src/data, no fetch.
"""

import pandas as pd
import pytest

from hybrid_search_rrf_dataset.lanes import LANES, Lane
from hybrid_search_rrf_dataset.retrieval import MaterializedDataset
from scripts.materialize_corpora import LaneCorpora


class FakeLane(MaterializedDataset):
    """A 50-doc stand-in for a streamed source, borrowing a real lane key."""

    name = "quest"

    def load_metadata(self):
        raise AssertionError("pass 1 is on disk — hydrate() should have served it")

    def _iter_corpus(self):
        for index in range(50):
            yield {"doc_id": f"d{index}", "title": "", "text": f"text {index}"}


def qrels(path, doc_ids, relevance=1):
    """Write a minimal qrels snapshot for one lane under tmp_path."""
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "query_id": ["q1"] * len(doc_ids),
            "doc_id": doc_ids,
            "relevance": [relevance] * len(doc_ids),
        }
    ).to_parquet(path / "qrels.parquet", index=False)


def test_lane_dir_uses_the_source_name(tmp_path):
    corpora = LaneCorpora(tmp_path)
    assert corpora.lane_dir("beir-nfcorpus") == tmp_path / "beir-nfcorpus"
    with pytest.raises(KeyError):
        corpora.lane_dir("nfcorpus")


def test_plan_skips_the_recipe_for_lanes_without_one(tmp_path):
    row = LaneCorpora(tmp_path).plan(("beir-nfcorpus",)).iloc[0]
    assert row["sizing"] == "full"
    assert pd.isna(row["target"])


def test_plan_skips_the_recipe_for_a_lane_that_sizes_itself(tmp_path):
    qrels(tmp_path / "msmarco-passage-dev", ["d1"])
    row = LaneCorpora(tmp_path).plan(("msmarco-passage-dev",)).iloc[0]
    assert row["sizing"] == "own"
    assert pd.isna(row["target"])


def test_plan_reports_pass_1_before_qrels_exist(tmp_path):
    row = LaneCorpora(tmp_path).plan(("quest",)).iloc[0]
    assert pd.isna(row["target"])
    assert "pass 1" in row["verdict"]


def test_plan_computes_the_recipe_target_from_qrels(tmp_path):
    qrels(tmp_path / "quest", [f"d{i}" for i in range(4_000)])
    row = LaneCorpora(tmp_path).plan(("quest",)).iloc[0]
    # 4,000 / 0.2 = 20,000, above the 10,000 floor
    assert (row["relevant"], row["target"], row["budget"]) == (4_000, 20_000, 16_000)
    assert (row["sizing"], row["verdict"], row["on_disk"]) == ("recipe", "ready", False)


def test_plan_ignores_judged_zero_rows(tmp_path):
    qrels(tmp_path / "quest", [f"d{i}" for i in range(50)], relevance=0)
    row = LaneCorpora(tmp_path).plan(("quest",)).iloc[0]
    # nothing is force-included, so the floor decides the whole corpus
    assert (row["relevant"], row["target"]) == (0, 10_000)


def test_plan_honours_a_pinned_target(tmp_path):
    qrels(tmp_path / "limit", [f"d{i}" for i in range(46)])
    row = LaneCorpora(tmp_path).plan(("limit",)).iloc[0]
    assert (row["sizing"], row["target"]) == ("pinned", LANES["limit"].corpus_target)


def test_plan_flags_a_negative_budget(tmp_path):
    pinned = LANES["limit"].corpus_target
    qrels(tmp_path / "limit", [f"d{i}" for i in range(pinned + 1)])
    row = LaneCorpora(tmp_path).plan(("limit",)).iloc[0]
    assert row["budget"] == -1
    assert "NEGATIVE BUDGET" in row["verdict"]


def test_plan_marks_a_corpus_already_on_disk(tmp_path):
    lane_dir = tmp_path / "quest"
    qrels(lane_dir, ["d1"])
    pd.DataFrame({"doc_id": ["d1"], "title": [""], "text": ["t"]}).to_parquet(
        lane_dir / "corpus.parquet", index=False
    )
    assert LaneCorpora(tmp_path).plan(("quest",)).iloc[0]["on_disk"]


def test_passes_skip_what_is_already_there(tmp_path, capsys):
    """Both passes must be no-ops on a complete snapshot — a fetch here would
    need the network."""
    lane_dir = tmp_path / "quest"
    qrels(lane_dir, ["d1"])
    pd.DataFrame({"doc_id": ["d1"], "title": [""], "text": ["t"]}).to_parquet(
        lane_dir / "corpus.parquet", index=False
    )
    corpora = LaneCorpora(tmp_path)
    corpora.build_metadata("quest")
    corpora.build("quest")
    assert corpora.build_all(("quest",)) == []
    assert "no fetch" in capsys.readouterr().out


def test_pass_2_hydrates_then_force_includes_the_judged_docs(tmp_path, monkeypatch):
    lane_dir = tmp_path / "quest"
    qrels(lane_dir, ["d0", "d1"])
    pd.DataFrame({"query_id": ["q1"], "text": ["t"]}).to_parquet(
        lane_dir / "queries.parquet", index=False
    )
    monkeypatch.setitem(LANES, "quest", Lane(FakeLane(), corpus_target=10))
    LaneCorpora(tmp_path).build("quest")

    corpus = pd.read_parquet(lane_dir / "corpus.parquet")
    assert len(corpus) == 10
    assert {"d0", "d1"} <= set(corpus["doc_id"])


def test_every_lane_is_planned_offline(tmp_path):
    assert len(LaneCorpora(tmp_path).plan()) == len(LANES)

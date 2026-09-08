"""HomeDepotLane maps local Kaggle CSVs to the retrieval contract: queries
deduped per search_term, float relevance preserved, corpus joined by product_uid."""

from __future__ import annotations

import pandas as pd
import pytest

from hybrid_search_rrf_dataset.retrieval.wave3 import HomeDepotLane


def _fixture(tmp_path):
    pd.DataFrame(
        [
            {"id": 1, "product_uid": 100, "product_title": "Angle Bracket",
             "search_term": "angle bracket", "relevance": 2.33},
            {"id": 2, "product_uid": 101, "product_title": "Deck Screws",
             "search_term": "deck screws", "relevance": 3.00},
            {"id": 3, "product_uid": 102, "product_title": "Angle Bracket XL",
             "search_term": "Angle Bracket", "relevance": 1.67},
        ]
    ).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(
        [
            {"product_uid": 100, "product_description": "Galvanized bracket for framing."},
            {"product_uid": 101, "product_description": "Exterior deck screws, 100-pack."},
            {"product_uid": 102, "product_description": "Heavy-duty angle bracket."},
            {"product_uid": 103, "product_description": "Unrated mystery product."},
        ]
    ).to_csv(tmp_path / "product_descriptions.csv", index=False)
    return HomeDepotLane(tmp_path)


def test_queries_dedupe_per_search_term_case_insensitive(tmp_path):
    lane = _fixture(tmp_path)
    lane.load_metadata()
    # "angle bracket" and "Angle Bracket" collapse to one query.
    assert set(lane.queries()["text"]) == {"angle bracket", "deck screws"}
    assert len(lane.queries()) == 2
    assert lane.queries()["query_id"].is_unique


def test_qrels_preserve_float_relevance_and_dedup_target(tmp_path):
    lane = _fixture(tmp_path)
    lane.load_metadata()
    qrels = lane.qrels()
    assert len(qrels) == 3  # one row per rated (query, product) pair
    assert qrels["relevance"].tolist() == [2.33, 3.00, 1.67]  # not rounded
    bracket = lane._query_id("angle bracket")
    assert set(qrels[qrels["query_id"] == bracket]["doc_id"]) == {"100", "102"}


def test_corpus_joins_title_and_description(tmp_path):
    lane = _fixture(tmp_path)
    corpus = {row["doc_id"]: row for row in lane._iter_corpus()}
    assert corpus["100"]["title"] == "Angle Bracket"
    assert "Galvanized" in corpus["100"]["text"]
    assert corpus["103"]["title"] == ""  # in the catalog, never rated -> no title


def test_missing_files_fail_loud(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"train\.csv"):
        HomeDepotLane(tmp_path).load_metadata()

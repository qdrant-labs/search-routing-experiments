from collections import namedtuple

import pandas as pd

from hybrid_search_rrf_dataset.retrieval.wave3 import (
    AmazonEsciLane,
    FinderLane,
    TechQaLane,
    Touche2020Lane,
    TrecCast2020HistoryLane,
    WandsLane,
)

CastTurn = namedtuple(
    "CastTurn",
    ("query_id", "raw_utterance", "topic_number", "turn_number"),
)
Qrel = namedtuple("Qrel", ("query_id", "doc_id", "relevance", "iteration"))
BeirQuery = namedtuple("BeirQuery", ("query_id", "text"))


class _FakeDataset:
    def __init__(self, queries=(), qrels=()):
        self._queries = list(queries)
        self._qrels = list(qrels)

    def queries_iter(self):
        return iter(self._queries)

    def qrels_iter(self):
        return iter(self._qrels)


def test_amazon_esci_exposes_english_hard_queries_products_and_qrels(monkeypatch):
    examples = [
        {
            "query_id": 10,
            "query": "acme 18v drill",
            "product_id": "p1",
            "product_locale": "us",
            "esci_label": "E",
            "small_version": 1,
        },
        {
            "query_id": 10,
            "query": "acme 18v drill",
            "product_id": "p2",
            "product_locale": "us",
            "esci_label": "S",
            "small_version": 1,
        },
        {
            "query_id": 11,
            "query": "drill battery",
            "product_id": "p3",
            "product_locale": "us",
            "esci_label": "C",
            "small_version": 1,
        },
        {
            "query_id": 12,
            "query": "taladro",
            "product_id": "p4",
            "product_locale": "es",
            "esci_label": "E",
            "small_version": 1,
        },
        {
            "query_id": 13,
            "query": "easy drill",
            "product_id": "p5",
            "product_locale": "us",
            "esci_label": "E",
            "small_version": 0,
        },
    ]
    products = [
        {
            "product_id": "p1",
            "product_locale": "us",
            "product_title": "Acme Drill",
            "product_brand": "Acme",
            "product_color": "Red",
            "product_bullet_point": "Cordless",
            "product_description": "An 18V drill.",
        },
        {
            "product_id": "p2",
            "product_locale": "us",
            "product_title": "Acme Drill Kit",
            "product_brand": "Acme",
            "product_color": None,
            "product_bullet_point": None,
            "product_description": None,
        },
        {
            "product_id": "p3",
            "product_locale": "us",
            "product_title": "18V Battery",
            "product_brand": None,
            "product_color": None,
            "product_bullet_point": None,
            "product_description": None,
        },
        {
            "product_id": "p4",
            "product_locale": "es",
            "product_title": "Taladro",
            "product_brand": None,
            "product_color": None,
            "product_bullet_point": None,
            "product_description": None,
        },
    ]

    def fake_load_dataset(reader, *, data_files, **kwargs):
        assert reader == "parquet"
        rows = products if "products" in data_files else examples
        return iter(rows)

    monkeypatch.setattr(
        "hybrid_search_rrf_dataset.retrieval.wave3.load_dataset",
        fake_load_dataset,
    )
    lane = AmazonEsciLane()

    lane.load_metadata()
    lane.materialize()

    assert lane.queries().to_dict("records") == [
        {"query_id": "10", "text": "acme 18v drill"},
        {"query_id": "11", "text": "drill battery"},
    ]
    assert lane.qrels().to_dict("records") == [
        {"query_id": "10", "doc_id": "p1", "relevance": 2},
        {"query_id": "10", "doc_id": "p2", "relevance": 1},
        {"query_id": "11", "doc_id": "p3", "relevance": 0},
    ]
    pd.testing.assert_frame_equal(
        lane.corpus(),
        pd.DataFrame(
            [
                {
                    "doc_id": "p1",
                    "title": "Acme Drill",
                    "text": (
                        "Brand: Acme\nColor: Red\nBullet points: Cordless\n"
                        "Description: An 18V drill."
                    ),
                },
                {
                    "doc_id": "p2",
                    "title": "Acme Drill Kit",
                    "text": "Brand: Acme",
                },
                {"doc_id": "p3", "title": "18V Battery", "text": ""},
            ]
        ),
    )


def test_wands_exposes_queries_full_products_and_graded_qrels(monkeypatch):
    source_rows = {
        "query.csv": [
            {"query_id": 7, "query": "blue reading chair", "query_class": "Chairs"}
        ],
        "label.csv": [
            {"id": 1, "query_id": 7, "product_id": 21, "label": "Exact"},
            {"id": 2, "query_id": 7, "product_id": 22, "label": "Partial"},
            {"id": 3, "query_id": 7, "product_id": 23, "label": "Irrelevant"},
        ],
        "product.csv": [
            {
                "product_id": 21,
                "product_name": "Blue Reading Chair",
                "product_class": "Accent Chairs",
                "category hierarchy": "Furniture/Seating",
                "product_description": "A comfortable chair.",
                "product_features": "color:blue|material:linen",
            }
        ],
    }

    def fake_load_dataset(reader, *, data_files, **kwargs):
        assert reader == "csv"
        assert kwargs["delimiter"] == "\t"
        filename = data_files.rsplit("/", 1)[-1]
        return iter(source_rows[filename])

    monkeypatch.setattr(
        "hybrid_search_rrf_dataset.retrieval.wave3.load_dataset",
        fake_load_dataset,
    )
    lane = WandsLane()

    lane.load_metadata()
    lane.materialize()

    assert lane.queries().to_dict("records") == [
        {"query_id": "7", "text": "blue reading chair"}
    ]
    assert lane.qrels().to_dict("records") == [
        {"query_id": "7", "doc_id": "21", "relevance": 2},
        {"query_id": "7", "doc_id": "22", "relevance": 1},
        {"query_id": "7", "doc_id": "23", "relevance": 0},
    ]
    assert lane.corpus().to_dict("records") == [
        {
            "doc_id": "21",
            "title": "Blue Reading Chair",
            "text": (
                "Product class: Accent Chairs\n"
                "Category hierarchy: Furniture/Seating\n"
                "Description: A comfortable chair.\n"
                "Features: color:blue|material:linen"
            ),
        }
    ]


def test_finder_deduplicates_reference_passages_into_a_retrieval_corpus(monkeypatch):
    rows = [
        {
            "_id": "q1",
            "text": "CBOE revenue in 2023",
            "reasoning": False,
            "category": "Financials",
            "references": ["Revenue  was $10 million.\n"],
        },
        {
            "_id": "q2",
            "text": "CBOE margin",
            "reasoning": True,
            "category": "Financials",
            "references": [
                "Revenue was $10 million.",
                "Operating margin was 12%.",
            ],
        },
    ]

    def fake_load_dataset(repo, *, split, streaming):
        assert (repo, split, streaming) == (
            "Linq-AI-Research/FinDER",
            "train",
            True,
        )
        return iter(rows)

    monkeypatch.setattr(
        "hybrid_search_rrf_dataset.retrieval.wave3.load_dataset",
        fake_load_dataset,
    )
    lane = FinderLane()

    lane.load_metadata()
    lane.materialize()

    revenue_id = "finder-4e137061ae7245a5af67f76d"
    margin_id = "finder-f9ac31063fd6591aced9e659"
    assert lane.queries().to_dict("records") == [
        {"query_id": "q1", "text": "CBOE revenue in 2023"},
        {"query_id": "q2", "text": "CBOE margin"},
    ]
    assert lane.qrels().to_dict("records") == [
        {"query_id": "q1", "doc_id": revenue_id, "relevance": 1},
        {"query_id": "q2", "doc_id": revenue_id, "relevance": 1},
        {"query_id": "q2", "doc_id": margin_id, "relevance": 1},
    ]
    assert lane.corpus().to_dict("records") == [
        {
            "doc_id": revenue_id,
            "title": "",
            "text": "Revenue was $10 million.",
        },
        {
            "doc_id": margin_id,
            "title": "",
            "text": "Operating margin was 12%.",
        },
    ]


def test_trec_cast_2020_history_serializes_dialogue_history_per_topic():
    lane = TrecCast2020HistoryLane()
    turns = [
        CastTurn("81_2", "second follow-up", 81, 2),
        CastTurn("81_1", "first turn on paris", 81, 1),
        CastTurn("82_1", "different topic", 82, 1),
    ]
    qrels = [
        Qrel("81_1", "doc-a", 2, 0),
        Qrel("81_2", "doc-b", 1, 0),
    ]
    lane._test_dataset = _FakeDataset(queries=turns, qrels=qrels)

    lane.load_metadata()

    assert lane.queries().to_dict("records") == [
        {"query_id": "81_1", "text": "[CURRENT] first turn on paris"},
        {
            "query_id": "81_2",
            "text": "first turn on paris [CURRENT] second follow-up",
        },
        {"query_id": "82_1", "text": "[CURRENT] different topic"},
    ]
    assert lane.qrels().to_dict("records") == [
        {"query_id": "81_1", "doc_id": "doc-a", "relevance": 2},
        {"query_id": "81_2", "doc_id": "doc-b", "relevance": 1},
    ]


def test_touche_2020_serves_queries_and_qrels_from_the_beir_v2_dataset():
    lane = Touche2020Lane()
    queries = [
        BeirQuery("1", "Should teachers get tenure?"),
        BeirQuery("2", "Is climate change man-made?"),
    ]
    qrels = [Qrel("1", "arg-A", 2, 0), Qrel("1", "arg-B", 0, 0)]
    lane._test_dataset = _FakeDataset(queries=queries, qrels=qrels)

    lane.load_metadata()

    assert lane.queries().to_dict("records") == [
        {"query_id": "1", "text": "Should teachers get tenure?"},
        {"query_id": "2", "text": "Is climate change man-made?"},
    ]
    assert lane.qrels().to_dict("records") == [
        {"query_id": "1", "doc_id": "arg-A", "relevance": 2},
        {"query_id": "1", "doc_id": "arg-B", "relevance": 0},
    ]


def test_techqa_lane_streams_question_document_pairs_and_dedups_the_corpus(monkeypatch):
    rows = {
        "train": [
            {
                "id": "TRAIN_Q000",
                "question": "Streams jobs missing env vars?",
                "document": "IBM Streams  4.1.1.1 env vars ...",
                "answer": "set env vars in the instance",
            }
        ],
        "validation": [
            {
                "id": "DEV_Q000",
                "question": "How to debug Streams env vars?",
                "document": "IBM Streams 4.1.1.1 env vars ...",
                "answer": "look at instance settings",
            }
        ],
        "test": [
            {
                "id": "TEST_Q000",
                "question": "Different question entirely?",
                "document": "Another Technote content.",
                "answer": "unrelated",
            }
        ],
    }

    def fake_load_dataset(repo, *, split, streaming):
        assert (repo, streaming) == ("rojagtap/tech-qa", True)
        return iter(rows[split])

    monkeypatch.setattr(
        "hybrid_search_rrf_dataset.retrieval.wave3.load_dataset",
        fake_load_dataset,
    )
    lane = TechQaLane()

    lane.load_metadata()
    lane.materialize()

    shared_doc = "techqa-" + __import__("hashlib").sha256(
        b"IBM Streams 4.1.1.1 env vars ..."
    ).hexdigest()[:24]
    other_doc = "techqa-" + __import__("hashlib").sha256(
        b"Another Technote content."
    ).hexdigest()[:24]
    assert lane.queries().to_dict("records") == [
        {"query_id": "TRAIN_Q000", "text": "Streams jobs missing env vars?"},
        {"query_id": "DEV_Q000", "text": "How to debug Streams env vars?"},
        {"query_id": "TEST_Q000", "text": "Different question entirely?"},
    ]
    assert lane.qrels().to_dict("records") == [
        {"query_id": "TRAIN_Q000", "doc_id": shared_doc, "relevance": 1},
        {"query_id": "DEV_Q000", "doc_id": shared_doc, "relevance": 1},
        {"query_id": "TEST_Q000", "doc_id": other_doc, "relevance": 1},
    ]
    pd.testing.assert_frame_equal(
        lane.corpus().sort_values("doc_id").reset_index(drop=True),
        pd.DataFrame(
            [
                {
                    "doc_id": shared_doc,
                    "title": "",
                    "text": "IBM Streams 4.1.1.1 env vars ...",
                },
                {
                    "doc_id": other_doc,
                    "title": "",
                    "text": "Another Technote content.",
                },
            ]
        )
        .sort_values("doc_id")
        .reset_index(drop=True),
    )

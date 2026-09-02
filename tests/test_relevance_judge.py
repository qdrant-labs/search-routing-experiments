"""Invariants of the relevance-atom judge: verdict parsing, population regime,
human-truth precedence, above-gold selection, and artifact separation."""

from __future__ import annotations

import pandas as pd

from hybrid_search_rrf_dataset.qrels import QrelSource, QrelStore
from relevance_judge.config import RelevanceJudgeConfig
from relevance_judge.judge import _parse
from relevance_judge.queue import JudgeQueue, regime


def test_parse_reads_yes_no_and_rejects_garbage():
    assert _parse("yes - answers it") == (True, "answers it")
    assert _parse("No, unrelated")[0] is False
    assert _parse("maybe later") is None
    assert _parse("") is None


def test_regime_buckets():
    assert regime({"a": 0.0, "b": 0.0, "c": 0.0}) == "all_zero"
    assert regime({"a": 0.4, "b": 0.4, "c": 0.4}) == "all_tied"
    assert regime({"a": 1.0, "b": 0.95, "c": 0.9}) == "low_margin"
    assert regime({"a": 1.0, "b": 0.1, "c": 0.0}) == "decisive_strong"


def test_human_wins_over_llm_on_conflict():
    """A judged atom NEVER overrides a human judgment — the merge is precedence,
    so human truth stays authoritative (doc §5a)."""
    human = QrelStore(pd.DataFrame([
        {"dataset": "d", "query_id": "q", "doc_id": "x", "relevance": 1,
         "source": str(QrelSource.HUMAN)},
    ]))
    llm = QrelStore(pd.DataFrame([
        {"dataset": "d", "query_id": "q", "doc_id": "x", "relevance": 0,
         "source": str(QrelSource.LLM)},
        {"dataset": "d", "query_id": "q", "doc_id": "y", "relevance": 1,
         "source": str(QrelSource.LLM)},
    ]))
    merged = QrelStore.concat([human, llm]).lookup("d")
    assert merged["q"]["x"] == 1  # human's 1 wins over llm's 0
    assert merged["q"]["y"] == 1  # llm fills a hole human never judged


def test_judged_qrels_artifact_is_separate_from_human_qrels():
    config = RelevanceJudgeConfig()
    assert config.judged_qrels.name == "judged_qrels.parquet"
    assert config.judged_qrels.parent.name == "relevance_judge"
    # never a per-lane human qrels file
    assert config.judged_qrels != config.data_dir / "clerc" / "qrels.parquet"


class _StubSources:
    def __init__(self, ranks, gold):
        self._ranks, self._gold = ranks, gold

    def rankings(self, dataset):
        return self._ranks

    def manifest_gold(self, dataset):
        return self._gold


def test_above_gold_takes_docs_ranked_above_gold_and_excludes_it():
    queue = JudgeQueue.__new__(JudgeQueue)
    queue.sources = _StubSources(
        ranks={"q": {
            "dense_only": ["a", "b", "g", "c"],   # gold at idx 2 -> above = a,b
            "sparse_only": ["d", "g", "e"],        # gold at idx 1 -> above = d
            "pure_rrf": ["f", "h"],                # no gold -> all above
        }},
        gold={"q": {"g"}},
    )
    above = queue._above_gold("any", "q")
    assert above == {"a", "b", "d", "f", "h"}
    assert "g" not in above


def test_above_gold_missing_rankings_is_none():
    queue = JudgeQueue.__new__(JudgeQueue)
    queue.sources = _StubSources(ranks={}, gold={})
    assert queue._above_gold("any", "missing") is None


class _FakeJudge:
    """Returns a canned verdict without any LLM call."""

    def judge_one(self, query, doc_text, *, budget=None):
        return True, "canned", "hash", None


def test_judge_rows_banks_every_n_and_final_matches(tmp_path):
    from relevance_judge.config import RelevanceJudgeConfig
    from relevance_judge.validation import ValidationHarness

    config = RelevanceJudgeConfig(data_dir=tmp_path)
    harness = ValidationHarness(config, judge=_FakeJudge(), sources=object())
    harness.config = config

    calls = {"n": 0}
    original = harness._persist_predictions

    def counting(preds):
        calls["n"] += 1
        original(preds)

    harness._persist_predictions = counting

    rows = pd.DataFrame([
        {"dataset": "d", "query_id": str(i), "doc_id": str(i),
         "human_relevant": True, "query": "q", "doc_text": "text"}
        for i in range(120)
    ])
    preds = harness.judge_rows(rows, bank_every=50)

    # writes at 50, 100, and a final flush -> at least 3 snapshots, none lost
    assert calls["n"] >= 3
    assert len(preds) == 120
    on_disk = pd.read_parquet(config.validation_predictions)
    assert len(on_disk) == 120

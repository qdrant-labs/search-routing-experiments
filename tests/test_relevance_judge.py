"""Invariants of the relevance-atom judge: verdict parsing, population regime,
human-truth precedence, above-gold selection, and artifact separation."""

from __future__ import annotations

import pandas as pd

from hybrid_search_rrf_dataset.qrels import QrelSource, QrelStore
from relevance_judge.config import RelevanceJudgeConfig
from relevance_judge.judge import _parse
from relevance_judge.residual import JudgeQueue, V2Labels, above_gold, regime
from relevance_judge.validation import Confusion, Gate


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
    assert config.judged_qrels != config.lanes.lane_qrels("clerc")


def test_every_path_derives_from_the_config_root(tmp_path):
    """No module may build a path from a literal: re-rooting the config must
    move EVERY read and write, inputs included."""
    config = RelevanceJudgeConfig(data_dir=tmp_path)
    paths = [
        config.draw, config.depth_probe, config.manifest, config.labels,
        config.artifacts, config.judged_qrels, config.validation_predictions,
        config.lanes.lane_corpus("clerc"), config.lanes.lane_queries("clerc"),
        config.lanes.lane_qrels("clerc"), *config.oracle_caches("clerc"),
    ]
    for path in paths:
        assert path.is_relative_to(tmp_path), path


def test_oracle_caches_reuse_the_rung_prefix(tmp_path):
    """The rung is spelled once — `v2_100k` — so a rung bump cannot leave a
    stale cache path behind."""
    config = RelevanceJudgeConfig(data_dir=tmp_path)
    rung = config.oracle_caches("clerc")[: len(config.oracle_stages)]
    assert all(p.is_relative_to(config.v2_100k) for p in rung)
    assert [p.parent.parent.name for p in rung] == list(config.oracle_stages)


def test_above_gold_is_one_definition_for_live_and_persisted_orders():
    assert above_gold({"r": ["a", "b", "g", "c"]}, {"g"}) == {"a", "b"}
    assert above_gold({"r": ["f", "h"]}, {"g"}) == {"f", "h"}   # no gold -> all above
    assert above_gold({"r": ["g", "a"]}, {"g"}) == set()        # gold at rank 1


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


def test_population_is_an_input_not_a_hardcoded_file():
    """The queue reads a score triple from whatever population it is handed —
    the v2 stack classifies the same rows differently from l2, which is the
    whole reason the source must be injectable."""

    class _Stub:
        def triples(self):
            return pd.DataFrame([
                {"dataset": "d", "query_id": "tied", "query": "q",
                 "triple": {"dense_only": 0.5, "sparse_only": 0.5, "pure_rrf": 0.5}},
                {"dataset": "d", "query_id": "one", "query": "q",
                 "triple": {"dense_only": 1.0, "sparse_only": 1.0, "pure_rrf": 1.0}},
                {"dataset": "d", "query_id": "clear", "query": "q",
                 "triple": {"dense_only": 1.0, "sparse_only": 0.1, "pure_rrf": 0.1}},
            ])

        def absent(self):
            return pd.DataFrame(columns=["dataset", "query_id", "query"])

    queue = JudgeQueue.__new__(JudgeQueue)
    queue.population = _Stub()
    res = queue.residual()
    assert set(res["query_id"]) == {"tied", "one"}          # decisive row excluded
    assert res.set_index("query_id").loc["tied", "sub1"]    # ties below 1.0
    assert not res.set_index("query_id").loc["one", "sub1"]


def test_v2labels_reads_the_configured_labels_path(tmp_path):
    config = RelevanceJudgeConfig(data_dir=tmp_path)
    config.labels.parent.mkdir(parents=True)
    pd.DataFrame([
        {"dataset": "d", "query_id": "q1", "route": "dense_only", "query": "text",
         "score_dense_only": 0.4, "score_sparse_only": 0.4, "score_pure_rrf": 0.4},
        {"dataset": "d", "query_id": "q1", "route": "pure_rrf", "query": "text",
         "score_dense_only": 0.4, "score_sparse_only": 0.4, "score_pure_rrf": 0.4},
    ]).to_parquet(config.labels, index=False)

    triples = V2Labels(config).triples()
    assert len(triples) == 1                                    # one row per query
    assert triples.iloc[0]["triple"] == {
        "dense_only": 0.4, "sparse_only": 0.4, "pure_rrf": 0.4
    }
    assert V2Labels(config).absent().empty                      # no depth probe exists


def test_gate_needs_negatives_and_refuses_a_degenerate_judge():
    config = RelevanceJudgeConfig()
    gate = Gate(config)
    positives_only = Confusion(tp=10, fp=0, fn=0, tn=0)
    assert not positives_only.has_negatives
    assert not gate.passed(positives_only, Confusion(tp=8, fp=0, fn=2, tn=0))

    always_no = Confusion(tp=0, fp=0, fn=10, tn=5)
    assert not gate.passed(Confusion(tp=10, fp=0, fn=0, tn=5), always_no)

    clean = Confusion(tp=10, fp=0, fn=0, tn=5)
    assert gate.passed(clean, Confusion(tp=8, fp=0, fn=2, tn=0))


def test_unmeasurable_rate_is_nan_not_a_passing_zero():
    """A rate with no denominator must not read as 0.0 — that would let an
    unmeasurable gate look merely failing rather than unmeasurable."""
    empty = Confusion(tp=0, fp=0, fn=0, tn=0)
    assert empty.precision != empty.precision   # nan
    assert empty.recall != empty.recall

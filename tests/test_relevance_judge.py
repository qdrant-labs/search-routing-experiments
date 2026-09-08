"""Invariants of the relevance-atom judge: verdict parsing, population regime,
human-truth precedence, tail-doc selection, and artifact separation."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from augmentation.engine import BudgetExceeded
from relevance_judge.judge import completion as _real_completion

from hybrid_search_rrf_dataset.qrels import QrelSource, QrelStore
from relevance_judge.config import RelevanceJudgeConfig
from relevance_judge.judge import _parse
from relevance_judge.residual import (
    JudgeQueue,
    V2Labels,
    above_gold,
    regime,
    tail_docs,
)
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


def test_tail_docs_serves_a_perfect_tie_that_above_gold_cannot():
    """The regression this exists for: gold at rank 1 in every route leaves no
    above-gold doc, so the old work list skipped these rows entirely — yet the
    routes disagree below rank 1, which is what the NDCG term reads."""
    orders = {
        "dense_only": ["g", "a", "b"],
        "sparse_only": ["g", "b", "a"],
        "pure_rrf": ["g", "a", "c"],
    }
    assert above_gold(orders, {"g"}) == set()          # nothing to judge
    assert set(tail_docs(orders, {"g"})) == {"a", "b", "c"}


def test_tail_docs_drops_same_rank_docs_and_keeps_differing_ranks():
    """A doc every route ranks identically adds the same DCG and IDCG to each,
    so it cannot break a tie; one every route ranks differently can — even
    when it sits in all of them."""
    orders = {
        "r1": ["g", "same", "x"],
        "r2": ["g", "same", "y"],
    }
    spreads = tail_docs(orders, {"g"})
    assert "same" not in spreads          # rank 1 in both -> inert
    assert set(spreads) == {"x", "y"}

    everywhere = {"r1": ["g", "m", "n"], "r2": ["g", "n", "m"]}
    assert set(tail_docs(everywhere, {"g"})) == {"m", "n"}   # in both, swapped


def test_tail_docs_excludes_gold_and_reports_spread_widest_first():
    orders = {"r1": ["g", "near", "far"], "r2": ["near", "g", "far"]}
    spreads = tail_docs(orders, {"g"})
    assert "g" not in spreads
    assert spreads["near"] == 1          # idx 1 vs 0
    assert "far" not in spreads          # idx 2 in both -> inert


def test_tail_docs_missing_rankings_is_none():
    queue = JudgeQueue.__new__(JudgeQueue)
    queue.sources = _StubSources(ranks={}, gold={})
    assert queue._tail_docs("any", "missing") is None


class _FakeJudge:
    """Returns a canned verdict without any LLM call. The 4th element is a real
    `Spend` because the production `judge_one` always returns one — a fake that
    hands back None hides a contract break rather than exercising it."""

    def judge_one(self, query, doc_text, *, dataset='', budget=None):
        from augmentation.engine import Spend

        from relevance_judge.judge import Verdict

        return Verdict(True, "canned", "hash", Spend())


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


def _pred(lane, i, human, pred, pseudo=False):
    return {"dataset": lane, "query_id": str(i), "doc_id": str(i),
            "human_relevant": human, "pred_relevant": pred,
            "reason": "", "pseudo": pseudo}


def test_pseudo_negatives_never_enter_the_precision_gate(tmp_path):
    """Random-corpus negatives are easy to reject; pooling them into the referee
    confusion would inflate precision exactly where it is least earned."""
    from relevance_judge.validation import ValidationHarness

    harness = ValidationHarness(RelevanceJudgeConfig(data_dir=tmp_path))
    human_only = pd.DataFrame([
        _pred("ref", 1, True, True), _pred("ref", 2, False, True),   # 1 fp
        _pred("ref", 3, False, False),
    ])
    with_pseudo = pd.concat([human_only, pd.DataFrame([
        _pred("deploy", 9, False, False, pseudo=True),               # easy reject
        _pred("deploy", 10, False, False, pseudo=True),
    ])], ignore_index=True)

    base = harness.score(human_only, ["ref"])
    plus = harness.score(with_pseudo, ["ref"], deploy_lanes=["deploy"])
    assert plus["precision_relevant"] == base["precision_relevant"]
    assert plus["confusion"] == base["confusion"]
    assert plus["n_validation"] == base["n_validation"]


def test_score_flags_precision_as_transfer_when_deploy_is_unrefereed(tmp_path):
    """The defect this exists for: precision measured on referee lanes was read
    as precision on the deploy lanes, which share no rows with them."""
    from relevance_judge.validation import ValidationHarness

    harness = ValidationHarness(RelevanceJudgeConfig(data_dir=tmp_path))
    preds = pd.DataFrame([
        _pred("ref", 1, True, True), _pred("ref", 2, False, False),
        _pred("deploy", 3, True, True),                     # positive-only lane
        _pred("deploy", 4, False, True, pseudo=True),        # random doc, called relevant
    ])
    report = harness.score(preds, ["ref", "deploy"], deploy_lanes=["deploy"])
    assert report["precision_is_transfer_estimate"] is True
    assert report["deploy_lanes_unrefereed"] == ["deploy"]
    assert report["deploy_lanes_refereed"] == []
    assert report["deploy_false_positive_rate"] == 1.0     # 1 of 1 pseudo called relevant

    # a deploy lane that DOES carry human negatives is a real measurement
    refereed = pd.DataFrame([
        _pred("deploy", 1, True, True), _pred("deploy", 2, False, False),
    ])
    ok = harness.score(refereed, ["deploy"], deploy_lanes=["deploy"])
    assert ok["precision_is_transfer_estimate"] is False
    assert ok["deploy_lanes_refereed"] == ["deploy"]


def test_a_budget_stop_keeps_the_verdicts_already_paid_for(tmp_path):
    """The defect this exists for: BudgetExceeded reaches judge_pairs through
    windowed_map's future.result(), so a buffer of already-PAID verdicts was
    dropped on the way out. Paid work must land on disk even on the error path."""
    from augmentation.engine import BudgetExceeded, Spend
    from relevance_judge.judge import RelevanceJudge

    config = RelevanceJudgeConfig(data_dir=tmp_path, llm_workers=1)
    judge = RelevanceJudge(config)

    calls = {"n": 0}

    def dying_judge_one(query, doc_text, *, dataset='', budget=None):
        calls["n"] += 1
        if calls["n"] > 3:
            raise BudgetExceeded("ceiling hit")
        from relevance_judge.judge import Verdict

        return Verdict(True, "canned", "hash", Spend())

    judge.judge_one = dying_judge_one
    pairs = pd.DataFrame([
        {"dataset": "d", "query_id": str(i), "doc_id": str(i),
         "query": "q", "doc_text": "text"}
        for i in range(10)
    ])
    try:
        judge.judge_pairs(pairs, run_id="run")
    except BudgetExceeded:
        pass
    banked = judge.load()
    assert len(banked) == 3, f"paid verdicts lost: banked {len(banked)}, paid 3"


def test_a_non_transient_provider_error_does_not_abandon_the_run(tmp_path):
    """The defect this exists for: only TRANSIENT_PROVIDER_ERRORS was caught, so
    a rejected param or filtered document propagated out of windowed_map and
    abandoned every remaining pair — after two verdicts, in a real run."""
    from litellm.exceptions import BadRequestError

    from relevance_judge.judge import RelevanceJudge

    judge = RelevanceJudge(RelevanceJudgeConfig(data_dir=tmp_path, llm_workers=1))
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise BadRequestError("bad param", model="m", llm_provider="openrouter")
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            choices=[SimpleNamespace(message=SimpleNamespace(content="yes - fine"))],
        )

    import relevance_judge.judge as mod
    mod.completion = flaky
    try:
        pairs = pd.DataFrame([
            {"dataset": "d", "query_id": str(i), "doc_id": str(i),
             "query": "q", "doc_text": "t"} for i in range(5)
        ])
        counts = judge.judge_pairs(pairs, run_id="run")
    finally:
        mod.completion = _real_completion

    assert counts["judged"] == 4, f"run abandoned early: {counts}"
    assert judge.dropped["BadRequestError"] == 1
    assert "bad param" in judge.dropped_detail["BadRequestError"]


def test_budget_exceeded_still_stops_the_run(tmp_path):
    """The broad catch must NOT swallow the ceiling — that one has to propagate."""
    from relevance_judge.judge import RelevanceJudge

    judge = RelevanceJudge(RelevanceJudgeConfig(data_dir=tmp_path, llm_workers=1))

    def broke(**kwargs):
        raise BudgetExceeded("ceiling")

    import relevance_judge.judge as mod
    mod.completion = broke
    try:
        pairs = pd.DataFrame([{"dataset": "d", "query_id": "1", "doc_id": "1",
                               "query": "q", "doc_text": "t"}])
        with pytest.raises(BudgetExceeded):
            judge.judge_pairs(pairs, run_id="run")
    finally:
        mod.completion = _real_completion


def test_parse_prefers_the_labelled_verdict_over_the_asked_line():
    """The ASKED line exists so the model states evidence BEFORE committing; it
    must never be read as the verdict, in either direction."""
    from relevance_judge.judge import _parse

    yes = (
        "ASKED: how to retrieve stored embeddings\n"
        'EVIDENCE: "db.get(include=[\'embeddings\'])"\n'
        "MISSING: NOTHING\n"
        "VERDICT: yes - shows the include argument"
    )
    assert _parse(yes) == (True, "shows the include argument")

    # the exact failure the format targets: the gap is conceded BEFORE the verdict
    no = (
        "ASKED: fix for embeddings showing None\n"
        "EVIDENCE: NOTHING\n"
        "MISSING: the retrieval fix\n"
        "VERDICT: no - never shows retrieval"
    )
    assert _parse(no)[0] is False

    # an ASKED line containing the word 'no' must not decide the verdict
    tricky = "ASKED: whether no-op transforms apply; doc gives the matrix\nVERDICT: yes - gives the transform matrix"
    assert _parse(tricky) == (True, "gives the transform matrix")

    # the bare legacy form still parses, so banked runs stay reproducible
    assert _parse("yes - answers it") == (True, "answers it")
    assert _parse("maybe later") is None


def test_the_four_line_reply_fits_the_token_cap():
    """12+15+10-word fields + 10-word VERDICT must fit max_tokens=128, or the
    reply is truncated mid-verdict and counts unreadable."""
    reply = ("ASKED: " + " ".join(["word"] * 12) + "\n"
             + "EVIDENCE: " + " ".join(["word"] * 15) + "\n"
             + "MISSING: " + " ".join(["word"] * 10) + "\n"
             + "VERDICT: no - " + " ".join(["word"] * 10))
    assert len(reply) / 4 < 128, "worst-case reply exceeds the completion cap"


def _resp(content, finish_reason="stop", out_tokens=20):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=out_tokens),
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=finish_reason)],
    )


def test_a_truncated_reply_is_retried_with_a_bigger_cap(tmp_path):
    """The defect this exists for: a reply cut off at the token cap loses its
    VERDICT line, parses to nothing, and threw away 249 PAID pairs. The provider
    reports finish_reason='length', so it must be retried, not discarded."""
    from relevance_judge.judge import RelevanceJudge

    judge = RelevanceJudge(RelevanceJudgeConfig(data_dir=tmp_path, max_answer_tokens=128))
    caps = []

    def truncate_then_succeed(**kwargs):
        caps.append(kwargs["max_tokens"])
        if len(caps) == 1:                      # first call: cut off mid-ASKED
            return _resp("ASKED: the query needs the specific fix and the document",
                         finish_reason="length")
        return _resp("ASKED: needs X; doc gives X\nVERDICT: yes - gives X")

    import relevance_judge.judge as mod
    mod.completion = truncate_then_succeed
    try:
        relevant, reason = judge.judge_one("q", "d")[:2]
    finally:
        mod.completion = _real_completion

    assert relevant is True, "the retry should have recovered the pair"
    assert reason == "gives X"
    assert caps == [128, 256], f"cap should double on retry, got {caps}"
    assert judge.dropped["truncated_retried"] == 1


def test_a_reply_truncated_twice_is_reported_as_truncated_not_unparsed(tmp_path):
    """Distinguishing the two causes is what made this diagnosable at all."""
    from relevance_judge.judge import RelevanceJudge

    judge = RelevanceJudge(RelevanceJudgeConfig(data_dir=tmp_path, max_answer_tokens=64))

    import relevance_judge.judge as mod
    mod.completion = lambda **kw: _resp("ASKED: still too long", finish_reason="length")
    try:
        relevant = judge.judge_one("q", "d").relevant
    finally:
        mod.completion = _real_completion

    assert relevant is None
    assert judge.dropped["truncated"] == 1
    assert judge.dropped["truncated_retried"] == 1     # it did try
    assert "finish=length" in judge.unreadable[0]      # and says why


def test_garbage_that_is_not_truncated_is_not_retried(tmp_path):
    """A model answering unreadably will answer unreadably again — retrying it
    just pays twice."""
    from relevance_judge.judge import RelevanceJudge

    judge = RelevanceJudge(RelevanceJudgeConfig(data_dir=tmp_path))
    calls = {"n": 0}

    def garbage(**kwargs):
        calls["n"] += 1
        return _resp("I cannot determine that", finish_reason="stop")

    import relevance_judge.judge as mod
    mod.completion = garbage
    try:
        relevant = judge.judge_one("q", "d").relevant
    finally:
        mod.completion = _real_completion

    assert relevant is None
    assert calls["n"] == 1, "non-truncated garbage must not be retried"
    assert judge.dropped["unparsed"] == 1


def test_the_pre_verdict_fields_are_parsed_and_banked(tmp_path):
    """A gold label is permanent, so the EVIDENCE quote that justifies it and the
    MISSING gap that refused it must survive on the atom — discarding them made
    every audit restart from the raw corpus."""
    from relevance_judge.judge import RATIONALE_FIELDS, RelevanceJudge, parse_fields

    reply = (
        "ASKED: how to retrieve stored embeddings\n"
        "EVIDENCE: db.get(include=[\"embeddings\"]) returns them\n"
        "MISSING: NOTHING\n"
        "VERDICT: yes - shows the include argument"
    )
    fields = parse_fields(reply)
    assert fields["asked"] == "how to retrieve stored embeddings"
    assert 'db.get(include=["embeddings"])' in fields["evidence"]
    assert fields["missing"] == "NOTHING"

    judge = RelevanceJudge(RelevanceJudgeConfig(data_dir=tmp_path, llm_workers=1))
    import relevance_judge.judge as mod
    mod.completion = lambda **kw: _resp(reply)
    try:
        pairs = pd.DataFrame([{"dataset": "d", "query_id": "1", "doc_id": "1",
                               "query": "q", "doc_text": "t"}])
        judge.judge_pairs(pairs, run_id="run")
    finally:
        mod.completion = _real_completion

    atom = judge.load().iloc[0]
    for field in RATIONALE_FIELDS:
        assert field in atom, f"{field} not banked"
    assert atom["reason"] == "shows the include argument"
    assert atom["missing"] == "NOTHING"
    assert 'db.get' in atom["evidence"]


def test_missing_fields_are_absent_not_invented():
    """A model that skips a line must leave the field empty, never guessed."""
    from relevance_judge.judge import parse_fields

    assert parse_fields("VERDICT: no - nothing here") == {}
    only_asked = parse_fields("ASKED: the fix\nVERDICT: no - absent")
    assert only_asked == {"asked": "the fix"}


def test_deploy_recall_is_reported_but_never_gates(tmp_path):
    """Recall on positive-only deploy lanes is measurable (recall needs only
    positives) and must be visible — but it measures agreement with each corpus's
    own notion of relevant, which differs from the judge's, so it cannot gate."""
    from relevance_judge.validation import ValidationHarness

    harness = ValidationHarness(RelevanceJudgeConfig(data_dir=tmp_path))
    preds = pd.DataFrame([
        # referee lane: carries a negative, so precision is measurable and passes
        _pred("ref", 1, True, True), _pred("ref", 2, False, False),
        # an anchor lane the judge does well on, so the recall GATE passes
        *[_pred("rarb-math", i, True, True) for i in range(10, 14)],
        # deploy lane: positives only, judge finds 1 of 3 -> recall 0.333
        _pred("dep", 3, True, True), _pred("dep", 4, True, False),
        _pred("dep", 5, True, False),
    ])
    report = harness.score(preds, ["ref", "rarb-math", "dep"], deploy_lanes=["dep"])
    assert report["anchor_recall"] == pytest.approx(1.0)   # the gate's own recall

    assert report["deploy_recall"] == pytest.approx(1 / 3)
    assert report["deploy_recall_by_lane"]["dep"] == pytest.approx(1 / 3)
    assert report["deploy_lanes_measured"] == 1
    # the abysmal deploy recall must NOT change the verdict
    assert report["passed"] is True, "deploy recall must not gate"
    assert "deploy_recall" not in Gate.ENFORCED


def test_deploy_recall_is_nan_when_no_deploy_lanes_were_sampled(tmp_path):
    """An unmeasured diagnostic reads nan, never a passing 0."""
    from math import isnan

    from relevance_judge.validation import ValidationHarness

    harness = ValidationHarness(RelevanceJudgeConfig(data_dir=tmp_path))
    preds = pd.DataFrame([_pred("ref", 1, True, True), _pred("ref", 2, False, False)])
    report = harness.score(preds, ["ref"])
    assert isnan(report["deploy_recall"])
    assert report["deploy_lanes_measured"] == 0


def test_lane_context_reaches_the_prompt_and_the_hash():
    """A lane card changes what the judge is told, so it MUST change the prompt
    hash — otherwise two lanes' verdicts record identical provenance."""
    from relevance_judge.judge import _prompt_hash
    from relevance_judge.lane_context import LANE_CONTEXT, context_for

    quest = context_for("quest")
    assert "an article simply not mentioning it is NOT a gap" in quest
    assert context_for("no-such-lane") == "", "unknown lanes fall back to no card"

    bare = _prompt_hash("q", "d")
    with_ctx = _prompt_hash("q", "d", quest)
    assert bare != with_ctx, "lane context must be provenanced in the hash"
    assert _prompt_hash("q", "d", context_for("clerc")) != with_ctx

    # crumb-legal-qa is deliberately absent — its gold is unreliable
    assert "crumb-legal-qa" not in LANE_CONTEXT


def test_lane_card_is_sent_in_the_system_turn(tmp_path):
    """The card narrows relevance for the lane; it must arrive alongside the
    universal rules, not replace them."""
    from relevance_judge.judge import INSTRUCTION, RelevanceJudge

    judge = RelevanceJudge(RelevanceJudgeConfig(data_dir=tmp_path))
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return _resp("ASKED: x\nEVIDENCE: y\nMISSING: NOTHING\nVERDICT: yes - ok")

    import relevance_judge.judge as mod
    mod.completion = capture
    try:
        judge.judge_one("q", "d", dataset="quest")
    finally:
        mod.completion = _real_completion

    system = seen["messages"][0]["content"]
    assert INSTRUCTION in system, "universal rules must survive"
    assert "set expression" in system, "lane card must be present"

    seen.clear()
    mod.completion = capture
    try:
        judge.judge_one("q", "d", dataset="unknown-lane")
    finally:
        mod.completion = _real_completion
    assert seen["messages"][0]["content"] == INSTRUCTION, "no card -> bare instruction"


def test_the_contract_does_not_presuppose_a_conjunctive_query():
    """The defect this exists for: `ASKED: the ONE thing` plus `MISSING: whatever
    THE QUERY needs` made a disjunctive query structurally unanswerable — a
    document satisfying one of 'A or B or C' put A and B in MISSING, and
    'MISSING names anything -> no' rejected it. 56 of quest's 167 false
    negatives were exactly this, and no lane card can override a mechanical rule."""
    from relevance_judge.judge import INSTRUCTION

    assert "the one thing the query needs" not in INSTRUCTION
    assert "alternatives" in INSTRUCTION, "ASKED must acknowledge disjunction"
    # MISSING must be judged against ASKED, not against the whole query
    assert "whatever ASKED names" in INSTRUCTION
    assert "an alternative ASKED did not name is not missing" in INSTRUCTION
    # the evidence bar itself must survive unchanged
    assert "MISSING is NOTHING" in INSTRUCTION
    assert "a wrong yes becomes permanent gold" in INSTRUCTION


def _capturing_completion(sink):
    def capture(**kwargs):
        sink.append(kwargs["messages"][0]["content"])
        return _resp("ASKED: x\nEVIDENCE: y\nMISSING: NOTHING\nVERDICT: yes - ok")
    return capture


def test_each_row_is_judged_under_its_OWN_lane_card(tmp_path):
    """A card is per-row context, not a global switch: a quest row must be judged
    as a set expression while a clerc row in the same batch is judged as a
    truncated citation."""
    from relevance_judge.judge import RelevanceJudge

    judge = RelevanceJudge(RelevanceJudgeConfig(data_dir=tmp_path, llm_workers=1))
    sent: list[str] = []
    rows = pd.DataFrame([
        {"dataset": d, "query_id": str(i), "doc_id": str(i), "query": "q", "doc_text": "t"}
        for i, d in enumerate(["quest", "clerc", "crumb-legal-qa"])
    ])
    import relevance_judge.judge as mod
    mod.completion = _capturing_completion(sent)
    try:
        judge.judge_pairs(rows, run_id="r")
    finally:
        mod.completion = _real_completion

    quest, clerc, uncarded = sent
    assert "NOT a gap" in quest
    assert "TRUNCATED at the point" in clerc
    # the lane with no card falls back to the bare universal instruction
    assert "About this collection:" not in uncarded
    assert len({quest, clerc, uncarded}) == 3, "each lane must get its own prompt"


def test_validation_judges_under_the_same_cards_as_banking(tmp_path):
    """The gate must measure the instrument that actually writes gold. If stage 1
    judged without the cards stage 2 uses, the measured precision would describe
    a different judge."""
    from relevance_judge.validation import ValidationHarness

    harness = ValidationHarness(RelevanceJudgeConfig(data_dir=tmp_path, llm_workers=1))
    sent: list[str] = []
    rows = pd.DataFrame([
        {"dataset": d, "query_id": "1", "doc_id": "1", "human_relevant": True,
         "query": "q", "doc_text": "t"}
        for d in ["quest", "beir-touche-2020"]
    ])
    import relevance_judge.judge as mod
    mod.completion = _capturing_completion(sent)
    try:
        harness.judge_rows(rows)
    finally:
        mod.completion = _real_completion

    carded, referee = sent
    assert "NOT a gap" in carded
    assert "About this collection:" not in referee, "referee lanes keep the bare rules"


def test_a_refused_connection_is_retried_not_discarded(tmp_path):
    """The defect this exists for: '[Errno 61] Connection refused' lost 2,115 of
    3,600 pairs in one run. Such a call never reached the model, so it cost
    nothing and the work is purely lost — it must be retried."""
    from relevance_judge.judge import RelevanceJudge

    judge = RelevanceJudge(RelevanceJudgeConfig(
        data_dir=tmp_path, llm_workers=1, connect_backoff_s=0.0))
    calls = {"n": 0}

    def refuse_then_answer(**kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError(
                "litellm.APIError: OpenrouterException - [Errno 61] Connection refused")
        return _resp("ASKED: x\nEVIDENCE: y\nMISSING: NOTHING\nVERDICT: yes - ok")

    import relevance_judge.judge as mod
    mod.completion = refuse_then_answer
    try:
        verdict = judge.judge_one("q", "d")
    finally:
        mod.completion = _real_completion

    assert verdict.relevant is True, "the retry should have recovered the pair"
    assert calls["n"] == 3, f"expected 2 retries then success, got {calls['n']} calls"
    assert judge.dropped["connection_retried"] == 2
    assert "RuntimeError" not in judge.dropped, "a recovered fault is not a drop"


def test_a_rejected_request_is_not_retried(tmp_path):
    """A provider that rejects the REQUEST will reject it again — retrying only
    burns time. Only transport faults are worth a second attempt."""
    from relevance_judge.judge import RelevanceJudge

    judge = RelevanceJudge(RelevanceJudgeConfig(
        data_dir=tmp_path, llm_workers=1, connect_backoff_s=0.0))
    calls = {"n": 0}

    def reject(**kwargs):
        calls["n"] += 1
        raise RuntimeError("BadRequestError: unsupported parameter 'reasoning_effort'")

    import relevance_judge.judge as mod
    mod.completion = reject
    try:
        verdict = judge.judge_one("q", "d")
    finally:
        mod.completion = _real_completion

    assert verdict.relevant is None
    assert calls["n"] == 1, "a request rejection must not be retried"
    assert judge.dropped["RuntimeError"] == 1


def test_connection_retries_are_bounded(tmp_path):
    """A provider that is simply down must not spin forever."""
    from relevance_judge.judge import RelevanceJudge

    cfg = RelevanceJudgeConfig(data_dir=tmp_path, llm_workers=1,
                               connect_retries=3, connect_backoff_s=0.0)
    judge = RelevanceJudge(cfg)
    calls = {"n": 0}

    def always_refuse(**kwargs):
        calls["n"] += 1
        raise RuntimeError("Connection refused")

    import relevance_judge.judge as mod
    mod.completion = always_refuse
    try:
        verdict = judge.judge_one("q", "d")
    finally:
        mod.completion = _real_completion

    assert verdict.relevant is None
    assert calls["n"] == cfg.connect_retries + 1, "bounded by connect_retries"
    assert judge.dropped["connection_retried"] == cfg.connect_retries


def test_quest_card_covers_all_three_set_operators():
    """Red-team finding: the card taught DISJUNCTION only. Recall on negated
    queries was 0.200 vs 0.608 on plain ones, because an article never states
    what a thing is NOT — so the unstated exclusion landed in MISSING. And the
    'any one alternative' rule could over-generalise to 'and', which is a live
    FALSE-POSITIVE channel on a lane with no negative labels."""
    from relevance_judge.lane_context import context_for

    quest = context_for("quest")
    assert "ALTERNATIVES" in quest and "any one is" in quest      # disjunction
    assert "REQUIRED" in quest and "all must hold" in quest       # conjunction
    assert "NEGATION" in quest                                     # negation
    # the operative negation rule: silence is not a gap
    assert "an article simply not mentioning it is NOT a gap" in quest
    # ...but an AFFIRMATIVE membership in the excluded set still is
    assert "put the entity IN" in quest


def test_rarb_card_forbids_the_correctness_check_procedurally():
    """Red-team finding: 'correctness is not the retrieval question' is a
    declarative negative, and a non-reasoning model reverts to its prior that
    solving == getting it right. The card must name what NOT to write."""
    from relevance_judge.lane_context import context_for

    card = context_for("rarb-math")
    assert "Do not check the arithmetic" in card
    assert "Never write 'wrong answer'" in card
    # but it must not license a solution to a DIFFERENT problem
    assert "DIFFERENT problem is still not relevant" in card


def test_universal_contract_no_longer_contradicts_itself_on_paraphrase():
    """Red-team finding, and the largest measured cluster: 'wording need not
    match' fought 'EVIDENCE that describes the document counts as NOTHING', and
    the mechanical rule won — 54.8% of ALL false negatives were the model
    describing the document. The trap must catch topic-naming only."""
    from relevance_judge.judge import INSTRUCTION

    assert "only names the document's topic or genre" in INSTRUCTION
    assert "IS evidence even if its wording differs" in INSTRUCTION
    # the hedge trigger must scope to the judge, not to quoted source text
    assert "Hedging in YOUR OWN assessment" in INSTRUCTION
    assert "appearing inside the document you quote mean nothing" in INSTRUCTION
    # the asymmetry that justifies the whole gate must survive
    assert "a wrong yes becomes permanent gold" in INSTRUCTION

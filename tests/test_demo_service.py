"""Demo runner and API: budget/deadline/guard enforcement, fallback atomicity,
and export round-trips. No live LLM, no network — litellm is stubbed and
pipeline-service is reached through its own TestClient via a requests shim."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import demo_service.runner as runner_mod
from augmentation.config import EngineSettings
from demo_service.config import DemoConfig, SourceDoc
from demo_service.runner import (
    DeadlineExceeded,
    DeadlineGate,
    DemoUnavailable,
    NotEligibleToSave,
    RunRecord,
    SearchTestDemo,
)

DOC_TEXT = (
    "Segmented diamond cup wheel with a reinforced hub for fast, "
    "low-chip material removal. Fits 5 in. concrete planer models "
    "PC5000C and PC5001C."
)
"""Stand-in copy, not the real corpus row: the Kaggle text may not be
committed, and every check here reads shape, not wording."""


@pytest.fixture(scope="module")
def pipeline_client():
    from pipeline_service.api import app as pipeline_app

    with TestClient(pipeline_app) as client:
        yield client


class _PipelineShim:
    """requests-compatible facade routing into pipeline-service's TestClient."""

    def __init__(self, client: TestClient) -> None:
        self._client = client

    def get(self, url: str, **_kw):
        return self._client.get("/" + url.split("/", 3)[-1])

    def post(self, url: str, *, json: dict, **_kw):
        return self._client.post("/" + url.split("/", 3)[-1], json=json)


def _reply(text: str):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
    )


@pytest.fixture()
def demo(tmp_path, monkeypatch, pipeline_client):
    pd.DataFrame(
        [{"doc_id": "d1", "title": "5 in. Segmented Diamond Cup Wheel for PC5000C",
          "text": DOC_TEXT},
         {"doc_id": "d2", "title": "Angle bracket", "text": "A steel angle bracket."}]
    ).to_parquet(tmp_path / "corpus.parquet")
    pd.DataFrame([{"query_id": "r1", "text": "angle bracket"}]).to_parquet(
        tmp_path / "queries.parquet"
    )
    pd.DataFrame([{"query_id": "m1", "text": "existing minted query"}]).to_parquet(
        tmp_path / "minted_queries.parquet"
    )
    monkeypatch.setattr(runner_mod, "requests", _PipelineShim(pipeline_client))
    config = DemoConfig(
        data_dir=tmp_path, replay_dir=tmp_path / "replays",
        export_dir=tmp_path / "demo_tests",
        docs=(
            SourceDoc(doc_id="d1", identifier_surface="PC5000C", identifier_feature="sku"),
            SourceDoc(doc_id="d2"),
        ),
        default_doc_id="d1",
        engine=EngineSettings(max_spend_usd=1.0),
        generate_deadline_s=5.0, retrieve_deadline_s=2.0,
    )
    return SearchTestDemo(config, qdrant_factory=lambda: None)


def _stub_model(monkeypatch, replies: list[str]) -> list[str]:
    calls: list[str] = []

    def fake_completion(*, model, messages, **_kw):
        calls.append(messages[0]["content"])
        return _reply(replies[min(len(calls) - 1, len(replies) - 1)])

    monkeypatch.setattr(runner_mod, "completion", fake_completion)
    return calls


# ---- deadline --------------------------------------------------------------


def test_deadline_gate_discards_a_late_result():
    gate = DeadlineGate(0.05)
    with pytest.raises(DeadlineExceeded):
        gate.run(lambda: time.sleep(0.5) or "late")


def test_an_expired_gate_refuses_before_submitting():
    gate = DeadlineGate(-1)
    with pytest.raises(DeadlineExceeded):
        gate.run(lambda: "never runs")


# ---- budget and call ceiling -------------------------------------------------


def test_budget_refuses_before_the_call_is_made(demo, monkeypatch):
    def must_not_run(**_kw):  # pragma: no cover - the point is it never runs
        raise AssertionError("budget should have refused before the call")

    monkeypatch.setattr(runner_mod, "completion", must_not_run)
    demo._budget.max_usd = 1e-9
    with pytest.raises(DemoUnavailable):
        demo.generate("identifier", feature="exact_id")


def test_budget_is_the_only_call_cap(demo, monkeypatch):
    """SPEC d73a: no call-count ceiling — the budget alone stops runaway calls.
    Generate + repair + repair all succeed under a comfortable budget; a
    depleted budget refuses the next call via reserve()."""
    _stub_model(monkeypatch, ["diamond cup wheel PC5000C grinder"])
    demo.generate("identifier", feature="exact_id")
    demo.check()
    demo.repair()   # no 2-call ceiling any more
    demo.repair()   # would have raised CallLimitExceeded before d73
    demo._budget.max_usd = 1e-9
    with pytest.raises(DemoUnavailable):
        demo.repair()  # budget, not a call count, is the cap


def test_repair_draws_from_the_same_budget(demo, monkeypatch):
    _stub_model(monkeypatch, ["diamond cup wheel PC5000C grinder"])
    demo.generate("identifier", feature="exact_id")
    spent_after_one = demo._budget.spent_usd
    assert spent_after_one > 0
    demo.check()
    demo.repair()
    assert demo._budget.spent_usd > spent_after_one
    assert demo._budget.calls == 2


# ---- guards ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "reason_fragment"),
    [
        ("I cannot write a query for this document.", "refusal"),
        ("existing minted query", "duplicate"),
        ("Segmented diamond cup wheel with a reinforced hub", "verbatim"),
    ],
)
def test_generation_guards_reject_with_a_reason(demo, monkeypatch, reply, reason_fragment):
    _stub_model(monkeypatch, [reply])
    out = demo.generate("identifier", feature="exact_id")
    assert out["guard_reason"] is not None
    assert reason_fragment in out["guard_reason"]
    checks = demo.check()
    assert checks.guard_rejected
    demo.answerable("answers")
    with pytest.raises(NotEligibleToSave):
        demo.save()


# ---- shape checks -------------------------------------------------------------


def test_identifier_behavior_requires_the_exact_surface(demo, monkeypatch):
    _stub_model(monkeypatch, ["diamond cup wheel for concrete planer"])
    demo.generate("identifier", feature="exact_id")
    checks = demo.check()
    assert not checks.shape_passed
    verbatim = [c for c in checks.shape_checks if "verbatim" in c["target"]]
    assert verbatim and not verbatim[0]["passed"]


def test_identifier_behavior_passes_with_the_surface_present(demo, monkeypatch):
    _stub_model(monkeypatch, ["diamond cup wheel PC5000C grinder"])
    demo.generate("identifier", feature="exact_id")
    checks = demo.check()
    assert checks.shape_passed, checks.shape_checks
    assert not checks.guard_rejected


# ---- the three statuses gate saving -------------------------------------------


def test_save_needs_shape_guard_and_answerability(demo, monkeypatch):
    _stub_model(monkeypatch, ["diamond cup wheel PC5000C grinder"])
    demo.generate("identifier", feature="exact_id")
    demo.check()
    with pytest.raises(NotEligibleToSave):  # answerability not yet inspected
        demo.save()
    demo.answerable("does_not_answer")
    with pytest.raises(NotEligibleToSave):  # inspected and found ungrounded
        demo.save()
    demo.answerable("answers")
    record = demo.save()
    assert record.accepted and record.saved_at


def test_export_round_trips_and_rerun_only_touches_retrieval(demo, monkeypatch):
    _stub_model(monkeypatch, ["diamond cup wheel PC5000C grinder"])
    demo.generate("identifier", feature="exact_id")
    demo.check()
    demo.answerable("answers")
    demo.retrieve(strategies={
        "dense_only": lambda q: {"d2": 1.0, "d1": 0.9},
        "sparse_only": lambda q: {"d1": 5.0},
        "pure_rrf": lambda q: {"d9": 1.0},
    })
    saved = demo.save()

    loaded = demo.suite()
    assert [r.query_id for r in loaded] == [saved.query_id]
    by_name = {s.strategy: s for s in loaded[0].retrieval}
    assert by_name["dense_only"].source_rank == 2
    assert by_name["sparse_only"].source_rank == 1
    assert by_name["pure_rrf"].source_rank is None  # absent -> None, never 0

    # rerun-retrieval-only, with injected strategies standing in for Qdrant
    demo._current["checks"] = saved.checks
    demo.retrieve(strategies={"dense_only": lambda q: {"d1": 1.0}})
    assert demo._current["retrieval"][0].source_rank == 1


def test_a_rejected_attempt_is_saved_only_as_rejected(demo, monkeypatch):
    _stub_model(monkeypatch, ["I cannot write a query for this."])
    demo.generate("identifier", feature="exact_id")
    demo.check()
    record = demo.save_rejected()
    files = list(demo.config.export_dir.glob("rejected_*.json"))
    assert len(files) == 1
    assert not record.accepted
    assert demo.suite()[0].query_id == record.query_id


# ---- API: fallback atomicity and contract -------------------------------------


@pytest.fixture()
def api_client(demo, monkeypatch, tmp_path):
    import demo_service.api as api_mod

    monkeypatch.setitem(api_mod._state, "demo", demo)
    monkeypatch.setitem(api_mod._state, "active_behavior_id", None)
    # TestClient without lifespan: _state is injected, and lifespan would
    # build a real runner against real paths.
    return TestClient(api_mod.app)


def _write_replay(demo, behavior_id: str) -> RunRecord:
    bundle = RunRecord(
        mode="replay", behavior_id=behavior_id, collection="home-depot",
        doc_id="d1", query_id="replay-1",
        query_text="prepared replay query PC5000C",
        targets={"spans": [], "stats": []},
        checks={
            "shape_passed": True, "shape_checks": (), "guard_rejected": False,
            "guard_reason": None, "answerable_status": "answers",
        },
        retrieval=[{"strategy": "dense_only", "ranked_doc_ids": ["d1"], "source_rank": 1}],
        fetch_limit=20,
    )
    demo.config.replay_dir.mkdir(parents=True, exist_ok=True)
    (demo.config.replay_dir / f"{behavior_id}.json").write_text(bundle.model_dump_json())
    return bundle


def test_fallback_swaps_the_whole_bundle(api_client, demo, monkeypatch):
    _write_replay(demo, "identifier")

    def refuse(*_a, **_kw):
        raise DemoUnavailable("model key missing")

    monkeypatch.setattr(demo, "generate", refuse)
    body = api_client.post("/run/generate", json={"behavior": "identifier"}).json()
    assert body["mode"] == "replay"
    assert body["fallback_reason"] == "model key missing"
    # query, checks and rankings arrive together, from one bundle
    assert body["query_text"] == "prepared replay query PC5000C"
    assert body["checks"]["shape_passed"] is True
    assert body["retrieval"][0]["source_rank"] == 1


def test_fallback_without_a_prepared_replay_is_503(api_client, demo, monkeypatch):
    def refuse(*_a, **_kw):
        raise DemoUnavailable("down")

    monkeypatch.setattr(demo, "generate", refuse)
    r = api_client.post("/run/generate", json={"behavior": "identifier"})
    assert r.status_code == 503


def test_unknown_behavior_is_422(api_client):
    assert api_client.post("/run/generate", json={"behavior": "nope"}).status_code == 422


def test_unknown_request_field_is_422(api_client):
    r = api_client.post("/run/generate", json={"behavior": "identifier", "extra": 1})
    assert r.status_code == 422


def test_save_before_any_check_is_409(api_client):
    assert api_client.post("/run/save").status_code == 409


def test_check_before_generate_is_409(api_client):
    assert api_client.post("/run/check").status_code == 409


# ---- the document shelf --------------------------------------------------------


def test_source_lists_the_shelf_with_behavior_availability(demo):
    shelf = demo.source()
    docs = {d["doc_id"]: d for d in shelf["documents"]}
    assert set(docs) == {"d1", "d2"}
    assert docs["d1"]["identifier_surface"] == "PC5000C"
    assert set(docs["d1"]["behaviors"]) == {"identifier", "conversational", "real_traffic"}
    # identifier docs offer the id variants first; clean docs offer shapes only
    assert [o["id"] for o in docs["d1"]["structured_options"]] == [
        "exact_id", "wrong_id", "negation", "operator_syntax",
    ]
    assert docs["d2"]["identifier_surface"] is None
    assert [o["id"] for o in docs["d2"]["structured_options"]] == [
        "negation", "operator_syntax",
    ]
    assert shelf["default_doc_id"] == "d1"


def test_structured_on_a_clean_doc_rolls_a_shape_feature(demo, monkeypatch):
    """Without an explicit variant, the SYSTEM rolls the structure; on a clean
    doc that can only be a shape feature, verified by its bank."""
    _stub_model(monkeypatch, ["security bar no drilling"])
    monkeypatch.setattr(demo, "_pick_structured", lambda entry: ("negation",))
    demo.generate("identifier", doc_id="d2")
    checks = demo.check()
    assert checks.shape_passed, checks.shape_checks
    assert any("negation" in c["target"] for c in checks.shape_checks)


def test_the_roll_stays_inside_the_documents_options(demo):
    """The random pick never pairs exact with mistyped, never exceeds two
    structures, never goes empty, and never rolls an id variant on a clean doc."""
    id_doc = demo.config.doc_entry("d1")
    clean = demo.config.doc_entry("d2")
    for _ in range(200):
        roll = demo._pick_structured(id_doc)
        assert 1 <= len(roll) <= 2
        assert not ({"exact_id", "wrong_id"} <= set(roll))
        assert set(roll) <= {"exact_id", "wrong_id", "negation", "operator_syntax"}
    for _ in range(50):
        roll = demo._pick_structured(clean)
        assert roll and set(roll) <= {"negation", "operator_syntax"}


def test_structured_rejects_an_unsupported_variant(demo, monkeypatch):
    def must_not_run(**_kw):  # pragma: no cover
        raise AssertionError("no model call for a refused setup")

    monkeypatch.setattr(runner_mod, "completion", must_not_run)
    with pytest.raises(ValueError, match="does not support"):
        demo.generate("identifier", doc_id="d2", feature="exact_id")


def test_wrong_id_gate_demands_the_mistake_and_forbids_the_truth(demo, monkeypatch):
    from demo_service.runner import mistype

    wrong = mistype("PC5000C", demo.config.corruption_seed)
    _stub_model(monkeypatch, [f"planer wheel {wrong} makita"])
    demo.generate("identifier", doc_id="d1", feature="wrong_id")
    checks = demo.check()
    assert checks.shape_passed, checks.shape_checks
    targets = {c["target"]: c for c in checks.shape_checks}
    assert targets[f"mistyped identifier present: {wrong}"]["passed"]
    assert targets["correct identifier absent: PC5000C"]["passed"]


def test_wrong_id_gate_fails_when_the_model_uses_the_real_id(demo, monkeypatch):
    _stub_model(monkeypatch, ["planer wheel PC5000C makita"])
    demo.generate("identifier", doc_id="d1", feature="wrong_id")
    checks = demo.check()
    assert not checks.shape_passed
    absent_row = next(c for c in checks.shape_checks if "absent" in c["target"])
    assert not absent_row["passed"]


def test_generate_grounds_in_the_chosen_document(demo, monkeypatch):
    calls = _stub_model(monkeypatch, ["a steel bracket for my patio door"])
    demo.generate("conversational", doc_id="d2")
    assert "angle bracket" in calls[0]  # d2's text reached the prompt
    record_doc = demo._current["doc_id"]
    assert record_doc == "d2"


def test_unknown_doc_id_is_422(api_client):
    r = api_client.post(
        "/run/generate", json={"behavior": "identifier", "doc_id": "nope"}
    )
    assert r.status_code == 422


# ---- real traffic (messy) behavior ----------------------------------------------

MESSY_ROWS_PASS = [
    {"target": "measurable damage: typo span, or unknown_token_rate >= 0.15",
     "measured": 1, "passed": True},
    {"target": "ordered language measured: es or de (langid, not asserted)",
     "measured": "es", "passed": True},
]


def test_messy_applies_deterministic_damage(demo, monkeypatch):
    """The model writes clean text; the seeded QwertyTypo damages it — same
    seed, same damage, and the damage happens before the guards run."""
    _stub_model(monkeypatch, ["makita wheel para concreto"])
    monkeypatch.setattr(demo, "_measure_messy", lambda text, clean_text: list(MESSY_ROWS_PASS))
    out = demo.generate("real_traffic")
    assert out["query_text"] != "makita wheel para concreto"  # damaged
    again_seed = demo.config.corruption_seed
    from random import Random

    from augmentation.corruption import QwertyTypo
    assert out["query_text"] == QwertyTypo().apply("makita wheel para concreto", Random(again_seed))


def test_messy_gate_includes_measured_language_and_damage(demo, monkeypatch):
    _stub_model(monkeypatch, ["makita wheel para concreto"])
    monkeypatch.setattr(demo, "_measure_messy", lambda text, clean_text: list(MESSY_ROWS_PASS))
    demo.generate("real_traffic")
    checks = demo.check()
    targets = [c["target"] for c in checks.shape_checks]
    assert any("length_words" in t for t in targets)          # via pipeline verify
    assert any("measurable damage" in t for t in targets)     # runner-measured
    assert any("ordered language" in t for t in targets)
    assert checks.shape_passed


def test_messy_gate_fails_when_language_not_measured(demo, monkeypatch):
    _stub_model(monkeypatch, ["makita wheel replacement"])
    rows = [dict(MESSY_ROWS_PASS[0]),
            {"target": "ordered language measured: es or de (langid, not asserted)",
             "measured": "en, sv", "passed": False}]
    monkeypatch.setattr(demo, "_measure_messy", lambda text, clean_text: rows)
    demo.generate("real_traffic")
    checks = demo.check()
    assert not checks.shape_passed
    demo.answerable("answers")
    with pytest.raises(NotEligibleToSave):
        demo.save()


def test_messy_measured_checks_use_the_real_extractor_if_available(demo, monkeypatch):
    """Integration: the runner's own all-engines extractor sees the typo and
    the language. Skipped where the spaCy/langid models are not installed."""
    try:
        rows = demo._measure_messy("makita whel para concreto", "makita wheel para concreto")
    except Exception:
        pytest.skip("all-engines extractor unavailable in this environment")
    by_target = {r["target"]: r for r in rows}
    damage = by_target["measurable damage: typo span, or unknown_token_rate >= 0.15"]
    language = by_target["ordered language measured: es or de (langid, not asserted)"]
    assert damage["passed"], rows
    assert language["passed"] and "es" in str(language["measured"]), rows


# ---- retaining both outcomes -----------------------------------------------------


def test_save_retains_a_not_answering_verdict_as_rejected(api_client, demo, monkeypatch):
    """'Source does not answer it' is feedback worth keeping: the save returns
    200 with a labeled rejected attempt, never a refusal."""
    _stub_model(monkeypatch, ["diamond cup wheel PC5000C grinder"])
    demo.generate("identifier", feature="exact_id")
    demo.check()
    demo.answerable("does_not_answer")
    body = api_client.post("/run/save").json()
    assert body["accepted"] is False
    assert "not answering" in body["rejected_reason"]
    assert not body["rejected_reason"].lower().startswith("not saved")
    files = list(demo.config.export_dir.glob("rejected_*.json"))
    assert len(files) == 1

    suite = api_client.get("/suite").json()
    assert suite[0]["accepted"] is False


def test_save_accepted_reports_accepted_true(api_client, demo, monkeypatch):
    _stub_model(monkeypatch, ["diamond cup wheel PC5000C grinder"])
    demo.generate("identifier", feature="exact_id")
    demo.check()
    demo.answerable("answers")
    body = api_client.post("/run/save").json()
    assert body["accepted"] is True
    assert body["rejected_reason"] is None

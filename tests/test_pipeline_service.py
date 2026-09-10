"""Contract tests for the generation/augmentation HTTP surface."""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pipeline_service import qrels
from pipeline_service.api import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:      # `with` runs lifespan, which loads the registries
        yield c


def test_health_reports_both_registries(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["tools"] == ["generate_surface", "list_features", "verify"]
    assert "inject" in body["operators"]


def test_tools_expose_their_json_schema(client: TestClient) -> None:
    """The HTTP adapter must publish what tools.py already carries, so a caller
    can drive it without reading our source."""
    tools = {t["name"]: t for t in client.get("/generate/tools").json()}
    assert tools["generate_surface"]["input_schema"]["required"] == ["feature"]
    assert tools["list_features"]["description"]


def test_generate_surface_is_seed_deterministic(client: TestClient) -> None:
    body = {"feature": "structured_identifiers:uuid", "n": 2, "seed": 7}
    first = client.post("/generate/generate_surface", json=body).json()
    again = client.post("/generate/generate_surface", json=body).json()
    assert len(first["surfaces"]) == 2
    assert first["surfaces"] == again["surfaces"]


def test_unknown_tool_is_404_not_500(client: TestClient) -> None:
    r = client.post("/generate/does_not_exist", json={})
    assert r.status_code == 404


def test_unknown_feature_is_422_not_500(client: TestClient) -> None:
    r = client.post("/generate/generate_surface", json={"feature": "nope:nope"})
    assert r.status_code == 422


def test_operators_publish_their_declaration(client: TestClient) -> None:
    decls = {d["operator"]: d for d in client.get("/augment/operators").json()}
    assert "inject" in decls
    assert decls["inject"]["surface_origin"]
    assert "meaning_preserved" in decls["inject"]


@pytest.mark.parametrize(
    ("floor", "operator"),
    [
        ("id:uuid", "inject"),
        ("marker:politeness", "decorate"),
        ("logical:operator_syntax", "operator_syntax_rewrite"),
    ],
)
def test_dispatch_matches_the_operator_registry(
    client: TestClient, floor: str, operator: str
) -> None:
    assert client.get("/augment/dispatch", params={"floor": floor}).json()[
        "operator"
    ] == operator


def test_dispatch_reports_null_for_an_unserved_floor(client: TestClient) -> None:
    assert client.get("/augment/dispatch", params={"floor": "length_words"}).json()[
        "operator"
    ] is None


def test_apply_on_an_unserved_floor_is_422(client: TestClient) -> None:
    r = client.post(
        "/augment/apply", json={"floor": "length_words", "text": "hello"}
    )
    assert r.status_code == 422


PARENT = {
    "query_id": "q1",
    "dataset": "d",
    "query": "reset admin password",
    "floors": [],
    "surfaces": [],
    "bank": "politeness",
}


def test_plan_returns_an_instruction_and_a_postcondition(client: TestClient) -> None:
    """The model-needing path hands the caller the prompt and the check rather
    than calling a model here."""
    body = client.post(
        "/augment/plan",
        json={
            "floor": "marker:politeness",
            "text": "reset admin password",
            "parent": PARENT,
        },
    ).json()
    assert body["operator"] == "decorate"
    assert body["instruction"].strip()
    assert body["targets"]
    assert body["declaration"]["operator"] == "decorate"


def test_apply_says_when_a_model_is_needed(client: TestClient) -> None:
    body = client.post(
        "/augment/apply",
        json={
            "floor": "marker:politeness",
            "text": "reset admin password",
            "parent": PARENT,
        },
    ).json()
    # Deterministic rewrite or an explicit hand-off — never a silent null.
    assert body["needs_model"] is (body["text"] is None)


def test_incomplete_parent_names_the_missing_column(client: TestClient) -> None:
    """Operators read named columns off the parent row. A caller who omits one
    should learn which, not get a 500 out of pandas."""
    r = client.post(
        "/augment/plan",
        json={"floor": "marker:politeness", "text": "x", "parent": {}},
    )
    assert r.status_code == 422
    assert "missing column" in r.json()["detail"]


def test_unknown_request_field_is_rejected(client: TestClient) -> None:
    """extra='forbid': a stale field name fails loudly instead of being ignored,
    which is the whole reason this is typed rather than argparse."""
    r = client.post(
        "/augment/apply",
        json={"floor": "id:uuid", "text": "x", "requirment": "typo"},
    )
    assert r.status_code == 422


# ---- examination -----------------------------------------------------------

SAMPLE_QUERIES = [
    "vector database",
    "please patch CVE-2024-3094",
    "192.168.1.1 timeout",
    "why do cats purr",
    "RTX 4090 vs RTX 4080 benchmark",
    "hello can you help me reset my password",
]


def test_examine_reports_coverage_over_declared_cells(client: TestClient) -> None:
    body = client.post("/examine", json={"queries": SAMPLE_QUERIES}).json()
    assert body["queries"] == len(SAMPLE_QUERIES)
    assert body["cells_declared"] > 0
    assert body["cells_covered"] <= body["cells_declared"]
    assert len(body["coverage"]) == body["cells_declared"]
    # coverage is sorted by rows descending, so the head is the fullest cell
    rows = [c["rows"] for c in body["coverage"]]
    assert rows == sorted(rows, reverse=True)


def test_examine_without_a_floor_emits_no_orders(client: TestClient) -> None:
    body = client.post("/examine", json={"queries": SAMPLE_QUERIES}).json()
    assert body["order_sheet"] == []


def test_a_floor_turns_thin_cells_into_orders(client: TestClient) -> None:
    """The order sheet is the point: a deficit plus the operator that mints it."""
    body = client.post(
        "/examine", json={"queries": SAMPLE_QUERIES, "floor": 5}
    ).json()
    assert body["order_sheet"], "6 queries cannot fill every cell to 5"
    first = body["order_sheet"][0]
    assert first["deficit"] == first["want"] - first["have"]
    assert first["want"] == 5
    deficits = [o["deficit"] for o in body["order_sheet"]]
    assert deficits == sorted(deficits, reverse=True)


def test_order_lines_name_an_operator_the_augment_endpoints_know(
    client: TestClient,
) -> None:
    """An order sheet nobody can act on is a report, not an order."""
    orders = client.post(
        "/examine", json={"queries": SAMPLE_QUERIES, "floor": 5}
    ).json()["order_sheet"]
    known = {d["operator"] for d in client.get("/augment/operators").json()}
    named = {o["operator"] for o in orders if o["operator"]}
    assert named, "no order line names a minting operator"
    assert named <= known


def test_empty_query_list_is_422(client: TestClient) -> None:
    assert client.post("/examine", json={"queries": []}).status_code == 422


def _zip(name: str, frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        payload = io.BytesIO()
        frame.to_parquet(payload)
        archive.writestr(name, payload.getvalue())
    return buffer.getvalue()


def test_archive_upload_profiles_the_lane_layout(client: TestClient) -> None:
    blob = _zip("mylane/queries.parquet", pd.DataFrame({"query": SAMPLE_QUERIES}))
    r = client.post(
        "/examine/archive", files={"file": ("lane.zip", blob, "application/zip")}
    )
    assert r.status_code == 200
    assert r.json()["queries"] == len(SAMPLE_QUERIES)


def test_archive_without_queries_names_what_was_expected(client: TestClient) -> None:
    blob = _zip("mylane/corpus.parquet", pd.DataFrame({"text": ["a doc"]}))
    r = client.post(
        "/examine/archive", files={"file": ("lane.zip", blob, "application/zip")}
    )
    assert r.status_code == 422
    assert "queries.parquet" in r.json()["detail"]


def test_a_queries_file_with_no_text_column_is_422(client: TestClient) -> None:
    blob = _zip("queries.parquet", pd.DataFrame({"qid": [1, 2]}))
    r = client.post(
        "/examine/archive", files={"file": ("lane.zip", blob, "application/zip")}
    )
    assert r.status_code == 422
    assert "column" in r.json()["detail"]


def test_a_non_zip_upload_is_422_not_500(client: TestClient) -> None:
    r = client.post(
        "/examine/archive",
        files={"file": ("lane.zip", b"not a zip at all", "application/zip")},
    )
    assert r.status_code == 422


# ---- qrels expansion -------------------------------------------------------

PAIRS = [
    {"dataset": "d", "query_id": "q1", "doc_id": "doc1",
     "query": "how do I rotate an API key", "doc_text": "To rotate a key, open settings."},
    {"dataset": "d", "query_id": "q2", "doc_id": "doc2",
     "query": "sourdough hydration ratio", "doc_text": "A 70% hydration dough is wetter."},
]


def _judge_body(**over) -> dict:
    body = {"pairs": PAIRS, "run_id": "t1", "max_spend_usd": 1.0}
    body.update(over)
    return body


def test_spending_without_a_ceiling_is_rejected(client: TestClient) -> None:
    """The one endpoint that costs money never defaults its ceiling."""
    r = client.post(
        "/qrels/expand", json={"pairs": PAIRS, "run_id": "t1", "dry_run": True}
    )
    assert r.status_code == 422


def test_a_nonpositive_ceiling_is_rejected(client: TestClient) -> None:
    r = client.post("/qrels/expand", json=_judge_body(max_spend_usd=0))
    assert r.status_code == 422


def test_dry_run_costs_the_work_without_calling(client: TestClient, monkeypatch) -> None:
    """A dry run must not reach a model. If it does, this fails loudly."""
    import relevance_judge.judge as judge_mod

    def explode(*a, **k):  # pragma: no cover - the point is that it never runs
        raise AssertionError("dry run reached the model")

    monkeypatch.setattr(judge_mod, "completion", explode)
    body = client.post("/qrels/expand", json=_judge_body(dry_run=True)).json()
    assert body["submitted"] == 2
    assert body["to_judge"] + body["already_judged"] == 2
    assert body["estimated_usd"] >= 0
    assert body["within_ceiling"] is True


def test_every_response_carries_the_measured_operating_point(
    client: TestClient,
) -> None:
    """98.7% precision at 30.8% recall: a negative is mostly 'not found', not
    'checked and rejected'. That has to travel with the verdicts."""
    body = client.post("/qrels/expand", json=_judge_body(dry_run=True)).json()
    point = body["operating_point"]
    assert point["precision_relevant"] == 0.987
    assert point["recall_relevant"] == 0.308


def test_over_the_batch_cap_is_422_and_says_to_loop(client: TestClient) -> None:
    many = [dict(PAIRS[0], doc_id=f"d{i}") for i in range(qrels.MAX_PAIRS + 1)]
    r = client.post("/qrels/expand", json=_judge_body(pairs=many, dry_run=True))
    assert r.status_code == 422
    assert "loop" in r.json()["detail"]


def test_expansion_banks_verdicts_from_a_stubbed_model(
    client: TestClient, monkeypatch, tmp_path
) -> None:
    """The spending path, with the model stubbed — no network, no charge."""
    import relevance_judge.judge as judge_mod
    from relevance_judge.config import RelevanceJudgeConfig
    from relevance_judge.judge import RelevanceJudge

    class _Msg:
        content = "evidence: the document explains key rotation\nverdict: yes"

    class _Choice:
        message = _Msg()

    class _Usage:
        prompt_tokens, completion_tokens = 100, 10

    class _Reply:
        choices, usage = [_Choice()], _Usage()

    monkeypatch.setattr(judge_mod, "completion", lambda *a, **k: _Reply())
    from pipeline_service.api import _state
    _state["judge"] = RelevanceJudge(RelevanceJudgeConfig(data_dir=tmp_path))
    try:
        body = client.post("/qrels/expand", json=_judge_body()).json()
        assert body["run_id"] == "t1"
        assert body["judged"] == 2
        assert body["spent_usd"] > 0
        assert body["calls"] == 2
        assert body["stopped_on_budget"] is False
    finally:
        _state.pop("judge", None)


# ---- cell-driven augmentation ----------------------------------------------


def test_a_cell_resolves_into_per_requirement_steps(client: TestClient) -> None:
    """This is the handoff /examine's order sheet depends on: a cell NAME in,
    stages and prompts out. /augment/plan takes a floor key and cannot serve it."""
    body = client.post(
        "/augment/cell",
        json={
            "cell": "conversational_courtesy_wrapper",
            "text": "cordless drill 20v",
            "parent": dict(PARENT, dataset="home-depot"),
        },
    ).json()
    assert body["cell"] == "conversational_courtesy_wrapper"
    assert body["steps"], "a declared cell has at least one requirement"
    assert body["actionable"] == sum(1 for s in body["steps"] if s["operator"])
    served = [s for s in body["steps"] if s["operator"]]
    assert served, "this cell is mintable, so at least one step must be served"
    for step in served:
        assert step["instruction"].strip()
        assert step["targets"] is not None
    for step in body["steps"]:
        if step["operator"] is None:
            # An unserved requirement must not pretend to be actionable.
            assert step["instruction"] is None and step["targets"] is None


def test_every_examine_order_line_can_be_planned(client: TestClient) -> None:
    """The loop must actually close: each mintable order line's cell resolves."""
    orders = client.post(
        "/examine", json={"queries": SAMPLE_QUERIES, "floor": 5}
    ).json()["order_sheet"]
    mintable = [o for o in orders if o["operator"]][:5]
    assert mintable
    for line in mintable:
        r = client.post(
            "/augment/cell",
            json={"cell": line["cell"], "text": "reset admin password",
                  "parent": PARENT},
        )
        assert r.status_code == 200, (line["cell"], r.json())
        assert r.json()["actionable"] >= 1


def test_an_unknown_cell_is_422_and_says_where_names_come_from(
    client: TestClient,
) -> None:
    r = client.post(
        "/augment/cell", json={"cell": "no_such_cell", "text": "x"}
    )
    assert r.status_code == 422
    assert "/examine" in r.json()["detail"]

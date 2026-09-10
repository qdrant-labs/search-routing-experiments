"""HTTP surface for the search-test demo: preflight, the five presentation
stages, save/suite, and a bundled static frontend. Every stage endpoint falls
back to a prepared `ReplayBundle` on `DemoUnavailable` — the presenter never
sees a stack trace, only a `mode: "replay"` banner."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from demo_service.config import BEHAVIORS, DemoConfig
from demo_service.runner import (
    DemoUnavailable,
    NotEligibleToSave,
    ReplayBundle,
    SearchTestDemo,
)

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend" / "dist"

_state: dict[str, Any] = {"active_behavior_id": None}


def _demo() -> SearchTestDemo:
    return _state["demo"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Same .env convention as the scripts: DemoConfig reads os.environ, and a
    # bare `uvicorn demo_service.api:app` shell has none of the keys exported.
    load_dotenv(".env") or load_dotenv("../.env")
    demo = SearchTestDemo(DemoConfig())
    demo.preflight()
    _state["demo"] = demo
    yield
    _state.clear()
    _state["active_behavior_id"] = None


app = FastAPI(
    title="demo-service",
    summary="From a collection to inspectable search tests.",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    behavior: str
    doc_id: str | None = None
    """Which shelf document to ground in; None means the pinned default."""
    feature: str | None = None
    """Structured-behavior variant (exact_id / wrong_id / negation /
    operator_syntax); None means the document's first option."""


class AnswerableRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str


def _fallback(reason: str) -> dict:
    behavior_id = _state.get("active_behavior_id")
    if behavior_id is None:
        raise HTTPException(503, f"no replay available and no active behavior: {reason}")
    path = _demo().config.replay_dir / f"{behavior_id}.json"
    if not path.exists():
        raise HTTPException(503, f"no prepared replay for {behavior_id!r}: {reason}")
    bundle = ReplayBundle.model_validate_json(path.read_text())
    bundle = bundle.model_copy(update={"mode": "replay"})
    return {"mode": "replay", "fallback_reason": reason, **bundle.model_dump()}


def _respond(action: Callable[[SearchTestDemo], Any]) -> dict:
    demo = _demo()
    try:
        action(demo)
    except DemoUnavailable as exc:
        return _fallback(str(exc))
    except ValueError as exc:
        # A stage clicked out of order. In replay mode the run state is empty
        # by construction (live stages never ran), so serve the prepared
        # bundle and keep the story moving; live, say plainly what to do.
        if demo.mode == "replay":
            return _fallback(str(exc))
        raise HTTPException(409, str(exc)) from exc
    return {"mode": "live", **demo.current_state()}


@app.get("/health", tags=["ops"])
def health() -> dict:
    demo = _demo()
    return {"mode": demo.mode, "reason": demo.replay_reason}


@app.get("/source", tags=["demo"])
def source() -> dict:
    return _demo().source()


@app.post("/run/generate", tags=["demo"])
def run_generate(req: GenerateRequest) -> dict:
    if req.behavior not in BEHAVIORS:
        raise HTTPException(422, f"unknown behavior {req.behavior!r}; see GET /source")
    try:
        entry = _demo().config.doc_entry(req.doc_id)
    except KeyError:
        raise HTTPException(
            422, f"unknown document {req.doc_id!r}; see GET /source"
        ) from None
    if req.feature is not None:
        options = [o["id"] for o in _demo().structured_options(entry)]
        if req.behavior == "identifier" and req.feature not in options:
            raise HTTPException(
                422,
                f"document {entry.doc_id} does not support {req.feature!r}; "
                f"it offers {options}",
            )
    _state["active_behavior_id"] = req.behavior
    return _respond(lambda d: d.generate(req.behavior, req.doc_id, req.feature))


@app.post("/run/check", tags=["demo"])
def run_check() -> dict:
    return _respond(lambda d: d.check())


@app.post("/run/repair", tags=["demo"])
def run_repair() -> dict:
    return _respond(lambda d: d.repair())


@app.post("/run/answerable", tags=["demo"])
def run_answerable(req: AnswerableRequest) -> dict:
    return _respond(lambda d: d.answerable(req.status))


@app.post("/run/retrieve", tags=["demo"])
def run_retrieve() -> dict:
    return _respond(lambda d: d.retrieve())


@app.post("/run/save", tags=["demo"])
def run_save() -> dict:
    """Retain the current run. An accepted candidate is saved as a reusable
    test; a checked-but-failing one (wrong shape, guard-rejected, or a source
    the presenter marked as not answering) is retained as a labeled rejected
    attempt — that verdict is feedback worth keeping, not a discard."""
    demo = _demo()
    try:
        record = demo.save()
        reason = None
    except NotEligibleToSave as exc:
        try:
            record = demo.save_rejected()
        except NotEligibleToSave as inner:
            raise HTTPException(409, str(inner)) from inner
        reason = str(exc)
    return {**record.model_dump(), "accepted": record.accepted,
            "rejected_reason": reason}


@app.get("/suite", tags=["demo"])
def suite() -> list[dict]:
    return [
        {**r.model_dump(), "accepted": r.accepted} for r in _demo().suite()
    ]


@app.post("/run/record-replay", tags=["rehearsal"])
def run_record_replay() -> dict:
    """Rehearsal-only: freeze the current completed run as the behavior's
    prepared replay. Not part of the timed presentation."""
    try:
        path = _demo().record_replay()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"replay": path.name}


@app.post("/run/reset", tags=["demo"])
def run_reset() -> dict:
    _demo().reset()
    _state["active_behavior_id"] = None
    return {"status": "reset"}


# Declared last: Starlette matches routes in registration order, so every
# explicit path above wins before this catch-all mount is ever tried.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:

    @app.get("/", include_in_schema=False)
    def frontend_missing() -> dict:
        return {
            "detail": "frontend not built",
            "fix": "cd src/demo_service/frontend && npm install && npm run build",
        }

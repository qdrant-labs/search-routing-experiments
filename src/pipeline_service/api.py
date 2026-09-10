"""HTTP surface for query generation and augmentation — the fourth adapter over
`taxonomy_generators.build_tools()`, plus the operator registry. Deterministic work
runs here; anything needing a model returns its instruction and postcondition instead."""

from __future__ import annotations

import os
import zipfile
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from augmentation.config import AugmentationConfig
from augmentation.core import Operator
from augmentation.operators import default_operators, operator_for
from pipeline_service.examine import Examination, examine, queries_from_zip
from pipeline_service import qrels
from pipeline_service.augment import CellPlan, plan_cell
from pipeline_service.qrels import JudgePair, JudgePlan, JudgeResult
from query_taxonomy.features import FeatureExtractor
from taxonomy_generators.tools import ToolSpec, build_tools

# Regex-only by default: the spaCy engines cost a model load per worker and only
# matter when a verify target names a stat signal.
ALL_ENGINES = os.environ.get("PIPELINE_ALL_ENGINES", "").lower() in ("1", "true", "yes")

_state: dict[str, Any] = {}


def _tools() -> dict[str, ToolSpec]:
    return _state["tools"]


def _operators() -> tuple[Operator, ...]:
    return _state["operators"]


def _judge():
    """Built on first use — it reads a verdict bank off disk, and every
    other endpoint works without one."""
    if "judge" not in _state:
        from relevance_judge.judge import RelevanceJudge
        _state["judge"] = RelevanceJudge()
    return _state["judge"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """One extractor and one operator set for the process — both are pure and
    the extractor holds the compiled banks."""
    extractor = FeatureExtractor(engines=None) if ALL_ENGINES else FeatureExtractor()
    _state["extractor"] = extractor
    _state["tools"] = {s.name: s for s in build_tools(extractor=extractor)}
    _state["operators"] = default_operators(AugmentationConfig())
    yield
    _state.clear()


app = FastAPI(
    title="pipeline-service",
    summary="Query generation and augmentation over the taxonomy.",
    lifespan=lifespan,
)


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {
        "status": "ok",
        "tools": sorted(_tools()),
        "operators": [op.declaration.operator for op in _operators()],
        "all_engines": ALL_ENGINES,
    }


# ---- generation ------------------------------------------------------------
# One endpoint over the registry rather than a hand-written route per tool:
# tools.py already carries name, description and JSON schema, and its docstring
# states adapters consume it "without this package knowing which".


@app.get("/generate/tools", tags=["generation"])
def list_tools() -> list[dict]:
    return [
        {"name": s.name, "description": s.description, "input_schema": s.input_schema}
        for s in _tools().values()
    ]


@app.post("/generate/{tool}", tags=["generation"])
def run_tool(tool: str, body: dict[str, Any] | None = None) -> dict:
    spec = _tools().get(tool)
    if spec is None:
        raise HTTPException(404, f"unknown tool {tool!r}; see GET /generate/tools")
    try:
        return spec.run(**(body or {}))
    except TypeError as exc:            # wrong/missing arguments for this tool
        raise HTTPException(422, f"{tool}: {exc}") from exc
    except KeyError as exc:             # unknown feature name reaching the registry
        raise HTTPException(422, f"{tool}: no such feature {exc}") from exc


# ---- examination -----------------------------------------------------------


class ExamineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: list[str] = Field(min_length=1, description="Query text to profile.")
    floor: int = Field(
        default=0, ge=0,
        description="Per-cell target. 0 reports coverage without an order sheet.",
    )


@app.post("/examine", tags=["examination"])
def examine_queries(req: ExamineRequest) -> Examination:
    """Cell coverage over a query set, plus an order sheet for the thin cells.
    Each order line's `cell`/`operator` feeds /augment/plan."""
    try:
        return examine(req.queries, _state["extractor"], floor=req.floor)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/examine/archive", tags=["examination"])
async def examine_archive(file: UploadFile, floor: int = 0) -> Examination:
    """Same, from an uploaded lane archive — any `queries.parquet`/`queries.csv`
    in the zip, so the contract is the `LanePaths` layout already in use."""
    blob = await file.read()
    try:
        return examine(queries_from_zip(blob), _state["extractor"], floor=floor)
    except (ValueError, zipfile.BadZipFile) as exc:
        raise HTTPException(422, str(exc)) from exc


# ---- qrels expansion -------------------------------------------------------
# The only endpoint here that spends money, so `max_spend_usd` carries no
# default: omitting it is a 422, never a silent charge.


class JudgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pairs: list[JudgePair] = Field(min_length=1)
    run_id: str = Field(min_length=1, description="Groups a run in the bank.")
    max_spend_usd: float = Field(gt=0, description="Hard ceiling. Required.")
    dry_run: bool = Field(
        default=False, description="Cost the work and return without calling."
    )


@app.post("/qrels/expand", tags=["qrels"])
def expand_qrels(req: JudgeRequest) -> JudgePlan | JudgeResult:
    """Judge (query, document) pairs the answer key never covered.

    Read `operating_point` on the response before acting on a negative:
    `relevant=false` mostly means the judge did not find support, not that it
    checked and rejected."""
    judge = _judge()
    try:
        if req.dry_run:
            return qrels.plan(req.pairs, judge, max_spend_usd=req.max_spend_usd)
        return qrels.expand(
            req.pairs, judge, run_id=req.run_id, max_spend_usd=req.max_spend_usd
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


# ---- augmentation ----------------------------------------------------------


class AugmentRequest(BaseModel):
    """`requirement` (cell AxisBands) is deliberately not exposed — floor-driven
    calls cover the current surface and it would drag composition.cells in."""

    model_config = ConfigDict(extra="forbid")

    floor: str = Field(description="Floor key, e.g. 'id:uuid' or a stat-axis key.")
    text: str = Field(description="Working text; a cell's calls feed each other.")
    parent: dict[str, Any] = Field(
        default_factory=dict, description="The parent selection row."
    )


@app.get("/augment/operators", tags=["augmentation"])
def list_operators() -> list[dict]:
    return [op.declaration.model_dump() for op in _operators()]


@app.get("/augment/dispatch", tags=["augmentation"])
def dispatch(floor: str) -> dict:
    op = operator_for(floor, _operators())
    return {"floor": floor, "operator": op.declaration.operator if op else None}


def _resolve(floor: str) -> Operator:
    op = operator_for(floor, _operators())
    if op is None:
        raise HTTPException(422, f"no operator serves floor {floor!r}")
    return op


@contextmanager
def _parent_columns(operator: str):
    """Operators read named columns off the parent row (`query`, `surfaces`,
    `bank`, ...). An incomplete parent is a caller error, so name the missing
    column rather than surfacing pandas' bare KeyError as a 500."""
    try:
        yield
    except KeyError as exc:
        raise HTTPException(
            422, f"{operator}: parent row is missing column {exc}"
        ) from exc


@app.post("/augment/apply", tags=["augmentation"])
def apply(req: AugmentRequest) -> dict:
    """Deterministic rewrite. `text: null` means this floor needs a model —
    call /augment/plan for the instruction and postcondition."""
    op = _resolve(req.floor)
    with _parent_columns(op.declaration.operator):
        rewritten = op.apply(pd.Series(req.parent), req.floor, req.text)
    return {
        "operator": op.declaration.operator,
        "text": rewritten,
        "needs_model": rewritten is None,
    }


@app.post("/augment/plan", tags=["augmentation"])
def plan(req: AugmentRequest) -> dict:
    """What a model would be told, and what its output must satisfy. The caller
    runs its own model, then posts the result to /augment/check."""
    op = _resolve(req.floor)
    parent = pd.Series(req.parent)
    with _parent_columns(op.declaration.operator):
        return {
            "operator": op.declaration.operator,
            "declaration": op.declaration.model_dump(),
            "instruction": op.instruction(req.floor, parent),
            "targets": op.targets(req.floor, parent).model_dump(),
        }


class CellRequest(BaseModel):
    """Cell-driven planning. Takes a cell NAME from /examine's order sheet, so
    the caller never constructs an AxisBand."""

    model_config = ConfigDict(extra="forbid")

    cell: str = Field(description="Cell name, as reported by /examine.")
    text: str = Field(description="Working text — the parent query.")
    parent: dict[str, Any] = Field(default_factory=dict)


@app.post("/augment/cell", tags=["augmentation"])
def augment_cell(req: CellRequest) -> CellPlan:
    """Plan every requirement of one cell: which stage serves it, which operator
    mints it, and the instruction plus postcondition for the ones that can be.

    This is what an /examine order line feeds into — /augment/plan takes a floor
    key, and a cell reaches its operators through bands instead."""
    op = _operators()
    with _parent_columns(req.cell):
        try:
            return plan_cell(req.cell, req.text, req.parent, op)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


class CheckRequest(AugmentRequest):
    pass


@app.post("/augment/check", tags=["augmentation"])
def check(req: CheckRequest) -> dict:
    """Re-measure a candidate against the operator's postcondition and its
    parent-relative structural checks."""
    op = _resolve(req.floor)
    parent = pd.Series(req.parent)
    with _parent_columns(op.declaration.operator):
        targets = op.targets(req.floor, parent)
        structural = op.structural(parent, req.text, targets)
    report = _tools()["verify"].run(text=req.text, targets=targets.model_dump())
    return {
        "operator": op.declaration.operator,
        "verify": report,
        "structural_failures": structural,
        "passed": bool(report.get("passed")) and not structural,
    }

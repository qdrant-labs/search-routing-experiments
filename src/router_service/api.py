"""FastAPI serving for the encoder router. Configure the arm and hedge via env vars:
ROUTER_DIR (path to a saved arm dir), RRF_DELTA (near-tie -> pure_rrf threshold)."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from router_service.serve import DEFAULT_RRF_DELTA, SavedRouter

ROUTER_DIR = os.environ.get(
    "ROUTER_DIR", "src/data/encoder_router/classifiers_union_200k/no_branches"
)
RRF_DELTA = float(os.environ.get("RRF_DELTA", DEFAULT_RRF_DELTA))

_state: dict[str, object] = {"dir": ROUTER_DIR}   # live arm dir; /reload may repoint it


def _weights_mtime(router_dir: str) -> str | None:
    """UTC mtime of the model file — the definitive 'which version is live' signal."""
    for name in ("router.pt", "lgbm.joblib"):
        f = Path(router_dir) / name
        if f.exists():
            return datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    return None


def _version() -> dict:
    r = _state.get("router")
    meta = getattr(r, "meta", {}) or {}
    router_dir = _state["dir"]
    return {"arm": meta.get("arm"), "holdout_lane": meta.get("holdout_lane"),
            "trained_at": meta.get("trained_at"), "weights_mtime_utc": _weights_mtime(router_dir),
            "router_dir": router_dir, "rrf_delta": RRF_DELTA}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["router"] = SavedRouter.load(_state["dir"])   # warm-load once at startup
    print(f"[router] loaded {_version()}", flush=True)  # log the live version at boot
    yield
    _state.clear()


app = FastAPI(title="Encoder Router", lifespan=lifespan)


class ClassifyRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1)
    delta: float | None = Field(None, description="override RRF_DELTA for this request")


@app.get("/health")
def health() -> dict:
    if _state.get("router") is None:
        raise HTTPException(503, "router not loaded")
    return {"status": "ok", **_version()}


@app.post("/reload")
def reload(router_dir: str | None = None) -> dict:
    """Re-read weights from disk without a restart — call after retraining.
    Pass ?router_dir=... to repoint the live service at a different saved arm."""
    if router_dir is not None:
        _state["dir"] = router_dir
    _state["router"] = SavedRouter.load(_state["dir"])
    v = _version()
    print(f"[router] reloaded {v}", flush=True)
    return {"status": "reloaded", **v}


@app.post("/classify")
def classify(req: ClassifyRequest) -> dict:
    r = _state.get("router")
    if r is None:
        raise HTTPException(503, "router not loaded")
    delta = req.delta if req.delta is not None else RRF_DELTA
    return {"arm": r.meta.get("arm"), "trained_at": r.meta.get("trained_at"),
            "results": r.classify(req.queries, delta)}


@app.post("/classify/pure")
def classify_pure(req: ClassifyRequest) -> list[str]:
    """Routes in request order, no wrapper — delegates so the two endpoints
    cannot disagree about routing."""
    return [result["route"] for result in classify(req)["results"]]

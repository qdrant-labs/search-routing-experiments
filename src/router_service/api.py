"""FastAPI serving for the encoder router. Configure the arm and hedge via env vars:
ROUTER_DIR (path to a saved arm dir), RRF_DELTA (near-tie -> pure_rrf threshold).
The arm is fixed at startup unless ROUTER_ALLOW_RELOAD is set, which exposes POST
/reload to hot-swap arms in the model library (dev/staging); prod leaves it off."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from router_service.serve import DEFAULT_ARM_DIR, DEFAULT_RRF_DELTA, SavedRouter

ROUTER_DIR = os.environ.get("ROUTER_DIR", str(DEFAULT_ARM_DIR))
RRF_DELTA = float(os.environ.get("RRF_DELTA", DEFAULT_RRF_DELTA))
# Runtime arm-swap (dev/staging): off by default so prod stays "arm fixed at startup".
ARM_LIBRARY = DEFAULT_ARM_DIR.parent
ALLOW_RELOAD = os.environ.get("ROUTER_ALLOW_RELOAD", "").lower() in ("1", "true", "yes")

_state: dict[str, object] = {}   # holds the one router, loaded at startup


def _weights_mtime(router_dir: str) -> str | None:
    """UTC mtime of the model file — the definitive 'which version is live' signal."""
    for name in ("router.pt", "lgbm.joblib"):
        f = Path(router_dir) / name
        if f.exists():
            return datetime.fromtimestamp(f.stat().st_mtime, UTC).isoformat(timespec="seconds")
    return None


def _version() -> dict:
    meta = getattr(_state.get("router"), "meta", {}) or {}
    return {"arm": meta.get("arm"), "holdout_lane": meta.get("holdout_lane"),
            "trained_at": meta.get("trained_at"), "weights_mtime_utc": _weights_mtime(ROUTER_DIR),
            "router_dir": ROUTER_DIR, "rrf_delta": RRF_DELTA}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["router"] = SavedRouter.load(ROUTER_DIR)   # warm-load once at startup
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


class ReloadRequest(BaseModel):
    arm: str = Field(..., description="arm name in the model library, e.g. 'zipf_shape_nocorpus'")


@app.post("/reload")
def reload_arm(req: ReloadRequest) -> dict:
    """Hot-swap the served arm without restart (dev/staging). Off unless
    ROUTER_ALLOW_RELOAD is set; only arms inside the model library may load."""
    if not ALLOW_RELOAD:
        raise HTTPException(403, "reload disabled (set ROUTER_ALLOW_RELOAD=1 to enable)")
    target = (ARM_LIBRARY / req.arm).resolve()
    if ARM_LIBRARY.resolve() not in target.parents or not (target / "meta.json").exists():
        raise HTTPException(404, f"no arm {req.arm!r} in {ARM_LIBRARY}")
    try:
        router = SavedRouter.load(target)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    global ROUTER_DIR
    _state["router"] = router          # atomic reference swap; in-flight requests keep the old arm
    ROUTER_DIR = str(target)
    print(f"[router] reloaded {_version()}", flush=True)
    return {"status": "reloaded", **_version()}

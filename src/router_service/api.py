"""FastAPI serving for the encoder router. Configure the arm and hedge via env vars:
ROUTER_DIR (path to a saved arm dir), RRF_DELTA (near-tie -> pure_rrf threshold).
The arm is fixed at startup unless ROUTER_ALLOW_RELOAD is set, which exposes POST
/reload to hot-swap arms in the model library (dev/staging); prod leaves it off."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from router_service.serve import (
    DEFAULT_ARM_DIR,
    DEFAULT_RRF_DELTA,
    SavedRouter,
    zipf_classify,
)

ROUTER_DIR = os.environ.get("ROUTER_DIR", str(DEFAULT_ARM_DIR))
RRF_DELTA = float(os.environ.get("RRF_DELTA", DEFAULT_RRF_DELTA))
# Runtime arm-swap (dev/staging): off by default so prod stays "arm fixed at startup".
ARM_LIBRARY = DEFAULT_ARM_DIR.parent
ALLOW_RELOAD = os.environ.get("ROUTER_ALLOW_RELOAD", "").lower() in ("1", "true", "yes")

_state: dict[str, object] = {}   # holds the one router, loaded at startup

ARMS = (
    "zipf_shape_nocorpus",
    "zipf_input_nocorpus",
    "no_branches",
    "design_branches",
    "shuffled_targets",
    "features_input_nocorpus",
    "lightgbm_nocorpus",
)


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


app = FastAPI(
    title="Encoder Router",
    version="1.0",
    summary="Routes queries to dense, sparse, or RRF search.",
    description="Send queries to classify. Close dense and sparse scores use the RRF route.",
    lifespan=lifespan,
)


class ClassifyRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1, examples=[["why do cats purr", "CVE-2021-44228"]])
    delta: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Optional near-tie threshold. Defaults to the service setting.",
    )


class ClassifyResult(BaseModel):
    query: str
    route: str = Field(..., examples=["dense_only", "sparse_only", "pure_rrf"])
    p_dense: float = Field(..., ge=0.0, le=1.0)
    p_sparse: float = Field(..., ge=0.0, le=1.0)


class ClassifyResponse(BaseModel):
    arm: str | None
    trained_at: str | None = None
    results: list[ClassifyResult]


class HealthResponse(BaseModel):
    status: str
    arm: str | None = None
    holdout_lane: str | None = None
    trained_at: str | None = None
    weights_mtime_utc: str | None = None
    router_dir: str
    rrf_delta: float


class ArmInfo(BaseModel):
    name: str
    serve_safe: bool = Field(..., description="Only serve_safe arms can be loaded via /reload.")
    trained_at: str | None = None


class ArmsResponse(BaseModel):
    live: str | None = Field(None, description="Arm currently loaded and serving /classify.")
    reload_enabled: bool
    arms: list[ArmInfo] = Field(..., description="Model arms in the library.")
    direct: list[str] = Field(..., description="Model-free heuristics, always at POST /classify_direct.")


@app.get("/health", tags=["ops"], response_model=HealthResponse,
         summary="Liveness + which arm is live")
def health() -> dict:
    if _state.get("router") is None:
        raise HTTPException(503, "router not loaded")
    return {"status": "ok", **_version()}


@app.get("/arms", tags=["ops"], response_model=ArmsResponse,
         summary="Every routable arm: model arms in the library plus the always-on direct heuristic.")
def arms() -> dict:
    library = []
    if ARM_LIBRARY.exists():
        for p in sorted(ARM_LIBRARY.iterdir()):
            mp = p / "meta.json"
            if not mp.exists():
                continue
            try:
                m = json.loads(mp.read_text())
            except (OSError, ValueError):
                continue
            library.append({"name": p.name,
                            "serve_safe": bool(m.get("serve_safe", not m.get("feature_inputs"))),
                            "trained_at": m.get("trained_at")})
    live = (getattr(_state.get("router"), "meta", {}) or {}).get("arm")
    return {"live": live, "reload_enabled": ALLOW_RELOAD, "arms": library, "direct": ["zipf_direct"]}


@app.post(
    "/classify", tags=["route"], response_model=ClassifyResponse,
    summary="Return a route and scores for each query.",
)
def classify(req: ClassifyRequest) -> dict:
    r = _state.get("router")
    if r is None:
        raise HTTPException(503, "router not loaded")
    delta = req.delta if req.delta is not None else RRF_DELTA
    return {"arm": r.meta.get("arm"), "trained_at": r.meta.get("trained_at"),
            "results": r.classify(req.queries, delta)}


@app.post(
    "/classify/pure", tags=["route"], response_model=list[str],
    summary="Return only routes, in request order.",
)
def classify_pure(req: ClassifyRequest) -> list[str]:
    return [result["route"] for result in classify(req)["results"]]


@app.post(
    "/classify_direct", tags=["route"], response_model=ClassifyResponse,
    summary="Direct heuristic route — no model, always available.",
    description="Routes by query term rarity alone (wordfreq): rare/OOV -> sparse, all-common -> dense.",
)
def classify_direct(req: ClassifyRequest) -> dict:
    return {"arm": "zipf_direct", "trained_at": None, "results": zipf_classify(req.queries)}


@app.post(
    "/classify_direct/pure", tags=["route"], response_model=list[str],
    summary="Direct heuristic routes only, in request order.",
)
def classify_direct_pure(req: ClassifyRequest) -> list[str]:
    return [result["route"] for result in zipf_classify(req.queries)]


class ReloadRequest(BaseModel):
    arm: str = Field(
        ..., examples=["zipf_shape_nocorpus"],
        description=(
            "Arm to load. Available: " + ", ".join(ARMS) + "."
        ),
    )


@app.post(
    "/reload", tags=["ops"],
    summary="Load another model arm without a restart.",
    description=(
        "Reload may be disabled by the environment. In that case, it returns 403."
    ),
)
def reload_arm(req: ReloadRequest) -> dict:
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

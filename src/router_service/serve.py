"""Serving surface for a saved encoder-router arm: load weights + SVD basis + thresholds and
route raw queries (tuned thresholds with a pure_rrf hedge on near-ties). Serve-safe arms only."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from encoder_router.model import EncoderRouter
from encoder_router.table import (
    HEAD_ROUTES,
    NgramSvd,
    ZipfStats,
    serve_from_probabilities,
)

DEFAULT_RRF_DELTA = 0.15
"""|p_dense - p_sparse| below this -> serve pure_rrf (hedge overlaid on threshold serving)."""


class SavedRouter:
    """A loaded arm: raw query text -> route. The exact serving path the notebook uses."""

    def __init__(self, meta, svd, thresholds, prob_fn, encoder, prefix, zstat) -> None:
        self.meta = meta
        self._svd = svd
        self._thr = thresholds
        self._prob = prob_fn
        self._enc = encoder
        self._prefix = prefix
        self._zstat = zstat

    @classmethod
    def load(cls, arm_dir: str | Path) -> "SavedRouter":
        from sentence_transformers import SentenceTransformer

        p = Path(arm_dir)
        meta = json.loads((p / "meta.json").read_text())
        if not meta.get("serve_safe", not meta.get("feature_inputs")):
            raise ValueError(
                f"{p.name} needs the taxonomy extractor at inference (serve_safe=False); "
                "serve-safe arms only: no_branches / shuffled_targets / zipf_input"
            )
        svd = NgramSvd.load(p / "svd.joblib")
        thresholds = np.load(p / "thresholds.npy")
        if meta["learner"] == "lgbm":
            models = joblib.load(p / "lgbm.joblib")
            prob = lambda x: pd.DataFrame(  # noqa: E731
                np.column_stack([m.predict_proba(x)[:, 1] for m in models]), columns=HEAD_ROUTES
            )
        else:
            prob = EncoderRouter.load(p / "router.pt").probabilities
        encoder = SentenceTransformer(meta["embedding_model"])
        zstat = (
            (np.load(p / "zipf_mean.npy"), np.load(p / "zipf_std.npy"))
            if meta.get("zipf_inputs") else None
        )
        return cls(meta, svd, thresholds, prob, encoder, meta.get("prefix", ""), zstat)

    def _inputs(self, queries: list[str]) -> np.ndarray:
        blocks = [
            np.asarray(self._enc.encode(
                [self._prefix + q for q in queries], normalize_embeddings=True
            )),
            self._svd.transform(pd.Series(queries)),
        ]
        if self._zstat is not None:
            z = ZipfStats().frame(pd.Series(queries)).to_numpy(np.float32)
            blocks.append((z - self._zstat[0]) / self._zstat[1])
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def classify(self, queries: str | list[str], delta: float = DEFAULT_RRF_DELTA) -> list[dict]:
        qs = [queries] if isinstance(queries, str) else list(queries)
        if not qs:
            return []
        probs = self._prob(self._inputs(qs))
        d = probs["dense_only"].to_numpy()
        s = probs["sparse_only"].to_numpy()
        base = np.asarray(serve_from_probabilities(probs, self._thr))
        route = np.where(np.abs(d - s) < delta, "pure_rrf", base)
        return [
            {"query": q, "route": str(route[i]),
             "p_dense": float(d[i]), "p_sparse": float(s[i])}
            for i, q in enumerate(qs)
        ]


if __name__ == "__main__":  # runnable check against a saved arm
    import sys

    arm = sys.argv[1] if len(sys.argv) > 1 else (
        "src/data/encoder_router/classifiers_union_200k/no_branches"
    )
    r = SavedRouter.load(arm)
    out = r.classify(["CVE-2021-44228 log4j remote code execution", "why do cats purr"])
    assert {o["route"] for o in out} <= {"dense_only", "sparse_only", "pure_rrf"}
    assert all(0.0 <= o["p_dense"] <= 1.0 for o in out)
    print(f"[{r.meta['arm']}] ok:", out)

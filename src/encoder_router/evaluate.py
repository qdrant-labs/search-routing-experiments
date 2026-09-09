"""Leave-one-lane-out evaluation shared by every arm: each fold fits its
lexical SVD, z-scoring, and outcome rates on training rows only, trains the
arm's learner, and reads out serve agreement and captured score on the held
lane. The fine-tuned-encoder ceiling arm is notebook-driven and not here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# lightgbm must load before torch — the reversed order segfaults on macOS
# (two OpenMP runtimes; import order decides which second init crashes).
import lightgbm  # noqa: F401
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from encoder_router.table import (
    HEAD_ROUTES,
    HEDGE,
    PRIORITY,
    ROUTES,
    LexicalShape,
    NgramSvd,
    ZipfStats,
    serve_from_probabilities,
    serve_indices,
)
from encoder_router.training import QueryEmbeddings, TrainingTable
from encoder_router.targets import OUT_DIR
from hybrid_search_rrf_dataset.objective import RouterObjective

THRESHOLD_GRID = np.arange(0.30, 0.91, 0.05)
TIE_WEIGHT = 0.25
VAL_LANE_SHARE = 0.1
COST_STEP = RouterObjective().ndcg_weight
"""Placeholder serving-cost exchange rate, pending a real number from
production economics: the serve rule's own tolerance says a route up to
`ndcg_weight` worse is acceptable when cheaper, so one cost rank is priced
at exactly that much score. Removing it is triple-measured: the tuner then
converges to whichever constant the serving rule leaves reachable."""


def tuned_thresholds(
    probs: pd.DataFrame, frame: pd.DataFrame, cost_step: float = COST_STEP
) -> np.ndarray:
    """Head thresholds (sparse, dense), coordinate-ascended on routes_differ
    rows to maximize cost-adjusted captured score — the objective whose
    argmax reproduces the serve oracle on 100% of differ rows. `cost_step=0`
    prices every route equally: a cost-free quality-only tuning."""
    differ = (
        (frame["shape"] == "routes_differ") & frame["serve"].notna()
    ).to_numpy()
    thresholds = np.full(len(HEAD_ROUTES), 0.5)
    if not differ.any():
        return thresholds
    choices = [*HEAD_ROUTES, HEDGE]
    ordered = probs[list(HEAD_ROUTES)].to_numpy()[differ]
    rewards = (
        frame[[f"score_{r}" for r in choices]].to_numpy()[differ]
        - cost_step * np.array([PRIORITY.index(r) for r in choices])
    )
    rows = np.arange(len(ordered))
    best = -np.inf
    for _ in range(2):
        for head in range(len(HEAD_ROUTES)):
            for t in THRESHOLD_GRID:
                trial = thresholds.copy()
                trial[head] = float(t)
                reward = rewards[rows, serve_indices(ordered, trial)].mean()
                if reward > best + 1e-9:
                    best, thresholds = float(reward), trial
    return thresholds

BGE = "BAAI/bge-small-en-v1.5"
E5 = "intfloat/multilingual-e5-small"
GEMINI = "openrouter/google/gemini-embedding-001"
"""Hosted (dim 3072) via OpenRouter — QueryEmbeddings gates the paid call behind
ROUTER_EMBED_LIVE=1. The complement matching the cascade's l2 dense leg."""


@dataclass(frozen=True)
class Arm:
    """One experiment configuration over the shared table and harness."""

    name: str
    feature_inputs: bool = False
    zipf_inputs: bool = False
    shape_inputs: bool = False
    svd_inputs: bool = True
    cell_branch: bool = True
    corpus_branch: bool = True
    feature_branch: bool = False
    shuffle_targets: bool = False
    decisive_only: bool = False
    learner: str = "mlp"
    embedding_model: str = BGE
    aux_embedding_model: str | None = None
    """A second, complementary query encoder concatenated onto the primary —
    off when None. `[bge ; aux]`, not a swap: the router keeps bge and gains a
    view where bge is weak."""


ARMS: tuple[Arm, ...] = (
    Arm("design"),
    Arm("features_as_input", feature_inputs=True, cell_branch=False),
    Arm("no_branches", cell_branch=False, corpus_branch=False),
    Arm("shuffled_targets", shuffle_targets=True),
    Arm("lightgbm", feature_inputs=True, learner="lgbm",
        cell_branch=False, corpus_branch=False),
    Arm("e5_control", embedding_model=E5),
    # query-local rarity as input (serve-safe), and taxonomy features as a
    # third privileged branch's TARGETS — the serve-safe form of
    # features_as_input, which needs the extractor at inference.
    Arm("zipf_channel", zipf_inputs=True),
    Arm("zipf_input_nocorpus", zipf_inputs=True,
        cell_branch=False, corpus_branch=False),
    Arm("zipf_shape_nocorpus", zipf_inputs=True, shape_inputs=True,
        cell_branch=False, corpus_branch=False),
    # complementary second encoder (free to test: e5 is already cached). Gemini
    # plugs into the same aux slot once QueryEmbeddings speaks it.
    Arm("bge_plus_e5_nocorpus", aux_embedding_model=E5,
        cell_branch=False, corpus_branch=False),
    Arm("bge_plus_gemini_nocorpus", aux_embedding_model=GEMINI,
        cell_branch=False, corpus_branch=False),
    Arm("feature_branch", feature_branch=True),
)


def _zscore(train: pd.DataFrame, full: pd.DataFrame) -> np.ndarray:
    mean = train.mean()
    std = train.std().replace(0.0, 1.0).fillna(1.0)
    return ((full - mean) / std).to_numpy(dtype=np.float32)


class LaneCV:
    """Runs one arm across every leave-one-lane-out fold and reports the
    per-lane readout."""

    def __init__(
        self,
        table: TrainingTable,
        *,
        seed: int = 0,
        tie_weight: float = TIE_WEIGHT,
    ) -> None:
        self.table = table
        self.seed = seed
        self.tie_weight = tie_weight

    def row_weights(self, arm: Arm) -> np.ndarray:
        """Per-row route-loss weight. `decisive_only` zeroes every
        non-decisive row rather than dropping it: the route loss normalizes by
        the weight sum, so a zero row is absent from the objective while the
        matrices stay aligned to the frame."""
        frame = self.table.frame
        if arm.decisive_only:
            decisive = (frame["shape"] == "routes_differ") & frame["serve"].notna()
            return decisive.to_numpy().astype(np.float32)
        return np.where(
            frame["shape"].to_numpy() == "all_tied", self.tie_weight, 1.0
        ).astype(np.float32)

    def run(self, arm: Arm, *, lanes: tuple[str, ...] | None = None) -> pd.DataFrame:
        frame = self.table.frame
        embeddings = QueryEmbeddings(arm.embedding_model).matrix(frame)
        if arm.aux_embedding_model:
            aux = QueryEmbeddings(arm.aux_embedding_model).matrix(frame)
            embeddings = np.concatenate([embeddings, aux], axis=1)
        held = lanes or tuple(sorted(frame["dataset"].unique()))
        rows = [
            self._fold(arm, lane, frame, embeddings)
            for lane in tqdm(held, desc=arm.name)
        ]
        return pd.DataFrame(rows)

    def _fold(
        self, arm: Arm, lane: str, frame: pd.DataFrame, embeddings: np.ndarray
    ) -> dict[str, float | str]:
        train = (frame["dataset"] != lane).to_numpy()
        x = self._inputs(arm, frame, embeddings, train)
        route = self.table.route_targets().to_numpy(dtype=np.float32)
        served, thresholds, fit_info = self._served(arm, x, route, train, frame)
        return self._readout(arm, lane, frame[~train], served, thresholds, fit_info)

    def _inputs(
        self, arm: Arm, frame, embeddings: np.ndarray, train: np.ndarray
    ) -> np.ndarray:
        blocks = [embeddings]
        if arm.svd_inputs:
            svd = NgramSvd(seed=self.seed).fit(frame.loc[train, "query"])
            blocks.append(svd.transform(frame["query"]))
        if arm.feature_inputs:
            features = self.table.feature_matrix
            blocks.append(_zscore(features[train], features))
        if arm.zipf_inputs:
            zipf = ZipfStats().frame(frame["query"])
            blocks.append(_zscore(zipf[train], zipf))
        if arm.shape_inputs:
            shape = LexicalShape().frame(frame["query"])
            blocks.append(_zscore(shape[train], shape))
        return np.concatenate(blocks, axis=1)

    def _targets(self, arm: Arm, train: np.ndarray):
        cell = (
            self.table.cell_targets.to_numpy(dtype=np.float32)
            if arm.cell_branch else None
        )
        corpus = None
        if arm.corpus_branch:
            joined = pd.concat(
                [self.table.corpus_targets(),
                 self.table.outcome_rates(train)],
                axis=1,
            )
            corpus = _zscore(joined[train], joined)
        feature = None
        if arm.feature_branch:
            features = self.table.feature_matrix
            feature = _zscore(features[train], features)
        if arm.shuffle_targets:
            # permute ONLY among training rows: held-out-lane targets must not land in
            # training positions (the caller fits on a subset of `train`).
            rng = np.random.default_rng(self.seed)
            idx = np.flatnonzero(train)
            src = idx[rng.permutation(len(idx))]

            def _shuf(a):
                if a is None:
                    return None
                b = a.copy()
                b[idx] = a[src]
                return b

            cell, corpus, feature = _shuf(cell), _shuf(corpus), _shuf(feature)
        return cell, corpus, feature

    def _served(
        self, arm, x, route, train, frame
    ) -> tuple[list[str], np.ndarray, dict[str, float]]:
        weights = self.row_weights(arm)
        probe = self._lgbm_probs if arm.learner == "lgbm" else self._mlp_probs
        probs_train, probs_test, fit_info = probe(arm, x, route, train, weights)
        thresholds = tuned_thresholds(probs_train, frame[train])
        return (
            serve_from_probabilities(probs_test, thresholds),
            thresholds,
            fit_info,
        )

    def _fit_val_split(
        self, train: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Whole LANES held out of the train pool for early stopping — never
        the held lane, and never a random row slice: outcome shape is
        lane-bound, so a within-lane slice stops the fit on an axis the
        readout does not measure."""
        lanes = self.table.frame["dataset"].to_numpy()
        pool = np.unique(lanes[train])
        rng = np.random.default_rng(self.seed)
        held = rng.choice(
            pool, max(int(len(pool) * VAL_LANE_SHARE), 1), replace=False
        )
        val = train & np.isin(lanes, held)
        return np.flatnonzero(train & ~val), np.flatnonzero(val)

    def _mlp_probs(
        self, arm, x, route, train, weights
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
        from encoder_router.model import EncoderRouter

        cell, corpus, feature = self._targets(arm, train)
        fit, val = self._fit_val_split(train)
        router = EncoderRouter(seed=self.seed).fit(
            x[fit], route[fit],
            None if cell is None else cell[fit],
            None if corpus is None else corpus[fit],
            None if feature is None else feature[fit],
            route_weights=weights[fit],
            x_val=x[val], val_route_targets=route[val],
            val_route_weights=weights[val],
        )
        fit_info = {
            "train_val_loss": float(router.best_val_loss),
            "train_epochs": float(router.best_epoch + 1),
        }
        return (
            router.probabilities(x[train]),
            router.probabilities(x[~train]),
            fit_info,
        )

    def _lgbm_probs(
        self, arm, x, route, train, weights
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        from lightgbm import LGBMClassifier

        del arm
        probs = {
            "train": np.zeros((int(train.sum()), len(HEAD_ROUTES))),
            "test": np.zeros((int((~train).sum()), len(HEAD_ROUTES))),
        }
        for i, _ in enumerate(HEAD_ROUTES):
            ok = ~np.isnan(route[:, i]) & train
            positives = route[ok, i].sum()
            # n_jobs=1: parallel LightGBM segfaults in this process — its
            # OpenMP pool collides with torch/onnxruntime's runtimes.
            model = LGBMClassifier(
                n_estimators=400, learning_rate=0.05, n_jobs=1,
                scale_pos_weight=float(
                    np.clip((ok.sum() - positives) / max(positives, 1), 1, 100)
                ),
                random_state=self.seed, verbose=-1,
            )
            model.fit(x[ok], route[ok, i], sample_weight=weights[ok])
            probs["train"][:, i] = model.predict_proba(x[train])[:, 1]
            probs["test"][:, i] = model.predict_proba(x[~train])[:, 1]
        return (
            pd.DataFrame(probs["train"], columns=HEAD_ROUTES),
            pd.DataFrame(probs["test"], columns=HEAD_ROUTES),
            {"train_val_loss": float("nan"), "train_epochs": float("nan")},
        )

    def _readout(
        self,
        arm: Arm,
        lane: str,
        test: pd.DataFrame,
        served: list[str],
        thresholds: np.ndarray,
        fit_info: dict[str, float],
    ) -> dict[str, float | str]:
        """Agreements and scores over answerable rows only — an all-zero row
        has no right answer and must not count against anyone."""
        judged = test.assign(served=served).dropna(subset=["serve"])
        differ = judged[judged["shape"] == "routes_differ"]
        scores = {
            route: judged[f"score_{route}"].to_numpy() for route in ROUTES
        }
        pick = lambda routes: np.array(  # noqa: E731
            [scores[route][i] for i, route in enumerate(routes)]
        )
        return {
            "arm": arm.name,
            "lane": lane,
            "rows": len(test),
            "judged": len(judged),
            **{
                f"threshold_{route}": float(t)
                for route, t in zip(HEAD_ROUTES, thresholds)
            },
            **{
                f"served_{route}": float((judged["served"] == route).mean())
                for route in ROUTES
            },
            "serve_agreement": float(
                (judged["served"] == judged["serve"]).mean()
            ),
            "differ_agreement": float(
                (differ["served"] == differ["serve"]).mean()
            ) if len(differ) else np.nan,
            "captured": float(pick(judged["served"]).mean()),
            "serve_captured": float(pick(judged["serve"]).mean()),
            "oracle": float(
                np.column_stack(list(scores.values())).max(axis=1).mean()
            ),
            "const_dense": float(scores["dense_only"].mean()),
            "const_sparse": float(scores["sparse_only"].mean()),
            "const_rrf": float(scores["pure_rrf"].mean()),
            **fit_info,
        }


def run_arms(
    table: TrainingTable,
    arms: tuple[Arm, ...] = ARMS,
    *,
    lanes: tuple[str, ...] | None = None,
    seed: int = 0,
    tie_weight: float = TIE_WEIGHT,
    out_path: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Every arm through the same folds, persisted after EACH arm — a
    crashed sweep loses at most the arm in flight and resumes past the
    completed ones. `lanes` narrows the holdout panel; training rows stay
    full either way, so a panel run measures the same thing with fewer
    readout points."""
    path = out_path if out_path is not None else OUT_DIR / "arm_results.parquet"
    done = (
        pd.read_parquet(path)
        if path.exists() and not force
        else pd.DataFrame(columns=["arm"])
    )
    cv = LaneCV(table, seed=seed, tie_weight=tie_weight)
    for arm in arms:
        if (done["arm"] == arm.name).any():
            tqdm.write(f"[{arm.name}] already in {path.name} — skipped")
            continue
        done = pd.concat([done, cv.run(arm, lanes=lanes)], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        done.to_parquet(path, index=False)
    return done

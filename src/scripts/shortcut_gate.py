"""Shortcut gate: how much of the route label do NUISANCE features alone explain?

A draw whose route is predictable from lane, archetype and surface shape teaches a
classifier to recognise its own lanes rather than to route. The comparator is a
seeded draw from an eligible population frozen on first build, matched to the
artifact on route class AND certification — gaps measured on different shapes,
or on labels of different reliability, are not comparable numbers.

    poetry run python src/scripts/shortcut_gate.py
    poetry run python src/scripts/shortcut_gate.py --artifact <path>
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from composition.pool_v3 import LabelledPool, native_mask
from composition.recipe import Recipe
from scripts.label_routes import DATA_DIR

SEED = 20260825
BANDS = ("corpus_idf", "corpus_oov", "corpus_pmi", "corruption_degree")
MEASURED = frozenset({
    "score", "score_dense_only", "score_sparse_only", "score_pure_rrf",
    "oracle", "margin", "depth", "winner", "kind", "certified", "route",
    "is_decisive", "is_hybrid", "is_waste", "route_class", "route_class_any",
})
"""Everything retrieval or judging produced. The gate is meaningless if any of
these reaches the feature matrix — they are the answer, not the context."""


class NuisanceMatrix:
    """Context-only design matrix: lane, cell membership, corpus/corruption bands
    and surface length, with every measured column excluded by construction."""

    CONTEXT = ("dataset", "cells", "query", *BANDS)
    """The only columns `X` can see. Narrowed at construction rather than merely
    asserted, so a later feature reading `margin` fails on a missing column
    instead of silently compromising the gate."""

    def __init__(self, frame: pd.DataFrame) -> None:
        assert not (MEASURED & set(self.CONTEXT)), "measured column in CONTEXT"
        self._frame = frame[list(self.CONTEXT)].reset_index(drop=True)
        self._y = frame["route_class"].astype(str).to_numpy()

    @property
    def lanes(self) -> pd.Series:
        return self._frame["dataset"].astype(str)

    @property
    def y(self) -> np.ndarray:
        return self._y

    @property
    def X(self) -> np.ndarray:
        frame = self._frame
        cells = frame["cells"].map(lambda v: tuple(v) if v is not None else ())
        blocks = [
            pd.get_dummies(self.lanes, prefix="lane").to_numpy(float),
            MultiLabelBinarizer().fit_transform(cells).astype(float),
            pd.get_dummies(
                frame[list(BANDS)].astype(str), columns=list(BANDS)
            ).to_numpy(float),
            np.log1p(
                frame["query"].astype(str).str.len().to_numpy(float)
            ).reshape(-1, 1),
            np.log1p(
                frame["query"].astype(str).str.split().str.len().to_numpy(float)
            ).reshape(-1, 1),
        ]
        return np.hstack(blocks)


@dataclass(frozen=True)
class GateScore:
    """One draw's two gaps: over its own majority class, and over the per-lane
    best constant that `VERDICT.md` used to call the shipped router no-go."""

    name: str
    rows: int
    certified_share: float
    in_dist_acc: float
    own_constant: float
    shortcut_gap: float
    hide_lane_acc: float
    per_lane_constant: float
    transfer: float
    macro_f1: float


class ShortcutGate:
    """Owns the frozen random comparator and scores any draw against it."""

    def __init__(self, out_dir: Path | None = None) -> None:
        self._out = out_dir if out_dir is not None else DATA_DIR / "v3"

    @property
    def population_path(self) -> Path:
        return self._out / "shortcut_population.parquet"

    @property
    def report_path(self) -> Path:
        return self._out / "shortcut_gate.md"

    def population(self) -> pd.DataFrame:
        """The eligible pool the comparator draws from, frozen on first build so
        the bar cannot drift as the pool grows."""
        if self.population_path.exists():
            return pd.read_parquet(self.population_path)
        pool = LabelledPool(Recipe.v3()).selectable()
        live = pool[native_mask(pool) & ~pool["is_waste"]].assign(
            route_class=lambda f: f["route_class_any"].astype(str),
            cells=lambda f: f["cells"].map(sorted),
        )
        # only what the matrix reads: a frozen bar should not carry columns
        # nobody consumes, each of which is another way for it to drift
        kept = live[
            ["dataset", "query_id", "route_class", "certified", "cells", "query",
             *BANDS]
        ]
        self._out.mkdir(parents=True, exist_ok=True)
        kept.to_parquet(self.population_path, index=False)
        return kept

    def comparator(self, target: pd.DataFrame) -> pd.DataFrame:
        """A seeded draw matched on route class AND certification: an
        uncertified label is a bare argmax, so arms with different certified
        shares are scored on labels of different reliability."""
        population = self.population()
        cells = target.groupby(
            [target["route_class"].astype(str), target["certified"]]
        ).size()
        parts, deficit = [], {}
        for (route, certified), want in cells.items():
            supply = population[
                (population["route_class"] == route)
                & (population["certified"] == certified)
            ]
            take = min(int(want), len(supply))
            if take < want:
                deficit[f"{route}/certified={certified}"] = int(want) - take
            parts.append(supply.sample(n=take, random_state=SEED))
        drawn = pd.concat(parts, ignore_index=True)
        drawn.attrs["match_deficit"] = deficit
        return drawn

    @staticmethod
    def _fit(x_train, y_train, x_test) -> np.ndarray:
        """Scaled because the two length features dwarf the 0/1 blocks, and an
        unconverged bar drifts with the solver version."""
        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)
        )
        model.fit(x_train, y_train)
        return model.predict(x_test)

    @staticmethod
    def gateable(frame: pd.DataFrame, name: str) -> pd.Series:
        """Non-waste route-class counts, or a named refusal — an empty or
        all-waste draw otherwise dies inside `concat` or `StratifiedKFold`, and
        a crash reads as a broken pipeline rather than an ungateable draw."""
        counts = (
            frame.loc[frame["route_class"].astype(str) != "waste", "route_class"]
            .astype(str)
            .value_counts()
        )
        if len(counts) < 2 or int(counts.min()) < 2:
            raise ValueError(
                f"{name}: not gateable — needs >=2 route classes holding >=2 "
                f"non-waste rows each, got {counts.to_dict()}"
            )
        return counts

    def score(self, frame: pd.DataFrame, name: str) -> GateScore:
        live = frame[frame["route_class"].astype(str) != "waste"]
        counts = self.gateable(live, name)
        matrix = NuisanceMatrix(live)
        x, y, lanes = matrix.X, matrix.y, matrix.lanes.to_numpy()

        folds = StratifiedKFold(
            n_splits=min(5, int(counts.min())), shuffle=True, random_state=SEED
        )
        pred = np.empty_like(y)
        for train, test in folds.split(x, y):
            pred[test] = self._fit(x[train], y[train], x[test])
        in_dist = float((pred == y).mean())

        held = np.empty_like(y)
        for lane in np.unique(lanes):
            mask = lanes == lane
            if mask.sum() == len(y) or len(np.unique(y[~mask])) < 2:
                held[mask] = y[~mask][0] if (~mask).any() else y[mask][0]
                continue
            held[mask] = self._fit(x[~mask], y[~mask], x[mask])

        own = float(pd.Series(y).value_counts(normalize=True).max())
        per_lane = float(pd.crosstab(lanes, y).max(axis=1).sum() / len(y))
        hide_lane = float((held == y).mean())
        return GateScore(
            name=name,
            rows=int(len(y)),
            certified_share=round(float(live["certified"].mean()), 4),
            in_dist_acc=round(in_dist, 4),
            own_constant=round(own, 4),
            shortcut_gap=round(in_dist - own, 4),
            hide_lane_acc=round(hide_lane, 4),
            per_lane_constant=round(per_lane, 4),
            transfer=round(hide_lane - per_lane, 4),
            macro_f1=round(float(f1_score(y, pred, average="macro")), 4),
        )

    @staticmethod
    def _match(target: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
        """Cut the target down to the control's per-cell counts. `comparator`
        matches one direction only, so a short population leaves the two arms on
        different class mixes — and a gap measured across different mixes is not
        a comparison, whatever the verdict line above it says."""
        want = control.groupby(
            [control["route_class"].astype(str), control["certified"]]
        ).size()
        kept = [
            part.sample(n=int(want.get(cell, 0)), random_state=SEED)
            for cell, part in target.groupby(
                [target["route_class"].astype(str), target["certified"]]
            )
            if want.get(cell, 0)
        ]
        return pd.concat(kept) if kept else target.iloc[:0]

    def run(self, artifact: Path) -> pd.DataFrame:
        """Score the artifact and the frozen comparator, then write the verdict."""
        full = pd.read_parquet(artifact)
        full = full[full["route_class"].astype(str) != "waste"]
        # before the comparator: an ungateable draw dies in its concat() first,
        # and the refusal must name the artifact, not the pandas internals
        self.gateable(full, artifact.stem)
        control = self.comparator(full)
        deficit = control.attrs.get("match_deficit", {})
        live = self._match(full, control) if deficit else full
        coverage = len(live) / len(full)

        try:
            scores = [
                self.score(live, artifact.stem),
                self.score(control, "matched_random"),
            ]
        except ValueError as exc:
            # a pair the population cannot build must not also destroy the
            # artifact's own numbers, so the verdict downgrades and the
            # unmatched artifact is still reported, alone and proving nothing
            scores = [self.score(full, artifact.stem)]
            verdict = (
                f"**INCONCLUSIVE** — no matched pair could be built, so the "
                f"artifact's numbers stand alone and nothing is proven: {exc}\n"
            )
        else:
            new, ref = scores
            ok = (
                new.shortcut_gap <= ref.shortcut_gap
                and new.transfer >= ref.transfer
            )
            verdict = (
                f"**{'PASS' if ok else 'FAIL'}** — the draw must not be more "
                f"nuisance-solvable in-distribution than a random draw matched "
                f"on route class AND certification (`shortcut_gap` "
                f"{new.shortcut_gap:+.4f} vs {ref.shortcut_gap:+.4f}) nor "
                f"transfer worse out-of-lane (`transfer` {new.transfer:+.4f} "
                f"vs {ref.transfer:+.4f}). Scored on {coverage:.1%} of the "
                f"artifact.\n"
            )
        note = (
            f"\n**Match deficit** — the population was short "
            f"{sum(deficit.values())} rows ({deficit}), so BOTH arms were cut "
            f"to the counts it could supply and {coverage:.1%} of the artifact "
            f"was gated. An artifact built before the current reserve fence "
            f"will show one here.\n"
            if deficit else ""
        )
        out = pd.DataFrame([asdict(s) for s in scores])
        self._out.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(
            f"# Shortcut gate\n\n{out.to_markdown(index=False)}\n\n{verdict}{note}"
        )
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path, default=DATA_DIR / "v3" / "dataset_v3.parquet"
    )
    args = parser.parse_args()
    gate = ShortcutGate()
    gate.run(args.artifact)
    print(gate.report_path.read_text())


if __name__ == "__main__":
    main()

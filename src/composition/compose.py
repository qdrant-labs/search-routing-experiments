"""SPEC d32/d33 target-composition artifact owner (the `FeatureTable`
pattern): one build → `selection.parquet` + `order_sheet.parquet` +
`summary.md` under `src/data/composition/`, skipped when already on disk.

Fill order is d33c: B (no-preference, uniform over the span pool) draws
FIRST so it is a real sample rather than orcas leftovers; A fills its
evidence floors from the remainder; C stratifies the zero-span pool; D
draws feature-blind from the champions. One global selected set — no row
enters twice. Query text joins from the registry caches at the end."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_registry import DATASETS
from dataset_registry.core import RegistryDataset

from composition.fill import DEFERRED, QRELS, FloorLedger
from composition.floors import SpanFloorDeriver, span_mask, with_derived
from composition.recipe import Recipe
from composition.slices import (
    DarkForestSlice,
    NoPreferenceSlice,
    SliceResult,
    SpanTargetSlice,
    StatStrataSlice,
)

DEFAULT_CATALOG = (
    Path(__file__).resolve().parent.parent
    / "data" / "feature_table" / "catalog.parquet"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "composition"


class TargetComposition:
    """Builds and owns the d32 selection artifact."""

    def __init__(
        self,
        recipe: Recipe | None = None,
        catalog_path: Path | None = None,
        out_dir: Path | None = None,
    ) -> None:
        self._recipe = recipe if recipe is not None else Recipe()
        self._catalog_path = (
            catalog_path if catalog_path is not None else DEFAULT_CATALOG
        )
        self._out_dir = out_dir if out_dir is not None else DEFAULT_OUT_DIR
        self._datasets = {dataset.name: dataset for dataset in DATASETS}

    @property
    def selection_path(self) -> Path:
        return self._out_dir / "selection.parquet"

    @property
    def order_sheet_path(self) -> Path:
        return self._out_dir / "order_sheet.parquet"

    @property
    def summary_path(self) -> Path:
        return self._out_dir / "summary.md"

    def build(self, *, force: bool = False) -> pd.DataFrame:
        if self.selection_path.exists() and not force:
            return pd.read_parquet(self.selection_path)
        catalog = with_derived(
            pd.read_parquet(self._catalog_path)
            .sort_values(["dataset", "query_id"], kind="stable")
            .reset_index(drop=True)
        )
        rng = np.random.default_rng(self._recipe.seed)
        results = self._fill(catalog, rng)
        selection = pd.concat(
            [result.frame for result in results.values()], ignore_index=True,
        ).sort_values(["slice", "dataset", "query_id"], ignore_index=True)
        self._assert_invariants(selection, results)
        selection["query"] = self._join_text(selection)
        order_sheet = self._order_sheet(results)
        self._write(selection, order_sheet, results)
        return selection

    def _fill(
        self, catalog: pd.DataFrame, rng: np.random.Generator,
    ) -> dict[str, SliceResult]:
        spans = span_mask(catalog)
        span_pool = catalog[spans]
        # derived once over the FULL span pool: B and A share the same
        # floor definitions, so attribution stays comparable across slices
        floors = SpanFloorDeriver(self._recipe).derive(span_pool)

        results: dict[str, SliceResult] = {}
        results["B"] = NoPreferenceSlice(self._recipe).run(
            span_pool, floors, rng,
        )
        taken = self._taken(results)
        results["A"] = SpanTargetSlice(self._recipe).run(
            span_pool.loc[~span_pool.index.isin(taken)], floors, rng,
        )
        results["C"] = StatStrataSlice(self._recipe).run(catalog[~spans], rng)
        taken = self._taken(results)
        champions = catalog["dataset"].isin(self._recipe.champions)
        results["D"] = DarkForestSlice(self._recipe).run(
            catalog.loc[champions & ~catalog.index.isin(taken)], rng,
        )
        return results

    @staticmethod
    def _taken(results: dict[str, SliceResult]) -> pd.Index:
        return pd.Index(
            np.concatenate([
                result.frame.index.to_numpy() for result in results.values()
            ])
        )

    def _assert_invariants(
        self, selection: pd.DataFrame, results: dict[str, SliceResult],
    ) -> None:
        recipe = self._recipe
        sizes = selection["slice"].value_counts()
        expected = {
            "A": recipe.span_target_rows,
            "B": recipe.no_preference_rows,
            "C": recipe.stat_strata_rows,
            "D": recipe.dark_forest_rows,
        }
        for name, amount in expected.items():
            assert sizes.get(name, 0) == amount, (
                f"slice {name}: {sizes.get(name, 0)} != {amount}"
            )
        duplicated = selection.duplicated(["dataset", "query_id"])
        assert not duplicated.any(), "duplicate (dataset, query_id) rows"
        lanes = np.where(selection["checkable"], QRELS, DEFERRED)
        assert (selection["label_lane"] == lanes).all(), "lane/checkable drift"
        forest = selection[selection["slice"] == "D"]
        assert set(forest["dataset"]) <= set(recipe.champions), (
            "non-champion in dark forest"
        )
        cap = recipe.floor_rules.cap_frac * recipe.dark_forest_rows
        assert (forest["dataset"].value_counts() <= cap).all(), (
            "champion over the dark-forest cap"
        )
        for name, result in results.items():
            if result.ledger is None:
                continue
            met = result.ledger.credit >= result.ledger.amounts
            short = {line.key for line in result.shortfalls}
            unaccounted = [
                key for key, ok in zip(result.ledger.keys, met)
                if not ok and key not in short
            ]
            assert not unaccounted, f"slice {name}: floors neither met nor on the order sheet: {unaccounted}"

    def _join_text(self, selection: pd.DataFrame) -> pd.Series:
        return join_text(selection, self._datasets)

    def _order_sheet(self, results: dict[str, SliceResult]) -> pd.DataFrame:
        lines: list[dict[str, object]] = []
        for name, result in results.items():
            for line in result.shortfalls:
                lines.append({
                    "slice": name,
                    "floor": line.key,
                    "amount": line.amount,
                    "credit": line.credit,
                    "missing": line.missing,
                    "reason": line.reason,
                })
        return pd.DataFrame(
            lines,
            columns=["slice", "floor", "amount", "credit", "missing", "reason"],
        )

    def _write(
        self,
        selection: pd.DataFrame,
        order_sheet: pd.DataFrame,
        results: dict[str, SliceResult],
    ) -> None:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        selection.to_parquet(self.selection_path, index=False)
        order_sheet.to_parquet(self.order_sheet_path, index=False)
        self.summary_path.write_text(self._summary(selection, order_sheet, results))

    def _summary(
        self,
        selection: pd.DataFrame,
        order_sheet: pd.DataFrame,
        results: dict[str, SliceResult],
    ) -> str:
        recipe = self._recipe
        lane_counts = selection["label_lane"].value_counts()
        parts = [
            "# Target composition — SPEC d32/d33 fill\n",
            f"{len(selection):,} rows | seed={recipe.seed} | "
            f"qrels lane {lane_counts.get(QRELS, 0):,} / "
            f"deferred {lane_counts.get(DEFERRED, 0):,}\n",
            "## Slice × dataset\n",
            _block(pd.crosstab(selection["dataset"], selection["slice"])),
        ]
        for name in ("A", "C"):
            ledger = results[name].ledger
            if ledger is not None:
                parts.append(f"## Slice {name} floors\n")
                parts.append(_block(_floor_table(ledger)))
        parts.append("## Order sheet\n")
        parts.append(
            _block(order_sheet) if len(order_sheet)
            else "every floor met\n"
        )
        return "\n".join(parts)


def join_text(
    frame: pd.DataFrame, datasets: Mapping[str, RegistryDataset],
) -> pd.Series:
    """Query text for (dataset, query_id) rows, read from the registry caches."""
    texts = pd.Series(pd.NA, index=frame.index, dtype="object")
    for name, group in frame.groupby("dataset"):
        ids = group["query_id"].astype(str)
        cache = pd.read_parquet(
            datasets[name].cache_path,
            columns=["query_id", "text"],
            filters=[("query_id", "in", ids.tolist())],
        )
        lookup = cache.set_index(cache["query_id"].astype(str))["text"]
        texts.loc[group.index] = ids.map(lookup).to_numpy()
    missing = texts.isna()
    assert not missing.any(), (
        f"text join missed {int(missing.sum())} rows — cache drift"
    )
    return texts


def _floor_table(ledger: FloorLedger) -> pd.DataFrame:
    return pd.DataFrame({
        "floor": ledger.keys,
        "amount": ledger.amounts,
        "credit": ledger.credit.round(1),
        "ratio": (ledger.credit / ledger.amounts).round(2),
        "qrels": ledger.by_lane[QRELS].round(1),
        "deferred": ledger.by_lane[DEFERRED].round(1),
    })


def _block(frame: pd.DataFrame) -> str:
    return f"```\n{frame.to_string()}\n```\n"

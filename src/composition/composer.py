"""V3Composition — the v3 dataset's artifact owner: one build from the
labelled pool through the three objective layers to `data/v3/`, plus the
admit door that credits generated rows back against the order sheet. CellFill
stays the frozen v2 owner; this class never touches its artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from augmentation.core import CreditGate
from composition.mini_catalog import mini_catalog
from composition.objectives import (
    CORRUPTION_SLICE,
    UNCOVERED_SLICE,
    DiversityFloors,
    InversionBound,
    SelectionOrder,
    UtilityObjective,
)
from composition.pool_v3 import CEILING, REUSED, LabelledPool
from composition.recipe import Recipe
from query_taxonomy.features import FeatureExtractor

from composition.floors import CORRUPTION_SPANS, with_derived


class V3Composition:
    """Builds and owns `data/v3/`: the selected dataset, the order sheet the
    augmentation loop consumes, the frozen eval reserve, the per-lane
    labelling demands and the representation reports."""

    def __init__(
        self,
        recipe: Recipe | None = None,
        pool: LabelledPool | None = None,
        out_dir: Path | None = None,
    ) -> None:
        self._recipe = recipe if recipe is not None else Recipe.v3()
        self._pool = pool if pool is not None else LabelledPool(self._recipe)
        self._out = out_dir if out_dir is not None else (
            self._pool.v3_catalog_path.parent
        )
        cells = self._pool.active_cells()
        self._floors = DiversityFloors(self._recipe, cells)
        self._utility = UtilityObjective(self._recipe)
        self._bound = InversionBound(self._recipe, cells)

    @property
    def catalog_path(self) -> Path:
        return self._pool.v3_catalog_path

    @property
    def dataset_path(self) -> Path:
        return self._out / "dataset_v3.parquet"

    @property
    def order_sheet_path(self) -> Path:
        return self._out / "order_sheet.parquet"

    @property
    def selection_order_path(self) -> Path:
        return self._out / "selection_order.parquet"

    @property
    def selection_summary_path(self) -> Path:
        return self._out / "selection_summary.parquet"

    @property
    def eval_reserve_path(self) -> Path:
        return self._out / "eval_reserve.parquet"

    @property
    def per_dataset_path(self) -> Path:
        return self._out / "per_dataset.parquet"

    @property
    def inversion_path(self) -> Path:
        return self._out / "inversion_bound.parquet"

    @property
    def admitted_path(self) -> Path:
        return self._out / "admitted.parquet"

    @property
    def report_path(self) -> Path:
        return self._out / "report.md"

    # ------------------------------------------------------------------ build ---
    def build(self, *, force: bool = False) -> pd.DataFrame:
        if self.dataset_path.exists() and not force:
            return pd.read_parquet(self.dataset_path)
        pool = self._pool.frame()
        reserve = self._pool.reserve()
        selected = self._utility.select(self._pool.selectable())

        marginals = self._floors.marginals(selected)
        sheet = self._floors.order_sheet(selected, pool)
        per_ds = self._per_dataset(pool)
        debt = self._floors.class_debt(pool)
        order, order_summary = self._selection_order(pool, per_ds, debt, sheet)
        inversion = self._bound.report(selected, pool)
        realism = self._bound.realism(selected)

        self._out.mkdir(parents=True, exist_ok=True)
        flat = selected.assign(cells=selected["cells"].map(sorted))
        flat.to_parquet(self.dataset_path, index=False)
        sheet.to_parquet(self.order_sheet_path, index=False)
        order.to_parquet(self.selection_order_path, index=False)
        order_summary.to_parquet(self.selection_summary_path)
        reserve[["dataset", "query_id", "route_class"]].to_parquet(
            self.eval_reserve_path, index=False
        )
        per_ds.to_parquet(self.per_dataset_path)
        inversion.to_parquet(self.inversion_path, index=False)
        marginals.to_parquet(self._out / "marginals.parquet", index=False)
        (self._out / "dataset_v3.provenance.json").write_text(
            json.dumps(self._sidecar(selected, reserve), indent=2) + "\n"
        )
        self.report_path.write_text(
            self._report(pool, selected, reserve, marginals, sheet, per_ds,
                         inversion, realism, debt, order_summary)
        )
        return selected

    def _selection_order(
        self, pool: pd.DataFrame, per_ds: pd.DataFrame, debt: pd.DataFrame,
        sheet: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The labelling rung's demand — built against the same catalog the
        strata came from, with cells filtered to the columns it carries."""
        from composition.floors import read_catalog, with_derived

        catalog = with_derived(read_catalog(self.catalog_path)).astype(
            {"query_id": str}
        )
        order = SelectionOrder(
            self._recipe,
            self._pool.active_cells(catalog),
            self._pool.v3_catalog_path.parent.parent,
        )
        return order.build(pool, per_ds, debt, sheet, catalog)

    # ------------------------------------------------------------------ admit ---
    def admit(
        self,
        pool: pd.DataFrame,
        *,
        extractor: FeatureExtractor | None = None,
        coherence_passed: set[str] | None = None,
    ) -> pd.DataFrame:
        """Credit generated rows against the hungry sheet lines: cell demands
        re-verified by the cell's own predicate over a mini catalog,
        corruption demands by the derived span total — a row minted FOR a
        floor still has to measurably serve it."""
        sheet = pd.read_parquet(self.order_sheet_path)
        selection = pd.read_parquet(self.dataset_path)
        fresh = self._admissible(pool, selection, sheet, coherence_passed)
        if fresh.empty:
            print("v3 admit: nothing admissible in the pool")
            return fresh
        cells = {cell.name: cell for cell in self._pool.active_cells()}
        mini = with_derived(mini_catalog(
            fresh,
            extractor or FeatureExtractor(engines=None),
            columns=tuple(
                {band.column for cell in cells.values() for band in cell.bands}
            ),
        ))
        admitted, gained = [], {}
        rng = np.random.default_rng(self._recipe.seed)
        for line in sheet[sheet["missing"] > 0].itertuples(index=False):
            members = fresh[fresh["floor"] == line.floor]
            if line.slice == CORRUPTION_SLICE:
                served = mini.loc[members.index, CORRUPTION_SPANS] >= 1
            elif line.floor in cells:
                served = cells[line.floor].select(mini.loc[members.index])
            else:
                continue
            members = members[served.to_numpy(dtype=bool)]
            if members.empty:
                continue
            take = members.sample(
                n=min(int(line.missing), len(members)),
                random_state=rng.integers(2**31),
            )
            gained[line.floor] = len(take)
            admitted.append(take.assign(credited_floor=line.floor))
        if not admitted:
            print("v3 admit: no generated row measurably serves a hungry line")
            return fresh.iloc[:0]
        rows = pd.concat(admitted, ignore_index=True)
        self._assert_natural_share(selection, rows)
        credited = sheet["floor"].map(gained).fillna(0.0)
        sheet["credit"] = sheet["credit"] + credited
        sheet["missing"] = (sheet["missing"] - credited).clip(lower=0.0)
        sheet.to_parquet(self.order_sheet_path, index=False)
        rows.to_parquet(self.admitted_path, index=False)
        print(
            f"v3 admit: {len(rows):,} rows into {len(gained)} lines | "
            f"hungry {int((sheet['missing'] > 0).sum())} remain -> label them, "
            f"then rebuild"
        )
        return rows

    @staticmethod
    def _admissible(
        pool: pd.DataFrame,
        selection: pd.DataFrame,
        sheet: pd.DataFrame,
        coherence_passed: set[str] | None = None,
    ) -> pd.DataFrame:
        """The pool rows this admission may credit: `coherence_passed` is
        `CoherenceJudge.passed()`, and None (the default) keeps every gated row
        waiting on its human audit."""
        rows = pool
        if "credit_gate" in rows.columns:
            gate = rows["credit_gate"].fillna(str(CreditGate.NONE))
            cleared = gate.eq(CreditGate.COHERENCE_GATE) & rows["query_id"].isin(
                coherence_passed or set()
            )
            gated = gate.ne(str(CreditGate.NONE)) & ~cleared
            if gated.any() or cleared.any():
                print(
                    f"v3 admit: skipping {int(gated.sum())} gated rows (d42h), "
                    f"{int(cleared.sum())} cleared by the coherence judge"
                )
            rows = rows[~gated]
        hungry = set(sheet.loc[sheet["missing"] > 0, "floor"])
        return rows[
            rows["floor"].isin(hungry)
            & ~rows["query_id"].isin(set(selection["query_id"]))
        ].assign(
            dataset=lambda frame: frame["home_lane"],
        ).reset_index(drop=True)

    def _assert_natural_share(
        self, selection: pd.DataFrame, rows: pd.DataFrame
    ) -> None:
        natural = (
            len(selection)
            / max(len(selection) + len(rows), 1)
        )
        assert natural >= self._recipe.min_natural_share - 1e-9, (
            f"natural share {natural:.3f} fell below the recipe minimum "
            f"{self._recipe.min_natural_share}"
        )

    # ---------------------------------------------------------------- reports ---
    def _per_dataset(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Per-lane labelling demands: floor credit, gap, and what a FRESH
        label is worth there (decisive-at-margin over BLIND rows only — reused
        rows entered for already having won and estimate nothing)."""
        recipe = self._recipe
        g = pool.groupby("dataset")
        out = pd.DataFrame({
            "labelled": g.size(),
            "floor_credit": g.apply(
                lambda d: int((d["is_decisive"] | d["is_hybrid"]).sum()),
                include_groups=False,
            ),
            "median_depth": g["depth"].median(),
        })
        out["floor"] = recipe.stratum_floor
        out["floor_met"] = out["floor_credit"] >= recipe.stratum_floor
        out["floor_gap"] = (recipe.stratum_floor - out["floor_credit"]).clip(lower=0)
        out["single_answer"] = out["median_depth"] <= 1
        blind = pool[pool["stage"] != REUSED].groupby("dataset")
        out["blind_labelled"] = blind.size().reindex(out.index).fillna(0).astype(int)
        out["blind_decisive"] = (
            blind["is_decisive"].sum().reindex(out.index).fillna(0).astype(int)
        )
        out["yield_rate"] = (
            out["blind_decisive"] / out["blind_labelled"].clip(lower=1)
        ).round(3)
        pooled = out["blind_decisive"].sum() / max(int(out["blind_labelled"].sum()), 1)
        out["fresh_left"] = self._fresh_left(out)

        def _action(r) -> str:
            if r["floor_met"]:
                return "ok"
            if r["single_answer"]:
                return "waive (single-answer)"
            if r["blind_labelled"] == 0:
                return "unmeasured (no blind rows)"
            if r["yield_rate"] < pooled:
                return "source/deepen"
            return "label more" if r["fresh_left"] > 0 else "exhausted (source/deepen)"

        out["floor_action"] = out.apply(_action, axis=1)
        out.attrs["pooled_blind_yield"] = round(pooled, 4)
        return out.sort_values("floor_gap", ascending=False)

    def _fresh_left(self, out: pd.DataFrame) -> pd.Series:
        """Source queries not yet labelled, per lane — 'label more' with zero
        fresh queries is a dead instruction."""
        from pyarrow.parquet import ParquetFile

        from scripts.label_routes import _source_name

        data = self._pool.v3_catalog_path.parent.parent
        left = {}
        for lane in out.index:
            qpath = data / _source_name(str(lane)) / "queries.parquet"
            total = ParquetFile(qpath).metadata.num_rows if qpath.exists() else 0
            left[lane] = max(0, total - int(out.at[lane, "labelled"]))
        return pd.Series(left)

    def _sidecar(self, selected: pd.DataFrame, reserve: pd.DataFrame) -> dict:
        """What this artifact was built from — the record whose absence once
        let a campaign run against a report that no longer existed."""
        import hashlib
        import subprocess
        from datetime import UTC, datetime

        root = Path(__file__).resolve().parent.parent.parent

        def rev(spec: str) -> str:
            return subprocess.run(
                ["git", "rev-parse", spec], capture_output=True, text=True,
                cwd=root,
            ).stdout.strip()

        data = self._pool.v3_catalog_path.parent.parent
        inputs = {}
        for path in (
            data / "v3" / "labels_rederived.parquet",
            data / "v3" / "labels.parquet",
            data / "v3" / "synthetic" / "labels.parquet",
            self._pool.v3_catalog_path,
            data / "route_labels" / "query_corpus_stats.parquet",
        ):
            if path.exists():
                inputs[path.name] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()[:16]
        tiers = selected["certified"].value_counts().to_dict()
        return {
            "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_head": rev("HEAD"),
            "recipe": self._recipe.model_dump(),
            "rows": len(selected),
            "certified": int(tiers.get(True, 0)),
            "uncertified": int(tiers.get(False, 0)),
            "eval_reserve_rows": len(reserve),
            "inputs_sha256_16": inputs,
        }

    def _report(self, pool, selected, reserve, marginals, sheet, per_ds,
                inversion, realism, debt, order_summary) -> str:
        recipe = self._recipe
        unmet_m = marginals[~marginals["met"]]
        unmet_l = per_ds[~per_ds["floor_met"]]
        breach = inversion[~inversion["floor_forced"]]
        k_inv = float(breach["max_ratio"].replace(np.inf, np.nan).max())
        lines = [
            "# v3 composition",
            "",
            f"Pool: {len(pool):,} labelled rows, {pool['dataset'].nunique()} lanes. "
            f"Target {recipe.target_total:,} rows at split {recipe.target_split}, "
            f"stratum floor {recipe.stratum_floor}, "
            f"class_margin {recipe.class_margin}, k_cap {recipe.k_cap}.",
            "",
            f"## Generation debt to {recipe.target_total:,}",
            debt.to_markdown(index=False),
            "",
            f"The order sheet owes **{int(sheet['missing'].sum()):,}** generated "
            f"rows across {len(sheet)} lines "
            f"({int((sheet['slice'] == 'cell').sum())} cell, "
            f"{int((sheet['slice'] == CORRUPTION_SLICE).sum())} corruption, "
            f"{int((sheet['slice'] == UNCOVERED_SLICE).sum())} uncovered).",
            "",
            "## Labelling order (the rung BEFORE generation)",
            f"{int(order_summary['labels_ordered'].sum()):,} labels ordered "
            f"across {len(order_summary)} lanes "
            f"({int(order_summary['queries_named'].sum()):,} queries named, "
            f"answer-coverage gated); estimated residual debt after labelling: "
            f"{order_summary.attrs.get('residual_debt', {})} — generation's "
            f"true share.",
            order_summary.round(4).to_markdown(),
            "",
            "## Layers",
            f"- Utility: certified tier {selected.attrs['total_certified']:,} "
            f"(feasible max under the lane cap), tier-0 {selected.attrs['total_tier0']:,}, "
            f"waste {int((selected['route_class'] == 'waste').sum())} of "
            f"{selected.attrs['waste_budget']} dictated. Selected {len(selected):,} rows; "
            f"reserve {len(reserve):,} frozen before selection.",
            f"- Diversity: {int((~marginals['met']).sum())}/{len(marginals)} marginal "
            f"strata under the floor of {recipe.stratum_floor}. "
            f"Lane gaps: {len(unmet_l)}/{len(per_ds)} lanes under floor "
            f"(labelling demand, per_dataset.parquet).",
            f"- Representation: max non-floor-forced inversion ratio "
            f"**{k_inv if np.isfinite(k_inv) else float('nan'):.2f}** "
            f"(K_inv candidate — freeze after reading inversion_bound.parquet); "
            f"damaged-query share {realism['selected_damaged_share'].iloc[0]:.1%} "
            f"vs census {realism['census_any_span_rate'].iloc[0]:.1%}.",
            "",
            "## Class split (tier-0 artifact)",
            selected["route_class"].value_counts().to_frame("rows").to_markdown(),
            "",
            "## Worst inversion ratios",
            inversion.head(10).to_markdown(index=False),
            "",
            "## Undermet marginals",
            unmet_m.head(20).to_markdown(index=False),
            "",
            f"## Lane floor shortfalls ({len(unmet_l)})",
            unmet_l[["labelled", "floor_credit", "floor_gap", "yield_rate",
                     "floor_action"]].head(24).to_markdown(),
            "",
            "## Waste taxonomy",
            f"- genuine ties {int((pool['kind'] == 'genuine_tie').sum()):,} · "
            f"fake ties {int((pool['kind'] == 'fake_tie').sum()):,} · "
            f"all_zero {int((pool['kind'] == 'all_zero').sum()):,} · "
            f"at-ceiling genuine "
            f"{int(((pool['kind'] == 'genuine_tie') & (pool['oracle'] >= CEILING)).sum()):,}.",
        ]
        return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    composer = V3Composition()
    selected = composer.build(force=args.force)
    print(composer.report_path.read_text())
    print(f"\nselected {len(selected):,} rows -> {composer.dataset_path.parent}")


if __name__ == "__main__":
    main()

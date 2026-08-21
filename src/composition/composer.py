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
from composition.cellfill import NATURAL
from composition.mini_catalog import mini_catalog
from composition.objectives import (
    CORRUPTION_SLICE,
    DiversityFloors,
    InversionBound,
    LaneOrder,
    SelectionOrder,
    UtilityObjective,
)
from composition.pool_v3 import CEILING, REUSED, LabelledPool, native_mask
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
    def lane_order_path(self) -> Path:
        return self._out / "lane_order.parquet"

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
        pool = self._pool.refresh().frame() if force else self._pool.frame()
        reserve = self._pool.reserve()
        selected = self._utility.select(self._pool.selectable())

        marginals = self._floors.marginals(selected)
        sheet = self._floors.order_sheet(selected, pool)
        per_ds = self._per_dataset(pool)
        debt = self._floors.class_debt(pool)
        order, order_summary = self._selection_order(pool, per_ds, debt, sheet)
        lane_order = self._lane_order(pool, order_summary)
        inversion = self._bound.report(selected, pool)
        realism = self._bound.realism(selected)

        self._out.mkdir(parents=True, exist_ok=True)
        flat = selected.assign(cells=selected["cells"].map(sorted))
        flat.to_parquet(self.dataset_path, index=False)
        sheet.to_parquet(self.order_sheet_path, index=False)
        order.to_parquet(self.selection_order_path, index=False)
        order_summary.to_parquet(self.selection_summary_path)
        lane_order.to_parquet(self.lane_order_path)
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
                         inversion, realism, debt, order_summary, lane_order)
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

    def _lane_order(
        self, pool: pd.DataFrame, order_summary: pd.DataFrame
    ) -> pd.DataFrame:
        """The class-residual carrier's quotas: capacity = a lane's corpus
        docs minus the grounding docs the lane rung already keyed; corpora
        over 1M docs wait for row-group sampling."""
        from pyarrow.parquet import ParquetFile

        from augmentation.config import AugmentationConfig
        from augmentation.supply import lane_dirs

        residual = order_summary.attrs.get("residual_debt")
        if residual is None:
            footer = order_summary.loc["(residual after labelling)"]
            residual = {
                name: float(footer[f"expected_{name}"])
                for name in ("dense", "sparse", "hybrid")
            }
        yields = SelectionOrder.yields(pool)
        paths = AugmentationConfig().paths
        used = (
            pd.read_parquet(paths.qrels, columns=["query_id"])["query_id"]
            .astype(str)
            if paths.qrels.exists() else pd.Series(dtype=str)
        )
        dirs = lane_dirs()
        pool_ids = pool["query_id"].astype(str)
        capacity: dict[str, int] = {}
        pending: dict[str, int] = {}
        for lane in yields.index:
            corpus = paths.lane_corpus(dirs.get(str(lane), str(lane)))
            if not corpus.exists():
                continue
            total = ParquetFile(corpus).metadata.num_rows
            if total > 1_000_000:
                continue
            prefix = f"lane-{lane}-"
            spent = int(used.str.startswith(prefix).sum()) if len(used) else 0
            labelled = int(pool_ids.str.startswith(prefix).sum())
            capacity[str(lane)] = max(0, total - spent)
            pending[str(lane)] = max(0, spent - labelled)
        return LaneOrder(self._recipe).build(
            residual, yields, pool, capacity, pending
        )

    # ------------------------------------------------------------------ admit ---
    def admit(
        self,
        pool: pd.DataFrame,
        *,
        extractor: FeatureExtractor | None = None,
        coherence_passed: set[str] | None = None,
        audit_passed: set[str] | None = None,
    ) -> pd.DataFrame:
        """Credit generated rows against the hungry sheet lines: cell demands
        re-verified by the cell's own predicate over a mini catalog,
        corruption demands by the derived span total — a row minted FOR a floor
        still has to measurably serve it, and pays every other hungry line it
        measurably serves too."""
        sheet = pd.read_parquet(self.order_sheet_path)
        selection = pd.read_parquet(self.dataset_path)
        from augmentation.campaign import audit_opened
        from augmentation.config import AugmentationConfig

        config = AugmentationConfig()
        opened_floors = audit_opened(
            pool, audit_passed,
            rate=config.coherence_pass_rate, pilot_n=config.pilot_n,
        )
        fresh = self._admissible(
            pool, selection, sheet, coherence_passed, audit_passed,
            opened_floors,
        )
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
        hungry = sheet[sheet["missing"] > 0]
        admitted, gained = [], {}
        rng = np.random.default_rng(self._recipe.seed)
        for line in hungry.itertuples(index=False):
            members = fresh[fresh["floor"] == line.floor]
            if members.empty:
                continue
            members = members[self._serves(line, cells, mini, members.index)]
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
        rows = pd.concat(admitted)
        extra = self._extra_credit(hungry, rows, gained, cells, mini)
        record = self._admission_record(self.admitted_path, rows)
        self._assert_natural_share(
            selection, record, self._recipe.min_natural_share
        )
        credited = sheet["floor"].map(
            {floor: gained.get(floor, 0) + extra.get(floor, 0)
             for floor in set(gained) | set(extra)}
        ).fillna(0.0)
        sheet["credit"] = sheet["credit"] + credited
        sheet["missing"] = (sheet["missing"] - credited).clip(lower=0.0)
        sheet.to_parquet(self.order_sheet_path, index=False)
        record.to_parquet(self.admitted_path, index=False)
        print(
            f"v3 admit: {len(rows):,} rows into {len(gained)} lines, plus "
            f"{sum(extra.values()):,} credits on {len(extra)} lines they also "
            f"serve | {len(record):,} rows on record | hungry "
            f"{int((sheet['missing'] > 0).sum())} remain -> label them, "
            f"then rebuild"
        )
        return rows

    @staticmethod
    def _serves(line, cells: dict, mini: pd.DataFrame, index) -> np.ndarray:
        """Which of these mini-catalog rows measurably serve the line — the
        cell's own predicate, the derived span total for corruption, and
        nothing for a line no rung can verify."""
        if line.slice == CORRUPTION_SLICE:
            return (mini.loc[index, CORRUPTION_SPANS] >= 1).to_numpy(dtype=bool)
        if line.floor in cells:
            return cells[line.floor].select(mini.loc[index]).to_numpy(dtype=bool)
        return np.zeros(len(index), dtype=bool)

    @classmethod
    def _extra_credit(
        cls,
        hungry: pd.DataFrame,
        rows: pd.DataFrame,
        gained: dict[str, int],
        cells: dict,
        mini: pd.DataFrame,
    ) -> dict[str, int]:
        """What the admitted rows serve BEYOND the line they were minted for,
        capped at what that line still needs — selection credits every stratum
        a row touches, so generation may not be billed for less."""
        extra: dict[str, int] = {}
        for line in hungry.itertuples(index=False):
            room = int(line.missing) - gained.get(line.floor, 0)
            others = rows.index[rows["credited_floor"] != line.floor]
            if room <= 0 or others.empty:
                continue
            serving = int(cls._serves(line, cells, mini, others).sum())
            if serving:
                extra[line.floor] = min(room, serving)
        return extra

    @staticmethod
    def _admission_record(path, rows: pd.DataFrame) -> pd.DataFrame:
        """Every batch ever admitted, newest verdict per query_id — the label
        sweep reads this file, so a round appends to it instead of replacing
        it."""
        old = pd.read_parquet(path) if path.exists() else rows.iloc[:0]
        return pd.concat([old, rows], ignore_index=True).drop_duplicates(
            "query_id", keep="last"
        )

    @staticmethod
    def _admissible(
        pool: pd.DataFrame,
        selection: pd.DataFrame,
        sheet: pd.DataFrame,
        coherence_passed: set[str] | None = None,
        audit_passed: set[str] | None = None,
        opened_floors: set[str] | None = None,
    ) -> pd.DataFrame:
        """The pool rows this admission may credit: per-row verdicts for the
        coherence gate (every row is judged), and for the declaration audit
        either the row's own cleared id or its FLOOR's passed pilot — the
        sample-audit certifies the operator, not just the sampled rows."""
        rows = pool
        if "credit_gate" in rows.columns:
            gate = rows["credit_gate"].fillna(str(CreditGate.NONE))
            cleared_by_gate = {
                str(CreditGate.COHERENCE_GATE): coherence_passed or set(),
                str(CreditGate.DECLARATION_AUDIT): audit_passed or set(),
            }
            cleared = pd.Series(False, index=rows.index)
            for value, ids in cleared_by_gate.items():
                cleared |= gate.eq(value) & rows["query_id"].isin(ids)
            cleared |= gate.eq(str(CreditGate.DECLARATION_AUDIT)) & rows[
                "floor"
            ].isin(opened_floors or set())
            gated = gate.ne(str(CreditGate.NONE)) & ~cleared
            if gated.any() or cleared.any():
                print(
                    f"v3 admit: {int(cleared.sum())} gated rows cleared, "
                    f"{int(gated.sum())} still waiting on their verdict "
                    f"{gate[gated].value_counts().to_dict()}"
                )
            rows = rows[~gated]
        hungry = set(sheet.loc[sheet["missing"] > 0, "floor"])
        return rows[
            rows["floor"].isin(hungry)
            & ~rows["query_id"].isin(set(selection["query_id"]))
        ].assign(
            dataset=lambda frame: frame["home_lane"],
        ).reset_index(drop=True)

    @staticmethod
    def _assert_natural_share(
        selection: pd.DataFrame, admitted: pd.DataFrame, minimum: float
    ) -> None:
        """Cumulative and by provenance: every generated row ever admitted
        counts against the share, whether or not this batch minted it."""
        natural = int((selection["provenance"] == NATURAL).sum())
        generated = set(admitted["query_id"]) | set(
            selection.loc[selection["provenance"] != NATURAL, "query_id"]
        )
        share = natural / max(natural + len(generated), 1)
        assert share >= minimum - 1e-9, (
            f"natural share {share:.3f} fell below the recipe minimum {minimum}"
        )

    # ---------------------------------------------------------------- reports ---
    def _per_dataset(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Per-lane labelling demands: floor credit, gap, and what a FRESH
        label is worth there (the floor-crediting rate over BLIND rows only,
        since reused rows entered for already having won and estimate
        nothing)."""
        recipe = self._recipe
        g = pool.groupby("dataset")
        out = pd.DataFrame({
            "labelled": g.size(),
            # the lane floor is a v3-dataset constraint, so only NATIVE rows
            # credit it, at the same tier-0 non-waste bar class_debt and the
            # labelling order count supply on; `labelled`/blind stats stay
            # all-era (measurement)
            "floor_credit": g.apply(
                lambda d: int((~d["is_waste"] & native_mask(d)).sum()),
                include_groups=False,
            ),
            "median_depth": g["depth"].median(),
        })
        out["floor"] = recipe.stratum_floor
        out["floor_met"] = out["floor_credit"] >= recipe.stratum_floor
        out["floor_gap"] = (recipe.stratum_floor - out["floor_credit"]).clip(lower=0)
        out["single_answer"] = out["median_depth"] <= 1
        blind_rows = pool[pool["stage"] != REUSED]
        blind = blind_rows.groupby("dataset")
        out["blind_labelled"] = blind.size().reindex(out.index).fillna(0).astype(int)
        # the rate answers "what does one fresh label credit?", so it counts the
        # rows `floor_credit` counts: a gap measured at one bar and divided by a
        # yield measured at another over-orders labels
        out["blind_creditable"] = (
            blind_rows[~blind_rows["is_waste"]].groupby("dataset").size()
            .reindex(out.index).fillna(0).astype(int)
        )
        out["yield_rate"] = (
            out["blind_creditable"] / out["blind_labelled"].clip(lower=1)
        ).round(3)
        pooled = out["blind_creditable"].sum() / max(int(out["blind_labelled"].sum()), 1)
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
                inversion, realism, debt, order_summary, lane_order) -> str:
        recipe = self._recipe
        unmet_m = marginals[~marginals["met"]]
        unmet_l = per_ds[~per_ds["floor_met"]]
        breach = inversion[~inversion["floor_forced"]]
        k_inv = float(breach["max_ratio"].replace(np.inf, np.nan).max())
        lines = [
            "# v3 composition",
            "",
            f"Pool: {len(pool):,} labelled rows, {pool['dataset'].nunique()} lanes — "
            f"**{int(native_mask(pool).sum()):,} v3-native (the only supply)**; "
            f"the re-scored v2 rows measure yields and priors, never the "
            f"dataset. Target {recipe.target_total:,} rows at split "
            f"{recipe.target_split}, stratum floor {recipe.stratum_floor}, "
            f"class_margin {recipe.class_margin}, k_cap {recipe.k_cap}.",
            "",
            f"## Class shortfall to {recipe.target_total:,}",
            debt.to_markdown(index=False),
            "",
            f"**Generation owes** the order sheet "
            f"**{int(sheet['missing'].sum()):,}** rows across {len(sheet)} "
            f"lines ({int((sheet['slice'] == 'cell').sum())} cell, "
            f"{int((sheet['slice'] == CORRUPTION_SLICE).sum())} corruption) — "
            f"cell floors and census-rate damage, nothing else.",
            "",
            f"**The lane rung owes** the class shortfall labelling cannot buy: "
            f"{order_summary.attrs.get('residual_debt', {})} rows toward "
            f"{recipe.target_total:,}, minted INTO lanes by their measured "
            f"yields (lane_order.parquet). Three owners, three numbers — "
            f"summing any two misprices all of them.",
            "",
            "## Labelling order (the rung BEFORE generation)",
            f"{int(order_summary['labels_ordered'].sum()):,} labels ordered "
            f"across {len(order_summary)} lanes "
            f"({int(order_summary['queries_named'].sum()):,} queries named, "
            f"answer-coverage gated); what this order cannot buy falls to the "
            f"lane rung below.",
            order_summary.round(4).to_markdown(),
            "",
            "## Lane rung (the class-residual carrier)",
            f"{int(lane_order['rows_to_mint'].dropna().sum()):,} doc-grounded "
            f"rows to mint across {max(len(lane_order) - 1, 0)} lanes; the "
            f"ceiling footer is what measured yields + lane caps + grounding "
            f"supply make reachable — the gap beyond it is the acquisition "
            f"conversation, not a generation dial.",
            lane_order.round(4).to_markdown(),
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

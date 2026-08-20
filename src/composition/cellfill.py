"""Route-signal fill: every archetype cell quotas both routes under a
lane-share cap computed from that cell's own supply, reusing the labels
already on disk and queueing the rest for labelling. Size is an output —
the artifact is the filled quotas plus a control group of unclaimed
queries, and a rebuild after fresh labels is the next iteration."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from augmentation.config import AugmentationConfig
from dataset_registry import DATASETS
from hybrid_search_rrf_dataset.router import decisive_rows
from query_taxonomy.features import FeatureExtractor

from composition.cells import CELLS, ArchetypeCell
from composition.compose import DEFAULT_CATALOG, DEFAULT_OUT_DIR, join_text
from composition.fill import QRELS, FloorLedger
from composition.floors import FloorSpec, with_derived
from composition.mini_catalog import mini_catalog
from composition.recipe import Recipe

DEFAULT_LABELS = DEFAULT_OUT_DIR.parent / "route_labels" / "labels.parquet"
DENSE = "dense_only"
SPARSE = "sparse_only"
ROUTES = (DENSE, SPARSE)
CELL_SLICE = "cell"
REUSED = "reused"
CANDIDATE = "candidate"
CONTROL = "control"
EXHAUSTED = "exhausted"
NATURAL = "natural"
ID_COLUMNS = ["dataset", "query_id", "checkable"]


def _key(frame: pd.DataFrame) -> pd.Series:
    """The label join key."""
    return frame["dataset"].astype(str) + "\x00" + frame["query_id"].astype(str)


def _top_share(counts: pd.Series, total: int) -> float:
    """The largest lane's share of a row set."""
    if not len(counts) or not total:
        return 0.0
    return round(float(counts.iloc[0] / total), 3)


class LaneCap:
    """The row share one lane may hold in one cell: the tightest share the
    cell's lane counts can satisfy at its draw size, and no tighter."""

    def __init__(self, counts: pd.Series, draw: int, target: float) -> None:
        self._counts = counts.to_numpy(dtype=int)
        self.draw = draw
        self.target = target
        self.floor = self._tightest()
        self.share = max(target, self.floor)

    def _tightest(self) -> float:
        for rows in range(1, self.draw + 1):
            if np.minimum(self._counts, rows).sum() >= self.draw:
                return rows / self.draw
        return 1.0

    def admits(self, counts: pd.Series | np.ndarray, want: int) -> int:
        """The largest draw up to `want` the share can spread over these lanes."""
        lanes = np.asarray(counts, dtype=int)
        take = want
        while take > 0:
            reachable = int(
                np.minimum(lanes, int(np.ceil(self.share * take))).sum()
            )
            if reachable >= take:
                return take
            take = reachable
        return 0

    @property
    def infeasible(self) -> bool:
        return self.floor > self.target

    @property
    def capacity(self) -> int:
        return self.admits(self._counts, self.draw)


class MassCap:
    """The rows one cell may quota per route for its own natural mass: a
    policy multiple of its share of the catalog, never above the flat quota."""

    def __init__(self, natural: int, pool: int, recipe: Recipe) -> None:
        self.p_natural = natural / pool if pool else 0.0
        self.flat = recipe.n_per_route
        self.floor = recipe.cell_floor
        self.cap = round(
            recipe.k_cap * self.p_natural * recipe.certified_total
        ) if recipe.k_cap is not None else self.flat
        self.quota = min(self.flat, self.cap)

    @property
    def generation_only(self) -> bool:
        """Whether the capped draw, both routes, stays under the floor."""
        return 2 * self.quota < self.floor


class CellPlan(NamedTuple):
    """One cell's draw plus the readout line a human reads before paying
    for labels."""

    rows: pd.DataFrame
    cell: str
    p_natural: float
    quota: int
    mass_cap: int
    generation_only: bool
    lanes: int
    top_lane: str
    top_share: float
    lane_share_target: float
    lane_share_floor: float
    lane_share_cap: float
    lane_share_achieved: float
    lane_share_infeasible: bool
    natural_rows: int
    natural_capacity: int
    reused_available: int
    reused_dense: int
    reused_sparse: int
    dense_short: int
    sparse_short: int
    labels_wanted: int
    labels_queued: int
    augmentation_rows: int
    pilot_dependent: bool

    @property
    def readout(self) -> dict[str, object]:
        return {
            name: value
            for name, value in self._asdict().items()
            if name != "rows"
        }


class CellFill:
    """Builds and owns the cell-quota artifact: selection, order sheet,
    per-cell readout, summary."""

    def __init__(
        self,
        recipe: Recipe | None = None,
        catalog_path: Path | None = None,
        labels_path: Path | None = None,
        out_dir: Path | None = None,
        cells: tuple[ArchetypeCell, ...] = CELLS,
    ) -> None:
        # the quota set is injectable so a v3 build (own cells, own out_dir)
        # can never rewrite the v2 artifacts; the default keeps v2 byte-identical
        self._cells = cells
        self._recipe = recipe if recipe is not None else Recipe()
        self._catalog_path = (
            catalog_path if catalog_path is not None else DEFAULT_CATALOG
        )
        self._labels_path = (
            labels_path if labels_path is not None else DEFAULT_LABELS
        )
        self._out_dir = out_dir if out_dir is not None else DEFAULT_OUT_DIR
        self._datasets = {dataset.name: dataset for dataset in DATASETS}
        self._rng = np.random.default_rng(self._recipe.seed)
        self._rates = pd.DataFrame(columns=list(ROUTES), dtype=float)
        self._pooled = pd.Series(0.0, index=list(ROUTES))

    @property
    def selection_path(self) -> Path:
        return self._out_dir / "cell_selection.parquet"

    @property
    def order_sheet_path(self) -> Path:
        return self._out_dir / "cell_order_sheet.parquet"

    @property
    def report_path(self) -> Path:
        return self._out_dir / "cell_report.parquet"

    @property
    def summary_path(self) -> Path:
        return self._out_dir / "cell_summary.md"

    def build(self, *, force: bool = False) -> pd.DataFrame:
        if self.selection_path.exists() and not force:
            return pd.read_parquet(self.selection_path)
        catalog = self._catalog()
        masks = {cell.name: cell.select(catalog).to_numpy() for cell in self._cells}
        plans = [
            self._plan(cell, catalog[masks[cell.name]], len(catalog))
            for cell in self._cells
        ]
        cells = pd.concat([plan.rows for plan in plans], ignore_index=True)
        unclaimed = catalog[~np.logical_or.reduce(list(masks.values()))]
        selection = pd.concat(
            [cells, self._control(unclaimed, len(cells))], ignore_index=True,
        )
        report = pd.DataFrame([plan.readout for plan in plans])
        sheet = self._order_sheet(report)
        self._assert_invariants(selection, report, sheet)
        selection["floors"] = self._membership(catalog, masks, selection)
        selection["query"] = join_text(selection, self._datasets)
        absent = sorted(set(self._datasets) - set(catalog["dataset"]))
        self._write(selection, sheet, report, absent, len(unclaimed) / len(catalog))
        return selection

    def admit(
        self,
        pool: pd.DataFrame,
        *,
        extractor: FeatureExtractor | None = None,
    ) -> pd.DataFrame:
        """Admit the pool rows that land in their own cell, each capped at that
        cell's shortfall and its lane share (d51j). Rewrites the selection and
        the order sheet; returns the admitted rows."""
        selection = pd.read_parquet(self.selection_path)
        if "provenance" not in selection.columns:
            # a selection written before d51k: the fill only ever drew catalog
            # rows, so every row it holds is natural
            selection["provenance"] = NATURAL
        sheet = pd.read_parquet(self.order_sheet_path)
        cells = {cell.name: cell for cell in self._cells}
        fresh = self._admissible(pool, selection, sheet)
        if fresh.empty:
            print("cell admit: nothing admissible in the pool")
            return selection.iloc[:0]

        mini = mini_catalog(
            fresh,
            extractor or FeatureExtractor(engines=None),
            columns=tuple(
                {band.column for cell in self._cells for band in cell.bands}
            ),
        )
        admitted, gained = [], {}
        for line in sheet[sheet["missing"] > 0].itertuples(index=False):
            cell = cells.get(line.floor)
            if cell is None:
                continue
            members = fresh[
                (fresh["floor"] == line.floor)
                & cell.select(mini).to_numpy()
            ]
            if members.empty:
                continue
            share = float(
                self._report_share(line.floor) or self._recipe.target_lane_share
            )
            taken = self._take(
                members, int(line.missing), share, f"admit:{line.floor}",
            )
            if taken.empty:
                continue
            gained[line.floor] = len(taken)
            admitted.append(self._admitted_rows(taken, line.floor))

        if not admitted:
            print("cell admit: no pool row landed in a hungry cell")
            return selection.iloc[:0]
        rows = pd.concat(admitted, ignore_index=True)[selection.columns]
        updated = pd.concat([selection, rows], ignore_index=True)
        self._assert_natural_share(updated)
        new_sheet = self._credit_sheet(sheet, gained)
        updated.to_parquet(self.selection_path, index=False)
        new_sheet.to_parquet(self.order_sheet_path, index=False)
        print(
            f"cell admit: {len(rows):,} rows into {len(gained)} cells | "
            f"selection {len(selection):,} -> {len(updated):,} | hungry cells "
            f"{int((sheet['missing'] > 0).sum())} -> "
            f"{int((new_sheet['missing'] > 0).sum())}"
        )
        return rows

    @staticmethod
    def _admissible(
        pool: pd.DataFrame, selection: pd.DataFrame, sheet: pd.DataFrame,
    ) -> pd.DataFrame:
        """Ungated pool rows for a still-hungry cell that the selection does
        not already hold."""
        rows = pool
        if "credit_gate" in rows.columns:
            gated = rows["credit_gate"].fillna("none") != "none"
            if gated.any():
                print(
                    f"cell admit: skipping {int(gated.sum())} feature-stock "
                    "rows (gated operators, d42h)"
                )
            rows = rows[~gated]
        hungry = set(sheet.loc[sheet["missing"] > 0, "floor"])
        return rows[
            rows["floor"].isin(hungry)
            & ~rows["query_id"].isin(set(selection["query_id"]))
        ].assign(
            # the lane a child is labelled in is the lane the ledger caps on
            dataset=lambda frame: frame["home_lane"],
        ).reset_index(drop=True)

    def _report_share(self, cell: str) -> float | None:
        """The lane-share cap the fill computed for this cell, so admitted rows
        obey the same bound the natural draw did."""
        if not self.report_path.exists():
            return None
        report = pd.read_parquet(self.report_path)
        line = report[report["cell"] == cell]
        return float(line["lane_share_cap"].iloc[0]) if len(line) else None

    def _admitted_rows(self, taken: pd.DataFrame, cell: str) -> pd.DataFrame:
        """Pool rows in selection schema: labelled in their home lane, staged
        for labelling, provenance carried from the operator that made them."""
        return pd.DataFrame({
            "dataset": taken["home_lane"].to_numpy(),
            "query_id": taken["query_id"].to_numpy(),
            "checkable": True,
            "cell": cell,
            "stage": CANDIDATE,
            "route": pd.NA,
            "provenance": taken["provenance"].to_numpy(),
            "floors": [[cell]] * len(taken),
            "query": taken["query"].to_numpy(),
        })

    def _assert_natural_share(self, selection: pd.DataFrame) -> None:
        natural = (selection["provenance"] == NATURAL).mean()
        assert natural >= self._recipe.min_natural_share - 1e-9, (
            f"natural share {natural:.3f} fell below the recipe minimum "
            f"{self._recipe.min_natural_share}"
        )

    @staticmethod
    def _credit_sheet(
        sheet: pd.DataFrame, gained: dict[str, int]
    ) -> pd.DataFrame:
        """Credit what was admitted and drop the lines that reached zero."""
        out = sheet.copy()
        credited = out["floor"].map(gained).fillna(0.0)
        out["credit"] = out["credit"] + credited
        out["missing"] = out["missing"] - credited
        return out[out["missing"] > 1e-9].reset_index(drop=True)

    @staticmethod
    def _membership(
        catalog: pd.DataFrame,
        masks: dict[str, np.ndarray],
        selection: pd.DataFrame,
    ) -> pd.Series:
        """Every cell each selected row claims — the membership the
        augmentation operators filter parents on."""
        names = np.array(list(masks))
        matrix = np.column_stack([masks[name] for name in names])
        keys = _key(catalog)
        wanted = keys.isin(set(_key(selection))).to_numpy()
        lists = pd.Series(
            [list(names[row]) for row in matrix[wanted]],
            index=keys[wanted].to_numpy(),
        )
        return _key(selection).map(lists)

    def _catalog(self) -> pd.DataFrame:
        """The catalog with the decisive labels joined on (dataset, query_id)."""
        catalog = with_derived(
            pd.read_parquet(self._catalog_path)
            .sort_values(["dataset", "query_id"], kind="stable")
            .reset_index(drop=True)
        )
        labels = pd.read_parquet(self._labels_path)
        decisive = decisive_rows(labels)
        winners = decisive["winner"].astype(str)
        keys = _key(catalog)
        catalog["winner"] = keys.map(
            pd.Series(winners.to_numpy(), index=_key(decisive))
        )
        catalog["labelled"] = keys.isin(set(_key(labels)))
        self._rates, self._pooled = self._yields(labels, decisive, winners)
        return catalog

    def _yields(
        self, labels: pd.DataFrame, decisive: pd.DataFrame, winners: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """The observed decisive-win rate per route, by lane and pooled."""
        per_lane = labels.groupby("dataset").size()
        won = (
            decisive.assign(winner=winners)
            .groupby(["dataset", "winner"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=per_lane.index, columns=list(ROUTES), fill_value=0)
        )
        pooled = pd.Series(
            {route: (winners == route).sum() / len(labels) for route in ROUTES}
        )
        return won.div(per_lane, axis=0).fillna(0.0), pooled

    def _plan(
        self, cell: ArchetypeCell, sub: pd.DataFrame, pool: int
    ) -> CellPlan:
        """One cell's reuse draw, labelling queue and readout line."""
        mass = MassCap(len(sub), pool, self._recipe)
        quota = mass.quota
        counts = sub["dataset"].value_counts()
        cap = LaneCap(counts, 2 * quota, self._recipe.target_lane_share)
        reused = {
            route: self._take(
                sub[sub["winner"] == route], quota, cap.share,
                f"{cell.name}:{route}",
            )
            for route in ROUTES
        }
        short = {route: quota - len(rows) for route, rows in reused.items()}
        fresh = sub[~sub["labelled"]]
        wanted = self._wanted(fresh, short)
        # the queue is the batch a human pays for, so its cap binds at the
        # size actually drawn, not at the size asked for
        batch = cap.admits(fresh["dataset"].value_counts(), wanted)
        queue = self._take(fresh, batch, cap.share, f"{cell.name}:queue")
        quota_rows = pd.concat(list(reused.values()))
        rows = pd.concat(
            [self._tag(rows, cell.name, route) for route, rows in reused.items()]
            + [self._tag(queue, cell.name, None)],
            ignore_index=True,
        )
        return CellPlan(
            rows=rows,
            cell=cell.name,
            p_natural=round(mass.p_natural, 6),
            quota=quota,
            mass_cap=mass.cap,
            generation_only=mass.generation_only,
            lanes=len(counts),
            top_lane=str(counts.index[0]) if len(counts) else "",
            top_share=_top_share(counts, len(sub)),
            lane_share_target=cap.target,
            lane_share_floor=round(cap.floor, 3),
            lane_share_cap=round(cap.share, 3),
            lane_share_achieved=_top_share(
                quota_rows["dataset"].value_counts(), cap.draw,
            ),
            lane_share_infeasible=cap.infeasible,
            natural_rows=len(sub),
            natural_capacity=cap.capacity,
            reused_available=int(sub["winner"].isin(ROUTES).sum()),
            reused_dense=len(reused[DENSE]),
            reused_sparse=len(reused[SPARSE]),
            dense_short=short[DENSE],
            sparse_short=short[SPARSE],
            labels_wanted=wanted,
            labels_queued=len(queue),
            augmentation_rows=max(0, cap.draw - cap.capacity),
            pilot_dependent=len(sub) < AugmentationConfig().pilot_n,
        )

    def _wanted(self, fresh: pd.DataFrame, short: dict[str, int]) -> int:
        """Rows to label for the deficits, at the rate the queue's own lanes
        have historically shown."""
        if fresh.empty:
            return 0
        weights = fresh["dataset"].value_counts(normalize=True)
        rates = self._rates.reindex(weights.index).fillna(self._pooled)
        needs = []
        for route in ROUTES:
            rate = float((weights * rates[route]).sum())
            if short[route] > 0 and rate > 0:
                needs.append(int(np.ceil(short[route] / rate)))
        return max(needs, default=0)

    def _take(
        self, pool: pd.DataFrame, amount: int, share: float, key: str,
    ) -> pd.DataFrame:
        """Lane-interleaved draw: every lane offers a row in turn and the
        ledger refuses whatever would break the share."""
        if pool.empty or amount <= 0:
            return pool.iloc[:0]
        shuffled = pool.sample(frac=1.0, random_state=self._rng)
        lanes = sorted(shuffled["dataset"].unique())
        # never waived: a route whose only winners sit in one lane is the
        # confound the share exists to break, not a sole-supplier exemption
        ledger = FloorLedger(
            [FloorSpec(
                key=key,
                amount=float(amount),
                credit=pd.Series(1.0, index=shuffled.index),
                cap_waived=False,
            )],
            lanes,
            share,
        )
        codes = shuffled["dataset"].map(
            {lane: index for index, lane in enumerate(lanes)}
        ).to_numpy()
        turns = shuffled.groupby("dataset", sort=False).cumcount().to_numpy()
        credit = np.array([1.0])
        picked: list[int] = []
        for position in np.lexsort((codes, turns)):
            if ledger.credit[0] >= amount:
                break
            if not ledger.uncapped()[0, codes[position]]:
                continue
            ledger.add(credit, int(codes[position]), QRELS)
            picked.append(int(position))
        return shuffled.iloc[picked]

    @staticmethod
    def _tag(rows: pd.DataFrame, cell: str, route: str | None) -> pd.DataFrame:
        frame = rows[ID_COLUMNS].copy()
        frame["cell"] = cell
        frame["stage"] = CANDIDATE if route is None else REUSED
        frame["route"] = pd.NA if route is None else route
        frame["provenance"] = NATURAL
        return frame

    def _control(self, unclaimed: pd.DataFrame, cell_rows: int) -> pd.DataFrame:
        """The never-trained slice: uniform draws from the queries no cell
        claimed."""
        take = min(
            round(self._recipe.control_share * cell_rows), len(unclaimed),
        )
        rows = (
            unclaimed.sample(n=take, random_state=self._rng) if take
            else unclaimed.iloc[:0]
        )
        frame = self._tag(rows, "", None)
        frame["stage"] = CONTROL
        return frame

    def _order_sheet(self, report: pd.DataFrame) -> pd.DataFrame:
        """Shortfall lines for the cells whose natural supply cannot reach
        the quota, keyed on the cell name the augmentation stack dispatches
        on."""
        short = report[report["augmentation_rows"] > 0]
        return pd.DataFrame({
            "floor": short["cell"].to_numpy(),
            "credit": short["natural_capacity"].to_numpy(dtype=float),
            "missing": short["augmentation_rows"].to_numpy(dtype=float),
        }).assign(
            slice=CELL_SLICE,
            amount=(2 * short["quota"]).to_numpy(dtype=float),
            reason=EXHAUSTED,
        )[["slice", "floor", "amount", "credit", "missing", "reason"]]

    def _assert_invariants(
        self,
        selection: pd.DataFrame,
        report: pd.DataFrame,
        sheet: pd.DataFrame,
    ) -> None:
        assert len(report) == len(self._cells), "readout lost a cell"
        duplicated = selection.duplicated(["cell", "stage", "dataset", "query_id"])
        assert not duplicated.any(), "duplicate (cell, stage, row) selection"
        over = report[
            report[["reused_dense", "reused_sparse"]].max(axis=1) > report["quota"]
        ]
        assert over.empty, f"cells over quota: {list(over['cell'])}"
        # one row of slack per route: the ledger admits the pick that crosses
        slack = 1.0 / report["quota"].clip(lower=1)
        broken = report[
            report["lane_share_achieved"] > report["lane_share_cap"] + slack
        ]
        assert broken.empty, f"cells over their lane cap: {list(broken['cell'])}"
        assert set(sheet["floor"]) == set(
            report.loc[report["augmentation_rows"] > 0, "cell"]
        ), "cell short of supply but not on the order sheet"

    def _write(
        self,
        selection: pd.DataFrame,
        sheet: pd.DataFrame,
        report: pd.DataFrame,
        absent: list[str],
        uncovered: float,
    ) -> None:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        selection.to_parquet(self.selection_path, index=False)
        sheet.to_parquet(self.order_sheet_path, index=False)
        report.to_parquet(self.report_path, index=False)
        self.summary_path.write_text(
            self._summary(selection, sheet, report, absent, uncovered)
        )

    def _summary(
        self,
        selection: pd.DataFrame,
        sheet: pd.DataFrame,
        report: pd.DataFrame,
        absent: list[str],
        uncovered: float,
    ) -> str:
        recipe = self._recipe
        stages = selection["stage"].value_counts()
        queue = selection[selection["stage"] == CANDIDATE]["dataset"]
        lanes = queue.value_counts().rename("queue_rows").to_frame()
        lanes["labelled_before"] = lanes.index.isin(self._rates.index)
        breached = list(report.loc[report["generation_only"], "cell"])
        return "\n".join([
            "# Cell fill — route-signal quotas\n",
            f"{len(selection):,} rows | "
            f"{selection.groupby(['dataset', 'query_id']).ngroups:,} distinct "
            f"queries | seed={recipe.seed} | n_per_route="
            f"{recipe.n_per_route} | k_cap={recipe.k_cap} of "
            f"{recipe.certified_total:,} | target lane share "
            f"{recipe.target_lane_share:.0%}\n",
            f"reused {stages.get(REUSED, 0):,} | queued for labelling "
            f"{stages.get(CANDIDATE, 0):,} | control "
            f"{stages.get(CONTROL, 0):,}\n",
            f"uncovered {uncovered:.1%} of the catalog — rows no cell claims, "
            "so the cap is applied to a partition missing that much mass\n",
            (
                f"generation-only {len(breached)} cells — capped below the "
                f"floor {recipe.cell_floor}, so no organic supply: "
                f"{', '.join(breached)}\n"
                if breached else
                f"every cell's allocation reaches the floor {recipe.cell_floor}\n"
            ),
            "## Per-cell readout\n",
            _block(report),
            "## Labelling queue by lane\n",
            _block(lanes),
            "## Order sheet\n",
            _block(sheet) if len(sheet)
            else "every cell reaches its quota from natural supply\n",
            "## Boxes with no catalog rows\n",
            (
                f"{len(absent)} registered, not in the catalog: "
                f"{', '.join(absent)} — every number above is recomputed per "
                "catalog, so a fetched box changes lane counts, caps and "
                "queues on the next build.\n"
                if absent else "every registered box is in the catalog\n"
            ),
        ])


def _block(frame: pd.DataFrame) -> str:
    return f"```\n{frame.to_string()}\n```\n"

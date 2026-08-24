"""The three v3 composition objectives as LAYERS (docs/v3_composition_
objectives.md): DiversityFloors = hard per-axis marginal constraints,
UtilityObjective = the sole scalar the draw maximizes, InversionBound =
representation checked as bounds at build and weighted only at eval."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pathlib import Path

from augmentation.config import AugmentationConfig
from composition.cellfill import CELL_SLICE, EXHAUSTED
from composition.pool_v3 import (
    CLASSES,
    CORPUS_AXES,
    REUSED,
    STRATA,
    UNKNOWN,
    native_mask,
)
from composition.recipe import Recipe

CORRUPTION_SLICE = "corruption"
SHEET_COLUMNS = ["slice", "floor", "amount", "credit", "missing", "reason"]
CLASS_ALLOCATION = "class_allocation"
ORDER_COLUMNS = ["dataset", "query_id", "reason"]


class DiversityFloors:
    """Layer 1: what MUST be covered, one floor per marginal axis. Cell and
    corruption deficits go on the generation sheet, corpus bands have no minter
    and stay report-only, the class shortfall belongs to the labelling rung,
    and the lane floor is `per_dataset`'s alone."""

    def __init__(self, recipe: Recipe, cells: tuple) -> None:
        self._recipe = recipe
        self._cells = cells

    def marginals(self, selected: pd.DataFrame) -> pd.DataFrame:
        """Every stratum the axes DECLARE — a band no selected row landed in
        reports 0 rows and fails, while `dark` carries the axis's uncovered
        rows, which are a coverage fact and never a floor."""
        rows = [
            {"axis": "cell", "stratum": cell.name, "dark": 0,
             "rows": int(selected["cells"].map(lambda s, n=cell.name: n in s).sum())}
            for cell in self._cells
        ]
        for axis, bands in STRATA.items():
            counts = selected[axis].value_counts()
            dark = int(counts.get(UNKNOWN, 0))
            for stratum, n in counts.reindex(list(bands), fill_value=0).items():
                rows.append({
                    "axis": axis, "stratum": str(stratum), "rows": int(n),
                    "dark": dark,
                })
        out = pd.DataFrame(rows)
        out["floor"] = self._recipe.stratum_floor
        out["met"] = out["rows"] >= self._recipe.stratum_floor
        return out

    def class_debt(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Each class's share of target_total minus the pool's NATIVE tier-0
        supply — the rows only acquisition (labelling or generation) can add."""
        live = pool[~pool["is_waste"] & native_mask(pool)]
        rows = []
        for name, share in zip(CLASSES, self._recipe.target_split):
            target = round(self._recipe.target_total * share)
            supply = int((live["route_class_any"] == name).sum())
            rows.append({
                "route_class": name, "target": target, "supply": supply,
                "debt": max(0, target - supply),
                "x_under": round(target / supply, 1) if supply else float("inf"),
            })
        return pd.DataFrame(rows)

    def order_sheet(
        self, selected: pd.DataFrame, pool: pd.DataFrame
    ) -> pd.DataFrame:
        """The generation debt ledger in CellFill's exact sheet schema: every
        floor shortfall generation alone can close — a cell's, net of both the
        selection and the unspent supply that already satisfies it, plus the
        corruption line — with credit at 0 for admit() to fill, while the class
        shortfall goes to the labelling rung (SelectionOrder) instead."""
        recipe = self._recipe
        spare = pool[~pool["is_waste"] & native_mask(pool)].drop(
            index=selected.index, errors="ignore"
        )
        lines = []
        for cell in self._cells:
            in_selection = int(
                selected["cells"].map(lambda s, n=cell.name: n in s).sum()
            )
            unspent = int(spare["cells"].map(lambda s, n=cell.name: n in s).sum())
            owed = recipe.stratum_floor - in_selection - unspent
            if owed <= 0:
                continue
            lines.append({
                "slice": CELL_SLICE, "floor": cell.name,
                "amount": float(owed), "credit": 0.0, "missing": float(owed),
                "reason": EXHAUSTED,
            })
        lines.append(self._corruption_line(pool))
        sheet = pd.DataFrame(lines, columns=SHEET_COLUMNS)
        return sheet[sheet["missing"] > 0].reset_index(drop=True)

    def _corruption_line(self, pool: pd.DataFrame) -> dict:
        """Damaged-query debt at the census's own pooled any-span rate of the
        full target_total, shipped as LIGHT — the minimal damage that counts."""
        census = pd.read_parquet(AugmentationConfig().paths.corruption_census)
        rate = float(
            (census["n_sampled"] * census["any_span"]).sum()
            / census["n_sampled"].sum()
        )
        target = round(rate * self._recipe.target_total)
        damaged = int(
            (pool["corruption_degree"].isin(["light", "heavy"])
             & native_mask(pool)).sum()
        )
        owed = float(max(0, target - damaged))
        return {
            "slice": CORRUPTION_SLICE, "floor": "corruption:light",
            "amount": owed, "credit": 0.0, "missing": owed, "reason": EXHAUSTED,
        }


class SelectionOrder:
    """The labelling rung's demand (the brief's Phase-3 order: label the
    datasets first, generate the residual): each class's target_total debt is
    allocated across lanes by their MEASURED blind yield, then the exact
    unlabelled catalog queries are named — answer-coverage gated, so a pick
    can never buy a guaranteed all_zero label."""

    def __init__(self, recipe: Recipe, cells: tuple, data_dir: Path) -> None:
        self._recipe = recipe
        self._cells = cells
        self._data = data_dir

    @staticmethod
    def yields(pool: pd.DataFrame) -> pd.DataFrame:
        """Per-lane tier-0 class yields from BLIND rows only — reused rows
        entered for already having won and estimate nothing."""
        blind = pool[pool["stage"] != REUSED]
        out = blind.groupby("dataset").size().to_frame("blind_labelled")
        for name in CLASSES:
            hits = blind[(blind["route_class_any"] == name) & ~blind["is_waste"]]
            out[f"yield_{name}"] = (
                hits.groupby("dataset").size().reindex(out.index).fillna(0)
                / out["blind_labelled"]
            )
        return out

    def allocate(
        self, debt: pd.DataFrame, yields: pd.DataFrame, per_ds: pd.DataFrame,
        pool: pd.DataFrame,
    ) -> pd.DataFrame:
        """Greedy on the biggest remaining debt: buy labels where the measured
        yield is best, capped by fresh supply and by each lane's remaining
        share of the class target, crediting every label's incidental
        other-class rows so the order never over-buys."""
        remaining = {r.route_class: float(r.debt) for r in debt.itertuples(index=False)}
        targets = {r.route_class: float(r.target) for r in debt.itertuples(index=False)}
        live = pool[~pool["is_waste"] & native_mask(pool)]
        supplied = {
            name: live[live["route_class_any"] == name].groupby("dataset").size()
            for name in CLASSES
        }
        cap = {
            (lane, name): max(
                0.0,
                self._recipe.target_lane_share * targets[name]
                - float(supplied[name].get(lane, 0)),
            )
            for lane in yields.index for name in CLASSES
        }
        fresh = per_ds["fresh_left"].to_dict()
        ordered: dict[str, float] = {}
        for name in sorted(CLASSES, key=lambda k: -remaining[k]):
            lanes = yields[yields[f"yield_{name}"] > 0].sort_values(
                f"yield_{name}", ascending=False
            )
            for lane in lanes.index:
                if remaining[name] < 1.0:
                    break
                rate = float(lanes.at[lane, f"yield_{name}"])
                room = float(fresh.get(lane, 0)) - ordered.get(lane, 0.0)
                take = min(room, cap[(lane, name)] / rate, remaining[name] / rate)
                if take < 1.0:
                    continue
                ordered[lane] = ordered.get(lane, 0.0) + take
                # credit every class this take buys, but never beyond the
                # lane's remaining share cap — rows the draw cannot take
                # must not retire debt
                for k in CLASSES:
                    buy = min(
                        take * float(yields.at[lane, f"yield_{k}"]),
                        cap[(lane, k)],
                    )
                    remaining[k] -= buy
                    cap[(lane, k)] -= buy
        out = yields.loc[sorted(ordered, key=lambda k: -ordered[k])].copy()
        out["labels_ordered"] = [round(ordered[lane]) for lane in out.index]
        for name in CLASSES:
            out[f"expected_{name}"] = (
                out["labels_ordered"] * out[f"yield_{name}"]
            ).round().astype(int)
        out["fresh_left"] = out.index.map(fresh)
        out.attrs["residual_debt"] = {
            k: int(max(0.0, v)) for k, v in remaining.items()
        }
        return out

    def picks(
        self,
        allocation: pd.DataFrame,
        pool: pd.DataFrame,
        catalog: pd.DataFrame,
        sheet: pd.DataFrame,
        *,
        oversample: float = 1.5,
    ) -> pd.DataFrame:
        """The named queries per allocated lane, drawn seeded from the lane's
        FRESH source queries and answer-coverage gated; hungry-cell targeting
        applies only where the catalog already carries the row's features
        (today: labelled rows only, so cells are served at natural rates
        until catalog_v3 extends to source queries)."""
        from scripts.label_routes import _source_name

        rng = np.random.default_rng(self._recipe.seed)
        hungry_names = set(sheet.loc[sheet["slice"] == CELL_SLICE, "floor"])
        hungry = [cell for cell in self._cells if cell.name in hungry_names]
        frames = []
        for lane in allocation.index:
            need = int(allocation.at[lane, "labels_ordered"])
            qpath = self._data / _source_name(str(lane)) / "queries.parquet"
            if need <= 0 or not qpath.exists():
                continue
            queries = pd.read_parquet(qpath, columns=["query_id"]).astype(
                {"query_id": str}
            )
            done = set(pool.loc[pool["dataset"] == lane, "query_id"].astype(str))
            fresh = queries[~queries["query_id"].isin(done)]
            if fresh.empty:
                continue
            cat_rows = catalog[
                (catalog["dataset"] == lane)
                & catalog["query_id"].isin(set(fresh["query_id"]))
            ]
            reason: dict[str, str] = {}
            for cell in hungry:
                for qid in cat_rows.loc[cell.select(cat_rows), "query_id"]:
                    reason.setdefault(str(qid), f"cell:{cell.name}")
            rest = fresh[~fresh["query_id"].isin(reason)].sample(
                frac=1.0, random_state=rng.integers(2**31)
            )
            candidates = (list(reason) + rest["query_id"].tolist())[
                : max(need, round(need * oversample))
            ]
            covered = self._answer_covered(lane, pd.Series(candidates), pool)
            take = [qid for qid in candidates if qid in covered][:need]
            frames.append(pd.DataFrame({
                "dataset": lane,
                "query_id": take,
                "reason": [reason.get(qid, CLASS_ALLOCATION) for qid in take],
            }))
        if not frames:
            return pd.DataFrame(columns=ORDER_COLUMNS)
        return pd.concat(frames, ignore_index=True)[ORDER_COLUMNS]

    def _answer_covered(
        self, lane: str, query_ids: pd.Series, pool: pd.DataFrame
    ) -> set[str]:
        """Query ids whose every graded answer doc exists in the lane corpus —
        labelling the rest buys a guaranteed all_zero row."""
        from scripts.label_routes import _source_name

        source = self._data / _source_name(lane)
        qrels_path, corpus_path = source / "qrels.parquet", source / "corpus.parquet"
        if not qrels_path.exists() or not corpus_path.exists():
            return set()
        lane_rows = pool.loc[pool["dataset"] == lane, "min_relevance"]
        min_rel = int(lane_rows.iloc[0]) if len(lane_rows) else 1
        qrels = pd.read_parquet(qrels_path).astype({"query_id": str, "doc_id": str})
        graded = qrels[
            (qrels["relevance"] >= min_rel)
            & qrels["query_id"].isin(set(query_ids.astype(str)))
        ]
        present: set[str] = set()
        doc_ids = graded["doc_id"].unique().tolist()
        for start in range(0, len(doc_ids), 2000):
            batch = doc_ids[start:start + 2000]
            present |= set(
                pd.read_parquet(
                    corpus_path, columns=["doc_id"],
                    filters=[("doc_id", "in", batch)],
                )["doc_id"].astype(str)
            )
        ok = graded.groupby("query_id")["doc_id"].agg(
            lambda docs: all(d in present for d in docs)
        )
        return set(ok[ok].index)

    def build(
        self, pool: pd.DataFrame, per_ds: pd.DataFrame, debt: pd.DataFrame,
        sheet: pd.DataFrame, catalog: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        allocation = self.allocate(debt, self.yields(pool), per_ds, pool)
        order = self.picks(allocation, pool, catalog, sheet)
        named = order.groupby("dataset").size()
        summary = allocation.assign(
            queries_named=named.reindex(allocation.index).fillna(0).astype(int)
        )
        summary.attrs.update(allocation.attrs)
        # a footer row, not attrs: attrs do not survive the parquet round-trip
        # and the residual is the number generation gets sized by
        summary.loc["(residual after labelling)"] = {
            f"expected_{k}": v
            for k, v in allocation.attrs["residual_debt"].items()
        }
        return order, summary


class LaneOrder:
    """The class-residual carrier: rows to MINT into each lane, sized by its
    measured class yield under the remaining lane-share cap and its unspent
    grounding documents — recomputed every build and never credited, because
    labelled minted rows re-net the residual through the pool."""

    FOOTER = "(reachable ceiling)"

    def __init__(self, recipe: Recipe) -> None:
        self._recipe = recipe

    def build(
        self,
        residual: dict[str, float],
        yields: pd.DataFrame,
        pool: pd.DataFrame,
        capacity: dict[str, int],
        pending: dict[str, int] | None = None,
    ) -> pd.DataFrame:
        """Greedy on the biggest residual class, lanes ranked by that class's
        yield, each take bounded by grounding capacity and the lane's
        remaining class share; incidental buys credited under the same caps."""
        remaining = {name: float(residual.get(name, 0)) for name in CLASSES}
        targets = {
            name: self._recipe.target_total * share
            for name, share in zip(CLASSES, self._recipe.target_split)
        }
        live = pool[~pool["is_waste"] & native_mask(pool)]
        supplied = {
            name: live[live["route_class_any"] == name].groupby("dataset").size()
            for name in CLASSES
        }
        lanes = yields[yields.index.isin(capacity)]
        cap = {
            (lane, name): max(
                0.0,
                self._recipe.target_lane_share * targets[name]
                - float(supplied[name].get(lane, 0)),
            )
            for lane in lanes.index for name in CLASSES
        }
        room = {lane: float(capacity.get(lane, 0)) for lane in lanes.index}
        ordered: dict[str, float] = {}
        for name in sorted(CLASSES, key=lambda k: -remaining[k]):
            ranked = lanes[lanes[f"yield_{name}"] > 0].sort_values(
                f"yield_{name}", ascending=False
            )
            for lane in ranked.index:
                if remaining[name] < 1.0:
                    break
                rate = float(ranked.at[lane, f"yield_{name}"])
                open_room = room[lane] - ordered.get(lane, 0.0)
                take = min(
                    open_room, cap[(lane, name)] / rate, remaining[name] / rate
                )
                if take < 1.0:
                    continue
                ordered[lane] = ordered.get(lane, 0.0) + take
                for k in CLASSES:
                    buy = min(
                        take * float(lanes.at[lane, f"yield_{k}"]),
                        cap[(lane, k)],
                    )
                    remaining[k] -= buy
                    cap[(lane, k)] -= buy
        # minted-but-unlabelled rows already carry paid demand — they re-net
        # through the pool only at labelling, so subtract them here or a
        # mid-crank restart re-orders rows it already owns
        owed = {
            lane: max(0, round(total) - (pending or {}).get(lane, 0))
            for lane, total in ordered.items()
        }
        # ranked and filtered in one index, because a boolean LIST that comes
        # out empty selects columns rather than rows — a fully-paid residual
        # then loses every yield column
        out = lanes.loc[
            [lane for lane in sorted(owed, key=lambda k: -owed[k]) if owed[lane] > 0]
        ].copy()
        out["rows_to_mint"] = [owed[lane] for lane in out.index]
        for name in CLASSES:
            out[f"expected_{name}"] = (
                out["rows_to_mint"] * out[f"yield_{name}"]
            ).round().astype(int)
        out["grounding_docs"] = [
            int(capacity.get(lane, 0)) for lane in out.index
        ]
        out.loc[self.FOOTER] = {
            f"expected_{name}": int(out[f"expected_{name}"].sum())
            for name in CLASSES
        }
        return out


class UtilityObjective:
    """Layer 2, the sole scalar: lane x route coverage with qrels-depth-vetted
    classes; cells and the other diversity strata only break ties. leg-1
    labels only — no confidence term exists until its referent, direction and
    mechanism are written down."""

    def __init__(self, recipe: Recipe) -> None:
        self._recipe = recipe

    @staticmethod
    def lane_counts(pool: pd.DataFrame) -> dict[str, pd.Series]:
        return {
            c: pool.loc[pool["route_class"] == c, "dataset"].value_counts()
            for c in CLASSES
        }

    @staticmethod
    def capped_supply(counts: pd.Series, target: int, lane_share_cap: float) -> int:
        """Rows actually drawable toward `target` when no lane may exceed
        the cap."""
        if target <= 0:
            return 0
        per_lane = max(1, math.ceil(lane_share_cap * target))
        return int(np.minimum(counts.to_numpy(), per_lane).sum())

    @staticmethod
    def feasible_total(
        lane_counts: dict[str, pd.Series],
        split: tuple[float, float, float],
        lane_share_cap: float = 1.0,
    ) -> int:
        """The largest total whose per-class targets fit the CAPPED supply —
        computed UNDER the lane cap, so the binding class never silently
        under-fills while the others fill to uncapped targets."""
        def fits(total: int) -> bool:
            return all(
                UtilityObjective.capped_supply(
                    lane_counts[c], round(total * s), lane_share_cap
                ) >= round(total * s)
                for c, s in zip(CLASSES, split) if s > 0
            )

        lo, hi = 0, min(
            int(lane_counts[c].sum() / s) for c, s in zip(CLASSES, split) if s > 0
        )
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

    @staticmethod
    def draw_class(
        cand: pd.DataFrame,
        target: int,
        per_lane_cap: int,
        pre_lane: dict[str, int] | None = None,
        pre_covered: set | None = None,
    ) -> list:
        """One class's draw: a TRUE greedy coverage pass (gain recomputed per
        pick, lane x route exhausted before diversity strata break ties) and
        then a seeded lane-capped fill; `pre_*` seed lane usage and covered
        strata with an outer tier's picks so a nested draw respects one
        shared cap."""
        rows = [
            (
                i,
                cand.at[i, "dataset"],
                frozenset(cand.at[i, "cells"])
                | {("corr", cand.at[i, "corruption_degree"])}
                | {(ax, cand.at[i, ax]) for ax in CORPUS_AXES},
            )
            for i in cand.index
        ]
        covered: set = set(pre_covered or ())
        per_lane: dict[str, int] = dict(pre_lane or {})
        lanes_drawn: set = set(per_lane)
        taken: list = []

        def admit(i, ds, diversity) -> None:
            taken.append(i)
            per_lane[ds] = per_lane.get(ds, 0) + 1
            lanes_drawn.add(ds)
            covered.update(diversity)

        remaining = rows
        while len(taken) < target:
            best, best_gain = None, (0, 0)
            for row in remaining:
                i, ds, diversity = row
                if per_lane.get(ds, 0) >= per_lane_cap:
                    continue
                gain = (ds not in lanes_drawn, len(diversity - covered))
                if gain > best_gain:
                    best, best_gain = row, gain
            if best is None or best_gain == (0, 0):
                break  # nothing uncovered remains — the fill takes over
            admit(*best)
            remaining = [r for r in remaining if r[0] != best[0]]
        for i, ds, diversity in remaining:
            if len(taken) >= target:
                break
            if per_lane.get(ds, 0) >= per_lane_cap:
                continue
            admit(i, ds, diversity)
        return taken

    def select(self, pool: pd.DataFrame) -> pd.DataFrame:
        """The two-tier draw: certified tier first at its own capped feasible
        max, an UNCERTIFIED-only top-up nested under one shared lane cap so
        the whole artifact also hits the split, then the dictated waste draw.
        `certified` is a per-row label-quality flag, never a draw record."""
        recipe = self._recipe
        counts_c = self.lane_counts(pool)
        total_c = self.feasible_total(
            counts_c, recipe.target_split, recipe.target_lane_share
        )
        targets_c = {
            c: round(total_c * s) for c, s in zip(CLASSES, recipe.target_split)
        }
        counts_0 = {
            c: pool.loc[pool["route_class_any"] == c, "dataset"].value_counts()
            for c in CLASSES
        }
        total_0 = self.feasible_total(
            counts_0, recipe.target_split, recipe.target_lane_share
        )
        targets_0 = {
            c: round(total_0 * s) for c, s in zip(CLASSES, recipe.target_split)
        }

        chosen: list = []
        for c in CLASSES:
            cand = pool[pool["route_class"] == c].sample(
                frac=1.0, random_state=recipe.seed
            )
            cap_c = max(1, math.ceil(recipe.target_lane_share * targets_c[c]))
            certified_taken = self.draw_class(cand, targets_c[c], cap_c)
            chosen.extend(certified_taken)

            pre_lane = pool.loc[certified_taken, "dataset"].value_counts().to_dict()
            pre_covered = set().union(
                *(pool.at[i, "cells"] for i in certified_taken), *(
                    {("corr", pool.at[i, "corruption_degree"]),
                     *((ax, pool.at[i, ax]) for ax in CORPUS_AXES)}
                    for i in certified_taken
                ),
            ) if certified_taken else set()
            # UNCERTIFIED rows only: admitting other certified rows of class c
            # here is how the certified TIER once ended up wider than its own
            # draw and off-split — the top-up may only add supply the
            # certified pass could not reach.
            top_up = pool[
                (pool["route_class_any"] == c) & ~pool["certified"]
            ].sample(frac=1.0, random_state=recipe.seed)
            cap_0 = max(1, math.ceil(recipe.target_lane_share * targets_0[c]))
            chosen.extend(self.draw_class(
                top_up, targets_0[c] - len(certified_taken), cap_0,
                pre_lane=pre_lane, pre_covered=pre_covered,
            ))

        waste_budget = round(recipe.waste_cap * total_0)
        waste: list = []
        if waste_budget > 0:
            wcand = pool[pool["is_waste"]].sample(
                frac=1.0, random_state=recipe.seed
            )
            wcap = max(1, math.ceil(recipe.target_lane_share * waste_budget))
            per_lane: dict[str, int] = {}
            for i in wcand.index:
                if len(waste) >= waste_budget:
                    break
                ds = wcand.at[i, "dataset"]
                if per_lane.get(ds, 0) >= wcap:
                    continue
                waste.append(i)
                per_lane[ds] = per_lane.get(ds, 0) + 1

        selected = pool.loc[chosen + waste].assign(selected=True)
        # the artifact's class is the tier-0 one (identical for certified
        # rows); waste is the typed fourth value and never certified
        selected["route_class"] = selected["route_class_any"]
        selected.loc[selected["is_waste"], "route_class"] = "waste"
        selected.loc[selected["is_waste"], "certified"] = False
        selected.attrs["waste_budget"] = waste_budget
        selected.attrs["total_certified"] = total_c
        selected.attrs["total_tier0"] = total_0
        return selected


class InversionBound:
    """Layer 3: no archetype over-represented >=K x under ANY plausible source
    weighting — an outcome check on the composed artifact, distinct from the
    K_cap ALLOCATION dial. Numbers only; the user freezes K."""

    def __init__(self, recipe: Recipe, cells: tuple) -> None:
        self._recipe = recipe
        self._cells = cells

    @staticmethod
    def weightings(pool_counts: pd.Series, lane_share_cap: float) -> dict[str, pd.Series]:
        """The plausible-source family. `pool_share` is our own acquisition
        mix (a candidate, never an anchor — it is proxy-circular as a prior);
        `lane_uniform` and the capped variant bracket it."""
        share = pool_counts / pool_counts.sum() if pool_counts.sum() else pool_counts
        return {
            "lane_uniform": pd.Series(1.0 / len(pool_counts), index=pool_counts.index),
            "pool_share": share,
            "pool_share_capped": (capped := share.clip(upper=lane_share_cap))
            / capped.sum(),
        }

    def report(self, selected: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
        """Per cell: the selected share against the minimum share any
        candidate weighting implies. `floor_ratio` is the over-representation
        the stratum floor itself MANDATES; a cell is floor_forced when the
        observed ratio does not exceed the mandated one."""
        weightings = self.weightings(
            pool["dataset"].value_counts(), self._recipe.target_lane_share
        )
        lane_of = pool.groupby("dataset")
        rows = []
        floor_share = self._recipe.stratum_floor / max(len(selected), 1)
        for cell in self._cells:
            in_cell = pool["cells"].map(lambda s, n=cell.name: n in s)
            p_lane = lane_of.apply(
                lambda g, m=in_cell: float(m.loc[g.index].mean()),
                include_groups=False,
            )
            p_w = {
                name: float((w * p_lane.reindex(w.index).fillna(0.0)).sum())
                for name, w in weightings.items()
            }
            min_p = min(p_w.values())
            s = float(
                selected["cells"].map(lambda x, n=cell.name: n in x).mean()
            )
            max_ratio = s / min_p if min_p > 0 else np.inf
            floor_ratio = floor_share / min_p if min_p > 0 else np.inf
            rows.append({
                "cell": cell.name, "selected_share": round(s, 6),
                "min_weighted_share": round(min_p, 6),
                **{f"share_{k}": round(v, 6) for k, v in p_w.items()},
                "max_ratio": round(max_ratio, 2) if np.isfinite(max_ratio) else np.inf,
                "floor_ratio": round(floor_ratio, 2) if np.isfinite(floor_ratio) else np.inf,
                "floor_forced": bool(max_ratio <= floor_ratio + 1e-9),
            })
        return pd.DataFrame(rows).sort_values(
            "max_ratio", ascending=False
        ).reset_index(drop=True)

    def realism(self, selected: pd.DataFrame) -> pd.DataFrame:
        """Row realism: the artifact's corruption-degree mix against the
        census's own pooled rate — the one external fact about how damaged
        real traffic is."""
        census = pd.read_parquet(AugmentationConfig().paths.corruption_census)
        pooled = float(
            (census["n_sampled"] * census["any_span"]).sum()
            / census["n_sampled"].sum()
        )
        shares = selected["corruption_degree"].value_counts(normalize=True)
        return pd.DataFrame([{
            "selected_damaged_share": round(
                float(shares.get("light", 0) + shares.get("heavy", 0)), 4
            ),
            "census_any_span_rate": round(pooled, 4),
            "selected_unknown_share": round(float(shares.get("unknown", 0)), 4),
        }])

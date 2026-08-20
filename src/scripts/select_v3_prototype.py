"""v3 selector PROTOTYPE (Phase 0) — feasibility + shortfall report over the
EXISTING labelled pool, no retrieval, no generation.

Proves whether the v3 objectives are jointly reachable and, where they are not,
what the 200K generation debt is. Route classes: dense/sparse = decisive
routes_differ wins (margin >= class_margin); hybrid = genuine ties (all_tied
with >= 2 judged docs — a shallow tie is unreadable at any score) + rrf-decisive. Fake ties + all_zero are
a capped waste budget. Diversity strata (additive, never crossed): cell,
corruption degree, and three marginal corpus-relative axes (corpus_idf,
corpus_oov, corpus_pmi). Utility coverage: (lane x decisive-route). Per-dataset
decisive floor is a hard constraint whose shortfall is the headline.

Corpus-relative values live here as stratum axes and NOT as cells: a cell must
be recomputable from query text alone, while these need a CorpusIndex and
differ per collection.

    poetry run python src/scripts/select_v3_prototype.py                 # 45/45/10, floor 25
    poetry run python src/scripts/select_v3_prototype.py --floor 50 --split 40 40 20
    poetry run python src/scripts/select_v3_prototype.py --class-margin 0.0   # routes_differ, not decisive

Reads: data/route_labels/labels.parquet, data/feature_table/catalog.parquet,
data/<dataset>/qrels.parquet. Writes: data/route_labels/v3_feasibility/.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from composition.cells import CELLS
from composition.cells_v3 import CELLS_V3
from composition.floors import CORRUPTION_SPANS, with_derived

DATA = Path(__file__).resolve().parent.parent / "data"
LABELS = DATA / "route_labels" / "labels.parquet"
CATALOG = DATA / "feature_table" / "catalog.parquet"
OUT = DATA / "route_labels" / "v3_feasibility"

V3_CATALOG = DATA / "v3" / "catalog_v3.parquet"
REDERIVED = DATA / "v3" / "labels_rederived.parquet"

REUSED = "reused"
"""Selection stage whose rows were drawn BECAUSE they already won a route, so
their decisive rate is conditioned on the outcome and estimates nothing."""

SCORES = ["score_dense_only", "score_pure_rrf", "score_sparse_only"]
ROUTE_TO_CLASS = {"dense_only": "dense", "sparse_only": "sparse", "pure_rrf": "hybrid"}
CLASSES = ("dense", "sparse", "hybrid")
CORPUS_AXES = ("corpus_idf", "corpus_oov", "corpus_pmi")
"""Corpus-relative diversity, one MARGINAL axis each — never crossed with each
other, with cell, or with lane."""
CEILING = 0.999
"""Reporting bar only: splits each tie kind into at-ceiling (everyone found a
judged doc at rank 1) vs all-routes-missed. Classification is depth-based."""
TOL = 1e-9


@dataclass(frozen=True)
class SelectorRecipe:
    """Every dial for the prototype; folds into composition/recipe.py when the
    selector graduates. Fold mapping: lane_share_cap unifies with
    Recipe.target_lane_share (cap vs measured-against semantics must merge) and
    seed with Recipe.seed — same numbers, currently different contracts."""

    target_split: tuple[float, float, float] = (0.45, 0.45, 0.10)  # dense, sparse, hybrid
    per_dataset_floor: int = 25
    waste_cap: float = 0.0  # dictated waste as a fraction of T (0 = exclude waste)
    genuine_tie_depth: int = 2  # a tie with fewer judged docs than this is fake
    class_margin: float = 0.4  # decisive if oracle - runner_up >= this
    lane_share_cap: float = 0.2  # no single dataset > this share of a class
    seed: int = 0
    target_total: int = 200_000  # for the 200K extrapolation


# ---------------------------------------------------------------- classify ---
def _load_labels() -> pd.DataFrame:
    """The re-derived pool (v2's labels rescored from the oracle caches at each
    lane's current min_relevance) PLUS the additive v3 labels, aligned on the
    base's columns. Falls back to v2's labels.parquet until the re-derivation has
    run; both are read-only. New (dataset, query_id) pairs are disjoint, so the
    dedup only guards against a re-labelled row."""
    base_path = REDERIVED if REDERIVED.exists() else LABELS
    base = pd.read_parquet(base_path).astype({"query_id": str})
    v3_path = DATA / "v3" / "labels.parquet"
    if not v3_path.exists():
        return base
    v3 = pd.read_parquet(v3_path).astype({"query_id": str})
    v3 = v3.reindex(columns=base.columns)  # absent cols (cell/stage) -> NaN
    combined = pd.concat([base, v3], ignore_index=True)
    return combined.drop_duplicates(["dataset", "query_id"], keep="first")


def _qrels_depth(labels: pd.DataFrame) -> pd.Series:
    """Judged-relevant doc count per row at its own min_relevance — the
    fake-tie discriminator, from on-disk qrels, no retrieval."""
    out = []
    for dataset, grp in labels.groupby("dataset"):
        qrels = pd.read_parquet(DATA / dataset / "qrels.parquet").astype(
            {"query_id": str}
        )
        want = grp[["query_id", "min_relevance"]].drop_duplicates()
        hit = qrels.merge(want, on="query_id", how="inner")
        depth = (
            hit[hit["relevance"] >= hit["min_relevance"]]
            .groupby("query_id")
            .size()
            .rename("depth")
        )
        joined = grp[["query_id"]].merge(depth, on="query_id", how="left")
        joined["dataset"] = dataset
        out.append(joined)
    depths = pd.concat(out).fillna({"depth": 0})
    return labels.merge(
        depths, on=["dataset", "query_id"], how="left"
    )["depth"].to_numpy()


def assign_classes(oracle, runner, low, winner, depth, recipe: SelectorRecipe):
    """Pure policy: score vectors + qrels depth -> (kind, route_class) arrays.
    kind in {decisive, genuine_tie, fake_tie, all_zero, undecisive};
    route_class in {dense, sparse, hybrid, ''}. No I/O — unit-tested directly."""
    zero = oracle <= TOL
    tied = (~zero) & (oracle - low <= TOL)
    differ = (~zero) & (~tied)
    decisive = differ & (oracle - runner >= recipe.class_margin)
    # a shallow tie is unreadable at ANY score: at ceiling everyone found the
    # one judged doc, below it everyone missed it — both are the qrels-depth
    # artifact. The old rule required the ceiling too, which misfiled 491
    # all-routes-missed rows as genuine hybrid supply.
    fake_tie = tied & (depth < recipe.genuine_tie_depth)
    genuine_tie = tied & ~fake_tie

    kind = np.full(len(oracle), "undecisive", dtype=object)
    kind[zero] = "all_zero"
    kind[fake_tie] = "fake_tie"
    kind[genuine_tie] = "genuine_tie"
    kind[decisive] = "decisive"

    cls = np.full(len(oracle), "", dtype=object)
    dec_class = np.array([ROUTE_TO_CLASS.get(w, "") for w in winner], dtype=object)
    cls[decisive] = dec_class[decisive]
    cls[genuine_tie] = "hybrid"
    return kind, cls


def classify(labels: pd.DataFrame, recipe: SelectorRecipe) -> pd.DataFrame:
    """Attach oracle/margin/depth and the row's route class + kind."""
    ordered = np.sort(labels[SCORES].to_numpy(), axis=1)
    oracle, runner, low = ordered[:, -1], ordered[:, -2], ordered[:, 0]
    winner = labels[SCORES].idxmax(axis=1).str.replace("score_", "").to_numpy()
    depth = _qrels_depth(labels)
    kind, cls = assign_classes(oracle, runner, low, winner, depth, recipe)

    return labels.assign(
        oracle=oracle, margin=oracle - runner, depth=depth,
        winner=winner, kind=kind, route_class=cls,
        is_decisive=np.isin(cls, CLASSES) & (kind != "genuine_tie"),
        is_hybrid=cls == "hybrid",
        is_waste=np.isin(kind, ["fake_tie", "all_zero"]),
    )


# ------------------------------------------------------------------ strata ---
def _active_cells(catalog: pd.DataFrame | None = None):
    """v2 cells, plus the v3 cells when the v3 catalog (their columns) exists.
    Given a catalog, cells banding on a column it does not carry are dropped —
    a not-yet-backfilled column is a KeyError in `cell.select`, and the cell is
    BLOCKED until the build runs, not silently empty."""
    cells = (*CELLS, *CELLS_V3) if V3_CATALOG.exists() else CELLS
    if catalog is None:
        return cells
    return tuple(
        cell for cell in cells
        if all(band.column in catalog.columns for band in cell.bands)
    )


def attach_strata(pool: pd.DataFrame) -> pd.DataFrame:
    """cell membership (multi) and corruption degree from the catalog, corpus
    axes from query_corpus_stats. Lane is `dataset` (the corpus proxy here)."""
    # v3 catalog (v2 columns + new taxonomy columns) enables the v3 cells; fall
    # back to the v2 catalog + v2 cells when it has not been built yet.
    catalog_path = V3_CATALOG if V3_CATALOG.exists() else CATALOG
    catalog = with_derived(pd.read_parquet(catalog_path).astype({"query_id": str}))
    idx = catalog.set_index(["dataset", "query_id"]).index
    per_row: list[set] = [set() for _ in range(len(catalog))]
    for cell in _active_cells(catalog):
        for i in np.nonzero(cell.select(catalog).to_numpy())[0]:
            per_row[i].add(cell.name)
    cell_lookup = dict(zip(idx, per_row))

    # Corruption degree reads the catalog's own derived total — the TOKENIZER-
    # inclusive pass the v3 damage cells band on. The REGEX+WORDFREQ
    # re-extraction that used to live here disagreed on 1.0% of rows (0.9% of
    # degrees), always by seeing FEWER spans, so it demoted 367 damaged rows to
    # clean: two structures on one axis. Rows the catalog does not carry (the
    # additive v3 labels) are 'unknown', not 'clean'.
    spans = dict(zip(idx, catalog[CORRUPTION_SPANS]))
    pool_idx = list(zip(pool["dataset"], pool["query_id"]))
    total = np.array([spans.get(k, np.nan) for k in pool_idx], dtype=float)
    degree = pd.cut(total, [-np.inf, 0.5, 1.5, np.inf],
                    labels=["clean", "light", "heavy"]).astype(object)
    pool = pool.assign(
        cells=[frozenset(cell_lookup.get(k, set())) for k in pool_idx],
        corruption_spans=total,
        corruption_degree=pd.Series(degree).fillna("unknown").to_numpy(),
    )
    return _attach_corpus_strata(pool)


def _attach_corpus_strata(pool: pd.DataFrame) -> pd.DataFrame:
    """The three marginal corpus axes from query_corpus_stats.parquet: IDF
    quartile band (edges measured off the file, never hand numbers), OOV
    presence, PMI sentinel. Rows the file does not carry -> 'unknown'."""
    qcs_path = DATA / "route_labels" / "query_corpus_stats.parquet"
    if not qcs_path.exists():
        return pool.assign(**dict.fromkeys(CORPUS_AXES, "unknown"))
    qcs = (
        pd.read_parquet(qcs_path)
        .astype({"query_id": str})
        .drop_duplicates(["dataset", "query_id"])
        .set_index(["dataset", "query_id"])
    )
    key = pd.MultiIndex.from_arrays([pool["dataset"], pool["query_id"]])
    absent = ~key.isin(qcs.index)
    rows = qcs.reindex(key)
    p25, p75 = qcs["avg_idf"].quantile([0.25, 0.75])
    idf, oov, pmi = (rows[c].to_numpy() for c in ("avg_idf", "oov_share", "min_pmi"))
    return pool.assign(
        corpus_idf=np.where(
            absent | np.isnan(idf), "unknown",
            np.where(idf < p25, "low_idf",
                     np.where(idf >= p75, "high_idf", "mid_idf")),
        ),
        corpus_oov=np.where(
            absent | np.isnan(oov), "unknown",
            np.where(oov > 0, "has_oov", "in_vocab"),
        ),
        # -1.0 exactly is PMIBank's "never co-occurs" sentinel — measured. NaN
        # is the pair never being measurable at all; binning the two together
        # would call 698 unmeasured rows a structural miss.
        corpus_pmi=np.where(
            absent, "unknown",
            np.where(np.isnan(pmi), "unmeasured",
                     np.where(pmi == -1.0, "never_co_occurs", "co_occurring")),
        ),
    )


# ---------------------------------------------------------------- analytics ---
def _lane_counts(pool: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        c: pool.loc[pool["route_class"] == c, "dataset"].value_counts()
        for c in CLASSES
    }


def _capped_supply(counts: pd.Series, target: int, lane_share_cap: float) -> int:
    """Rows actually drawable toward `target` when no lane may exceed the cap."""
    if target <= 0:
        return 0
    per_lane = max(1, math.ceil(lane_share_cap * target))
    return int(np.minimum(counts.to_numpy(), per_lane).sum())


def feasible_total(
    lane_counts: dict[str, pd.Series],
    split: tuple[float, float, float],
    lane_share_cap: float = 1.0,
) -> int:
    """The largest total whose per-class targets fit the CAPPED supply. The old
    closed form used raw supply, so the binding class silently under-filled at
    draw time while the others filled to uncapped targets (clerc alone would
    have taken 26% of the sparse class); cap=1 recovers the closed form."""
    def fits(total: int) -> bool:
        return all(
            _capped_supply(lane_counts[c], round(total * s), lane_share_cap)
            >= round(total * s)
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


def split_report(pool: pd.DataFrame, recipe: SelectorRecipe) -> pd.DataFrame:
    lane_counts = _lane_counts(pool)
    rows = []
    for split, name in ((recipe.target_split, "target"), ((0.4, 0.4, 0.2), "40/40/20")):
        total = feasible_total(lane_counts, split, recipe.lane_share_cap)
        for c, s in zip(CLASSES, split):
            target_n = round(total * s)
            grown = round((total + 1) * s)
            rows.append({
                "split": name, "class": c, "target_share": s,
                "supply": int(lane_counts[c].sum()),
                "capped_supply": _capped_supply(
                    lane_counts[c], target_n, recipe.lane_share_cap
                ),
                "feasible_total": total,
                "target_n": target_n,
                # binding = this class cannot grow with the total
                "binding": s > 0 and _capped_supply(
                    lane_counts[c], grown, recipe.lane_share_cap
                ) < grown,
            })
    return pd.DataFrame(rows)


def per_dataset_report(pool: pd.DataFrame, recipe: SelectorRecipe) -> pd.DataFrame:
    g = pool.groupby("dataset")
    out = pd.DataFrame({
        "labelled": g.size(),
        "decisive_total": g.apply(lambda d: int(d["is_decisive"].sum()), include_groups=False),
        "dense": g.apply(lambda d: int((d["route_class"] == "dense").sum()), include_groups=False),
        "sparse": g.apply(lambda d: int((d["route_class"] == "sparse").sum()), include_groups=False),
        "hybrid": g.apply(lambda d: int((d["route_class"] == "hybrid").sum()), include_groups=False),
        # rows that credit the floor, counted ONCE: an rrf-decisive row is both
        # decisive and hybrid, so summing the two columns counts it twice
        "floor_credit": g.apply(
            lambda d: int((d["is_decisive"] | d["is_hybrid"]).sum()), include_groups=False
        ),
    })
    out["floor"] = recipe.per_dataset_floor
    out["floor_met"] = out["floor_credit"] >= recipe.per_dataset_floor
    out["floor_gap"] = (recipe.per_dataset_floor - out["floor_credit"]).clip(lower=0)
    # single-answer lanes (~1 judged doc/query) structurally cannot yield
    # decisive rows -> their floor is WAIVED, not a generation target.
    out["median_depth"] = g["depth"].median()
    out["single_answer"] = out["median_depth"] <= 1
    # What a FRESH query from this lane is worth: decisive-at-margin only, over
    # BLIND rows only. Genuine ties are excluded because a hybrid row is not what
    # a decisive floor buys, and `reused` rows are excluded because they entered
    # the pool for already having won — including either is what over-predicted
    # the last campaign by 10x.
    blind = pool[pool["stage"] != REUSED].groupby("dataset")
    out["blind_labelled"] = blind.size().reindex(out.index).fillna(0).astype(int)
    out["blind_decisive"] = (
        blind["is_decisive"].sum().reindex(out.index).fillna(0).astype(int)
    )
    out["yield_rate"] = (
        out["blind_decisive"] / out["blind_labelled"].clip(lower=1)
    ).round(3)
    # the bar is the pool's own blind rate, not a hand number: "label more" means
    # this lane converts fresh labels at least as well as the pool average does
    pooled = out["blind_decisive"].sum() / max(int(out["blind_labelled"].sum()), 1)

    def _action(r) -> str:
        if r["floor_met"]:
            return "ok"
        if r["single_answer"]:
            return "waive (single-answer)"
        if r["blind_labelled"] == 0:
            return "unmeasured (no blind rows)"
        return "label more" if r["yield_rate"] >= pooled else "source/deepen"

    out["floor_action"] = out.apply(_action, axis=1)
    out.attrs["pooled_blind_yield"] = round(pooled, 4)
    return out.sort_values("floor_gap", ascending=False)


def tie_zero_report(pool: pd.DataFrame) -> pd.DataFrame:
    """Ties decomposed by kind AND by ceiling: an at-ceiling tie means every
    route surfaced a judged doc at rank 1, an all-miss tie means none did —
    the second is the 'no route answers this' signal, tracked per lane."""
    g = pool.groupby("dataset")

    def n(mask_fn) -> pd.Series:
        return g.apply(lambda d: int(mask_fn(d).sum()), include_groups=False)

    return pd.DataFrame({
        "all_tied": n(lambda d: d["kind"].isin(["fake_tie", "genuine_tie"])),
        "genuine_tie": n(lambda d: d["kind"] == "genuine_tie"),
        "genuine_all_miss": n(
            lambda d: (d["kind"] == "genuine_tie") & (d["oracle"] < CEILING)
        ),
        "fake_tie": n(lambda d: d["kind"] == "fake_tie"),
        "fake_all_miss": n(
            lambda d: (d["kind"] == "fake_tie") & (d["oracle"] < CEILING)
        ),
        "all_zero": n(lambda d: d["kind"] == "all_zero"),
    }).sort_values("fake_tie", ascending=False)


def stratum_coverage(pool: pd.DataFrame) -> pd.DataFrame:
    """Per stratum cell: is there >=1 decisive/hybrid row to cover it?"""
    dec = pool[pool["route_class"].isin(CLASSES)]
    rows = []
    cell_hits = {}
    for cells in dec["cells"]:
        for c in cells:
            cell_hits[c] = cell_hits.get(c, 0) + 1
    for cell in _active_cells():
        rows.append({"axis": "cell", "stratum": cell.name,
                     "coverable_rows": cell_hits.get(cell.name, 0),
                     "coverable": cell_hits.get(cell.name, 0) > 0})
    for deg, n in dec["corruption_degree"].value_counts().items():
        rows.append({"axis": "corruption", "stratum": deg,
                     "coverable_rows": int(n), "coverable": True})
    for axis in CORPUS_AXES:
        for band, n in dec[axis].value_counts().items():
            rows.append({"axis": axis, "stratum": str(band),
                         "coverable_rows": int(n), "coverable": True})
    lr = dec.groupby(["dataset", "route_class"]).size()
    for (ds, rc), n in lr.items():
        rows.append({"axis": "lane_route", "stratum": f"{ds}:{rc}",
                     "coverable_rows": int(n), "coverable": True})
    return pd.DataFrame(rows)


def shortfalls_200k(split_df: pd.DataFrame, per_ds: pd.DataFrame,
                    pool: pd.DataFrame, recipe: SelectorRecipe) -> pd.DataFrame:
    """Extrapolate the natural supply to target_total and report the gap to
    generate per class and per dataset floor. THE Phase-B order sheet."""
    n_pool = len(pool)
    scale = recipe.target_total / n_pool
    rows = []
    for _, r in split_df[split_df["split"] == "target"].iterrows():
        target = recipe.target_total * r["target_share"]
        nat = r["supply"] * scale
        rows.append({"scope": "class", "key": r["class"],
                     "natural_supply": int(r["supply"]),
                     "target_200k": round(target),
                     "gap_to_generate": max(0, round(target - nat)),
                     "x_under": round(target / nat, 1) if nat else float("inf"),
                     "action": "generate"})
    for ds, r in per_ds.iterrows():
        if r["floor_gap"] > 0:
            act = r["floor_action"]
            rows.append({"scope": "dataset_floor", "key": ds,
                         "natural_supply": int(r["floor_credit"]),
                         "target_200k": recipe.per_dataset_floor,
                         "gap_to_generate": 0 if act.startswith("waive") else int(r["floor_gap"]),
                         "x_under": None,
                         "action": act})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- selector ---
def _draw_class(cand: pd.DataFrame, target: int, per_lane_cap: int) -> list:
    """One class's draw: a TRUE greedy coverage pass (gain recomputed per pick,
    lane×route exhausted before diversity strata break ties) and then a seeded
    lane-capped fill. Coverage resets per class, so cell coverage is keyed
    (cell, class) — a cell covered for dense still earns gain for sparse."""
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
    covered: set = set()
    lanes_drawn: set = set()
    per_lane: dict[str, int] = {}
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


def greedy_select(pool: pd.DataFrame, recipe: SelectorRecipe) -> pd.DataFrame:
    """Per class: coverage-first greedy then seeded fill, under the SAME lane
    cap the feasibility bound used, so the targets are reachable by
    construction and no escape hatch is needed. The dictated waste draw runs
    last, typed 'waste' in the artifact. Deterministic under seed."""
    lane_counts = _lane_counts(pool)
    total = feasible_total(lane_counts, recipe.target_split, recipe.lane_share_cap)
    targets = {c: round(total * s) for c, s in zip(CLASSES, recipe.target_split)}

    chosen: list = []
    for c in CLASSES:
        cand = pool[pool["route_class"] == c].sample(
            frac=1.0, random_state=recipe.seed
        )
        cap = max(1, math.ceil(recipe.lane_share_cap * targets[c]))
        chosen.extend(_draw_class(cand, targets[c], cap))

    waste_budget = round(recipe.waste_cap * total)
    waste: list = []
    if waste_budget > 0:
        wcand = pool[pool["is_waste"]].sample(frac=1.0, random_state=recipe.seed)
        wcap = max(1, math.ceil(recipe.lane_share_cap * waste_budget))
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
    selected.loc[selected["is_waste"], "route_class"] = "waste"
    selected.attrs["waste_budget"] = waste_budget
    return selected


# -------------------------------------------------------------------- write ---
def write_report(pool: pd.DataFrame, selected: pd.DataFrame,
                 split_df, per_ds, tie_zero, coverage, shortfalls,
                 recipe: SelectorRecipe) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    split_df.to_parquet(OUT / "split_achieved.parquet", index=False)
    per_ds.to_parquet(OUT / "per_dataset.parquet")
    tie_zero.to_parquet(OUT / "tie_zero.parquet")
    coverage.to_parquet(OUT / "stratum_coverage.parquet", index=False)
    shortfalls.to_parquet(OUT / "shortfalls_200k.parquet", index=False)
    selected[["dataset", "query_id", "route_class"]].to_parquet(
        OUT / "selected.parquet", index=False
    )

    tgt = split_df[split_df["split"] == "target"]
    binding = tgt[tgt["binding"]]["class"].tolist()
    unmet = per_ds[~per_ds["floor_met"]]
    n_fail = len(unmet)
    acts = unmet["floor_action"].value_counts()
    n_waive = int(acts.get("waive (single-answer)", 0))
    n_label = int(acts.get("label more", 0))
    n_source = int(acts.get("source/deepen", 0))
    cov = coverage[coverage["axis"] == "cell"]
    lines = [
        "# v3 selector prototype — feasibility report",
        "",
        f"Pool: {len(pool):,} labelled rows, {pool['dataset'].nunique()} datasets. "
        f"Split target {recipe.target_split}, per-dataset floor {recipe.per_dataset_floor}, "
        f"class_margin {recipe.class_margin}.",
        "",
        "## Verdict",
        f"- Feasible total at target split: **{round(tgt['feasible_total'].iloc[0]):,} rows** "
        f"(binding class: {', '.join(binding) or 'none'}).",
        f"- Selected: **{len(selected):,} rows** "
        f"(waste drawn {int((selected['route_class'] == 'waste').sum())} "
        f"of a dictated budget {selected.attrs.get('waste_budget', 0)}).",
        f"- Per-dataset floor {recipe.per_dataset_floor} unmet by **{n_fail}/{len(per_ds)}** "
        f"({n_label} label-more, {n_source} source/deepen, {n_waive} waive single-answer).",
        f"- Cells with zero decisive supply: **{int((~cov['coverable']).sum())}/{len(cov)}**.",
        f"- `yield_rate` is decisive-at-margin over BLIND rows (reused excluded); "
        f"the label-more bar is the pool's own blind yield, "
        f"**{per_ds.attrs.get('pooled_blind_yield', float('nan')):.1%}**.",
        "",
        "## Class split (target)",
        tgt[["class", "target_share", "supply", "capped_supply", "target_n", "binding"]].to_markdown(index=False),
        "",
        "## 200K generation debt (the Phase-B order sheet)",
        shortfalls[shortfalls["scope"] == "class"].to_markdown(index=False),
        "",
        f"## Per-dataset floor shortfalls — {n_label} label-more, "
        f"{n_source} source/deepen, {n_waive} waive",
        unmet[["labelled", "floor_credit", "floor_gap", "blind_labelled",
               "blind_decisive", "yield_rate", "floor_action"]]
        .head(24).to_markdown(),
        "",
        "## Tie / zero (waste) totals",
        f"- genuine ties {int((pool['kind']=='genuine_tie').sum()):,} · "
        f"fake ties {int((pool['kind']=='fake_tie').sum()):,} · "
        f"all_zero {int((pool['kind']=='all_zero').sum()):,} "
        f"({(pool['is_waste'].mean()):.0%} of pool is waste).",
    ]
    (OUT / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", nargs=3, type=float, default=None,
                        help="dense sparse hybrid, e.g. 45 45 10")
    parser.add_argument("--floor", type=int, default=None)
    parser.add_argument("--class-margin", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    kw = {}
    if args.split:
        s = np.array(args.split, dtype=float)
        kw["target_split"] = tuple(s / s.sum())
    if args.floor is not None:
        kw["per_dataset_floor"] = args.floor
    if args.class_margin is not None:
        kw["class_margin"] = args.class_margin
    if args.seed is not None:
        kw["seed"] = args.seed
    recipe = SelectorRecipe(**kw)

    labels = _load_labels()
    pool = attach_strata(classify(labels, recipe))

    split_df = split_report(pool, recipe)
    per_ds = per_dataset_report(pool, recipe)
    tie_zero = tie_zero_report(pool)
    coverage = stratum_coverage(pool)
    shortfalls = shortfalls_200k(split_df, per_ds, pool, recipe)
    selected = greedy_select(pool, recipe)

    write_report(pool, selected, split_df, per_ds, tie_zero, coverage,
                 shortfalls, recipe)
    print((OUT / "report.md").read_text())
    print(f"\nartifacts -> {OUT}")


if __name__ == "__main__":
    main()

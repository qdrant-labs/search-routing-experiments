"""v3 selector PROTOTYPE (Phase 0) — feasibility + shortfall report over the
EXISTING labelled pool, no retrieval, no generation.

Proves whether the v3 objectives are jointly reachable and, where they are not,
what the 200K generation debt is. Route classes: dense/sparse = decisive
routes_differ wins (margin >= class_margin); hybrid = genuine ties (all_tied
that is NOT a single-doc ceiling tie) + rrf-decisive. Fake ties + all_zero are
a capped waste budget. Diversity strata (additive, never crossed): cell,
corruption degree. Utility coverage: (lane x decisive-route). Per-dataset
decisive floor is a hard constraint whose shortfall is the headline.

Corpus-stat bands and per-query corpus stats are DEFERRED to Phase A (lane
identity is the corpus proxy here, via lane x route coverage).

    poetry run python src/scripts/select_v3_prototype.py                 # 45/45/10, floor 25
    poetry run python src/scripts/select_v3_prototype.py --floor 50 --split 40 40 20
    poetry run python src/scripts/select_v3_prototype.py --class-margin 0.0   # routes_differ, not decisive

Reads: data/route_labels/labels.parquet, data/feature_table/catalog.parquet,
data/<dataset>/qrels.parquet. Writes: data/route_labels/v3_feasibility/.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from composition.cells import CELLS
from composition.cells_v3 import CELLS_V3
from composition.floors import with_derived
from query_taxonomy.core import Engine
from query_taxonomy.features import FeatureExtractor
from query_taxonomy.taxonomy import FeatureGroup

DATA = Path(__file__).resolve().parent.parent / "data"
LABELS = DATA / "route_labels" / "labels.parquet"
CATALOG = DATA / "feature_table" / "catalog.parquet"
OUT = DATA / "route_labels" / "v3_feasibility"

V3_CATALOG = DATA / "v3" / "catalog_v3.parquet"

SCORES = ["score_dense_only", "score_pure_rrf", "score_sparse_only"]
ROUTE_TO_CLASS = {"dense_only": "dense", "sparse_only": "sparse", "pure_rrf": "hybrid"}
CLASSES = ("dense", "sparse", "hybrid")
CEILING = 0.999
TOL = 1e-9


@dataclass(frozen=True)
class SelectorRecipe:
    """Every dial for the prototype. Folds into composition/recipe.py when the
    selector graduates from prototype."""

    target_split: tuple[float, float, float] = (0.45, 0.45, 0.10)  # dense, sparse, hybrid
    per_dataset_floor: int = 25
    waste_cap: float = 0.0  # dictated waste as a fraction of T (0 = exclude waste)
    genuine_tie_depth: int = 2  # a tie with < this many judged docs AND at ceiling is fake
    class_margin: float = 0.4  # decisive if oracle - runner_up >= this
    lane_share_cap: float = 0.2  # no single dataset > this share of a class
    seed: int = 0
    target_total: int = 200_000  # for the 200K extrapolation


# ---------------------------------------------------------------- classify ---
def _load_labels() -> pd.DataFrame:
    """v2 labels PLUS the additive v3 labels (new query_ids from the in-loop
    labelling), aligned on v2's columns. v2 is read-only; v3 rows fill the
    label-more gaps. New (dataset, query_id) pairs are disjoint, so a dedup
    only guards against a re-labelled row (v2 kept)."""
    v2 = pd.read_parquet(LABELS).astype({"query_id": str})
    v3_path = DATA / "v3" / "labels.parquet"
    if not v3_path.exists():
        return v2
    v3 = pd.read_parquet(v3_path).astype({"query_id": str})
    v3 = v3.reindex(columns=v2.columns)  # v2-only cols (cell/stage) -> NaN, recomputed
    combined = pd.concat([v2, v3], ignore_index=True)
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
    fake_tie = tied & (depth < recipe.genuine_tie_depth) & (oracle >= CEILING)
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
def _active_cells():
    """v2 cells, plus the v3 cells when the v3 catalog (their columns) exists."""
    return (*CELLS, *CELLS_V3) if V3_CATALOG.exists() else CELLS


def attach_strata(pool: pd.DataFrame) -> pd.DataFrame:
    """cell membership (multi) from the catalog + corruption degree from the
    query text. Lane is `dataset` (the corpus proxy for the prototype)."""
    # v3 catalog (v2 columns + new taxonomy columns) enables the v3 cells; fall
    # back to the v2 catalog + v2 cells when it has not been built yet.
    catalog_path = V3_CATALOG if V3_CATALOG.exists() else CATALOG
    catalog = with_derived(pd.read_parquet(catalog_path).astype({"query_id": str}))
    idx = catalog.set_index(["dataset", "query_id"]).index
    per_row: list[set] = [set() for _ in range(len(catalog))]
    for cell in _active_cells():
        for i in np.nonzero(cell.select(catalog).to_numpy())[0]:
            per_row[i].add(cell.name)
    cell_lookup = dict(zip(idx, per_row))

    pool_idx = list(zip(pool["dataset"], pool["query_id"]))
    pool = pool.assign(
        cells=[frozenset(cell_lookup.get(k, set())) for k in pool_idx]
    )

    extractor = FeatureExtractor(engines=(Engine.REGEX, Engine.WORDFREQ))
    spans = []
    for text in pool["query"].fillna("").astype(str):
        res = extractor.resolve(text, groups=[FeatureGroup.CORRUPTION])
        group = res.spans.get(FeatureGroup.CORRUPTION, {})
        spans.append(sum(len(v) for v in group.values()))
    spans_arr = np.array(spans)
    degree = np.where(spans_arr == 0, "clean", np.where(spans_arr == 1, "light", "heavy"))
    pool = pool.assign(corruption_spans=spans_arr, corruption_degree=degree)
    return _attach_corpus_band(pool)


def _attach_corpus_band(pool: pd.DataFrame) -> pd.DataFrame:
    """Per-query corpus-rarity band from query_corpus_stats.parquet (Phase A).
    Absent until that build runs -> one 'unknown' band (graceful fallback; the
    lane axis still carries corpus identity via lane x route coverage)."""
    qcs_path = DATA / "route_labels" / "query_corpus_stats.parquet"
    if not qcs_path.exists():
        return pool.assign(corpus_band="unknown")
    qcs = pd.read_parquet(qcs_path).astype({"query_id": str})
    pool = pool.merge(
        qcs[["dataset", "query_id", "avg_idf"]], on=["dataset", "query_id"], how="left"
    )
    band = pd.cut(
        pool["avg_idf"], [-0.01, 0.2, 0.4, 1.01],
        labels=["low_idf", "mid_idf", "high_idf"],
    ).astype(object)
    return pool.assign(corpus_band=band.fillna("unknown")).drop(columns="avg_idf")


# ---------------------------------------------------------------- analytics ---
def feasible_total(supply: dict[str, int], split: tuple[float, float, float]) -> float:
    """Closed-form: the largest total whose per-class targets all fit supply."""
    return min(
        supply[c] / s for c, s in zip(CLASSES, split) if s > 0
    )


def split_report(pool: pd.DataFrame, recipe: SelectorRecipe) -> pd.DataFrame:
    supply = {c: int((pool["route_class"] == c).sum()) for c in CLASSES}
    rows = []
    for split, name in ((recipe.target_split, "target"), ((0.4, 0.4, 0.2), "40/40/20")):
        total = feasible_total(supply, split)
        for c, s in zip(CLASSES, split):
            rows.append({
                "split": name, "class": c, "target_share": s,
                "supply": supply[c], "feasible_total": round(total),
                "target_n": round(total * s),
                "binding": abs(supply[c] / s - total) < 1e-6 if s > 0 else False,
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
    })
    out["floor"] = recipe.per_dataset_floor
    out["floor_met"] = out["decisive_total"] + out["hybrid"] >= recipe.per_dataset_floor
    out["floor_gap"] = (recipe.per_dataset_floor - out["decisive_total"] - out["hybrid"]).clip(lower=0)
    # single-answer lanes (~1 judged doc/query) structurally cannot yield
    # decisive rows -> their floor is WAIVED, not a generation target.
    out["median_depth"] = g["depth"].median()
    out["single_answer"] = out["median_depth"] <= 1
    # decisive yield among the rows already labelled: distinguishes an
    # UNDER-LABELLED lane (good yield, few rows -> label more via Qdrant) from a
    # genuinely LOW-YIELD one (label more nets nothing -> source better data).
    out["yield_rate"] = (
        (out["decisive_total"] + out["hybrid"]) / out["labelled"].clip(lower=1)
    ).round(2)

    def _action(r) -> str:
        if r["floor_met"]:
            return "ok"
        if r["single_answer"]:
            return "waive (single-answer)"
        return "label more" if r["yield_rate"] >= 0.15 else "source/deepen"

    out["floor_action"] = out.apply(_action, axis=1)
    return out.sort_values("floor_gap", ascending=False)


def tie_zero_report(pool: pd.DataFrame) -> pd.DataFrame:
    g = pool.groupby("dataset")
    return pd.DataFrame({
        "all_tied": g.apply(lambda d: int(d["kind"].isin(["fake_tie", "genuine_tie"]).sum()), include_groups=False),
        "genuine_tie": g.apply(lambda d: int((d["kind"] == "genuine_tie").sum()), include_groups=False),
        "fake_tie": g.apply(lambda d: int((d["kind"] == "fake_tie").sum()), include_groups=False),
        "all_zero": g.apply(lambda d: int((d["kind"] == "all_zero").sum()), include_groups=False),
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
    for band, n in dec["corpus_band"].value_counts().items():
        rows.append({"axis": "corpus_idf", "stratum": str(band),
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
                         "natural_supply": int(r["decisive_total"] + r["hybrid"]),
                         "target_200k": recipe.per_dataset_floor,
                         "gap_to_generate": 0 if act.startswith("waive") else int(r["floor_gap"]),
                         "x_under": None,
                         "action": act})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- selector ---
def greedy_select(pool: pd.DataFrame, recipe: SelectorRecipe) -> pd.DataFrame:
    """Coverage-first draw per class up to the feasible split, lane-capped.
    Coverage pass (one row per uncovered cell / lane-route) then random fill.
    Deterministic under seed. Its exhaustion is the shortfall."""
    supply = {c: int((pool["route_class"] == c).sum()) for c in CLASSES}
    total = feasible_total(supply, recipe.target_split)
    targets = {c: round(total * s) for c, s in zip(CLASSES, recipe.target_split)}

    chosen: list[int] = []
    covered_cells: set = set()
    covered_flat: set = set()  # lane-route, corruption degree, corpus band

    def _gain(cand: pd.DataFrame, i: int, cls: str) -> int:
        ds = cand.at[i, "dataset"]
        flats = {("lr", ds, cls),
                 ("corr", cand.at[i, "corruption_degree"]),
                 ("corp", cand.at[i, "corpus_band"])}
        return len(cand.at[i, "cells"] - covered_cells) + len(flats - covered_flat)

    for c in CLASSES:
        cand = pool[pool["route_class"] == c].sample(frac=1.0, random_state=recipe.seed)
        cap = max(1, int(recipe.lane_share_cap * targets[c]))
        per_lane: dict[str, int] = {}
        taken: list[int] = []
        order = sorted(cand.index, key=lambda i: -_gain(cand, i, c))
        for i in order:
            if len(taken) >= targets[c]:
                break
            ds = cand.at[i, "dataset"]
            if per_lane.get(ds, 0) >= cap and len(taken) < targets[c] - 1:
                continue
            taken.append(i)
            per_lane[ds] = per_lane.get(ds, 0) + 1
            covered_cells |= cand.at[i, "cells"]
            covered_flat |= {("lr", ds, c),
                             ("corr", cand.at[i, "corruption_degree"]),
                             ("corp", cand.at[i, "corpus_band"])}
        chosen.extend(taken)
    return pool.loc[chosen].assign(selected=True)


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
        f"- Selected: **{len(selected):,} rows**.",
        f"- Per-dataset floor {recipe.per_dataset_floor} unmet by **{n_fail}/{len(per_ds)}** "
        f"({n_label} label-more, {n_source} source/deepen, {n_waive} waive single-answer).",
        f"- Cells with zero decisive supply: **{int((~cov['coverable']).sum())}/{len(cov)}**.",
        "",
        "## Class split (target)",
        tgt[["class", "target_share", "supply", "target_n", "binding"]].to_markdown(index=False),
        "",
        "## 200K generation debt (the Phase-B order sheet)",
        shortfalls[shortfalls["scope"] == "class"].to_markdown(index=False),
        "",
        f"## Per-dataset floor shortfalls — {n_label} label-more, "
        f"{n_source} source/deepen, {n_waive} waive",
        unmet[["labelled", "decisive_total", "hybrid", "floor_gap", "yield_rate", "floor_action"]]
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

"""Phase 3 primary ablation: does the two-tier selector's certified draw (arm A)
beat a lane x route_class-matched random control (arm B) on the eval_reserve
frozen BEFORE any selection ran? Paired per-row objective, lane-cluster
bootstrap CI, per-lane spread, near-dup leak report — no re-fit, no re-split.

    poetry run python src/scripts/run_ablation.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import wilcoxon

from hybrid_search_rrf_dataset.router import (
    Representation,
    StrategyRouter,
    _is_engineered,
    _score_matrix,
)
from scripts.select_v3_prototype import (
    CLASSES,
    DATA,
    OUT,
    V3_CATALOG,
    SelectorRecipe,
    _cluster_ids,
    _load_labels,
    classify,
)

DUP_CLUSTERS = DATA / "route_labels" / "dup_clusters.parquet"
ABLATION_OUT = OUT / "ablation"
B_SEED = 999  # deliberately distinct from the selector's own seed=0
N_BOOT = 5000
LANE_MIN_N = 20  # below this a lane's own mean is reported but flagged low-n


def _key(frame: pd.DataFrame) -> pd.Series:
    return frame["dataset"].astype(str) + "\x00" + frame["query_id"].astype(str)


def full_pool() -> pd.DataFrame:
    """Every labelled row, classified (route_class/certified) and joined to
    every v3 catalog ENGINEERED column — the one frame both arms and the
    frozen eval slice are all read out of, so no join can drift between them."""
    labels = classify(_load_labels(), SelectorRecipe())
    names = pq.read_schema(V3_CATALOG).names
    # query_corpus.* is NaN where a lane has no index — "unmeasured", not zero
    # (build_v3_catalog's own invariant) — so it cannot go through FeatureSpace's
    # reindex-fills-absent-with-0 contract. Excluded from the router's engineered
    # set here; it is already the selector's own stratum axis, not lost.
    feat_cols = [
        c for c in names if _is_engineered(c) and not c.startswith("query_corpus.")
    ]
    catalog = pd.read_parquet(
        V3_CATALOG, columns=["dataset", "query_id", *feat_cols]
    ).astype({"query_id": str})
    return labels.merge(catalog, on=["dataset", "query_id"], how="inner")


def _match_random(
    pool: pd.DataFrame,
    contingency: pd.Series,
    exclude_clusters: set[str],
    key_to_cluster: dict[str, str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Arm B: for each (dataset, route_class) cell, a random draw of the SAME
    size drawn from the cell MINUS every row sharing a near-dup CLUSTER with
    arm A first — not just row-identical matches, since a cluster-mate is the
    same signal wearing different characters and is just as forced. A cell
    where arm A already took most or all of the candidate supply otherwise
    forces near-total overlap regardless of how B is drawn (a hypergeometric
    fact, not a randomization bug); only when a cell's free supply is smaller
    than the count does it reuse an arm-A-clustered row, and that reuse is
    counted, not hidden. Returns the draw and a per-cell exhaustion report."""
    rng = np.random.default_rng(seed)
    parts, report = [], []
    for (lane, cls), n in contingency.items():
        cell = pool[(pool["dataset"] == lane) & (pool["route_class"] == cls)]
        in_exclude = _key(cell).map(key_to_cluster).isin(exclude_clusters)
        free = cell[~in_exclude]
        forced = max(0, n - len(free))
        drawn = free.sample(n=min(n, len(free)), random_state=rng.integers(2**31))
        if forced:
            reused = cell[in_exclude].sample(
                n=forced, random_state=rng.integers(2**31)
            )
            drawn = pd.concat([drawn, reused])
        parts.append(drawn)
        report.append({
            "dataset": lane, "route_class": cls, "candidate": len(cell),
            "free_supply": len(free), "drawn": n, "forced_reuse": forced,
        })
    return pd.concat(parts, ignore_index=False), pd.DataFrame(report)


def fit_router(train: pd.DataFrame, tune_frac: float, seed: int) -> StrategyRouter:
    """One arm's router: ENGINEERED representation only (the question here is
    selection strategy, not representation), min_fires=0 on BOTH arms so the
    engineered column set is identical regardless of which specific rows each
    arm happened to draw — a threshold difference would confound the compare."""
    tune = train.sample(frac=tune_frac, random_state=seed)
    model = train.drop(index=tune.index)
    router = StrategyRouter(Representation.ENGINEERED).fit(
        model, all_rows=False, min_fires=0
    )
    router.tune_thresholds(tune)
    return router


def row_objective(frame: pd.DataFrame, routes: list) -> np.ndarray:
    """Each row's score under its OWN predicted route — the paired unit."""
    from hybrid_search_rrf_dataset.router import _ROUTE_ORDER

    scores = _score_matrix(frame)
    idx = np.fromiter((_ROUTE_ORDER.index(r) for r in routes), dtype=int)
    return scores[np.arange(len(idx)), idx]


def lane_cluster_bootstrap(
    diff: np.ndarray, lane: np.ndarray, n_boot: int, seed: int
) -> np.ndarray:
    """Resample LANES with replacement (the population unit `shape_by_cell`
    identified), not rows — a row bootstrap understates uncertainty here."""
    rng = np.random.default_rng(seed)
    lanes = np.unique(lane)
    by_lane = {name: diff[lane == name] for name in lanes}
    means = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(lanes, size=len(lanes), replace=True)
        means[b] = np.concatenate([by_lane[name] for name in draw]).mean()
    return means


def dup_leak(train: pd.DataFrame, eval_frame: pd.DataFrame) -> dict[str, object]:
    """Share of eval rows whose near-dup cluster also appears in this arm's
    training rows — a leak the split cannot fix retroactively, only report."""
    clusters = pd.read_parquet(DUP_CLUSTERS).astype({"query_id": str})
    clusters["key"] = _key(clusters)
    train_clusters = set(
        clusters.loc[clusters["key"].isin(_key(train)), "cluster_id"]
    )
    eval_c = clusters[clusters["key"].isin(_key(eval_frame))]
    covered = len(eval_c)
    leaked = int(eval_c["cluster_id"].isin(train_clusters).sum())
    return {
        "eval_rows_in_dup_index": covered, "eval_rows_total": len(eval_frame),
        "leaked": leaked,
        "leak_share_of_covered": round(leaked / covered, 4) if covered else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-b", type=int, default=B_SEED)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()

    pool = full_pool()
    selected = pd.read_parquet(OUT / "selected.parquet").astype({"query_id": str})
    reserve = pd.read_parquet(OUT / "eval_reserve.parquet").astype({"query_id": str})

    eval_frame = pool.merge(
        reserve[["dataset", "query_id"]], on=["dataset", "query_id"], how="inner"
    )
    assert len(eval_frame) == len(reserve), "eval_reserve keys missing from pool"
    assert eval_frame["certified"].all(), "eval_reserve must be certified-only"

    reserve_keys = set(_key(reserve))
    candidate = pool[pool["certified"] & ~_key(pool).isin(reserve_keys)]

    arm_a_keys = selected.loc[selected["certified"], ["dataset", "query_id"]]
    arm_a = candidate.merge(arm_a_keys, on=["dataset", "query_id"], how="inner")
    assert len(arm_a) == len(arm_a_keys), "arm A keys missing from candidate pool"
    contingency = arm_a.groupby(["dataset", "route_class"]).size()

    key_to_cluster = dict(zip(_key(pool), _cluster_ids(pool)))
    arm_a_clusters = set(_key(arm_a).map(key_to_cluster))
    arm_b, exhaustion = _match_random(
        candidate, contingency, arm_a_clusters, key_to_cluster, args.seed_b
    )
    forced_reuse = int(exhaustion["forced_reuse"].sum())
    exhausted_cells = exhaustion[exhaustion["free_supply"] < exhaustion["drawn"]]

    manifest = {
        "eval_reserve_rows": len(eval_frame),
        "eval_reserve_lanes": sorted(eval_frame["dataset"].unique().tolist()),
        "arm_a_rows": len(arm_a),
        "arm_b_rows": len(arm_b),
        "contingency_lane_class_cells": len(contingency),
        "forced_reuse_rows": forced_reuse,
        "forced_reuse_share": round(forced_reuse / len(arm_a), 4),
        "fully_exhausted_cells": int((exhaustion["free_supply"] == 0).sum()),
        "endpoint": "paired per-row objective, arm A minus arm B, on eval_reserve",
        "resampling": "lane-cluster bootstrap, resample lanes with replacement",
        "direction": "positive = arm A (selector) beats arm B (disjoint-first random)",
        "note": "arm B draws disjoint from arm A at the near-dup CLUSTER level "
                "wherever a cell's free supply allows; forced_reuse_rows is the "
                "hypergeometric floor no randomization can avoid — read "
                "alongside the result, not hidden",
        "class_margin": 0.4, "seed_a_pool": 0, "seed_b": args.seed_b,
        "n_boot": args.n_boot,
    }
    ABLATION_OUT.mkdir(parents=True, exist_ok=True)
    (ABLATION_OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    exhaustion.to_parquet(ABLATION_OUT / "cell_exhaustion.parquet", index=False)

    router_a = fit_router(arm_a, tune_frac=0.25, seed=0)
    router_b = fit_router(arm_b, tune_frac=0.25, seed=args.seed_b)
    routes_a = router_a.predict_routes(eval_frame)
    routes_b = router_b.predict_routes(eval_frame)

    obj_a = row_objective(eval_frame, routes_a)
    obj_b = row_objective(eval_frame, routes_b)
    diff = obj_a - obj_b
    lane = eval_frame["dataset"].to_numpy()
    discordant = np.array([a != b for a, b in zip(routes_a, routes_b)])

    boot = lane_cluster_bootstrap(diff, lane, args.n_boot, seed=0)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    disc_a, disc_b = obj_a[discordant], obj_b[discordant]
    wilcoxon_p = (
        float(wilcoxon(disc_a, disc_b).pvalue) if discordant.sum() >= 2 else None
    )

    per_lane = (
        pd.DataFrame({"dataset": lane, "diff": diff})
        .groupby("dataset")["diff"].agg(["size", "mean"])
        .rename(columns={"size": "n", "mean": "mean_diff"})
        .sort_values("mean_diff", ascending=False)
    )
    per_lane["low_n"] = per_lane["n"] < LANE_MIN_N

    results = {
        "mean_diff": float(diff.mean()),
        "lane_cluster_bootstrap_ci95": [float(ci_lo), float(ci_hi)],
        "n_discordant": int(discordant.sum()),
        "n_total": int(len(eval_frame)),
        "wilcoxon_p_on_discordant": wilcoxon_p,
        "arm_a_mean_objective": float(obj_a.mean()),
        "arm_b_mean_objective": float(obj_b.mean()),
        "route_class_counts": {
            c: int((contingency.index.get_level_values("route_class") == c).sum())
            for c in CLASSES
        },
        "dup_leak_arm_a": dup_leak(arm_a, eval_frame),
        "dup_leak_arm_b": dup_leak(arm_b, eval_frame),
    }
    (ABLATION_OUT / "results.json").write_text(json.dumps(results, indent=2))
    per_lane.to_parquet(ABLATION_OUT / "per_lane.parquet")

    row_overlap = len(set(_key(arm_a)) & set(_key(arm_b)))
    print(f"arm A {len(arm_a):,} rows | arm B {len(arm_b):,} rows | "
          f"eval {len(eval_frame):,} rows over {len(np.unique(lane))} lanes\n")
    print(f"arm A/B row overlap: {row_overlap:,}/{len(arm_a):,} "
          f"({row_overlap / len(arm_a):.1%}) — of which "
          f"{forced_reuse:,} ({forced_reuse / len(arm_a):.1%}) were FORCED "
          f"(cell had no free supply left after excluding arm A) across "
          f"{len(exhausted_cells)}/{len(contingency)} cells, "
          f"{int((exhaustion['free_supply'] == 0).sum())} fully exhausted\n")
    print(f"mean diff (A - B): {results['mean_diff']:+.4f}")
    print(f"lane-cluster bootstrap 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]  "
          f"(n_boot={args.n_boot})")
    print(f"discordant rows: {results['n_discordant']}/{results['n_total']} "
          f"(agreement rows contribute 0 by construction)")
    if wilcoxon_p is not None:
        print(f"Wilcoxon on discordant pairs: p={wilcoxon_p:.4f}")
    print(f"\narm A mean objective: {results['arm_a_mean_objective']:.4f}  |  "
          f"arm B: {results['arm_b_mean_objective']:.4f}")
    print("\nper-lane mean diff (negative n = low-n, read with caution):")
    print(per_lane.to_string())
    for arm, leak in (("A", results["dup_leak_arm_a"]), ("B", results["dup_leak_arm_b"])):
        print(f"\ndup-leak arm {arm}: {leak}")
    print(f"\n-> {ABLATION_OUT}")


if __name__ == "__main__":
    main()

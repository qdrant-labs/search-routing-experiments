"""Phase 0.5 feasibility gate: build dup_clusters.parquet once, replay the
precommitted estimator on both designs, and report CI half-widths and power at
Δmin — deciding whether the dataset can resolve the margin before Phase 1 runs.
Reads stored embeddings and labels only; no retrieval, no new artifacts besides
the cluster map."""

from __future__ import annotations

import sys

import numpy as np

import pandas as pd
from scipy.stats import norm

from hybrid_search_rrf_dataset.router import (
    DATA_DIR,
    Representation,
    RouterExperiment,
    StrategyRouter,
)

DELTA_MIN = 0.02
COS_THRESHOLD = 0.95
TEST_FRAC = 0.2
N_BOOT = 1000
SEED = 0
EMBEDDINGS_PATH = DATA_DIR / "encoder_router" / "embeddings_bge-small-en-v1-5.parquet"
CLUSTERS_PATH = DATA_DIR / "route_labels" / "dup_clusters.parquet"
ROUTES = ["dense_only", "pure_rrf", "sparse_only"]
SCORE_BY_ROUTE = {r: f"score_{r}" for r in ROUTES}


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[rj] = ri


def build_clusters(frame: pd.DataFrame, block: int = 2048) -> pd.DataFrame:
    """cos > COS_THRESHOLD connected components over the stored bge embeddings."""
    emb = pd.read_parquet(EMBEDDINGS_PATH)
    keyed = frame[["dataset", "query_id"]].merge(
        emb, on=["dataset", "query_id"], how="left"
    )
    ecols = [c for c in emb.columns if c.startswith("e")]
    missing = keyed[ecols[0]].isna()
    if missing.any():
        print(f"WARNING: {int(missing.sum())} frame rows lack embeddings; "
              "they become singleton clusters")
    vecs = keyed[ecols].to_numpy(dtype=np.float32)
    vecs = np.nan_to_num(vecs)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.clip(norms, 1e-9, None)

    n = len(vecs)
    uf = UnionFind(n)
    for start in range(0, n, block):
        sims = vecs[start : start + block] @ vecs.T
        rows, cols = np.nonzero(sims > COS_THRESHOLD)
        for r, c in zip(rows + start, cols):
            if r < c:
                uf.union(r, c)
    roots = np.fromiter((uf.find(i) for i in range(n)), dtype=np.int64, count=n)
    out = frame[["dataset", "query_id"]].copy()
    out["cluster_id"] = roots
    return out


def grouped_split_random(
    data: pd.DataFrame, clusters: pd.Series, test_frac: float = TEST_FRAC
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The _split_random design with whole clusters kept on one side, per lane."""
    rng = np.random.default_rng(SEED)
    test_idx = []
    for _, lane_rows in data.groupby("dataset"):
        lane_clusters = clusters.loc[lane_rows.index]
        ids = lane_clusters.unique()
        rng.shuffle(ids)
        target = test_frac * len(lane_rows)
        sizes = lane_clusters.value_counts()
        picked, count = [], 0
        for cid in ids:
            if count >= target:
                break
            picked.append(cid)
            count += int(sizes[cid])
        test_idx.extend(lane_rows.index[lane_clusters.isin(picked)])
    test = data.loc[test_idx]
    return data.drop(index=test.index), test


def fit_and_deltas(
    train: pd.DataFrame, test: pd.DataFrame, baseline: str
) -> pd.DataFrame:
    """Fit the shipped router on train, return per-row (router − baseline)
    deltas over answerable test rows. baseline: 'per_lane' (C1) or 'global' (C3)."""
    answerable = test[test["shape"] != "all_zero"]
    if answerable.empty:
        return pd.DataFrame(columns=["dataset", "delta"])
    router = StrategyRouter(Representation.ENGINEERED).fit(train, all_rows="decisive")
    router.tune_thresholds(train)
    routes = router.predict_routes(answerable)
    chosen = np.array(
        [answerable.iloc[i][SCORE_BY_ROUTE[r.value]] for i, r in enumerate(routes)]
    )
    if baseline == "per_lane":
        lane_best = train.groupby("dataset")[list(SCORE_BY_ROUTE.values())].mean().idxmax(axis=1)
        base = np.array(
            [answerable.iloc[i][lane_best.get(answerable.iloc[i]["dataset"], "score_dense_only")]
             for i in range(len(answerable))]
        )
    else:
        best_col = train[list(SCORE_BY_ROUTE.values())].mean().idxmax()
        base = answerable[best_col].to_numpy()
    out = answerable[["dataset"]].copy()
    out["delta"] = chosen - base
    return out


def pooled_bootstrap(deltas: pd.DataFrame, clusters: pd.Series, n_boot: int = N_BOOT) -> float:
    """SE of the precommitted pooled statistic: resample lanes, then clusters
    within each sampled lane; statistic = unweighted mean of per-lane row-means."""
    rng = np.random.default_rng(SEED)
    per_lane = {
        lane: rows.groupby(clusters.loc[rows.index])["delta"].agg(["sum", "count"]).to_numpy()
        for lane, rows in deltas.groupby("dataset")
    }
    lanes = list(per_lane)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(len(lanes), size=len(lanes), replace=True)
        lane_means = np.empty(len(picked))
        for j, li in enumerate(picked):
            arr = per_lane[lanes[li]]
            take = rng.integers(0, len(arr), size=len(arr))
            s = arr[take]
            lane_means[j] = s[:, 0].sum() / max(s[:, 1].sum(), 1)
        stats[b] = lane_means.mean()
    return float(stats.std(ddof=1))


def power(se: float, true_delta: float) -> float:
    """P(CI lower bound > Δmin) when the true effect is true_delta."""
    return float(norm.cdf((true_delta - DELTA_MIN) / se - 1.96))


def report(name: str, deltas: pd.DataFrame, clusters: pd.Series) -> None:
    se = pooled_bootstrap(deltas, clusters)
    hw = 1.96 * se
    print(f"\n== {name} ==")
    print(f"lanes: {deltas['dataset'].nunique()}, answerable test rows: {len(deltas):,}")
    print(f"pooled bootstrap SE: {se:.4f}   CI half-width: ±{hw:.4f}")
    print(f"resolvable at Δmin={DELTA_MIN}: {'YES' if hw < DELTA_MIN else 'NO'} "
          f"(half-width {'<' if hw < DELTA_MIN else '>='} Δmin)")
    for d in (0.02, 0.03, 0.04, 0.05):
        print(f"power if true effect = {d:.2f}: {power(se, d):.2f}")


def main() -> None:
    exp = RouterExperiment()
    data = exp.load()

    if CLUSTERS_PATH.exists():
        cluster_map = pd.read_parquet(CLUSTERS_PATH)
        print(f"reusing {CLUSTERS_PATH.name} ({len(cluster_map):,} rows)")
    else:
        print("building duplicate clusters from stored bge embeddings ...")
        cluster_map = build_clusters(data)
        cluster_map.to_parquet(CLUSTERS_PATH, index=False)
        print(f"wrote {CLUSTERS_PATH}")
    merged = data.merge(cluster_map, on=["dataset", "query_id"], how="left")
    assert merged["cluster_id"].notna().all(), "cluster map must cover every row"
    clusters = merged["cluster_id"]
    sizes = clusters.value_counts()
    dup_rows = int(sizes[sizes > 1].sum())
    print(f"clusters: {clusters.nunique():,}; rows in >1-member clusters: "
          f"{dup_rows:,} ({dup_rows / len(merged):.1%})")

    # Design A — C1: grouped random split within lane, per-lane train-selected constant
    train, test = grouped_split_random(merged, clusters)
    deltas_a = fit_and_deltas(train, test, baseline="per_lane")
    report("Design A (C1: grouped random split, router − per-lane best constant)",
           deltas_a, clusters)

    # Design B — C3: leave-one-lane-out rotation, train-global constant
    lane_deltas = []
    for lane in sorted(merged["dataset"].unique()):
        tr = merged[merged["dataset"] != lane]
        te = merged[merged["dataset"] == lane]
        d = fit_and_deltas(tr, te, baseline="global")
        if d.empty:
            print(f"rotation: {lane} has no answerable rows — excluded from pooling")
            continue
        lane_deltas.append(d)
    deltas_b = pd.concat(lane_deltas, ignore_index=False)
    report("Design B (C3: hide-one-lane rotation, router − train-global constant)",
           deltas_b, clusters)


def selfcheck() -> None:
    uf = UnionFind(4)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(2) == uf.find(0) and uf.find(3) == 3
    rng = np.random.default_rng(1)
    fake = pd.DataFrame({
        "dataset": ["a"] * 500 + ["b"] * 500,
        "delta": rng.normal(0.02, 0.1, 1000),
    })
    fake_clusters = pd.Series(np.arange(1000), index=fake.index)
    se = pooled_bootstrap(fake, fake_clusters, n_boot=400)
    analytic = np.sqrt((0.1**2 / 500 + 0.1**2 / 500) / 4)
    assert 0.5 * analytic < se < 2.0 * analytic, (se, analytic)
    print("selfcheck OK")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()

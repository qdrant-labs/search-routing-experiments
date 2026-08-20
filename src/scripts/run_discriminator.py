"""Phase 5 discriminator: per lane x operator, can a text classifier separate
generated rows from natural rows of the SAME lane, judged against that cell's
own permutation null instead of a fixed AUC bar? Adds the matched-parent
control, a step-down max-T family correction, and the sibling-similarity test
no per-row classifier can do.

    poetry run python src/scripts/run_discriminator.py --plan
    poetry run python src/scripts/run_discriminator.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize

from scripts.run_ablation import DUP_CLUSTERS, _key
from scripts.select_v3_prototype import DATA, OUT, _load_labels

POOL = DATA / "augmentation" / "pool.parquet"
SELECTION = DATA / "composition" / "selection.parquet"
DISC_OUT = OUT / "discriminator"

MIN_N = 30
"""Smallest per-class n worth fitting. A balanced AUC's null sd is
sqrt((2n+1)/(12n^2)): 0.075 at n=30, so the null's OWN 95th percentile already
sits near 0.62 — at n=10 it reaches 0.72 and the fit describes the sample, not
the generator. Below this a cell is reported unfit, never fitted."""
N_PERM = 200
N_FOLDS = 5
NEAR_DUP = 0.95
"""dup_clusters.parquet's own cosine bar, reused so the sibling rate and the
organic rate are the same measurement on two row sets."""
CONTROL_REPEATS = 20
"""Matched random regroupings behind the sibling baseline — enough for a range,
not a CI; the sibling side's own n is the binding limit here."""
SEED = 0


# ------------------------------------------------------------------- stats ---
def vectorizer() -> FeatureUnion:
    """Char n-grams catch a typo or an injected token, word n-grams catch an
    added phrase; a generator that fools both is the bar."""
    return FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2)),
    ])


def cv_auc(features, y: np.ndarray, seed: int) -> float:
    """Out-of-fold ROC AUC. The vectorizer is fitted OUTSIDE this call and
    reused unchanged by every permutation, which is what makes the null an exact
    reference for the observed value rather than an approximate one."""
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    scores = cross_val_predict(
        LogisticRegression(solver="liblinear", max_iter=1000),
        features, y, cv=folds, method="decision_function",
    )
    return float(roc_auc_score(y, scores))


def permutation_null(features, y: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    """This cell's own null: the same fit at the same n with the labels
    shuffled, so the threshold is measured rather than assumed at 0.75."""
    rng = np.random.default_rng(seed)
    shuffled = y.copy()
    out = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(shuffled)
        out[i] = cv_auc(features, shuffled, seed)
    return out


def step_down_max_t(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Westfall-Young step-down max-T adjusted p-values: walk the hypotheses
    from largest observed statistic down, and at each step compare against the
    MAX null statistic over the hypotheses still in play, then enforce
    monotonicity. `null` is (n_hypotheses, n_resamples), column b being resample
    b of the whole family, so the family's dependence is carried, not assumed
    away."""
    order = np.argsort(-observed)
    n_perm = null.shape[1]
    adjusted = np.empty(len(observed))
    running = 0.0
    for step, idx in enumerate(order):
        max_null = null[order[step:], :].max(axis=0)
        raw = (1 + int((max_null >= observed[idx]).sum())) / (1 + n_perm)
        running = max(running, raw)
        adjusted[idx] = running
    return adjusted


def within_group_dup_share(
    texts: list[str], group_ids: np.ndarray, space
) -> tuple[int, int]:
    """Rows holding a within-group cosine twin above NEAR_DUP, over rows in
    groups of 2+ — dup_clusters.parquet's definition applied to a grouping."""
    matrix = normalize(space.transform(texts))
    frame = pd.DataFrame({"g": group_ids, "i": np.arange(len(texts))})
    twinned = considered = 0
    for _, grp in frame.groupby("g"):
        if len(grp) < 2:
            continue
        rows = matrix[grp["i"].to_numpy()]
        sim = (rows @ rows.T).toarray()
        np.fill_diagonal(sim, -1.0)
        considered += len(grp)
        twinned += int((sim.max(axis=1) > NEAR_DUP).sum())
    return twinned, considered


def organic_dup_share() -> tuple[float, int]:
    """The labelled pool's own near-duplicate rate — computed from
    dup_clusters.parquet, never carried as a literal."""
    clusters = pd.read_parquet(DUP_CLUSTERS)
    sizes = clusters.groupby("cluster_id").size()
    in_multi = int(sizes[sizes >= 2].sum())
    return in_multi / len(clusters), len(clusters)


# -------------------------------------------------------------------- data ---
def natural_frame(source: str) -> pd.DataFrame:
    """The lane-matched natural class. `selection` is the augmentation's own
    parent population, so the matched-parent control is drawn from the same
    frame the children were; `labels` is where v3's natural rows actually come
    from, under a different selection pressure — hence a flag, not a default."""
    if source == "labels":
        labels = _load_labels()
        return labels[["dataset", "query_id", "query"]]
    selection = pd.read_parquet(SELECTION).astype({"query_id": str})
    natural = selection[selection["generated_from"].isna()]
    return natural[["dataset", "query_id", "query"]]


def parent_text() -> pd.Series:
    """Parent query text keyed by dataset\\x00query_id, from the selection the
    operators drew parents out of, with the labelled pool as a fallback."""
    frames = [
        pd.read_parquet(SELECTION).astype({"query_id": str})[
            ["dataset", "query_id", "query"]
        ],
        _load_labels()[["dataset", "query_id", "query"]],
    ]
    joined = pd.concat(frames, ignore_index=True)
    joined["k"] = _key(joined)
    return joined.drop_duplicates("k").set_index("k")["query"]


def _sample(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    return frame.sample(n=n, random_state=int(rng.integers(2**31)))


def _auc_at(
    positives: list[str], negatives: list[str], seed: int
) -> tuple[float, object, np.ndarray]:
    texts = positives + negatives
    y = np.r_[np.ones(len(positives)), np.zeros(len(negatives))].astype(int)
    features = vectorizer().fit_transform(texts)
    return cv_auc(features, y, seed), features, y


def fit_cell(
    gen: pd.DataFrame,
    nat: pd.DataFrame,
    parents: pd.DataFrame,
    n_perm: int,
    min_n: int,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    """One lane x operator cell: balanced AUC against its own permutation null,
    plus the parent control run at a matched n on the same negatives so the
    selector's text signature subtracts out instead of being credited to the
    generator. Returns the row and the null draws, studentized by the caller."""
    rng = np.random.default_rng(seed)
    n_fit = min(len(gen), len(nat))
    pos = _sample(gen, n_fit, rng)["query"].tolist()
    neg = _sample(nat, n_fit, rng)["query"].tolist()
    auc, features, y = _auc_at(pos, neg, seed)
    null = permutation_null(features, y, n_perm, seed)
    mean, sd = float(null.mean()), float(null.std(ddof=1))
    p95 = float(np.percentile(null, 95))

    row: dict[str, object] = {
        "n_fit_per_class": n_fit,
        "auc": round(auc, 4),
        "null_mean": round(mean, 4),
        "null_sd": round(sd, 4),
        "null_p95": round(p95, 4),
        "z": round((auc - mean) / sd, 3) if sd > 0 else None,
        "exceeds_own_p95": bool(auc > p95),
        "parents_with_text": len(parents),
    }

    if len(parents) < min_n:
        row |= {
            "parent_control": f"skipped: {len(parents)} parents with text < {min_n}",
            "n_control_per_class": None, "auc_child_matched": None,
            "auc_parent": None, "delta": None,
        }
        return row, null

    n_ctl = min(n_fit, len(parents))
    ctl_neg = _sample(nat, n_ctl, rng)["query"].tolist()
    child = _sample(gen, n_ctl, rng)["query"].tolist()
    verbatim = _sample(parents, n_ctl, rng)["query"].tolist()
    auc_child, _, _ = _auc_at(child, ctl_neg, seed)
    auc_parent, _, _ = _auc_at(verbatim, ctl_neg, seed)
    # ponytail: the delta is a point estimate; giving it its own null needs a
    # paired child/parent swap permutation, which doubles the fit budget.
    row |= {
        "parent_control": "ok",
        "n_control_per_class": n_ctl,
        "auc_child_matched": round(auc_child, 4),
        "auc_parent": round(auc_parent, 4),
        "delta": round(auc_child - auc_parent, 4),
    }
    return row, null


def sibling_report(
    pool: pd.DataFrame, nat: pd.DataFrame, seed: int
) -> dict[str, object]:
    """The fingerprint AUC structurally cannot see: how concentrated within-group
    similarity is among generated rows, under both groupings the plan names
    (shared parent, shared cell), each against a same-lane natural regrouping of
    identical size structure and against the organic rate."""
    # only the pool's own lanes: the comparison is within-lane throughout, and
    # fitting the space over every other lane's text buys nothing
    nat = nat[nat["dataset"].isin(set(pool["home_lane"]))]
    space = vectorizer().fit(pool["query"].tolist() + nat["query"].tolist())
    organic, organic_rows = organic_dup_share()
    rng = np.random.default_rng(seed)
    by_lane = {
        lane: grp["query"].to_numpy() for lane, grp in nat.groupby("dataset")
    }
    out: dict[str, object] = {
        "organic_near_dup_share": round(organic, 4),
        "organic_rows": organic_rows,
        "threshold_cosine": NEAR_DUP,
        "similarity": "cosine on the classifier's own TF-IDF space (char_wb 2-4 "
                      "+ word 1-2), L2-renormalized after the union",
        "note": "the organic rate is a bge-small clustering; this side is TF-IDF, "
                "so the matched control is the like-for-like comparison and the "
                "organic rate is the plan's stated anchor",
    }
    for name, keys in (
        ("shared_parent", ["parent_dataset", "generated_from"]),
        ("shared_cell", ["home_lane", "floor"]),
    ):
        groups = pool.groupby(keys, sort=False).ngroup().to_numpy()
        twinned, considered = within_group_dup_share(
            pool["query"].tolist(), groups, space
        )
        multi = pool.groupby(keys, sort=False).agg(
            n=("query", "size"), lane=("home_lane", "first")
        )
        multi = multi[multi["n"] >= 2]
        # a group whose lane has no natural rows can be measured but not
        # controlled, so the control's own group count is reported separately
        sizes = multi[multi["lane"].isin(by_lane)]
        control = []
        for _ in range(CONTROL_REPEATS):
            texts, ids = [], []
            for gid, (n, lane) in enumerate(zip(sizes["n"], sizes["lane"])):
                pick = min(n, len(by_lane[lane]))
                texts += list(rng.choice(by_lane[lane], size=pick, replace=False))
                ids += [gid] * pick
            hit, seen = within_group_dup_share(texts, np.array(ids), space)
            control.append(hit / seen if seen else np.nan)
        out[name] = {
            "groups_of_2_plus": len(multi),
            "rows_in_those_groups": considered,
            "rows_with_a_twin": twinned,
            "sibling_share": round(twinned / considered, 4) if considered else None,
            "control_groups": len(sizes),
            "matched_control_mean": round(float(np.nanmean(control)), 4),
            "matched_control_range": [
                round(float(np.nanmin(control)), 4), round(float(np.nanmax(control)), 4)
            ],
        }
    return out


# ------------------------------------------------------------------- report ---
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural", choices=("selection", "labels"),
                        default="selection")
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--min-n", type=int, default=MIN_N)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--plan", action="store_true",
                        help="family + fitness census only, no fitting")
    args = parser.parse_args()

    if not POOL.exists():
        print(f"no generated pool at {POOL} — nothing to discriminate")
        return
    pool = pd.read_parquet(POOL)
    nat = natural_frame(args.natural)
    texts = parent_text()
    pool["parent_key"] = _key(pd.DataFrame({
        "dataset": pool["parent_dataset"],
        "query_id": pool["generated_from"].astype(str),
    }))
    pool["parent_query"] = pool["parent_key"].map(texts)

    census, family = [], []
    for (lane, operator), gen in pool.groupby(["home_lane", "operator"]):
        lane_nat = nat[nat["dataset"] == lane]
        free = lane_nat[~_key(lane_nat).isin(set(gen["parent_key"]))]
        entry = {
            "lane": lane, "operator": operator, "n_gen": len(gen),
            "n_natural_free": len(free),
            "n_parents_with_text": int(gen["parent_query"].notna().sum()),
        }
        n_fit = min(len(gen), len(free))
        if n_fit < args.min_n:
            entry["fit"] = False
            entry["reason"] = (
                f"n_fit={n_fit} < {args.min_n} "
                f"(gen {len(gen)}, free natural {len(free)})"
            )
        else:
            entry["fit"] = True
            entry["reason"] = ""
            family.append((lane, operator, gen, free))
        census.append(entry)
    census_frame = pd.DataFrame(census).sort_values(
        ["fit", "n_gen"], ascending=[False, False]
    )

    # one directory per natural class, so a comparison run never overwrites the
    # run it is being compared against
    out_dir = DISC_OUT / args.natural
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "pool_rows": len(pool),
        "operators": sorted(pool["operator"].unique().tolist()),
        "natural_class": args.natural,
        "family_size": len(family),
        "family": [f"{lane}:{op}" for lane, op, _, _ in family],
        "min_n_per_class": args.min_n,
        "min_n_rationale": "balanced-AUC null sd sqrt((2n+1)/(12n^2)) = 0.075 at "
                           "n=30; below that the cell's own null p95 exceeds 0.62 "
                           "and the fit describes the sample",
        "n_perm": args.n_perm, "n_folds": N_FOLDS, "seed": args.seed,
        "threshold_rule": "per-cell observed AUC vs that cell's own permutation "
                          "95th percentile — no fixed 0.75",
        "family_correction": "Westfall-Young step-down max-T on the studentized "
                             "AUC z = (auc - null_mean)/null_sd, because the null "
                             "sd differs per cell and a raw max-T would be "
                             "decided by the smallest lane",
        "direction": "higher AUC = more detectable generation",
        "parent_control": "AUC(children) - AUC(parents verbatim), both at a "
                          "matched n against the same negatives",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    census_frame.to_parquet(out_dir / "census.parquet", index=False)

    print(f"pool {len(pool):,} rows | operators "
          f"{', '.join(manifest['operators'])} | natural class: {args.natural}")
    print(f"\nlane x operator fitness census (min n per class = {args.min_n}):")
    print(census_frame.to_string(index=False))

    siblings = sibling_report(pool, nat, args.seed)
    (out_dir / "siblings.json").write_text(json.dumps(siblings, indent=2))
    print(f"\nMINIMAL-PAIR TEST (cosine > {NEAR_DUP}, organic anchor "
          f"{siblings['organic_near_dup_share']:.1%} over "
          f"{siblings['organic_rows']:,} labelled rows)")
    for name in ("shared_parent", "shared_cell"):
        s = siblings[name]
        share = "n/a" if s["sibling_share"] is None else f"{s['sibling_share']:.1%}"
        print(f"  {name:<14} {s['groups_of_2_plus']:>5} groups, "
              f"{s['rows_in_those_groups']:>5} rows -> {share:>6} twinned  vs "
              f"matched same-lane natural regrouping "
              f"{s['matched_control_mean']:.1%} "
              f"[{s['matched_control_range'][0]:.1%}, "
              f"{s['matched_control_range'][1]:.1%}]")

    if not family:
        print(f"\nINSUFFICIENT DATA: no lane x operator cell reaches n={args.min_n} "
              f"per class. No AUC is reported — a fit below the floor would "
              f"describe these rows, not the generator.")
        print(f"\n-> {out_dir}")
        return
    if args.plan:
        # a leftover cells.parquet under a freshly rewritten manifest is exactly
        # the stale-artifact trap Phase 0.4 is about
        (out_dir / "cells.parquet").unlink(missing_ok=True)
        print(f"\n--plan: {len(family)} cells would be fitted, nothing fitted")
        return

    print(f"\nfitting {len(family)} cells x ({args.n_perm} permutations + "
          f"2 control fits) x {N_FOLDS} folds")
    rows, nulls = [], []
    for lane, operator, gen, free in family:
        # one row per PARENT: siblings share a parent, and counting its text
        # twice would let the control fit memorize a duplicate
        parents = (
            gen.dropna(subset=["parent_query"])
            .drop_duplicates("parent_key")
            .assign(query=lambda f: f["parent_query"])
        )
        row, null = fit_cell(
            gen, free, parents, args.n_perm, args.min_n, args.seed
        )
        rows.append({"lane": lane, "operator": operator, **row})
        nulls.append(null)
        print(f"  {lane}:{operator} n={row['n_fit_per_class']} "
              f"auc={row['auc']} null_p95={row['null_p95']} "
              f"delta={row['delta']}")

    frame = pd.DataFrame(rows)
    means = np.array([n.mean() for n in nulls])
    sds = np.array([max(n.std(ddof=1), 1e-12) for n in nulls])
    observed_z = (frame["auc"].to_numpy() - means) / sds
    null_z = (np.vstack(nulls) - means[:, None]) / sds[:, None]
    frame["p_adj_max_t"] = step_down_max_t(observed_z, null_z).round(4)
    frame.to_parquet(out_dir / "cells.parquet", index=False)

    cols = ["lane", "operator", "n_fit_per_class", "auc", "null_p95",
            "exceeds_own_p95", "z", "p_adj_max_t", "n_control_per_class",
            "auc_child_matched", "auc_parent", "delta"]
    print(f"\nPER-CELL (threshold = each cell's OWN null p95):\n"
          f"{frame[cols].to_string(index=False)}")
    over = int(frame["exceeds_own_p95"].sum())
    controlled = frame["delta"].dropna()
    print(f"\n{over}/{len(frame)} cells exceed their own null p95; "
          f"{int((frame['p_adj_max_t'] <= 0.05).sum())}/{len(frame)} survive "
          f"step-down max-T at FWER 0.05 over the family")
    if len(controlled):
        print(f"parent-controlled delta over {len(controlled)} cells: "
              f"mean {controlled.mean():+.4f}, "
              f"range [{controlled.min():+.4f}, {controlled.max():+.4f}]")
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()

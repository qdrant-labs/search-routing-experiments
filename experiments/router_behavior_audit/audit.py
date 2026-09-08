"""Offline, paired audit of the two saved volume-probe arms. No fitting or tuning."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/router-diagnosis-mpl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from encoder_router.model import EncoderRouter
from encoder_router.table import NgramSvd, ZipfStats, ROUTES, serve_from_probabilities
from hybrid_search_rrf_dataset.labels import AcceptabilityLabels

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "src/data"
BASE = DATA / "encoder_router/classifiers_union_200k"
ARMS = ("no_branches", "zipf_input_nocorpus")
SHORT = {"D": "dense_only", "S": "sparse_only", "R": "pure_rrf"}
DELTA = .07
torch.set_num_threads(4)
torch.manual_seed(0)
np.random.seed(0)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return " ".join(str(text).lower().split())


def paired_ci(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0}
    rng = np.random.default_rng(2718)
    means = np.array([rng.choice(a, len(a), replace=True).mean() for _ in range(5000)])
    return {"n": len(a), "mean": float(a.mean()),
            "ci95": np.quantile(means, [.025, .975]).tolist()}


def main():
    cases = json.loads((HERE / "cases.json").read_text())
    metas = {arm: json.loads((BASE / arm / "meta.json").read_text()) for arm in ARMS}
    for field in ("embedding_model", "holdout_lane", "train_version", "prefix"):
        assert metas[ARMS[0]][field] == metas[ARMS[1]][field], field
    fingerprints = {arm: {p.name: digest(p) for p in sorted((BASE / arm).iterdir()) if p.is_file()}
                    for arm in ARMS}
    source_paths = [DATA / p for p in ("route_labels/labels.parquet", "v3/labels.parquet",
                    "v3/augmented/labels.parquet", "rungs/100k-v2/labeling/labels.parquet")]
    frames = [pd.read_parquet(p) for p in source_paths]
    shared = [c for c in frames[0].columns if all(c in f.columns for f in frames)]
    union = pd.concat([f[shared] for f in frames], ignore_index=True).drop_duplicates(
        ["dataset", "query_id"], keep="last")
    union = union[union["query"].notna() & union["query"].ne("")].reset_index(drop=True)
    holdout = metas[ARMS[0]]["holdout_lane"]
    train_texts = set(union.loc[union.dataset.ne(holdout), "query"].map(normalize))
    held = AcceptabilityLabels(union[union.dataset.eq(holdout)]).frame().reset_index(drop=True)
    # The notebook's shared-column union drops provenance because v2 lacks it.
    # Recover metadata from the SAME winning source row without changing scores.
    full_union = pd.concat([f.assign(label_source=str(p.relative_to(DATA)))
                            for f, p in zip(frames, source_paths)], ignore_index=True).drop_duplicates(
        ["dataset", "query_id"], keep="last")
    metadata_cols = [c for c in ("provenance", "scored_against", "label_source") if c in full_union]
    held = held.merge(full_union[["dataset", "query_id", *metadata_cols]],
                      on=["dataset", "query_id"], how="left", validate="one_to_one")
    print(f"Reconstructed union: {len(union):,}; holdout {holdout}: {len(held):,}", flush=True)

    rows = []
    for group in cases["groups"]:
        for variant, query in enumerate(group["queries"]):
            rows.append(dict(section="fresh", family=group["family"], category=group["category"],
                             variant=str(variant), query=query, preferred=SHORT[group["preferred"]],
                             acceptable=[SHORT[a] for a in group["acceptable"]]))
        base = group["queries"][0]
        for style, query in {"padding": "   " + base + "   ",
                             "spacing": "  ".join(base.split()),
                             "upper": base.upper(), "lower": base.lower()}.items():
            rows.append(dict(section="format", family=group["family"], category=group["category"],
                             variant=style, query=query, preferred=SHORT[group["preferred"]],
                             acceptable=[SHORT[a] for a in group["acceptable"]]))
    for i, row in enumerate(cases["user_examples"]):
        rows.append(dict(section="user_original", family=str(i), category="user_original", variant="0",
                         query=row["query"], preferred=SHORT[row["expected"]], acceptable=[SHORT[row["expected"]]]))
    for group in cases["user_paraphrases"]:
        for i, query in enumerate(group["queries"]):
            rows.append(dict(section="user_extension", family=group["family"], category="user_extension",
                             variant=str(i), query=query, preferred=SHORT[group["expected"]],
                             acceptable=[SHORT[group["expected"]]]))
    behavior = pd.DataFrame(rows)
    behavior["seen_training_text"] = behavior["query"].map(normalize).isin(train_texts)
    held["seen_training_text"] = held["query"].map(normalize).isin(train_texts)
    texts = pd.concat([behavior["query"], held["query"]], ignore_index=True)
    meta = metas[ARMS[0]]
    print(f"Encoding {len(texts)} queries locally; fixture SHA256 {digest(HERE / 'cases.json')}", flush=True)
    encoder = SentenceTransformer(meta["embedding_model"], local_files_only=True, device="cpu")
    emb = np.asarray(encoder.encode([meta["prefix"] + t for t in texts], normalize_embeddings=True,
                                   batch_size=32, show_progress_bar=True))
    zs = ZipfStats().frame(texts).to_numpy(np.float32)
    predictions = {}
    for arm in ARMS:
        p = BASE / arm
        blocks = [emb, NgramSvd.load(p / "svd.joblib").transform(texts)]
        if metas[arm]["zipf_inputs"]:
            blocks.append((zs - np.load(p / "zipf_mean.npy")) / np.load(p / "zipf_std.npy"))
        x = np.concatenate(blocks, axis=1).astype(np.float32)
        router = EncoderRouter.load(p / "router.pt")
        probs = router.probabilities(x)
        assert np.allclose(probs.to_numpy(), router.probabilities(x).to_numpy(), atol=1e-7), "nondeterministic predictions"
        thr = np.load(p / "thresholds.npy")
        margin = np.abs(probs["dense_only"].to_numpy() - probs["sparse_only"].to_numpy())
        base_routes = np.asarray(serve_from_probabilities(probs, thr))
        routes = np.where(margin < DELTA, "pure_rrf", base_routes)
        predictions[arm] = probs
        for frame, sl in ((behavior, slice(0, len(behavior))), (held, slice(len(behavior), None))):
            frame[arm] = routes[sl]
            frame[arm + ".sparse"] = probs["sparse_only"].to_numpy()[sl]
            frame[arm + ".dense"] = probs["dense_only"].to_numpy()[sl]
            frame[arm + ".base_route"] = base_routes[sl]
        print(arm, "thresholds", thr.tolist(), flush=True)

    fresh = behavior[behavior.section.eq("fresh") & ~behavior.seen_training_text].copy()
    stats = []
    for arm in ARMS:
        behavior[arm + ".accepted"] = [p in ok for p, ok in zip(behavior[arm], behavior.acceptable)]
        fresh[arm + ".accepted"] = [p in ok for p, ok in zip(fresh[arm], fresh.acceptable)]
        fresh[arm + ".preferred"] = fresh[arm].eq(fresh.preferred)
        for category, g in fresh.groupby("category"):
            base = g[g.variant.eq("0")].set_index("family")[arm]
            same = [r[arm] == base.get(r["family"]) for _, r in g[g.variant.ne("0")].iterrows()]
            stats.append(dict(arm=arm, category=category, n=len(g),
                              accepted=float(g[arm + ".accepted"].mean()),
                              preferred=float(g[arm + ".preferred"].mean()),
                              paraphrase_same_route=float(np.mean(same)),
                              all_family_variants_accepted=float(g.groupby("family")[arm + ".accepted"].all().mean()),
                              routes=g[arm].value_counts().to_dict()))
    a, z = ARMS
    paired = (fresh[z + ".accepted"].astype(float) - fresh[a + ".accepted"].astype(float))
    family_delta = paired.groupby(fresh.family).mean()
    format_stats = []
    bases = behavior[(behavior.section.eq("fresh")) & behavior.variant.eq("0")].set_index("family")
    for arm in ARMS:
        for style, g in behavior[behavior.section.eq("format")].groupby("variant"):
            changes = g[arm].to_numpy() != bases.loc[g.family, arm].to_numpy()
            format_stats.append(dict(arm=arm, style=style, n=len(g), changes=int(changes.sum())))
    baseline_acceptance = {r: float(np.mean([r in ok for ok in fresh.acceptable])) for r in ROUTES}
    # Post-hoc sensitivity analysis, explicitly separate from the frozen rubric:
    # allow hybrid for EVERY identifier lookup, as the user accepts it for CVE.
    lenient = {}
    for arm in ARMS:
        allowed = np.where(fresh.category.eq("concept"), fresh[arm].eq("dense_only"), fresh[arm].ne("dense_only"))
        lenient[arm] = float(allowed.mean())

    retrieval = []
    score_cols = [f"score_{r}" for r in ROUTES]
    scores = held[score_cols].to_numpy(float)
    train_dec = union[union.dataset.ne(holdout) & union["shape"].eq("routes_differ")]
    global_route = max(ROUTES, key=lambda r: train_dec[f"score_{r}"].mean())
    for arm in ARMS:
        held[arm + ".captured"] = scores[np.arange(len(held)), [ROUTES.index(r) for r in held[arm]]]
        held[arm + ".regret"] = scores.max(axis=1) - held[arm + ".captured"]
    masks = {"all": np.ones(len(held), dtype=bool),
             "routes_differ": held["shape"].eq("routes_differ").to_numpy()}
    if "provenance" in held:
        masks["natural"] = held.provenance.eq("natural").to_numpy()
        masks["natural_routes_differ"] = masks["natural"] & masks["routes_differ"]
        masks["natural_unseen_text"] = masks["natural"] & ~held.seen_training_text.to_numpy()
        masks["synthetic_routes_differ"] = held.provenance.eq("synthetic").to_numpy() & masks["routes_differ"]
    for label, mask in masks.items():
        g = held[mask]
        if not len(g):
            continue
        retrieval.append(dict(slice=label, n=len(g), overlap=int(g.seen_training_text.sum()),
                              captured={arm: float(g[arm + ".captured"].mean()) for arm in ARMS},
                              constants={r: float(g[f"score_{r}"].mean()) for r in ROUTES},
                              global_route=global_route,
                              oracle=float(g[score_cols].max(axis=1).mean()),
                              disagreements=int(g[a].ne(g[z]).sum()),
                              zipf_minus_base=paired_ci(g[z + ".captured"] - g[a + ".captured"])))
    held["zipf_minus_base"] = held[z + ".captured"] - held[a + ".captured"]
    # Same-probability policy swaps diagnose how much the distinct saved thresholds contribute.
    policy_swaps = []
    for arm in ARMS:
        probs = predictions[arm]
        gap = np.abs(probs["dense_only"].to_numpy() - probs["sparse_only"].to_numpy())
        for policy in ARMS:
            threshold = np.load(BASE / policy / "thresholds.npy")
            all_routes = np.where(gap < DELTA, "pure_rrf", np.asarray(serve_from_probabilities(probs, threshold)))
            routes = all_routes[:len(behavior)]
            fm = behavior.section.eq("fresh") & ~behavior.seen_training_text
            accepted = np.array([r in ok for r, ok in zip(routes, behavior.acceptable)])
            held_routes = all_routes[len(behavior):]
            held_capture = scores[np.arange(len(held)), [ROUTES.index(r) for r in held_routes]]
            policy_swaps.append(dict(model=arm, thresholds_from=policy, fresh_accepted=float(accepted[fm].mean()),
                                     heldout_differ_capture=float(held_capture[masks["routes_differ"]].mean()),
                                     heldout_natural_capture=float(held_capture[masks["natural"]].mean())))
    # Nearest training examples are evidence of template exposure, not proof of memorization.
    train = union[union.dataset.ne(holdout)].copy()
    neighbors = {}
    for pattern in ("Who likes ", "curling", "DHA", "CVE-2021-44228"):
        hits = train[train["query"].str.contains(r"\b" + re.escape(pattern) + r"\b" if pattern == "DHA" else re.escape(pattern), case=False, regex=True)]
        neighbors[pattern] = {"n": len(hits), "lanes": hits.dataset.value_counts().head(8).to_dict(),
                              "examples": hits[["dataset", "query"]].head(5).to_dict("records")}
    output = dict(fixture_sha256=digest(HERE / "cases.json"), checkpoint_hashes=fingerprints,
                  metadata=metas, union_rows=len(union), holdout=holdout,
                  fresh_total=int(behavior.section.eq("fresh").sum()), fresh_unseen=len(fresh),
                  behavior=stats, fresh_acceptance={arm: float(fresh[arm + ".accepted"].mean()) for arm in ARMS},
                  behavioral_zipf_minus_base_family_bootstrap=paired_ci(family_delta),
                  constant_behavioral_acceptance=baseline_acceptance, formatting=format_stats,
                  posthoc_hybrid_allowed_for_all_identifiers=lenient,
                  retrieval=retrieval, policy_swaps=policy_swaps, training_exposure=neighbors)
    (HERE / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    behavior.to_csv(HERE / "behavior_predictions.csv", index=False)
    held.to_csv(HERE / "heldout_predictions.csv", index=False)
    print(json.dumps({k: output[k] for k in ("fresh_unseen", "fresh_acceptance", "behavioral_zipf_minus_base_family_bootstrap", "retrieval", "policy_swaps", "formatting")}, indent=2))
    print("USER EXAMPLES\n", behavior[behavior.section.eq("user_original")][["query", *ARMS]].to_string(index=False))


if __name__ == "__main__":
    main()

"""Skeleton contrast probe: does anchor rarity decide the route?

Each pair holds one real query's frame fixed and swaps its rarest content token
for two anchors drawn from the SAME judged documents — one common, one rare
across the measured min_zipf quartile boundary. Both arms inherit the query's
own qrels, so a flipped winner can only come from the anchor.

Pairs live in their own artifact and never enter `pool.parquet`, so they carry no
floor credit and no order-sheet line; the labels land beside them, not in the v3
pool the selector reads.

    poetry run python src/scripts/probe_skeleton.py --plan
    poetry run python src/scripts/probe_skeleton.py
"""

from __future__ import annotations

import argparse
import os
import re
from functools import lru_cache
from pathlib import Path
from random import Random

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from scipy.stats import binomtest, chi2_contingency
from wordfreq import zipf_frequency

from hybrid_search_rrf_dataset.fusion import (
    DenseOnlyStrategy,
    PureRRFStrategy,
    SparseOnlyStrategy,
)
from hybrid_search_rrf_dataset.indexer import EmbeddingCache, EmbeddingConfig
from hybrid_search_rrf_dataset.labels import RouteLabels
from hybrid_search_rrf_dataset.lanes import LANES
from hybrid_search_rrf_dataset.objective import RouterObjective
from hybrid_search_rrf_dataset.retrieval import SnapshotDataset
from hybrid_search_rrf_dataset.retrieval.base import QuerySupplement
from scripts.label_routes import (
    DATA_DIR,
    DENSE_MODEL,
    DENSE_SIZE,
    SPARSE_MODEL,
    _collection,
    _source_name,
)

OUT = DATA_DIR / "v3" / "probe_skeleton"
WORD = re.compile(r"[A-Za-z]{4,}")
"""Hyphens are excluded deliberately: wordfreq resolves `h-hour` to 4.81 while
the retriever sees the literal string, so a hyphenated anchor's measured rarity
is not the rarity under test."""
RARE_MAX, COMMON_MIN = 2.31, 3.88
"""The measured min_zipf quartile boundaries — the axis under test."""
MIN_DEPTH = 2
SCORES = ["score_dense_only", "score_pure_rrf", "score_sparse_only"]
ROUTE_TO_CLASS = {"dense_only": "dense", "sparse_only": "sparse", "pure_rrf": "hybrid"}
MARGIN = RouterObjective().decisive_margin
TOL = 1e-9
STEM = 5
"""Shared leading characters that make an anchor a morphological variant of the
token it replaces (architecture -> architectural), which contrasts nothing."""
MIN_ANCHOR_SHARE = 0.10
"""Smallest share of the query the swapped token may be. Several lanes store
whole documents as their query (crumb-code-retrieval's median is 141 words), and
one token in 141 is a perturbation too small for any route to notice."""
PROVENANCE = "probe_pair"


@lru_cache(maxsize=200_000)
def _zipf(token: str) -> float:
    return zipf_frequency(token, "en")


def _anchor_slot(query: str) -> str | None:
    """The token a pair swaps: the query's rarest word-shaped token."""
    tokens = {m.group().lower() for m in WORD.finditer(query)}
    return min(tokens, key=_zipf) if tokens else None


class PairBuilder:
    """Builds contrast pairs for one lane from its judged documents."""

    def __init__(self, lane: str, seed: int = 0, anchor_source: str = "auto") -> None:
        self.lane = lane
        self.min_relevance = LANES[lane].min_relevance if lane in LANES else 1
        self._rng = Random(seed)
        source = DATA_DIR / _source_name(lane)
        self._queries = pd.read_parquet(source / "queries.parquet").astype(
            {"query_id": str}
        )
        self._qrels = pd.read_parquet(source / "qrels.parquet").astype(
            {"query_id": str, "doc_id": str}
        )
        corpus = pd.read_parquet(source / "corpus.parquet").astype({"doc_id": str})
        # a title is the better anchor source where one exists — it is the
        # entity name, so its tokens are content words by construction. Several
        # lanes carry the COLUMN but leave it empty, so fall back per document.
        # Forcing `text` on a titled lane is what separates "which lane" from
        # "which anchor source", since the two moved together across the first
        # two runs.
        title = (
            corpus["title"].fillna("") if "title" in corpus.columns
            else pd.Series("", index=corpus.index)
        )
        titled = (title.str.len() > 0).any()
        if anchor_source == "title" and not titled:
            raise ValueError(f"{lane!r} has no titles to draw anchors from")
        self.anchor_source = anchor_source if anchor_source != "auto" else (
            "title" if titled else "text"
        )
        text = (
            title.where(title.str.len() > 0, corpus["text"].fillna(""))
            if self.anchor_source == "title"
            else corpus["text"].fillna("")
        )
        self._anchor_text = dict(zip(corpus["doc_id"], text))

    def _judged(self) -> pd.Series:
        """Doc ids per query at this lane's relevance bar, deep queries only."""
        rel = self._qrels[self._qrels["relevance"] >= self.min_relevance]
        docs = rel.groupby("query_id")["doc_id"].agg(list)
        return docs[docs.map(len) >= MIN_DEPTH]

    def _anchor_pools(
        self, doc_ids: list[str], exclude: set[str], slot: str
    ) -> tuple[list, list]:
        """Anchor-source tokens of the judged docs, split rare vs common."""
        rare, common = [], []
        for doc_id in doc_ids:
            for match in WORD.finditer(self._anchor_text.get(doc_id, "")):
                token = match.group().lower()
                if token in exclude or token[:STEM] == slot[:STEM]:
                    continue
                z = _zipf(token)
                if z <= RARE_MAX:
                    rare.append(token)
                elif COMMON_MIN <= z:
                    common.append(token)
        return sorted(set(rare)), sorted(set(common))

    def build(self, n_pairs: int) -> pd.DataFrame:
        text = dict(zip(self._queries["query_id"], self._queries["text"]))
        rows: list[dict[str, object]] = []
        for query_id, doc_ids in self._judged().items():
            if len(rows) >= 2 * n_pairs:
                break
            query = str(text.get(query_id, ""))
            slot = _anchor_slot(query)
            if slot is None:
                continue
            frame_tokens = {m.group().lower() for m in WORD.finditer(query)}
            rare, common = self._anchor_pools(list(doc_ids), frame_tokens, slot)
            if not rare or not common:
                continue
            swap = re.compile(rf"\b{re.escape(slot)}\b", re.I)
            for arm, pool in (("common", common), ("rare", rare)):
                anchor = self._rng.choice(pool)
                rows.append({
                    "pair_id": f"{self.lane}:{query_id}",
                    "arm": arm,
                    "dataset": self.lane,
                    "query_id": f"probe-{query_id}-{arm}",
                    "query": swap.sub(anchor, query, count=1),
                    "anchor": anchor,
                    "anchor_zipf": round(_zipf(anchor), 3),
                    "replaced": slot,
                    "frame": query,
                    "source_query_id": query_id,
                    "depth": len(doc_ids),
                    "provenance": PROVENANCE,
                })
        return pd.DataFrame(rows)

    def extra_qrels(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Each arm inherits its source query's judged rows verbatim."""
        rel = self._qrels[self._qrels["relevance"] >= self.min_relevance]
        merged = pairs[["query_id", "source_query_id"]].merge(
            rel, left_on="source_query_id", right_on="query_id",
            suffixes=("", "_src"),
        )
        return merged[["query_id", "doc_id", "relevance"]]


def _label(
    pairs: pd.DataFrame, qrels: pd.DataFrame, lane: str, out_dir: Path
) -> pd.DataFrame:
    """Score both arms through the three routes, into the probe's own artifact."""
    load_dotenv()
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"), timeout=60,
    )
    dense = EmbeddingConfig(
        name="dense_base", model_id=DENSE_MODEL, kind="dense",
        size=DENSE_SIZE, distance=Distance.COSINE, parallel=4,
    )
    sparse = EmbeddingConfig(
        name="sparse_base", model_id=SPARSE_MODEL, kind="sparse",
    )
    source = QuerySupplement(
        SnapshotDataset(_source_name(lane), path=str(DATA_DIR)),
        pairs.rename(columns={"query": "text"})[["query_id", "text"]],
        qrels,
    )
    args = (
        client, _collection(lane), dense, sparse,
        EmbeddingCache(namespace=_source_name(lane)),
    )
    strategies = (
        DenseOnlyStrategy(*args[:4]), PureRRFStrategy(*args[:4]),
        SparseOnlyStrategy(*args[:4]),
    )
    labels = RouteLabels(
        pairs[["dataset", "query_id", "query"]],
        out_dir=out_dir,
        objective=RouterObjective(
            min_relevance=LANES[lane].min_relevance if lane in LANES else 1
        ),
    )
    return labels.label(source, *strategies, dataset=lane)


def report(pairs: pd.DataFrame, anchor_source: str) -> None:
    print(f"pairs: {pairs['pair_id'].nunique():,} ({len(pairs):,} arms) "
          f"in {pairs['dataset'].nunique()} lane(s), anchors from {anchor_source}")
    words = pairs["frame"].str.split().str.len()
    share = (1.0 / words).median()
    print(f"the swapped token is {share:.1%} of the median query "
          f"({words.median():.0f} words)")
    if share < MIN_ANCHOR_SHARE:
        print(f"  WARNING: below {MIN_ANCHOR_SHARE:.0%} — this lane's queries are "
              "long enough that one token cannot plausibly move the route, so a "
              "null here would measure the perturbation's size, not rarity.")
    by_arm = pairs.groupby("arm")["anchor_zipf"].describe()[["count", "mean", "min", "max"]]
    print(f"\nanchor zipf by arm:\n{by_arm.round(2).to_string()}")
    print(f"\nmedian judged depth per pair: {pairs['depth'].median():.0f}")
    print("\nexample pairs:")
    for pid, grp in list(pairs.groupby("pair_id"))[:5]:
        frame = grp["frame"].iloc[0]
        print(f"  frame {frame!r}  (swapping {grp['replaced'].iloc[0]!r})")
        for row in grp.sort_values("arm").itertuples(index=False):
            print(f"    {row.arm:<6} z={row.anchor_zipf:<5} {row.query!r}")


def readout(out_dir: Path) -> pd.DataFrame:
    """Per-pair arms side by side, plus the three paired tests, from the labels
    already on disk. Only the anchor varies, so a shift is attributable to
    rarity; arm AGREEMENT is not attributable to the frame, which the design
    holds constant alongside the lane and the documents."""
    pairs = pd.read_parquet(out_dir / "pairs.parquet")
    labels = pd.read_parquet(out_dir / "labels.parquet").astype({"query_id": str})
    ordered = np.sort(labels[SCORES].to_numpy(), axis=1)
    oracle, runner, low = ordered[:, -1], ordered[:, -2], ordered[:, 0]
    labels = labels.assign(
        top=labels[SCORES].idxmax(axis=1).str.replace("score_", "").map(ROUTE_TO_CLASS),
        answered=oracle > TOL,
        decisive=(oracle > TOL) & (oracle - low > TOL) & (oracle - runner >= MARGIN),
        oracle=oracle,
    )
    joined = pairs.merge(
        labels[["query_id", "top", "answered", "decisive", "oracle"]],
        on="query_id", how="left",
    )
    wide = joined.pivot(
        index="pair_id", columns="arm",
        values=["top", "answered", "decisive", "oracle"],
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]

    both = wide[wide["answered_common"] & wide["answered_rare"]]
    table = pd.crosstab(both["top_common"], both["top_rare"])
    chi2, p_assoc, dof, expected = chi2_contingency(table)

    def mcnemar(rare_only: pd.Series, common_only: pd.Series) -> tuple[int, int, float]:
        b, c = int(rare_only.sum()), int(common_only.sum())
        return b, c, binomtest(b, max(b + c, 1), 0.5).pvalue

    sparse_b, sparse_c, p_sparse = mcnemar(
        (both["top_rare"] == "sparse") & (both["top_common"] != "sparse"),
        (both["top_common"] == "sparse") & (both["top_rare"] != "sparse"),
    )
    dec_b, dec_c, p_dec = mcnemar(
        wide["decisive_rare"] & ~wide["decisive_common"],
        wide["decisive_common"] & ~wide["decisive_rare"],
    )
    ans_b, ans_c, p_ans = mcnemar(
        wide["answered_rare"] & ~wide["answered_common"],
        wide["answered_common"] & ~wide["answered_rare"],
    )
    ci = binomtest(sparse_b, max(sparse_b + sparse_c, 1), 0.5).proportion_ci(
        method="wilson"
    )
    wide.attrs["stats"] = {
        "run": out_dir.name,
        "pairs": len(wide),
        "both_answered": len(both),
        "sparse_rare_only": sparse_b,
        "sparse_common_only": sparse_c,
        "sparse_p": p_sparse,
        "sparse_lo_pp": (ci.low - 0.5) * 100,
        "sparse_hi_pp": (ci.high - 0.5) * 100,
        "agree": np.trace(table.to_numpy()) / table.to_numpy().sum(),
        "agree_chance": np.trace(expected) / expected.sum(),
        "agree_p": p_assoc,
        "dec_common": int(wide["decisive_common"].sum()),
        "dec_rare": int(wide["decisive_rare"].sum()),
        "dec_p": p_dec,
        "ans_common": int(wide["answered_common"].sum()),
        "ans_rare": int(wide["answered_rare"].sum()),
        "ans_p": p_ans,
        "oracle_common": wide["oracle_common"].mean(),
        "oracle_rare": wide["oracle_rare"].mean(),
    }

    print(f"pairs {len(wide):,} | both arms answered {len(both):,}")
    print("\nDOES THE WINNER GO SPARSE WHEN THE ANCHOR IS RARE?")
    print(f"  rare-only sparse {sparse_b} vs common-only {sparse_c}, p={p_sparse:.3f}")
    print(f"  directional effect bounded to "
          f"[{(ci.low - 0.5) * 100:+.1f}, {(ci.high - 0.5) * 100:+.1f}] pp")
    print("\nDO MATCHED ARMS AGREE MORE THAN CHANCE?")
    print(f"  arms agree {np.trace(table.to_numpy()) / table.to_numpy().sum():.1%} vs "
          f"{np.trace(expected) / expected.sum():.1%} under independence "
          f"(chi2={chi2:.1f}, p={p_assoc:.2e})")
    print("  NOT attributable to the frame: the arms share frame, lane, grounding "
          "documents and qrels, and only the anchor varies.")
    print("\nWHAT DOES RARITY MOVE INSTEAD?")
    print(f"  answered:  common {int(wide['answered_common'].sum())}/{len(wide)} | "
          f"rare {int(wide['answered_rare'].sum())}/{len(wide)} "
          f"(discordant {ans_b} vs {ans_c}, p={p_ans:.2e})")
    print(f"  decisive:  common {int(wide['decisive_common'].sum())}/{len(wide)} | "
          f"rare {int(wide['decisive_rare'].sum())}/{len(wide)} "
          f"(discordant {dec_b} vs {dec_c}, p={p_dec:.4f})")
    print(f"  mean oracle: common {wide['oracle_common'].mean():.3f} | "
          f"rare {wide['oracle_rare'].mean():.3f}")
    print(f"\nroute of the top arm (rows=common, cols=rare):\n{table.to_string()}")
    wide.to_parquet(out_dir / "readout.parquet")
    return wide


def compare() -> pd.DataFrame:
    """Every probe run side by side — the cross-lane question the single-lane
    readout cannot answer."""
    import contextlib
    import io

    rows = []
    for run in sorted(p for p in OUT.iterdir() if (p / "labels.parquet").exists()):
        with contextlib.redirect_stdout(io.StringIO()):
            stats = readout(run).attrs["stats"]
        pairs = pd.read_parquet(run / "pairs.parquet")
        anchors = pairs.groupby("arm")["anchor"].nunique().to_dict()
        rows.append({**stats, "distinct_anchors": min(anchors.values())})
    frame = pd.DataFrame(rows).set_index("run")
    view = frame.assign(
        sparse_shift=frame.apply(
            lambda r: f"{r.sparse_rare_only:>3}v{r.sparse_common_only:<3} "
                      f"p={r.sparse_p:.4f}", axis=1),
        sparse_ci=frame.apply(
            lambda r: f"[{r.sparse_lo_pp:+.0f},{r.sparse_hi_pp:+.0f}]pp", axis=1),
        decisive=frame.apply(
            lambda r: f"{r.dec_common:>3}->{r.dec_rare:<3} p={r.dec_p:.3f}", axis=1),
        answered=frame.apply(
            lambda r: f"{r.ans_common:>3}->{r.ans_rare:<3} p={r.ans_p:.1e}", axis=1),
    )
    print(view[["pairs", "sparse_shift", "sparse_ci", "decisive", "answered"]]
          .to_string())
    frame.to_parquet(OUT / "compare.parquet")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", default="dbpedia-entity")
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan", action="store_true", help="build + report, no retrieval")
    parser.add_argument("--readout", action="store_true", help="analyse existing labels")
    parser.add_argument("--anchor-source", choices=("auto", "title", "text"),
                        default="auto")
    parser.add_argument("--compare", action="store_true", help="all runs side by side")
    args = parser.parse_args()

    if args.compare:
        compare()
        return

    # a forced source gets its own directory, so the auto run it is compared
    # against is never overwritten
    suffix = "" if args.anchor_source == "auto" else f"@{args.anchor_source}"
    out_dir = OUT / f"{args.lane}{suffix}"
    if args.readout:
        readout(out_dir)
        return

    builder = PairBuilder(args.lane, seed=args.seed, anchor_source=args.anchor_source)
    pairs = builder.build(args.pairs)
    if pairs.empty:
        print(f"{args.lane}: no pair had both a rare and a common anchor")
        return
    report(pairs, builder.anchor_source)

    out_dir.mkdir(parents=True, exist_ok=True)
    qrels = builder.extra_qrels(pairs)
    if args.plan:
        print("\n--plan: nothing labelled")
        return
    pairs.to_parquet(out_dir / "pairs.parquet", index=False)
    qrels.to_parquet(out_dir / "qrels.parquet", index=False)
    out = _label(pairs, qrels, args.lane, out_dir)
    print(f"\nlabelled {len(out):,} arms -> {out_dir}")


if __name__ == "__main__":
    main()

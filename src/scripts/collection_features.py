"""Query-Corpus plumbing (SPEC d47e/f): one `CorpusIndex` per lane corpus
under `src/data/<lane>/corpus_index.parquet`, plus the 16-row side test that
asks whether six per-corpus numbers tell our lanes apart at all.

`CollectionIndexStore` owns the index artifacts the way `SupplyIndex` owns
`surfaces.parquet` — built lazily, skipped when on disk, `--force` rebuilds.
The taxonomy owns the six definitions; this module only counts documents and
reads the answer out.

    poetry run python src/scripts/collection_features.py          # build + side test
    poetry run python src/scripts/collection_features.py --force  # recount corpora
    poetry run python src/scripts/collection_features.py --per-query --v3
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Final

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from augmentation.config import AugmentationPaths
from augmentation.supply import lane_dirs
from composition.indexer import SPARSE_MODEL_ID
from hybrid_search_rrf_dataset.fusion import StrategyName
from hybrid_search_rrf_dataset.router import LABELS_PATH, SCORE_COLS, decisive_rows
from query_taxonomy.corpus_relative import (
    CORPUS_RELATIVE_BANKS,
    CorpusIndex,
    CorpusRelativeBank,
)
from query_taxonomy.metrics.general import STOPWORDS

SEED: Final[int] = 0

# the v3-NATIVE label files; labels_rederived.parquet is the re-scored v2 set,
# which query_corpus_stats.parquet already covers under its own keys
V3_LABELS: Final[tuple[Path, ...]] = (
    LABELS_PATH.parent.parent / "v3" / "labels.parquet",
    LABELS_PATH.parent.parent / "v3" / "synthetic" / "labels.parquet",
    LABELS_PATH.parent.parent / "v3" / "augmented" / "labels.parquet",
)

STAT_COLS: Final[tuple[str, ...]] = (
    "avg_idf",
    "max_idf",
    "oov_share",
    "collection_size",
    "avg_doc_length",
    "vocab_overlap",
)

# Pre-committed reads (SPEC d47f), on 16 lanes and a 3-class target.
_LANES: Final[int] = 16
_DISCRIMINATES: Final[int] = 12
_PARTIAL: Final[int] = 8


class Bm25Tokenizer:
    """The sparse route's own tokenization, so the features and the retriever
    share one vocabulary: fastembed's `Qdrant/bm25` regex + Snowball stemmer,
    punctuation and length filters included."""

    def __init__(self, model_id: str = SPARSE_MODEL_ID) -> None:
        from fastembed.common.utils import remove_non_alphanumeric
        from fastembed.sparse.bm25 import Bm25

        self._clean = remove_non_alphanumeric
        self._model = Bm25(
            model_id, specific_model_path=str(_snapshot_dir(model_id))
        )
        # The cached snapshot's english.txt blob is absent offline, which would
        # silently leave stopwords in the stream and turn avg_idf into a
        # stopword-ratio proxy. The taxonomy's closed list stands in.
        # ponytail: swap back to the model's own list when the cache is filled.
        self._model.stopwords = set(STOPWORDS)

    def tokens(self, text: str) -> list[str]:
        # the three calls Bm25.raw_embed makes, minus the mmh3 hashing: we
        # want the terms themselves, and re-deriving them would drift
        return self._model._stem(
            self._model.tokenizer.tokenize(self._clean(text))
        )


def _snapshot_dir(model_id: str) -> Path:
    """fastembed's cached snapshot for `model_id`. Resolved by hand because
    the cache carries no `refs/`, so the offline path cannot pin a revision
    and would fall through to a download."""
    from fastembed.common.utils import define_cache_dir

    repo = f"models--{model_id.replace('/', '--')}"
    snapshots = sorted((Path(define_cache_dir()) / repo / "snapshots").glob("*"))
    if not snapshots:
        raise FileNotFoundError(
            f"{model_id} is not in the fastembed cache — index a lane once "
            f"(or fetch the model) before building corpus indexes"
        )
    return snapshots[-1]


class CollectionIndexStore:
    """Owns the per-lane `corpus_index.parquet` artifacts: (term, df) rows
    with n_docs and avgdl in the file metadata."""

    def __init__(
        self,
        paths: AugmentationPaths | None = None,
        tokenizer: Bm25Tokenizer | None = None,
    ) -> None:
        self._paths = paths or AugmentationPaths()
        self.tokenizer = tokenizer or Bm25Tokenizer()

    def path(self, lane: str) -> Path:
        return self._paths.lane_corpus_index(lane)

    def indexable(self) -> dict[str, str]:
        """Lane key -> on-disk dir, for lanes whose corpus is materialized.
        The dir is `source.name`, never the key — `beir-nfcorpus` lives in
        `nfcorpus`, and the label frame is keyed by the key."""
        found = {
            key: lane
            for key, lane in lane_dirs().items()
            if self._paths.lane_corpus(lane).exists()
        }
        # two keys resolving to one dir would silently pool two lanes' corpora
        assert len(set(found.values())) == len(found), sorted(found.items())
        return found

    def build(self, lane: str, *, force: bool = False) -> CorpusIndex:
        """Count document frequencies over one lane corpus. Idempotent — the
        tokenization pass is the expensive leg."""
        target = self.path(lane)
        if target.exists() and not force:
            return self.load(lane)
        corpus = pd.read_parquet(self._paths.lane_corpus(lane))
        texts = corpus["text"].fillna("")
        if "title" in corpus.columns:
            # matches CorpusIndexer.item_text — the text that was indexed
            texts = (corpus["title"].fillna("") + " " + texts).str.strip()

        frequencies: Counter[str] = Counter()
        tokens_total = 0
        for text in tqdm(texts, desc=f"corpus_index:{lane}", unit="doc"):
            tokens = self.tokenizer.tokens(str(text))
            tokens_total += len(tokens)
            frequencies.update(set(tokens))
        n_docs = len(corpus)
        index = CorpusIndex(
            document_frequencies=dict(frequencies),
            n_docs=n_docs,
            avgdl=tokens_total / n_docs if n_docs else 0.0,
        )
        self._write(target, index)
        print(
            f"{lane}: {n_docs:,} docs, {len(frequencies):,} terms, "
            f"avgdl {index.avgdl:.1f} -> {target}"
        )
        return index

    def build_all(self, *, force: bool = False) -> None:
        for lane in sorted(set(self.indexable().values())):
            self.build(lane, force=force)

    def load(self, lane: str) -> CorpusIndex:
        table = pq.read_table(self.path(lane))
        meta = table.schema.metadata or {}
        return CorpusIndex(
            document_frequencies=dict(
                zip(
                    table.column("term").to_pylist(),
                    table.column("df").to_pylist(),
                    strict=True,
                )
            ),
            n_docs=int(meta[b"n_docs"]),
            avgdl=float(meta[b"avgdl"]),
        )

    @staticmethod
    def _write(target: Path, index: CorpusIndex) -> None:
        table = pa.table({
            "term": list(index.document_frequencies),
            "df": list(index.document_frequencies.values()),
        }).replace_schema_metadata({
            "n_docs": str(index.n_docs),
            "avgdl": repr(index.avgdl),
        })
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target)


class LaneCorpusStats:
    """The d47f side test: one row per lane (six mean stats + that lane's
    best-constant route) and a leave-one-out probe over it. n is ~16, so a
    single train/test split would mean nothing."""

    def __init__(
        self,
        store: CollectionIndexStore | None = None,
        labels_path: Path = LABELS_PATH,
        out_path: Path | None = None,
        *,
        decisive_only: bool = True,
    ) -> None:
        self._store = store or CollectionIndexStore()
        self._labels_path = labels_path
        self._decisive_only = decisive_only
        suffix = "" if decisive_only else "_all_rows"
        self._out_path = out_path or (
            labels_path.parent / f"lane_corpus_stats{suffix}.parquet"
        )

    def build(self, *, force: bool = False) -> pd.DataFrame:
        if self._out_path.exists() and not force:
            return pd.read_parquet(self._out_path)
        labels = pd.read_parquet(self._labels_path)
        # decisive-only is the pre-committed construction; it costs every lane
        # with no decisive row at all (bright-aops) and collapses the target
        # toward dense, so `--all-rows` keeps the same target definition over
        # the full label set as a diagnostic.
        frame_in = decisive_rows(labels) if self._decisive_only else labels
        dirs = self._store.indexable()
        rows: list[dict[str, object]] = []
        for key, group in frame_in.groupby("dataset", sort=True):
            lane = dirs.get(str(key))
            if lane is None or not self._store.path(lane).exists():
                print(f"[skip] {key}: no corpus index on disk")
                continue
            index = self._store.load(lane)
            banks = [cls(index) for cls in CORPUS_RELATIVE_BANKS]
            stats = pd.DataFrame(
                self._query_stats(query, banks) for query in group["query"]
            )
            rows.append({
                "dataset": key,
                "lane_dir": lane,
                "n_queries": len(group),
                **stats.mean().to_dict(),
                "target": self._best_constant(group),
            })
        frame = pd.DataFrame(rows)
        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(self._out_path, index=False)
        return frame

    def leave_one_out(self, frame: pd.DataFrame | None = None) -> pd.DataFrame:
        """Per-lane truth and held-out prediction. The lane under test is
        never in the training rows."""
        frame = self.build() if frame is None else frame
        x = frame[list(STAT_COLS)].to_numpy(dtype=float)
        y = frame["target"].to_numpy()
        predictions: list[str] = []
        for train, test in LeaveOneOut().split(x):
            model = Pipeline([
                ("scale", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=SEED,
                    ),
                ),
            ])
            model.fit(x[train], y[train])
            predictions.append(str(model.predict(x[test])[0]))
        return frame[["dataset", "n_queries", "target"]].assign(
            predicted=predictions,
            correct=[p == t for p, t in zip(predictions, y, strict=True)],
        )

    def _query_stats(
        self, query: str, banks: list[CorpusRelativeBank]
    ) -> dict[str, float]:
        tokens = self._store.tokenizer.tokens(str(query))
        return {
            stat.name: stat.value
            for bank in banks
            for stat in bank.compute(tokens)
        }

    @staticmethod
    def _best_constant(group: pd.DataFrame) -> str:
        """The route that wins this lane if you always pick one route."""
        means = {
            route.value: float(group[column].mean())
            for route, column in SCORE_COLS.items()
        }
        return max(means, key=lambda route: means[route])


class QueryCorpusStats:
    """Per-QUERY corpus-relative stats keyed (dataset, query_id) — the v3
    unit's corpus dimension. Where LaneCorpusStats keeps per-lane means, this
    keeps every row, and it populates each lane's pair co-occurrence so PMI has
    data (LaneCorpusStats leaves pair_document_frequencies empty -> no PMI)."""

    def __init__(
        self,
        store: CollectionIndexStore | None = None,
        labels_path: Path | tuple[Path, ...] = LABELS_PATH,
        out_path: Path | None = None,
    ) -> None:
        self._store = store or CollectionIndexStore()
        self._labels_paths = (
            (labels_path,) if isinstance(labels_path, Path) else tuple(labels_path)
        )
        self._out_path = out_path or (
            LABELS_PATH.parent / "query_corpus_stats.parquet"
        )

    @property
    def out_path(self) -> Path:
        return self._out_path

    def _labels(self) -> pd.DataFrame:
        """The label rows to measure, from every label file that exists."""
        frames = [
            pd.read_parquet(path, columns=["dataset", "query_id", "query"])
            for path in self._labels_paths
            if path.exists()
        ]
        if not frames:
            raise FileNotFoundError(f"no label file on disk: {self._labels_paths}")
        return (
            pd.concat(frames, ignore_index=True)
            .astype({"query_id": str})
            .drop_duplicates(["dataset", "query_id"])
        )

    def build(self, *, force: bool = False) -> pd.DataFrame:
        """Measure the label rows this artifact does not carry yet and append;
        `force` re-measures the ones it does."""
        frame = (
            pd.read_parquet(self._out_path)
            if self._out_path.exists()
            else pd.DataFrame(columns=["dataset", "query_id"])
        )
        labels = self._labels()
        if not force and not frame.empty:
            known = pd.MultiIndex.from_frame(
                frame[["dataset", "query_id"]].astype({"query_id": str})
            )
            labels = labels[
                ~pd.MultiIndex.from_frame(labels[["dataset", "query_id"]]).isin(known)
            ]
        dirs = self._store.indexable()
        for key, group in tqdm(labels.groupby("dataset", sort=True), unit="lane"):
            lane = dirs.get(str(key))
            if lane is None or not self._store.path(lane).exists():
                print(f"[skip] {key}: no corpus index on disk")
                continue
            tok = self._store.tokenizer
            tokens = {
                qid: tok.tokens(str(text))
                for qid, text in zip(group["query_id"], group["query"], strict=True)
            }
            pairs, vocab = self._wanted_pairs(tokens.values())
            index = replace(
                self._store.load(lane),
                pair_document_frequencies=self._pair_counts(lane, pairs, vocab),
            )
            banks = [cls(index) for cls in CORPUS_RELATIVE_BANKS]
            rows: list[dict[str, object]] = []
            for qid, toks in tokens.items():
                stats = {
                    stat.name: stat.value
                    for bank in banks
                    for stat in bank.compute(toks)
                }
                rows.append({"dataset": key, "query_id": qid, **stats})
            # written per lane, like RouteLabels' chunks: the corpus pass is the
            # expensive leg, so a crash costs one lane instead of the whole run
            frame = pd.concat(
                [frame, pd.DataFrame(rows)], ignore_index=True
            ).drop_duplicates(["dataset", "query_id"], keep="last")
            self._out_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(self._out_path, index=False)
        return frame

    @staticmethod
    def _wanted_pairs(
        token_lists,
    ) -> tuple[set[frozenset[str]], set[str]]:
        """Distinct query term-pairs (PMI needs exactly these) + the query
        vocabulary (to prune each document's tokens before pairing)."""
        vocab: set[str] = set()
        pairs: set[frozenset[str]] = set()
        for toks in token_lists:
            unique = set(toks)
            vocab |= unique
            for a, b in combinations(sorted(unique), 2):
                pairs.add(frozenset((a, b)))
        return pairs, vocab

    def _pair_counts(
        self, lane: str, pairs: set[frozenset[str]], vocab: set[str]
    ) -> dict[frozenset[str], int]:
        """Documents co-occurring each query pair — one corpus pass, restricted
        to the query vocabulary so only the wanted pairs are ever formed."""
        if not pairs:
            return {}
        corpus = pd.read_parquet(self._store._paths.lane_corpus(lane))
        texts = corpus["text"].fillna("")
        if "title" in corpus.columns:
            texts = (corpus["title"].fillna("") + " " + texts).str.strip()
        counts: Counter[frozenset[str]] = Counter()
        for text in tqdm(texts, desc=f"pairs:{lane}", unit="doc", leave=False):
            terms = sorted(set(self._store.tokenizer.tokens(str(text))) & vocab)
            for a, b in combinations(terms, 2):
                pair = frozenset((a, b))
                if pair in pairs:
                    counts[pair] += 1
        return dict(counts)


def _band(correct: int) -> str:
    if correct >= _DISCRIMINATES:
        return "corpus stats discriminate our lanes"
    if correct >= _PARTIAL:
        return "partial signal"
    return f"about chance for a {len(StrategyName)}-class target"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-lane CorpusIndex artifacts and run the d47f "
        "side test."
    )
    parser.add_argument(
        "--force", action="store_true", help="recount corpora and re-run"
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="target from every labelled row, not the decisive ones only "
        "(diagnostic: keeps lanes with no decisive row)",
    )
    parser.add_argument(
        "--per-query",
        action="store_true",
        help="build per-query corpus stats (incl. PMI) keyed (dataset, "
        "query_id) for the v3 unit, instead of the per-lane side test",
    )
    parser.add_argument(
        "--v3",
        action="store_true",
        help="with --per-query: measure the v3-native label rows, appending "
        "to the same artifact (v2's rows are already in it)",
    )
    args = parser.parse_args()

    store = CollectionIndexStore()
    print(f"indexable lanes: {sorted(store.indexable())}")
    store.build_all(force=args.force)

    if args.per_query:
        stats = QueryCorpusStats(store, V3_LABELS if args.v3 else LABELS_PATH)
        frame = stats.build(force=args.force)
        print(f"per-query corpus stats: {len(frame):,} rows, "
              f"cols {[c for c in frame.columns if c not in ('dataset', 'query_id')]}")
        print(f"-> {stats.out_path}")
        return

    probe = LaneCorpusStats(store, decisive_only=not args.all_rows)
    frame = probe.build(force=args.force)
    print(frame.to_string(index=False))
    readout = probe.leave_one_out(frame)
    correct = int(readout["correct"].sum())
    total = len(readout)
    print(readout.to_string(index=False))
    print(f"\nleave-one-out: {correct}/{total} — {_band(correct)}")
    majority = frame["target"].value_counts()
    print(f"majority-class baseline: {majority.iloc[0]}/{total} ({majority.index[0]})")
    if total != _LANES:
        print(f"WARNING: band was pre-committed on {_LANES} lanes, not {total}")
    if correct <= majority.iloc[0]:
        print("WARNING: the band label is unearned — guessing one route scores "
              "at least as well, so the stats add nothing")


if __name__ == "__main__":
    main()

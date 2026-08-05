"""Query-Corpus plumbing (SPEC d47e/f): one `CorpusIndex` per lane corpus
under `src/data/<lane>/corpus_index.parquet`, plus the 16-row side test that
asks whether six per-corpus numbers tell our lanes apart at all.

`CollectionIndexStore` owns the index artifacts the way `SupplyIndex` owns
`surfaces.parquet` — built lazily, skipped when on disk, `--force` rebuilds.
The taxonomy owns the six definitions; this module only counts documents and
reads the answer out.

    poetry run python src/scripts/collection_features.py          # build + side test
    poetry run python src/scripts/collection_features.py --force  # recount corpora
"""

from __future__ import annotations

import argparse
from collections import Counter
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
from query_taxonomy.corpus_relative import CORPUS_RELATIVE_BANKS, CorpusIndex
from query_taxonomy.metrics.general import STOPWORDS

SEED: Final[int] = 0

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
        return self._paths.data_dir / lane / "corpus_index.parquet"

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
        self._banks = [cls() for cls in CORPUS_RELATIVE_BANKS]

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
            stats = pd.DataFrame(
                self._query_stats(query, index) for query in group["query"]
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

    def _query_stats(self, query: str, index: CorpusIndex) -> dict[str, float]:
        tokens = self._store.tokenizer.tokens(str(query))
        return {
            stat.name: stat.value
            for bank in self._banks
            for stat in bank.compute(tokens, index)
        }

    @staticmethod
    def _best_constant(group: pd.DataFrame) -> str:
        """The route that wins this lane if you always pick one route."""
        means = {
            route.value: float(group[column].mean())
            for route, column in SCORE_COLS.items()
        }
        return max(means, key=lambda route: means[route])


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
    args = parser.parse_args()

    store = CollectionIndexStore()
    print(f"indexable lanes: {sorted(store.indexable())}")
    store.build_all(force=args.force)

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

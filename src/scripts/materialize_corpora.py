"""Lane corpora in two passes: pass 1 saves queries + qrels, pass 2 sizes the
corpus from those qrels and saves it. `--plan` prices pass 2 from the qrels
already on disk, because a lane costs hours and gigabytes to build.

`LaneCorpora` owns the snapshot directories the way `FeatureTable` owns the
feature parquet: one directory per lane, skipped when the artifact is already
there.

    poetry run python src/scripts/materialize_corpora.py --plan
    poetry run python src/scripts/materialize_corpora.py --metadata
    poetry run python src/scripts/materialize_corpora.py
    poetry run python src/scripts/materialize_corpora.py --only quest --force
"""

import argparse
from pathlib import Path

import pandas as pd
from pyarrow.parquet import ParquetFile
from tqdm.auto import tqdm

from composition import CellFill

from hybrid_search_rrf_dataset.lanes import LANES
from hybrid_search_rrf_dataset.retrieval import MaterializedDataset

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class LaneCorpora:
    """Owns every lane's snapshot directory across both acquisition passes.

    `plan()` reads only what is on disk, so the corpus bill can be inspected
    before any of it is paid.
    """

    PARTS = ("queries", "qrels", "excluded", "corpus")

    def __init__(self, data_dir: Path | None = None) -> None:
        self._dir = data_dir if data_dir is not None else DEFAULT_DATA_DIR
        self._selection: pd.DataFrame | None = None

    def _wanted(self, key: str) -> set[str] | None:
        """The composition's distinct query ids for this lane; None when the
        selection has no rows for it. Pinned onto every MaterializedDataset
        before its metadata loads, so a lane that samples its query set
        samples exactly the selection — a snapshot drawn independently of the
        selection silently drops selected, judged queries from the labels."""
        if self._selection is None:
            self._selection = pd.read_parquet(
                CellFill().selection_path, columns=["dataset", "query_id"]
            )
        ids = self._selection.loc[
            self._selection["dataset"].astype(str) == key, "query_id"
        ].astype(str)
        return set(ids) if len(ids) else None

    def lane_dir(self, key: str) -> Path:
        """Resolve through `source.name`, which owns the directory — the table
        key only points at the lane."""
        name = LANES[key].source.name
        assert name == key, f"{key}: lane table key disagrees with source {name!r}"
        return self._dir / name

    COUNTS = {"relevant": "Int64", "target": "Int64", "budget": "Int64"}

    def plan(self, keys: tuple[str, ...] | None = None) -> pd.DataFrame:
        """Price pass 2 per lane without fetching or materializing anything."""
        rows = [self._bill(key) for key in self._selected(keys)]
        return pd.DataFrame(rows).astype(self.COUNTS)

    def build_metadata(self, key: str, *, force: bool = False) -> None:
        """Pass 1: queries + qrels only, the input the corpus target is
        computed from."""
        out = self.lane_dir(key)
        if (out / "qrels.parquet").exists() and not force:
            tqdm.write(f"[{key}] metadata on disk -> {out} (no fetch)")
            return
        source = LANES[key].source
        if isinstance(source, MaterializedDataset):
            source.query_ids = self._wanted(key)
            source.load_metadata()
        source.save_metadata(self._dir)
        tqdm.write(f"[{key}] {self._counts(out)} -> {out}")

    def build(self, key: str, *, force: bool = False) -> None:
        """Pass 2: metadata from disk when pass 1 left it there, then the
        recipe-sized corpus."""
        out = self.lane_dir(key)
        if (out / "corpus.parquet").exists() and not force:
            tqdm.write(f"[{key}] corpus on disk -> {out} (no fetch)")
            return
        lane = LANES[key]
        source = lane.source
        # BeirDataset builds its corpus inside corpus(), so it owns neither
        # hydrate() nor load_metadata() and materialize() stays the base no-op
        if isinstance(source, MaterializedDataset):
            source.query_ids = self._wanted(key)
            if not source.hydrate(self._dir):
                source.load_metadata()
            source.corpus_target = lane.corpus_target
            self._narrow(key, source)
        source.materialize()
        source.save(self._dir)
        tqdm.write(f"[{key}] {self._counts(out)} -> {out}")

    def build_all(
        self,
        keys: tuple[str, ...] | None = None,
        *,
        metadata_only: bool = False,
        force: bool = False,
    ) -> list[str]:
        """Run one pass over the selected lanes, returning the keys that failed
        — an unverified schema or a gated source must not kill the batch."""
        failed: list[str] = []
        for key in self._selected(keys):
            try:
                if metadata_only:
                    self.build_metadata(key, force=force)
                else:
                    self.build(key, force=force)
            except Exception as error:
                failed.append(key)
                tqdm.write(f"[{key}] FAILED {type(error).__name__}: {error}")
        return failed

    def _selected(self, keys: tuple[str, ...] | None) -> list[str]:
        """`--only` order is the build order — expensive lanes go last by
        choice, not by table position."""
        return list(dict.fromkeys(keys)) if keys is not None else list(LANES)

    def _bill(self, key: str) -> dict[str, object]:
        lane = LANES[key]
        out = self.lane_dir(key)
        row: dict[str, object] = {
            "lane": key,
            "relevant": None,
            "target": None,
            "budget": None,
            "sizing": "",
            "on_disk": (out / "corpus.parquet").exists(),
            "sel_in_snap": self._sel_in_snap(key, out),
        }
        if not isinstance(lane.source, MaterializedDataset):
            return {**row, "sizing": "full", "verdict": "no recipe, corpus() is cheap"}
        # a lane that overrides materialize() sizes itself from constructor
        # arguments, so the recipe number would be a target it never uses
        if type(lane.source).materialize is not MaterializedDataset.materialize:
            return {
                **row,
                "sizing": "own",
                "verdict": "own materialize() — construct it with its own limits",
            }
        if not (out / "qrels.parquet").exists():
            return {**row, "verdict": "pass 1 first — target unknown"}
        relevant = self._relevant(
            pd.read_parquet(out / "qrels.parquet", columns=["doc_id", "relevance"])
        )
        target = lane.corpus_target or lane.source.recipe.target(relevant)
        budget = target - relevant
        return {
            **row,
            "relevant": relevant,
            "target": target,
            "budget": budget,
            "sizing": "pinned" if lane.corpus_target else "recipe",
            "verdict": (
                "ready"
                if budget >= 0
                else "NEGATIVE BUDGET — pinned target too low; materialize() raises"
                if lane.corpus_target
                else "over target — build narrows to the composition's queries"
            ),
        }

    def _relevant(self, qrels: pd.DataFrame) -> int:
        """Distinct force-included doc_ids, counted the way `materialize()`
        counts them rather than through `min_relevance`."""
        return qrels.loc[qrels["relevance"] >= 1, "doc_id"].astype(str).nunique()

    def _sel_in_snap(self, key: str, out: Path) -> str:
        """`selected ∩ snapshot queries / selected` — the staleness readout.
        A snapshot serving fewer selected queries than its source can supply
        is how 30.7K judged rows went silently unlabelled (2026-08). The
        column only reports; rebuilding stays an explicit `--metadata
        --force` decision."""
        wanted = self._wanted(key)
        if wanted is None or not (out / "queries.parquet").exists():
            return ""
        snap = set(
            pd.read_parquet(out / "queries.parquet", columns=["query_id"])[
                "query_id"
            ].astype(str)
        )
        return f"{len(wanted & snap)}/{len(wanted)}"

    def _narrow(self, key: str, source: MaterializedDataset) -> None:
        """Cut a lane whose relevant docs overflow the *computed* target down to
        the composition's queries, rather than pinning a target by hand."""
        # a pinned target is already a deliberate decision — overriding it here
        # would silently undo it, so let materialize() fail loud instead
        if source.corpus_target is not None:
            return
        relevant = self._relevant(source.qrels())
        target = source.recipe.target(relevant)
        if target >= relevant:
            return
        ids = source.query_ids
        if not ids:
            raise ValueError(
                f"{key}: {relevant:,} relevant docs exceed the {target:,} target "
                "and the composition selects none of its queries — pin an "
                "explicit corpus_target deliberately."
            )
        source.restrict(ids)
        tqdm.write(
            f"[{key}] {relevant:,} relevant over the {target:,} target -> "
            f"{len(ids):,} composition queries, "
            f"{self._relevant(source.qrels()):,} relevant"
        )

    def _counts(self, out: Path) -> str:
        return " ".join(
            f"{part} {ParquetFile(out / f'{part}.parquet').metadata.num_rows:,}"
            for part in self.PARTS
            if (out / f"{part}.parquet").exists()
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the lane corpora in two passes."
    )
    parser.add_argument(
        "--only",
        nargs="*",
        metavar="LANE",
        help="act on these lane keys only (default: every lane)",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="print the computed corpus target per lane and stop",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="pass 1 only: queries + qrels, never the corpus",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild lanes whose artifact is already on disk",
    )
    args = parser.parse_args()

    only = tuple(args.only) if args.only else None
    unknown = sorted(set(only or ()) - set(LANES))
    if unknown:
        parser.error(f"unknown lanes: {', '.join(unknown)}")

    corpora = LaneCorpora()
    print(corpora.plan(only).to_string(index=False))
    if args.plan:
        return
    failed = corpora.build_all(only, metadata_only=args.metadata, force=args.force)
    if failed:
        print(f"failed ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()

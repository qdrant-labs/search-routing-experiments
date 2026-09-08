"""Corruption census (query-taxonomy SPEC, corruption validation gate): run
the corruption detectors over a sample of each dataset's NATURAL queries and
report how often each fires. On natural text the hit rate is essentially the
detector's false-positive rate — the number that decides whether `clean` is a
real band and whether the heuristics (mojibake, paste, typo) over-fire.

Cached datasets only by default — never triggers a fetch (reads the registry
query cache; uncached datasets are skipped and listed, never silently
dropped). spaCy is not loaded: only the REGEX + WORDFREQ engines are selected.

    poetry run python src/scripts/corruption_census.py                # all cached
    poetry run python src/scripts/corruption_census.py --sample 5000  # bigger sample
    poetry run python src/scripts/corruption_census.py --only orcas miracl-en-dev

Read the result in the notebook:

    import pandas as pd
    pd.read_parquet("src/data/corruption_census.parquet").sort_values("any_span")
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from dataset_registry import DATASETS, RegistryDataset
from query_taxonomy.core import Engine, StatBank
from query_taxonomy.corruption import CORRUPTION_BANKS
from query_taxonomy.features import FeatureExtractor
from query_taxonomy.taxonomy import FeatureGroup

CENSUS_PATH = Path(__file__).resolve().parent.parent / "data" / "corruption_census.parquet"

# split the corruption features by output shape from the banks themselves, so
# this stays a view of the group, never a second hand-kept list.
_BANKS = [cls() for cls in CORRUPTION_BANKS]
SPAN_KINDS: tuple[str, ...] = tuple(
    b.name.value for b in _BANKS if not isinstance(b, StatBank)
)
STAT_KINDS: tuple[str, ...] = tuple(
    b.name.value for b in _BANKS if isinstance(b, StatBank)
)


class CorruptionCensus:
    """Per-dataset corruption hit rates over a cached query sample. Span banks
    report the share of queries with any detection; stat banks report the mean
    of their scalar over the queries that define it."""

    def __init__(
        self,
        datasets: tuple[RegistryDataset, ...] = DATASETS,
        sample_size: int = 2000,
        seed: int = 0,
        extractor: FeatureExtractor | None = None,
    ) -> None:
        # REGEX + WORDFREQ only: corruption needs no spaCy, so never load it.
        self._extractor = extractor or FeatureExtractor(
            engines=(Engine.REGEX, Engine.WORDFREQ)
        )
        self._datasets = datasets
        self._sample_size = sample_size
        self._seed = seed

    def _row(self, dataset: RegistryDataset) -> dict[str, float]:
        queries = dataset.sample_queries(self._sample_size, seed=self._seed)
        span_hits = dict.fromkeys(SPAN_KINDS, 0)
        stat_sum = dict.fromkeys(STAT_KINDS, 0.0)
        stat_seen = dict.fromkeys(STAT_KINDS, 0)
        any_span = 0
        for query in queries:
            features = self._extractor.resolve(
                query.text, groups=[FeatureGroup.CORRUPTION]
            )
            spans = features.spans.get(FeatureGroup.CORRUPTION, {})
            stats = features.stats.get(FeatureGroup.CORRUPTION, {})
            for kind in SPAN_KINDS:
                if spans.get(kind):
                    span_hits[kind] += 1
            if spans:
                any_span += 1
            for kind in STAT_KINDS:
                for stat in stats.get(kind, []):
                    stat_sum[kind] += stat.value
                    stat_seen[kind] += 1
        n = len(queries)
        row: dict[str, float] = {"n_sampled": float(n)}
        row.update({kind: span_hits[kind] / n if n else 0.0 for kind in SPAN_KINDS})
        row.update(
            {
                kind: stat_sum[kind] / stat_seen[kind] if stat_seen[kind] else 0.0
                for kind in STAT_KINDS
            }
        )
        row["any_span"] = any_span / n if n else 0.0
        return row

    def census(self) -> pd.DataFrame:
        """One row per cached dataset; uncached datasets are skipped and their
        names returned to the caller via the frame's `.attrs['skipped']`."""
        rows: dict[str, dict[str, float]] = {}
        skipped: list[str] = []
        for dataset in tqdm(self._datasets, desc="census", unit="dataset"):
            if dataset.queries_cached is None:
                skipped.append(dataset.name)
                continue
            rows[dataset.name] = self._row(dataset)
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.attrs["skipped"] = skipped
        return frame.sort_values("any_span", ascending=False) if len(frame) else frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=2000, help="queries per dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=None, help="dataset names")
    parser.add_argument("--out", type=Path, default=CENSUS_PATH)
    args = parser.parse_args()

    datasets = DATASETS
    if args.only:
        wanted = set(args.only)
        datasets = tuple(d for d in DATASETS if d.name in wanted)

    census = CorruptionCensus(datasets, sample_size=args.sample, seed=args.seed)
    frame = census.census()
    if frame.attrs.get("skipped"):
        print(f"skipped {len(frame.attrs['skipped'])} uncached: "
              f"{', '.join(frame.attrs['skipped'])}")
    if not len(frame):
        print("no cached datasets to census")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out)
    pd.set_option("display.width", 200, "display.max_columns", None)
    print(frame.round(4))
    print(f"\n{len(frame)} datasets -> {args.out}")


if __name__ == "__main__":
    main()

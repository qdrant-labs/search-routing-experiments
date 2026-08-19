"""RARE_ZIPF_MAX calibration (query-taxonomy): pool the wordfreq Zipf value of
every known query term across the cached lanes and report the distribution, so
the rarity cutoff is read off real data instead of hand-set. Same tokenization
and known-only filter as RarityBank, so the reported distribution IS the
population the cutoff is compared against.

Cached datasets only (never triggers a fetch); uncached are skipped and listed.

    poetry run python src/scripts/zipf_calibration.py
    poetry run python src/scripts/zipf_calibration.py --sample 1000
"""

import argparse
from statistics import quantiles

from tqdm.auto import tqdm
from wordfreq import tokenize, zipf_frequency

from dataset_registry import DATASETS

_LANG = "en"
_PERCENTILES = (1, 5, 10, 25, 50, 75, 90)
_CANDIDATES = (2.0, 2.5, 3.0, 3.5)  # rare_share these cutoffs would produce


def _known_zipfs(text: str) -> list[float]:
    values = (zipf_frequency(tok, _LANG) for tok in tokenize(text, _LANG))
    return [v for v in values if v > 0.0]


def collect(sample: int, seed: int) -> list[float]:
    pool: list[float] = []
    skipped: list[str] = []
    for dataset in tqdm(DATASETS, desc="zipf", unit="dataset"):
        if dataset.queries_cached is None:
            skipped.append(dataset.name)
            continue
        for query in dataset.sample_queries(sample, seed=seed):
            pool.extend(_known_zipfs(query.text))
    if skipped:
        print(f"skipped {len(skipped)} uncached: {', '.join(sorted(skipped))}\n")
    return pool


def report(pool: list[float]) -> None:
    pool.sort()
    n = len(pool)
    print(f"{n:,} known query terms pooled\n")
    cuts = quantiles(pool, n=100)  # cuts[i] = (i+1)-th percentile
    print("percentile  zipf")
    for p in _PERCENTILES:
        print(f"  {p:>3}       {cuts[p - 1]:.2f}")
    lo, hi = pool[0], pool[-1]
    width = (hi - lo) / 20 or 1.0
    print("\nhistogram (zipf -> share of terms)")
    for i in range(20):
        edge = lo + i * width
        share = sum(edge <= v < edge + width for v in pool) / n
        print(f"  {edge:4.1f} {'#' * round(share * 100)}")
    print("\nrare_share a cutoff would yield on this pool:")
    for c in _CANDIDATES:
        print(f"  RARE_ZIPF_MAX={c}: {sum(v < c for v in pool) / n:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=500, help="queries per dataset")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report(collect(args.sample, args.seed))


if __name__ == "__main__":
    main()

"""Engine performance stress test — throughput / latency / peak RSS per
extractor engine (REGEX / SPACY / GLINER) at scaling query counts.

Public surface: `benchmark()` and `BenchmarkResult`. Everything else is
implementation.

Usage — notebook or library:

    from benchmark_engines import benchmark
    results = benchmark(sizes=[100])                       # sanity check
    results = benchmark(sizes=[100], engines=[Engine.REGEX])  # single engine
    results = benchmark(csv_path=None, verbose=False)      # programmatic

Usage — CLI (defaults: all engines, 100/1K/10K/100K, writes CSV):

    poetry run python src/benchmark_engines.py

Rows append to `src/data/benchmarks/engines.csv` after every (engine, N) so
an interrupted long GLiNER run still leaves the smaller sizes on disk.
"""
from __future__ import annotations

import csv
import gc
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import ir_datasets
from tqdm.auto import tqdm

from query_taxonomy.core import Engine
from query_taxonomy.features import FeatureExtractor

DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "data" / "benchmarks" / "engines.csv"
DEFAULT_QUERY_SOURCE = "msmarco-passage/train"
DEFAULT_SIZES: tuple[int, ...] = (100, 1_000, 10_000, 100_000)
DEFAULT_ENGINES: tuple[Engine, ...] = (Engine.REGEX, Engine.SPACY, Engine.GLINER)
DEFAULT_WARMUP = 100


@dataclass(frozen=True)
class BenchmarkResult:
    """One (engine, N) measurement. Field order defines the CSV schema."""

    run_ts: str
    engine: str
    n: int
    wall_s: float
    throughput_qps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    peak_rss_mb: float
    rss_delta_mb: float


def benchmark(
    *,
    sizes: Sequence[int] = DEFAULT_SIZES,
    engines: Sequence[Engine] = DEFAULT_ENGINES,
    warmup: int = DEFAULT_WARMUP,
    query_source: str = DEFAULT_QUERY_SOURCE,
    csv_path: Path | None = DEFAULT_CSV_PATH,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """Run each `engine` at each `n` in `sizes`; return the results.

    Query pool comes from `query_source` (ir_datasets id); the tail
    `warmup` queries load lazy models and are never reused for timing.
    Timing batches are non-overlapping slices so cache spillover between
    sizes is zero.

    Side effects (opt-out by parameter):
        csv_path=None  -> no CSV write
        verbose=False  -> no progress bars, no printed summary
    """
    pool = _load_queries(query_source, sum(sizes) + warmup, verbose)
    warmup_texts, timing_pool = pool[-warmup:], pool[:-warmup]
    run_ts = datetime.now().isoformat(timespec="seconds")

    results: list[BenchmarkResult] = []
    for engine in engines:
        extractor = FeatureExtractor(engines=[engine])
        for text in _iter(warmup_texts, f"{engine.value} warmup", verbose):
            extractor.resolve(text)

        offset = 0
        for size in sizes:
            batch = timing_pool[offset : offset + size]
            offset += size
            metrics = _time_batch(extractor, batch, f"{engine.value} n={size:>6,}", verbose)
            result = BenchmarkResult(run_ts=run_ts, engine=engine.value, n=size, **metrics)
            results.append(result)
            if csv_path is not None:
                _append_csv(csv_path, result)
            if verbose:
                print(_summary_line(result))
    return results


def _load_queries(source: str, n: int, verbose: bool) -> list[str]:
    dataset = ir_datasets.load(source)
    texts: list[str] = []
    for query in _iter(dataset.queries_iter(), f"load {n:,} queries", verbose, total=n):
        texts.append(query.text)
        if len(texts) >= n:
            break
    if len(texts) < n:
        raise RuntimeError(
            f"only {len(texts):,} queries available from {source}, need {n:,}"
        )
    return texts


def _time_batch(
    extractor: FeatureExtractor, batch: list[str], label: str, verbose: bool
) -> dict[Any, Any]:
    gc.collect()
    baseline_rss = _peak_rss_mb()

    latencies_ms: list[float] = []
    wall_start = time.perf_counter()
    for text in _iter(batch, label, verbose):
        q_start = time.perf_counter()
        extractor.resolve(text)
        latencies_ms.append((time.perf_counter() - q_start) * 1000)
    wall = time.perf_counter() - wall_start

    peak = _peak_rss_mb()
    latencies_ms.sort()
    return {
        "wall_s": round(wall, 3),
        "throughput_qps": round(len(batch) / wall, 2),
        "p50_ms": round(_percentile(latencies_ms, 50), 3),
        "p95_ms": round(_percentile(latencies_ms, 95), 3),
        "p99_ms": round(_percentile(latencies_ms, 99), 3),
        "peak_rss_mb": round(peak, 1),
        "rss_delta_mb": round(peak - baseline_rss, 1),
    }


def _iter(iterable, desc: str, verbose: bool, total: int | None = None):
    return tqdm(iterable, desc=desc, total=total) if verbose else iterable


def _percentile(sorted_xs: list[float], pct: int) -> float:
    if not sorted_xs:
        return 0.0
    idx = min(len(sorted_xs) - 1, int(pct / 100 * len(sorted_xs)))
    return sorted_xs[idx]


def _peak_rss_mb() -> float:
    """High-water RSS since process start. macOS reports bytes, Linux kB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def _append_csv(path: Path, result: BenchmarkResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(BenchmarkResult)])
        if not exists:
            writer.writeheader()
        writer.writerow(asdict(result))


def _summary_line(result: BenchmarkResult) -> str:
    return (
        f"  {result.engine:>12}  n={result.n:>7,}  "
        f"{result.throughput_qps:>10,.1f} qps  "
        f"p50={result.p50_ms:>7.2f} ms  "
        f"p95={result.p95_ms:>7.2f} ms  "
        f"p99={result.p99_ms:>7.2f} ms  "
        f"rss_delta={result.rss_delta_mb:>+7.1f} MB"
    )


if __name__ == "__main__":
    benchmark()

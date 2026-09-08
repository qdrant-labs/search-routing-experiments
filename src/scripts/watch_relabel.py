"""Live progress for an os_distill relabel run, from outside the busy kernel:
polls the checkpoint parquet and prints rows banked, rate, and ETA.

    poetry run python src/scripts/watch_relabel.py [--target 36669] [--every 30]
"""

import argparse
import time
from pathlib import Path

import pyarrow.parquet as pq

LABELS = Path(__file__).resolve().parent.parent / "data" / "legb_pilot" / \
    "os_distill_relabel" / "labels.parquet"


def rows() -> int:
    return pq.ParquetFile(LABELS).metadata.num_rows if LABELS.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=36669, help="total rows this run should reach")
    parser.add_argument("--every", type=int, default=30, help="seconds between polls")
    args = parser.parse_args()

    start_rows, t0 = rows(), time.monotonic()
    print(f"watching {LABELS}\nstart: {start_rows:,} rows banked, target {args.target:,} "
          f"(ctrl-c to stop)")
    while True:
        time.sleep(args.every)
        n, dt = rows(), time.monotonic() - t0
        rate = (n - start_rows) / dt  # rows/s since the watcher started
        eta = (args.target - n) / rate / 3600 if rate > 0 else float("inf")
        lanes = pq.read_table(LABELS, columns=["dataset"])["dataset"].to_pylist() if n else []
        last = lanes[-1] if lanes else "-"
        print(f"{time.strftime('%H:%M:%S')}  {n:,}/{args.target:,} rows "
              f"({n / args.target:5.1%})  {rate:5.1f} rows/s  eta {eta:4.1f}h  lane: {last}")


if __name__ == "__main__":
    main()

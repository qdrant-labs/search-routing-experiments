"""Run Rung A selection, labelling, pool refresh, and v4 composition. Every
artifact this writes lives under src/data/v4; src/data/v3 is read-only supply."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PICKS = REPO_ROOT / "src/data/v4/rung_a_picks.parquet"


def _log(t0: float, message: str) -> None:
    print(
        f"[+{time.perf_counter() - t0:6.1f}s] {message}",
        file=sys.stderr,
        flush=True,
    )


def _run(command: Sequence[str]) -> int:
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def _qdrant_running() -> bool:
    try:
        result = subprocess.run(
            ["podman", "compose", "ps"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0 and "qdrant" in result.stdout.lower()


def _stage(
    number: int,
    name: str,
    action: Callable[[], int],
    t0: float,
) -> bool:
    _log(t0, f"stage {number} {name} start")
    try:
        code = action()
    except BaseException as error:
        print(f"stage {number} {name}: {error}", file=sys.stderr)
        code = 1
    _log(t0, f"stage {number} {name} end exit code={code}")
    if code == 0:
        return True
    print(
        f"STAGE {number} FAILED at [+{time.perf_counter() - t0:.1f}s]",
        file=sys.stderr,
        flush=True,
    )
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=200_000)
    parser.add_argument("--rung-a-target", type=int, default=100_000)
    parser.add_argument("--skip-picks", action="store_true")
    parser.add_argument("--skip-materialize", action="store_true")
    parser.add_argument("--skip-labelling", action="store_true")
    args = parser.parse_args()
    if args.target < 1 or args.rung_a_target < 1:
        parser.error("--target and --rung-a-target must be positive")
    if not args.skip_labelling and not _qdrant_running():
        print(
            "error: Qdrant is not running; start it with `docker compose up -d` "
            "before overnight labelling",
            file=sys.stderr,
        )
        return 1

    t0 = time.perf_counter()
    # picks always overwrite, for the same reason compose does: an unattended
    # run that dies on stage 1 because last night's file is still there has
    # burned the night.
    picks_command = [
        "poetry",
        "run",
        "python",
        "src/scripts/rung_a_picks.py",
        "--target",
        str(args.rung_a_target),
        "--out",
        str(PICKS),
        "--force",
    ]
    picks_action = (
        (lambda: (_log(t0, f"reusing {PICKS}"), 0)[1])
        if args.skip_picks and PICKS.exists()
        else lambda: _run(picks_command)
    )
    if not _stage(1, "Rung A picks", picks_action, t0):
        return 1

    materialize_action = (
        (lambda: (_log(t0, "reusing existing snapshots"), 0)[1])
        if args.skip_materialize
        else lambda: _run([
            "poetry", "run", "python", "src/scripts/materialize_rung_a.py",
            "--picks", str(PICKS),
        ])
    )
    if not _stage(2, "materialize picks", materialize_action, t0):
        return 1

    label_action = (
        (lambda: (_log(t0, "reusing existing labels"), 0)[1])
        if args.skip_labelling
        else lambda: _run([
            "poetry", "run", "python", "src/scripts/label_routes_v3.py", "--v4",
        ])
    )
    if not _stage(3, "labelling", label_action, t0):
        return 1

    if not _stage(
        4,
        "pool refresh",
        lambda: (_log(t0, "compose will instantiate a fresh LabelledPool"), 0)[1],
        t0,
    ):
        return 1

    # compose always overwrites — the whole point of an overnight run is a
    # fresh artifact against the freshly-labelled pool, and refusing to
    # overwrite defeats the point.
    compose_command = [
        "poetry",
        "run",
        "python",
        "src/scripts/compose_v4.py",
        "--target",
        str(args.target),
        "--force",
    ]
    if not _stage(5, "compose", lambda: _run(compose_command), t0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

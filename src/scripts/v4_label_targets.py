"""v4 label targets — split the composed targets into already-labelled (reuse,
never relabel) and needs-labelling, by a plain set difference against existing
labels.

Composition decided WHICH queries. This never does: existing labels are read
once, read-only, after composition — a cache that saves work, not a policy that
picks rows. No SelectionOrder, no yield weighting, no order sheet.

    poetry run python src/scripts/v4_label_targets.py [--out ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGETS = REPO_ROOT / "src" / "data" / "v4" / "composition_targets_v4.parquet"


def _known_label_keys() -> set[tuple[str, str]]:
    """(dataset, query_id) of every already-labelled query, all rungs."""
    from composition.pool_v3 import LabelledPool

    labels = LabelledPool(native_only=False).labels()[["dataset", "query_id"]]
    labels = labels.astype({"dataset": str, "query_id": str})
    return set(map(tuple, labels.itertuples(index=False, name=None)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    targets_path = args.targets if args.targets.is_absolute() else REPO_ROOT / args.targets
    if not targets_path.exists():
        print(f"error: {targets_path} not found; run compose_v4.py first")
        return 1
    targets = pd.read_parquet(targets_path).astype({"dataset": str, "query_id": str})

    known = _known_label_keys()
    keys = list(map(tuple, targets[["dataset", "query_id"]].itertuples(index=False, name=None)))
    needs_mask = [key not in known for key in keys]
    needs = targets.loc[needs_mask].reset_index(drop=True)

    already = len(targets) - len(needs)
    print(f"targets={len(targets):,} already_labelled={already:,} needs_labelling={len(needs):,}")

    out = args.out or targets_path.with_name("needs_label_v4.parquet")
    out = out if out.is_absolute() else REPO_ROOT / out
    needs[["dataset", "query_id"]].to_parquet(out, index=False)
    print(f"wrote {len(needs):,} needs-label keys -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

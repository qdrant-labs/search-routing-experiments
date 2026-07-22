"""Build the SPEC d32/d33 target composition (50K training selection).

    poetry run python scripts/compose_target.py           # skip if on disk
    poetry run python scripts/compose_target.py --force   # rebuild
"""

import argparse

from composition import TargetComposition


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the d32/d33 target composition from the catalog."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="rebuild even when the selection already exists on disk",
    )
    args = parser.parse_args()

    composition = TargetComposition()
    selection = composition.build(force=args.force)
    lanes = selection["label_lane"].value_counts()
    print(
        f"{len(selection):,} rows -> {composition.selection_path}\n"
        f"qrels lane {lanes.get('qrels', 0):,} | "
        f"deferred {lanes.get('deferred', 0):,}\n"
        f"order sheet -> {composition.order_sheet_path}\n"
        f"summary -> {composition.summary_path}"
    )


if __name__ == "__main__":
    main()

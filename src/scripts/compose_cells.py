"""Build the SPEC d48/d50 cell-quota composition from the catalog.

    poetry run python src/scripts/compose_cells.py           # skip if on disk
    poetry run python src/scripts/compose_cells.py --force    # rebuild
"""

import argparse

from composition.cellfill import CANDIDATE, CONTROL, REUSED, CellFill


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the archetype-cell selection, order sheet and report."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="rebuild even when the selection already exists on disk",
    )
    args = parser.parse_args()

    fill = CellFill()
    selection = fill.build(force=args.force)
    stages = selection["stage"].value_counts()
    queries = selection.groupby(["dataset", "query_id"]).ngroups
    print(
        f"{len(selection):,} rows / {queries:,} distinct queries "
        f"-> {fill.selection_path}\n"
        f"reused {stages.get(REUSED, 0):,} | queued {stages.get(CANDIDATE, 0):,} "
        f"| control {stages.get(CONTROL, 0):,}\n"
        f"order sheet -> {fill.order_sheet_path}\n"
        f"report -> {fill.report_path}\n"
        f"summary -> {fill.summary_path}"
    )


if __name__ == "__main__":
    main()

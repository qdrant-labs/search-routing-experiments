"""Per-cell dense-vs-sparse DIVERGENCE audit (SPEC d50e pre-label screen):
for each archetype cell, top-10 Jaccard of the dense_only vs sparse_only
retrieved doc lists — low overlap means the routes retrieve different docs,
so a routing signal is present. Grouped by cell, off stored labels, rather
than run live on Qdrant as the retired embeddings audit did.

    poetry run python src/scripts/cell_divergence.py

Reads doc lists from the persisted oracle rankings
(`data/route_labels/*_oracle/rows.parquet`, `GoldenRoutingDataset.route_rankings`).
Only datasets with such a directory contribute; every other cell row is
reported as uncovered, never guessed.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd

from hybrid_search_rrf_dataset.paths import LanePaths
from composition.cellfill import DENSE, SPARSE
from composition.compose import DEFAULT_OUT_DIR

CELL_SELECTION = DEFAULT_OUT_DIR / "cell_selection.parquet"
PATHS = LanePaths(data_dir=DEFAULT_OUT_DIR.parent)
ROUTE_LABELS = PATHS.oracle_dir()
TOP_K = 10


def jaccard(a: list[str], b: list[str]) -> float:
    """Overlap of two id sets, |a ∩ b| / |a ∪ b|; empty union scores 0.0."""
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


class CellDivergence(NamedTuple):
    """One cell's divergence readout: rows in the cell, how many had rankings
    to score, and the top-10 dense-vs-sparse Jaccard over the scored ones."""

    cell: str
    rows: int
    scored: int
    coverage: float
    mean_jaccard: float
    median_jaccard: float


def load_rankings(root: Path = ROUTE_LABELS) -> pd.DataFrame:
    """Concatenate every `<dataset>_oracle` directory's per-route doc lists,
    keyed by the composition dataset name the directory encodes."""
    frames = []
    for dataset in PATHS.oracle_lanes(under=root):
        rows = PATHS.oracle_rows(dataset, under=root)
        df = pd.read_parquet(rows, columns=["query_id", "route_rankings"])
        df = df.assign(dataset=dataset, query_id=df["query_id"].astype(str))
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["dataset", "query_id", "route_rankings"])
    return pd.concat(frames, ignore_index=True)


def _row_jaccard(rankings: dict[str, list[str]] | None) -> float | None:
    """Top-k dense-vs-sparse Jaccard for one row, None when it has no rankings
    or is missing a route."""
    if not isinstance(rankings, dict) or DENSE not in rankings or SPARSE not in rankings:
        return None
    return jaccard(list(rankings[DENSE])[:TOP_K], list(rankings[SPARSE])[:TOP_K])


class CellDivergenceAudit:
    """Joins cell membership to persisted oracle rankings and scores each
    cell's dense-vs-sparse disagreement."""

    def __init__(
        self,
        selection: pd.DataFrame | None = None,
        rankings: pd.DataFrame | None = None,
    ) -> None:
        self.selection = (
            selection if selection is not None else pd.read_parquet(CELL_SELECTION)
        )
        self.rankings = rankings if rankings is not None else load_rankings()

    def _joined(self) -> pd.DataFrame:
        """Named-cell rows with a per-row Jaccard column (NaN where no rankings)."""
        cells = self.selection[self.selection["cell"].astype(str).str.len() > 0].copy()
        cells["query_id"] = cells["query_id"].astype(str)
        joined = cells.merge(self.rankings, on=["dataset", "query_id"], how="left")
        joined["jaccard"] = joined["route_rankings"].map(_row_jaccard)
        return joined

    def readout(self) -> list[CellDivergence]:
        """Per-cell divergence, most divergent (lowest Jaccard) first; cells
        with no scored rows sort last."""
        joined = self._joined()
        out = []
        for cell, group in joined.groupby("cell"):
            scored = group["jaccard"].dropna()
            out.append(
                CellDivergence(
                    cell=str(cell),
                    rows=len(group),
                    scored=len(scored),
                    coverage=round(len(scored) / len(group), 3),
                    mean_jaccard=round(float(scored.mean()), 3) if len(scored) else float("nan"),
                    median_jaccard=round(float(scored.median()), 3) if len(scored) else float("nan"),
                )
            )
        return sorted(out, key=lambda c: (c.scored == 0, c.mean_jaccard))


def _demo() -> None:
    """Self-check the Jaccard and per-row scoring on hand cases."""
    assert jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert jaccard(["a", "b"], ["c", "d"]) == 0.0
    assert jaccard(["a", "b"], ["b", "c"]) == 1 / 3  # {b} over {a,b,c}
    assert jaccard([], []) == 0.0
    assert _row_jaccard({DENSE: ["a"], SPARSE: ["a"]}) == 1.0
    assert _row_jaccard({DENSE: ["a"]}) is None
    assert _row_jaccard(None) is None


def main() -> None:
    _demo()
    audit = CellDivergenceAudit()
    table = pd.DataFrame(c._asdict() for c in audit.readout())
    scored_rows = int(table["scored"].sum())
    total_rows = int(table["rows"].sum())
    print(
        f"cells: {len(table)}  cell rows: {total_rows}  "
        f"rows with rankings: {scored_rows} "
        f"({scored_rows / total_rows:.1%})\n"
    )
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()

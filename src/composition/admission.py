"""Mini-fill (d42i, d43a): admit generated-pool rows into the frozen base.

Start-from-base: the existing selection rows and their labels are never
touched — floors open at the order sheet's `missing` (the base credit is
already subtracted there), and only pool rows compete. The natural share
(recipe.min_natural_share) bounds the admission budget; the d29
anti-monoculture cap applies per home_lane over the REMAINING gap
(cap_frac x missing — the d43-review "cap within the top-up" ruling).

The one accounting authority stays the fill: credits come from
re-measuring the child texts with the same extractor configuration that
built the catalog, floor derivation is the same derivers, and the sheet
is re-emitted from the mini-fill's own ledger.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from query_taxonomy.features import FeatureExtractor

from composition.fill import QRELS, FillResult, WeakestFirstFill
from composition.floors import (
    FloorSpec,
    SpanFloorDeriver,
    StatFloorDeriver,
    span_mask,
)
from composition.mini_catalog import mini_catalog
from composition.recipe import Recipe


class SelectionArtifacts(NamedTuple):
    """Where a fill wrote its selection, order sheet and summary."""

    selection_path: Path
    order_sheet_path: Path
    summary_path: Path

    @classmethod
    def legacy(cls) -> "SelectionArtifacts":
        """The slice fill's artifacts — the only ones the pool was built against."""
        # local: keeps the slice fill out of the import graph of everything
        # that only needs paths
        from composition.compose import TargetComposition

        old = TargetComposition()
        return cls(old.selection_path, old.order_sheet_path, old.summary_path)


class MiniFill:
    """Admission from the generated pool into a selection artifact.

    Takes the three artifact paths, so it serves whichever fill wrote them."""

    def __init__(
        self,
        artifacts: SelectionArtifacts | None = None,
        recipe: Recipe | None = None,
        extractor: FeatureExtractor | None = None,
    ) -> None:
        self._artifacts = artifacts or SelectionArtifacts.legacy()
        self._recipe = recipe or Recipe()
        # engines=None: full parity with the catalog build — children's
        # floors and credits must mean the same thing as the base's
        self._extractor = extractor or FeatureExtractor(engines=None)

    def admit(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Admit as many pool rows as the hungry floors, the natural-share
        ceiling, and the per-lane cap allow. Returns the admitted rows in
        selection schema; artifacts (selection, order sheet, summary) are
        rewritten in place."""
        selection = pd.read_parquet(self._artifacts.selection_path)
        sheet = pd.read_parquet(self._artifacts.order_sheet_path)
        if "generated_from" not in selection.columns:
            selection["generated_from"] = pd.NA

        if "credit_gate" in pool.columns:
            gated = pool["credit_gate"].fillna("none") != "none"
            if gated.any():
                print(
                    f"mini-fill: skipping {int(gated.sum())} feature-stock "
                    "rows (gated operators, d42h)"
                )
            pool = pool[~gated]

        hungry_keys = set(sheet.loc[sheet["missing"] > 0, "floor"])
        fresh = pool[
            ~pool["query_id"].isin(set(selection["query_id"]))
            & pool["floor"].isin(hungry_keys)
        ].reset_index(drop=True)
        if fresh.empty:
            print("mini-fill: nothing new to admit")
            return selection.iloc[0:0]

        budget = self._headroom(selection)
        if budget <= 0:
            raise ValueError(
                "mini-fill: natural-share ceiling reached "
                f"(min_natural_share={self._recipe.min_natural_share}) — "
                "no admission budget left."
            )

        mini = self._mini_catalog(fresh)
        credits = self._derive_credits(mini)
        floors = self._residual_floors(sheet, credits)
        if not floors:
            print("mini-fill: no hungry floor is served by the pool")
            return selection.iloc[0:0]

        rng = np.random.default_rng(self._recipe.seed)
        result = WeakestFirstFill(mini, floors, self._recipe, rng).run(
            min(budget, len(fresh)), top_up=False
        )

        admitted = self._selection_rows(
            fresh.iloc[result.picked], sheet, credits, result.picked
        )[selection.columns]
        updated = pd.concat([selection, admitted], ignore_index=True)
        new_sheet = self._updated_sheet(sheet, result)
        self._assert_invariants(updated, new_sheet)

        updated.to_parquet(self._artifacts.selection_path, index=False)
        new_sheet.to_parquet(self._artifacts.order_sheet_path, index=False)
        self._append_summary(fresh.iloc[result.picked], sheet, new_sheet)
        print(
            f"mini-fill: admitted {len(admitted):,} of {len(fresh):,} fresh "
            f"pool rows (budget {budget:,}) | selection "
            f"{len(selection):,} -> {len(updated):,} | hungry floors "
            f"{len(sheet[sheet['missing'] > 0])} -> "
            f"{len(new_sheet[new_sheet['missing'] > 0])}"
        )
        return admitted

    def _headroom(self, selection: pd.DataFrame) -> int:
        """How many augmented rows the natural share still allows."""
        minimum = self._recipe.min_natural_share
        natural = int(selection["generated_from"].isna().sum())
        ceiling = int(natural * (1.0 - minimum) / minimum)
        return ceiling - (len(selection) - natural)

    def _mini_catalog(self, fresh: pd.DataFrame) -> pd.DataFrame:
        return mini_catalog(fresh, self._extractor)

    def _derive_credits(self, mini: pd.DataFrame) -> dict[str, pd.Series]:
        """Every floor's per-row credit over the mini catalog. Span floors
        derive over all rows; stat floors only over the zero-span rows
        (d33b — stat bands live on the zero-span pool), reindexed to the
        full frame."""
        credits: dict[str, pd.Series] = {}
        for spec in SpanFloorDeriver(self._recipe).derive(mini):
            credits[spec.key] = spec.credit
        zero_span = mini[~span_mask(mini)]
        if not zero_span.empty:
            for spec in StatFloorDeriver(self._recipe).derive(zero_span):
                credits[spec.key] = spec.credit.reindex(mini.index, fill_value=0.0)
        return credits

    def _residual_floors(
        self, sheet: pd.DataFrame, credits: dict[str, pd.Series]
    ) -> list[FloorSpec]:
        """Hungry sheet lines the pool can actually feed, opened at
        `missing`. cap_waived stays False: the d29 cap binds per home_lane
        at cap_frac x missing — the cap-within-the-top-up ruling."""
        floors = []
        for line in sheet[sheet["missing"] > 0].itertuples(index=False):
            credit = credits.get(line.floor)
            if credit is None or not (credit > 0).any():
                continue
            floors.append(FloorSpec(
                key=line.floor,
                amount=float(line.missing),
                credit=credit,
                cap_waived=False,
            ))
        return floors

    def _selection_rows(
        self,
        picked: pd.DataFrame,
        sheet: pd.DataFrame,
        credits: dict[str, pd.Series],
        positions: list[int],
    ) -> pd.DataFrame:
        """Admitted rows in selection schema. slice = the birth floor's
        sheet slice; floors = full re-measured membership (parity with the
        base rows' semantics)."""
        floor_slice = dict(zip(sheet["floor"], sheet["slice"]))
        keys = sorted(credits)
        memberships = [
            [key for key in keys if credits[key].iloc[position] > 0]
            for position in positions
        ]
        return pd.DataFrame({
            "dataset": picked["home_lane"].to_numpy(),
            "query_id": picked["query_id"].to_numpy(),
            "checkable": True,
            "slice": picked["floor"].map(floor_slice).to_numpy(),
            "label_lane": QRELS,
            "floors": memberships,
            "query": picked["query"].to_numpy(),
            "generated_from": picked["generated_from"].to_numpy(),
        })

    def _updated_sheet(
        self, sheet: pd.DataFrame, result: FillResult
    ) -> pd.DataFrame:
        """Credit the gains, recompute missing, drop met lines (the sheet
        lists shortfalls only). A mini-fill 'budget' stop means the
        natural-share ceiling bound — reported as such."""
        gained = dict(zip(result.ledger.keys, result.ledger.credit))
        reasons = {line.key: line.reason for line in result.shortfalls}
        out = sheet.copy()
        out["credit"] = [
            row.credit + gained.get(row.floor, 0.0)
            for row in out.itertuples(index=False)
        ]
        out["missing"] = out["amount"] - out["credit"]
        out["reason"] = [
            (
                "natural_share"
                if reasons.get(row.floor) == "budget"
                else reasons.get(row.floor, row.reason)
            )
            for row in out.itertuples(index=False)
        ]
        return out[out["missing"] > 1e-9].reset_index(drop=True)

    def _assert_invariants(
        self, updated: pd.DataFrame, sheet: pd.DataFrame
    ) -> None:
        duplicated = updated.duplicated(["dataset", "query_id"])
        assert not duplicated.any(), "mini-fill produced duplicate rows"
        natural = updated["generated_from"].isna().mean()
        assert natural >= self._recipe.min_natural_share - 1e-9, (
            f"natural share {natural:.3f} fell below the recipe minimum"
        )
        assert (sheet["missing"] > 0).all(), "met floor left on the sheet"
        assert updated["slice"].notna().all(), "admitted row without a slice"

    def _append_summary(
        self,
        picked: pd.DataFrame,
        old_sheet: pd.DataFrame,
        new_sheet: pd.DataFrame,
    ) -> None:
        """Append the admission record — picked rows still carry their
        birth floor, so the block reads per floor."""
        lines = [
            "\n## Mini-fill admission\n",
            f"admitted {len(picked):,} rows | hungry floors "
            f"{int((old_sheet['missing'] > 0).sum())} -> "
            f"{int((new_sheet['missing'] > 0).sum())}\n",
        ]
        if not picked.empty:
            lines.append(
                f"```\n{picked['floor'].value_counts().to_string()}\n```\n"
            )
        path = self._artifacts.summary_path
        path.write_text(path.read_text() + "\n".join(lines))

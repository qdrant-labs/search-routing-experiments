"""The augmentation loop (d42a): demand from the order sheet, dispatch by
floor, produce -> verify -> pool. Selection is deterministic; the LLM only
weaves. Running a batch is a user-initiated LLM spend."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from augmentation.config import AugmentationConfig
from augmentation.core import AugmentedCandidate, CreditGate, Operator
from augmentation.dispatch import (
    cell_targets,
    dispatch,
    rung_report,
    viable_rungs,
)
from augmentation.engine import Augmenter
from augmentation.operators import default_operators
from augmentation.parents import ParentPool
from augmentation.pool import GeneratedPool
from augmentation.qrels import AugmentationQrels
from composition.cells import CELLS_BY_NAME
from taxonomy_generators.verify import Targets


def _pair_line(sign: str, parent: dict, text: str, note: str) -> str:
    tag = "after: " if sign == "+" else "tried: "
    return (
        f"{sign} {parent['query_id']}{note}\n"
        f"    before: {str(parent['query'])[:80]!r}\n"
        f"    {tag} {text[:80]!r}"
    )


class AugmentationLoop:
    """Wires order sheet -> operator -> Augmenter -> GeneratedPool."""

    def __init__(
        self,
        selection: pd.DataFrame,
        *,
        config: AugmentationConfig | None = None,
        engine: Augmenter | None = None,
        operators: tuple[Operator, ...] | None = None,
        sheet_path: Path | str | None = None,
        pool: GeneratedPool | None = None,
        qrels: AugmentationQrels | None = None,
        parents: ParentPool | None = None,
    ) -> None:
        self.selection = selection
        self.parents = parents
        """Cell demands draw parents from here (d51g); floor demands still
        draw from `selection`."""
        self.config = config or AugmentationConfig()
        self.engine = engine or Augmenter(
            self.config.engine, seed=self.config.seed
        )
        self.operators = (
            operators if operators is not None else default_operators(self.config)
        )
        self.pool = pool or GeneratedPool(self.config.paths)
        self.qrels = qrels or AugmentationQrels(self.config.paths)
        self._sheet_path = Path(sheet_path or self.config.paths.order_sheet)

    def order_sheet(self) -> pd.DataFrame:
        sheet = pd.read_parquet(self._sheet_path)
        hungry = sheet[sheet["missing"] > 0]
        return hungry.sort_values("missing", ascending=False, ignore_index=True)

    def operator_for(self, floor: str) -> Operator | None:
        return next((op for op in self.operators if op.serves(floor)), None)

    def demand(self, floor: str) -> tuple[tuple[Operator, pd.DataFrame], ...]:
        """Who serves this demand and on which parents. A cell derives its
        operators from the bands each parent fails and may have several, one
        per parent slice, richest first (d51c/e); a floor has exactly one."""
        cell = CELLS_BY_NAME.get(floor)
        if cell is None:
            operator = self.operator_for(floor)
            if operator is None:
                raise ValueError(f"No registered operator serves {floor!r}.")
            return ((operator, operator.eligible(self.selection, floor)),)
        if self.parents is None:
            raise ValueError(
                f"{floor!r} is a cell: dispatch needs a parent pool — "
                "construct the loop with parents=ParentPool(...)."
            )
        rungs = viable_rungs(cell, self.parents.available(), self.operators)
        if not rungs:
            reason = dispatch(cell, self.parents.available(), self.operators).reason
            raise ValueError(f"{floor!r} has no parent rung: {reason}.")
        return tuple(
            (rung.operator, self.parents.hydrate(rung.parents)) for rung in rungs
        )

    def targets_for(self, floor: str, parent: pd.Series) -> Targets:
        """A cell demands its WHOLE predicate; a floor demands the operator's
        own postcondition (d51b)."""
        cell = CELLS_BY_NAME.get(floor)
        if cell is not None:
            return cell_targets(cell, parent)
        operator = self.operator_for(floor)
        assert operator is not None, floor
        return operator.targets(floor, parent)

    def hungry(self) -> pd.DataFrame:
        """The readout: every hungry floor, who serves it, and whether its
        credit gate is open — the loop's own coverage table."""
        rows = []
        for row in self.order_sheet().itertuples(index=False):
            operator = self.operator_for(row.floor)
            rows.append(
                {
                    "floor": row.floor,
                    "missing": row.missing,
                    "operator": operator.declaration.operator if operator else None,
                    "gate": str(operator.declaration.credit_gate) if operator else None,
                    "runnable": operator is not None
                    and operator.declaration.credit_gate is CreditGate.NONE,
                }
            )
        return pd.DataFrame(rows)

    def rungs(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Every hungry cell priced against a parent pool before any spend:
        the derived operator, its surface_origin, how many parents reach the
        cell, and — where none do — unmintable versus unsupplied (d51d)."""
        return rung_report(
            self.order_sheet(), pool, CELLS_BY_NAME, self.operators
        )

    def run(self, floor: str, *, n: int | None = None) -> pd.DataFrame:
        """Produce up to `n` ACCEPTED candidates for one floor (default:
        ceil of the floor's missing credit) and append them to the pool."""
        rungs = self.demand(floor)
        for operator, _ in rungs:
            if operator.declaration.credit_gate is not CreditGate.NONE:
                print(
                    f"NOTE: {operator.declaration.operator!r} is gated by "
                    f"{operator.declaration.credit_gate} (d42h) — rows are "
                    "produced as feature-stock (no floor credit; skipped by "
                    "the mini-fill) until the gate's pilot passes."
                )

        sheet = self.order_sheet()
        missing = sheet.loc[sheet["floor"] == floor, "missing"]
        if missing.empty:
            raise ValueError(
                f"{floor!r} is not hungry on the order sheet — nothing to "
                "produce. Re-read loop.hungry() after the last admission."
            )
        need = n if n is not None else math.ceil(float(missing.iloc[0]))

        # eligibility arrives in the operator's declared preference order
        # (d42c) — near-parents for StatRewrite, seeded shuffle elsewhere
        spent = self.pool.parents_used(floor)
        queue = [
            (operator, parent)
            for operator, parents in rungs
            for parent in parents[
                ~parents["query_id"].astype(str).isin(spent)
            ].to_dict("records")
        ]

        accepted: list[AugmentedCandidate] = []
        attempted = 0
        bar = tqdm(total=need, desc=f"augment:{floor}", unit="row")
        for operator, parent in queue:
            if len(accepted) >= need:
                break
            attempted += 1
            outcome = self.engine.run(
                operator.instruction(floor, parent),
                f"Query: {parent['query']}",
                self.targets_for(floor, parent),
                tool_loop=operator.declaration.tool_loop,
            )
            problems = (
                operator.structural(parent, outcome.text)
                if outcome.accepted
                else []
            )
            if outcome.accepted and not problems:
                candidate = operator.candidate(
                    parent, floor, outcome.text, outcome.attempts
                )
                accepted.append(candidate)
                self.pool.append([candidate])   # banked immediately — paid spend
                self.qrels.mint(candidate)      # answer key born with the row (d43d)
                bar.update(1)
                bar.write(_pair_line(
                    "+", parent, outcome.text,
                    f" ({outcome.attempts} attempt(s))",
                ))
            elif outcome.accepted:
                bar.write(_pair_line(
                    "-", parent, outcome.text,
                    f": dropped — structural: {problems}",
                ))
            else:
                failed = [c.target for c in outcome.checks if not c.passed]
                bar.write(_pair_line(
                    "-", parent, outcome.text, f": dropped — failed {failed}",
                ))
            bar.set_postfix(attempted=attempted, dropped=attempted - len(accepted))
        bar.close()
        print(
            f"{floor}: accepted {len(accepted)}/{attempted} attempts "
            f"(need {need}, parents available {len(queue):,} across "
            f"{len(rungs)} rung(s)) -> {self.pool.path}"
        )
        return pd.DataFrame([c.model_dump() for c in accepted])

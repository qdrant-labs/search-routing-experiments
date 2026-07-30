"""The augmentation loop (d42a): demand from the order sheet, dispatch by
floor, produce -> verify -> pool. Selection is deterministic; the LLM only
weaves. Running a batch is a user-initiated LLM spend."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from augmentation.core import AugmentedCandidate, CreditGate, Operator
from augmentation.engine import Augmenter
from augmentation.operators import OPERATORS
from augmentation.pool import GeneratedPool

DEFAULT_SHEET = Path("data") / "composition" / "order_sheet.parquet"


class AugmentationLoop:
    """Wires order sheet -> operator -> Augmenter -> GeneratedPool."""

    def __init__(
        self,
        selection: pd.DataFrame,
        *,
        engine: Augmenter | None = None,
        operators: tuple[Operator, ...] = OPERATORS,
        sheet_path: Path | str = DEFAULT_SHEET,
        pool: GeneratedPool | None = None,
        seed: int = 0,
    ) -> None:
        self.selection = selection
        self.engine = engine or Augmenter(seed=seed)
        self.operators = operators
        self.pool = pool or GeneratedPool()
        self._sheet_path = Path(sheet_path)
        self._seed = seed

    def order_sheet(self) -> pd.DataFrame:
        sheet = pd.read_parquet(self._sheet_path)
        hungry = sheet[sheet["missing"] > 0]
        return hungry.sort_values("missing", ascending=False, ignore_index=True)

    def operator_for(self, floor: str) -> Operator | None:
        return next((op for op in self.operators if op.serves(floor)), None)

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

    def run(self, floor: str, *, n: int | None = None) -> pd.DataFrame:
        """Produce up to `n` ACCEPTED candidates for one floor (default:
        ceil of the floor's missing credit) and append them to the pool."""
        operator = self.operator_for(floor)
        if operator is None:
            raise ValueError(f"No registered operator serves {floor!r}.")
        if operator.declaration.credit_gate is not CreditGate.NONE:
            raise ValueError(
                f"{operator.declaration.operator!r} is gated by "
                f"{operator.declaration.credit_gate} (d42h) — its rows would "
                "be feature-stock. Run the gate's pilot first."
            )

        sheet = self.order_sheet()
        missing = sheet.loc[sheet["floor"] == floor, "missing"]
        need = n if n is not None else math.ceil(float(missing.iloc[0]))

        parents = operator.eligible(self.selection, floor)
        parents = parents[
            ~parents["query_id"].astype(str).isin(self.pool.parents_used(floor))
        ].sample(frac=1.0, random_state=self._seed)

        accepted: list[AugmentedCandidate] = []
        attempted = 0
        bar = tqdm(total=need, desc=f"augment:{floor}", unit="row")
        for parent in parents.to_dict("records"):
            if len(accepted) >= need:
                break
            attempted += 1
            outcome = self.engine.run(
                operator.instruction(floor, parent),
                f"Query: {parent['query']}",
                operator.targets(floor),
                tool_loop=operator.declaration.tool_loop,
            )
            if outcome.accepted:
                candidate = operator.candidate(
                    parent, floor, outcome.text, outcome.attempts
                )
                accepted.append(candidate)
                self.pool.append([candidate])   # banked immediately — paid spend
                bar.update(1)
                bar.write(
                    f"+ {parent['query_id']} ({outcome.attempts} attempt(s))\n"
                    f"    before: {str(parent['query'])[:80]!r}\n"
                    f"    after:  {outcome.text[:80]!r}"
                )
            else:
                failed = [c.target for c in outcome.checks if not c.passed]
                bar.write(
                    f"- {parent['query_id']}: dropped — failed {failed}\n"
                    f"    before: {str(parent['query'])[:80]!r}\n"
                    f"    tried:  {outcome.text[:80]!r}"
                )
            bar.set_postfix(attempted=attempted, dropped=attempted - len(accepted))
        bar.close()
        print(
            f"{floor}: accepted {len(accepted)}/{attempted} attempts "
            f"(need {need}, parents available {len(parents):,}) -> {self.pool.path}"
        )
        return pd.DataFrame([c.model_dump() for c in accepted])

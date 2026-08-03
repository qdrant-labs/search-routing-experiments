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
from augmentation.engine import Augmenter
from augmentation.operators import default_operators
from augmentation.pool import GeneratedPool
from augmentation.qrels import AugmentationQrels


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
    ) -> None:
        self.selection = selection
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
            print(
                f"NOTE: {operator.declaration.operator!r} is gated by "
                f"{operator.declaration.credit_gate} (d42h) — rows are "
                "produced as feature-stock (no floor credit; skipped by the "
                "mini-fill) until the gate's pilot passes."
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
        parents = operator.eligible(self.selection, floor)
        parents = parents[
            ~parents["query_id"].astype(str).isin(self.pool.parents_used(floor))
        ]

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
                operator.targets(floor, parent),
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
            f"(need {need}, parents available {len(parents):,}) -> {self.pool.path}"
        )
        return pd.DataFrame([c.model_dump() for c in accepted])

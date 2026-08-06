"""The augmentation loop (d42a): demand from the order sheet, dispatch by
floor, produce -> verify -> pool. Selection is deterministic; the LLM only
weaves. Running a batch is a user-initiated LLM spend."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from augmentation.config import AugmentationConfig
from augmentation.core import (
    NOTHING_ELSE,
    AugmentedCandidate,
    CreditGate,
    Operator,
    SurfaceOrigin,
)
from augmentation.dispatch import (
    CellPlan,
    Stage,
    Step,
    calls_for,
    plan,
    plan_report,
    requirements,
    targets_of,
    unreachable,
)
from augmentation.engine import AugmentationOutcome, Augmenter, ErrorCase
from augmentation.operators import default_operators
from augmentation.parents import ParentPool
from augmentation.pool import GeneratedPool
from augmentation.qrels import AugmentationQrels
from composition.cells import CELLS_BY_NAME
from query_taxonomy.features import FeatureExtractor
from taxonomy_generators.verify import Targets


def _pair_line(sign: str, parent: dict, text: str | None, note: str) -> str:
    tag = "after: " if sign == "+" else "tried: "
    shown = "(nothing — never got that far)" if text is None else repr(text[:80])
    return (
        f"{sign} {parent['query_id']}{note}\n"
        f"    before: {str(parent['query'])[:80]!r}\n"
        f"    {tag} {shown}"
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
        # engines=None: cells band spaCy stats (natural_language_share,
        # nesting_depth), and a regex-only extractor reports those as
        # measured=None — a target the model can never satisfy, so it burns its
        # whole round budget chasing one. Same parity MiniFill keeps (d42g).
        self.engine = engine or Augmenter(
            self.config.engine,
            seed=self.config.seed,
            extractor=FeatureExtractor(engines=None),
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

    def demand(self, floor: str) -> tuple[CellPlan | None, pd.DataFrame]:
        """The plan for this demand and the parents that survived it. A cell
        runs the staged pipeline; a floor keeps its single operator and has no
        plan."""
        cell = CELLS_BY_NAME.get(floor)
        if cell is None:
            operator = self.operator_for(floor)
            if operator is None:
                raise ValueError(f"No registered operator serves {floor!r}.")
            return None, operator.eligible(self.selection, floor)
        if self.parents is None:
            raise ValueError(
                f"{floor!r} is a cell: the pipeline needs a parent pool — "
                "construct the loop with parents=ParentPool(...)."
            )
        result = plan(cell, self.parents.available(), self.operators)
        if not result.servable:
            unserved = ", ".join(
                f"{step.stage}:{step.requirement[0].member}"
                for step in result.unsatisfied
            )
            raise ValueError(f"{floor!r} has no parents left: {unserved}.")
        if result.needs_selection:
            raise ValueError(
                f"{floor!r} needs SELECTION, not augmentation: "
                f"{len(result.parents):,} unspent queries already satisfy it. "
                "Its shortfall comes from the fill's own constraints, so "
                "paying an LLM here would change nothing."
            )
        return result, self.parents.hydrate(result.parents)

    def _planned(
        self, floor: str, result: CellPlan | None
    ) -> tuple[Operator, ...]:
        """The operators this demand runs: a cell's planned mints, a floor's
        single family."""
        if result is not None:
            return result.operators
        operator = self.operator_for(floor)
        if operator is None:
            raise ValueError(f"No registered operator serves {floor!r}.")
        return (operator,)

    def grounded(self, result: CellPlan | None, parent: pd.Series) -> pd.Series:
        """The parent plus its gold document, read whenever Inject already
        resolved one — never for the pool, one lookup per parent actually used.

        NOT gated on `Stage.CONSTRAIN`: that classifies the PLAN, but
        `StatRewrite.instruction` decides to cut from the PARENT's own value,
        so a two-sided band the parent already overshoots also cuts — and
        did so blind before this fix, dropping content with no document to
        check it against. `grounding_doc_id` existing is the cheap proxy for
        "a document is already known"; fetching it costs one pushdown filter
        regardless of whether the eventual instruction ends up using it."""
        if self.parents is None or parent.get("grounding_doc_id") is None:
            return parent
        return pd.Series({**parent, "gold_text": self.parents.gold_text(parent)})

    def brief(self, floor: str, steps: tuple[Step, ...], parent: pd.Series) -> str:
        """One call's request: what to change, what the result must look like,
        then the exclusion ONCE. No operator asserts a shape of its own, so the
        cell's `looks_like` is the only voice on that and needs no override
        (d55d) — a floor demand simply has none."""
        cell = CELLS_BY_NAME.get(floor)
        moves = [
            step.operator.instruction(floor, parent, step.requirement)
            for step in steps
            if step.operator
        ]
        shape = cell.looks_like if cell else None
        return "\n\n".join(
            moves
            + ([f"The finished query must look like this: {shape}"] if shape else [])
            + [NOTHING_ELSE]
        )

    def produce(
        self, floor: str, result: CellPlan | None, parent: pd.Series
    ) -> tuple[AugmentationOutcome, Targets]:
        """Run the calls in harm order, feeding each result to the next (d55b).
        Checks accumulate, so a later call cannot undo an earlier one; a call
        that fails returns the last text that passed, which is a real row with
        fewer targets met (d55c)."""
        cell = CELLS_BY_NAME.get(floor)
        if cell is None or result is None:
            return self._one_call(floor, self._planned(floor, result), parent)

        text, best, targets = str(parent["query"]), None, Targets()
        calls = calls_for(result, cell, parent)
        if not calls:
            # every mint this parent could need is already free — measured,
            # not assumed (2026-08 follow-up): no operator spends a call on a
            # row that already is what the cell asks for
            targets = targets_of(requirements(cell), parent)
            report = self.engine.accept(text, targets)
            return AugmentationOutcome(
                text=text, accepted=report.passed, attempts=0, checks=report.checks,
            ), targets
        for call in calls:
            targets = targets_of(call.verified, parent)
            if unreachable(call, parent):
                # tokens already committed (a mint's own surfaces) already
                # break this call's ceiling — no rewrite fixes that, so this
                # is not spent on the model at all
                outcome = AugmentationOutcome(
                    text=None, accepted=False, attempts=0,
                    error=ErrorCase.INCOMPATIBLE_PARENT,
                )
                return (best or outcome), targets
            outcome = self.engine.run(
                self.brief(floor, call.steps, parent),
                f"Query: {text}",
                targets,
                tool_loop=any(op.declaration.tool_loop for op in call.operators),
            )
            if not outcome.accepted:
                return (best or outcome), targets
            text, best = outcome.text, outcome
        assert best is not None, f"{floor!r} planned no calls"
        return best, targets

    def _one_call(
        self, floor: str, planned: tuple[Operator, ...], parent: pd.Series
    ) -> tuple[AugmentationOutcome, Targets]:
        """The floor path: one operator, its own postcondition, one call."""
        operator = planned[0]
        targets = operator.targets(floor, parent)
        outcome = self.engine.run(
            self.brief(floor, (Step((), Stage.QUERY_ONLY, operator),), parent),
            f"Query: {parent['query']}",
            targets,
            tool_loop=operator.declaration.tool_loop,
        )
        return outcome, targets

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

    def plans(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Every hungry cell priced against a parent pool before any spend: how
        many parents survive the stages, who mints what, and which
        requirements nothing can serve."""
        return plan_report(
            self.order_sheet(), pool, CELLS_BY_NAME, self.operators
        )

    def run(self, floor: str, *, n: int | None = None) -> pd.DataFrame:
        """Produce up to `n` ACCEPTED candidates for one floor (default:
        ceil of the floor's missing credit) and append them to the pool."""
        result, parents = self.demand(floor)
        planned = self._planned(floor, result)
        for operator in planned:
            if operator.declaration.credit_gate is not CreditGate.NONE:
                print(
                    f"NOTE: {operator.declaration.operator!r} is gated by "
                    f"{operator.declaration.credit_gate} (d42h) — rows are "
                    "produced as feature-stock (no floor credit; skipped by "
                    "the mini-fill) until the gate's pilot passes."
                )
        if result and result.partial:
            print(
                f"NOTE: {floor!r} is served in part — "
                + ", ".join(
                    f"{step.stage}:{step.requirement[0].member}"
                    for step in result.unsatisfied
                )
                + " unserved, so rows land in whatever cell they measure into."
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
        queue = parents[
            ~parents["query_id"].astype(str).isin(spent)
        ].to_dict("records")
        # the answer key follows the strongest origin among the planned mints:
        # one doc-copied surface makes the whole child grounded in that doc
        owner = max(
            planned, key=lambda op: op.declaration.surface_origin is not
            SurfaceOrigin.NONE
        )

        accepted: list[AugmentedCandidate] = []
        attempted = 0
        bar = tqdm(total=need, desc=f"augment:{floor}", unit="row")
        for parent in queue:
            if len(accepted) >= need:
                break
            attempted += 1
            parent = self.grounded(result, parent)
            outcome, targets = self.produce(floor, result, parent)
            # structural checks read `targets` as the authorisation: a mint must
            # not veto the span another was asked to add (d52d)
            problems = (
                [
                    problem
                    for op in planned
                    for problem in op.structural(parent, outcome.text, targets)
                ]
                if outcome.accepted
                else []
            )
            if outcome.accepted and not problems:
                candidate = owner.candidate(
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
                # error is the reason nothing verifiable exists to grade —
                # attempted_tools is the trace that separates a stuck loop
                # from one that ran out of rounds doing varied real work
                if outcome.error is not None:
                    reason = str(outcome.error)
                    if outcome.attempted_tools:
                        reason += f" — rounds called: {list(outcome.attempted_tools)}"
                else:
                    failed = ", ".join(
                        f"{c.target} (measured {c.measured})"
                        for c in outcome.checks if not c.passed
                    )
                    reason = f"failed [{failed}]"
                bar.write(_pair_line(
                    "-", parent, outcome.text, f": dropped — {reason}",
                ))
            bar.set_postfix(attempted=attempted, dropped=attempted - len(accepted))
        bar.close()
        print(
            f"{floor}: accepted {len(accepted)}/{attempted} attempts "
            f"(need {need}, parents available {len(queue):,}, minting "
            f"{[op.declaration.operator for op in planned]}) -> {self.pool.path}"
        )
        return pd.DataFrame([c.model_dump() for c in accepted])

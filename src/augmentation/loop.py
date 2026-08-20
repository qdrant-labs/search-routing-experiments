"""The augmentation loop (d42a): demand from the order sheet, dispatch by
floor, produce -> verify -> pool. Selection is deterministic; the LLM only
weaves. Running a batch is a user-initiated LLM spend."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from augmentation.config import FAULT_STREAK, AugmentationConfig
from augmentation.core import (
    NOTHING_ELSE,
    AugmentedCandidate,
    CreditGate,
    Operator,
    SurfaceOrigin,
)
from augmentation.dispatch import (
    Call,
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
from augmentation.engine import AugmentationOutcome, Augmenter, ErrorCase, Spend
from augmentation.operators import default_operators
from augmentation.parents import ParentPool
from augmentation.constructed import ConstructedDocs
from augmentation.pool import GeneratedPool
from augmentation.qrels import AugmentationQrels
from augmentation.synthetic import SyntheticOperator
from composition.cells import CELLS_BY_NAME
from query_taxonomy.features import FeatureExtractor
from taxonomy_generators.verify import Targets


class NeedsSelection(ValueError):
    """A cell its own unspent parents already satisfy: the shortfall is the
    selection layer's, and no generated row can serve it."""


@dataclass
class FaultStreak:
    """One `run()` call's fault-streak bookkeeping (d59): every attempted
    parent id, in order, and whether the call stopped BECAUSE of the streak
    rather than by reaching `need` or exhausting `queue`. Mutable, in-loop,
    never crosses a boundary until `report()` — same shape as `Spend`, not a
    pydantic model, for the same reason: it's an accumulator, not a value
    object. `report()` feeds `AugmentationCampaign`'s chances scheduler via
    `.attrs` — no loose counter threaded through the attempt loop."""

    limit: int | None
    attempted_ids: list[str] = field(default_factory=list)
    consecutive: int = 0
    stopped: bool = False

    def tripped(self) -> bool:
        """Whether the streak just hit `limit` — marks `stopped` itself, so
        the call site never has to reach in and set the flag by hand."""
        self.stopped = self.limit is not None and self.consecutive >= self.limit
        return self.stopped

    def record(self, query_id: str, *, accepted: bool) -> None:
        self.attempted_ids.append(query_id)
        self.consecutive = 0 if accepted else self.consecutive + 1

    def report(self) -> dict[str, object]:
        return {
            "attempted_ids": tuple(self.attempted_ids),
            "stopped_early": self.stopped,
        }


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
        docs: ConstructedDocs | None = None,
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
        # whole round budget chasing one. Same parity the cell fill keeps (d42g).
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
        self.docs = docs or ConstructedDocs(self.config.paths)
        self._sheet_path = Path(sheet_path or self.config.paths.order_sheet)

    @property
    def sheet_path(self) -> Path:
        """Which sheet this loop serves, and so which composition owns it."""
        return self._sheet_path

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
            raise NeedsSelection(
                f"{floor!r} needs SELECTION, not augmentation: "
                f"{len(result.parents):,} unspent queries already satisfy it. "
                "Its shortfall comes from the fill's own constraints, so "
                "paying an LLM here would change nothing."
            )
        return result, self.parents.hydrate(result.parents)

    def planned(
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

    @staticmethod
    def owner(planned: tuple[Operator, ...]) -> Operator:
        """Which planned operator a banked row's answer key and credit gate
        follow: the strongest surface origin among the mints — one doc-copied
        surface makes the whole child grounded in that doc, regardless of how
        many other operators also touched it."""
        return max(
            planned,
            key=lambda op: op.declaration.surface_origin is not SurfaceOrigin.NONE,
        )

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
            return self._one_call(floor, self.planned(floor, result), parent)

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
        spend = Spend()
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
                return spend.stamp(best or outcome), targets
            produced = self._deterministic(floor, call, parent, text)
            if produced is not None:
                report = self.engine.accept(produced, targets)
                outcome = AugmentationOutcome(
                    text=produced, accepted=report.passed, attempts=1,
                    checks=report.checks,
                )
            else:
                outcome = self.engine.run(
                    self.brief(floor, call.steps, parent),
                    f"Query: {text}",
                    targets,
                    tool_loop=any(op.declaration.tool_loop for op in call.operators),
                )
            spend.add(outcome)
            if not outcome.accepted:
                return spend.stamp(best or outcome), targets
            text, best = outcome.text, outcome
        assert best is not None, f"{floor!r} planned no calls"
        return spend.stamp(best), targets

    @staticmethod
    def _deterministic(
        floor: str, call: Call, parent: pd.Series, text: str
    ) -> str | None:
        """The call's text when every one of its steps can be served without a
        model — None the moment one step needs the LLM, because the brief is
        written per call and a half-served call would lose the other half."""
        working = text
        for step in call.steps:
            if step.operator is None:
                return None
            produced = step.operator.apply(parent, floor, working, step.requirement)
            if produced is None:
                return None
            working = produced
        return working

    def _one_call(
        self, floor: str, planned: tuple[Operator, ...], parent: pd.Series
    ) -> tuple[AugmentationOutcome, Targets]:
        """The floor path: one operator, its own postcondition, one call."""
        operator = planned[0]
        targets = operator.targets(floor, parent)
        text = operator.apply(parent, floor, str(parent["query"]))
        if text is not None:
            # deterministic operator: local re-measure IS the acceptance, so
            # the row costs no completion, no tokens, and reproduces from seed
            report = self.engine.accept(text, targets)
            return AugmentationOutcome(
                text=text, accepted=report.passed, attempts=1, checks=report.checks,
            ), targets
        outcome = self.engine.run(
            self.brief(floor, (Step((), Stage.QUERY_ONLY, operator),), parent),
            f"Query: {parent['query']}",
            targets,
            tool_loop=operator.declaration.tool_loop,
        )
        return outcome, targets

    def synthesize(
        self,
        floor: str,
        n: int,
        *,
        source_dataset: str,
        docs_per_query: int = 2,
        max_consecutive_faults: int | None = FAULT_STREAK,
    ) -> pd.DataFrame:
        """The synthetic rung: rows for a cell no parent can reach.

        Query first — a query that misses its bands costs one completion, and
        documents are only written for a query that already measures into the
        cell. `docs_per_query` defaults to 2 = the genuine-tie depth bar:
        one doc is a depth classify() files as fake-tie waste, more buys
        nothing at that boundary and costs a completion each. `source_dataset`
        names the lane whose corpus lends the constructed collection its
        distractors AND whose min_relevance grades the minted key.
        """
        operator = SyntheticOperator(self.config)
        if not operator.serves(floor):
            raise ValueError(f"{floor!r} is not a cell — nothing to synthesize")
        from hybrid_search_rrf_dataset.lanes import LANES

        relevance = (
            LANES[source_dataset].min_relevance
            if source_dataset in LANES else 1
        )
        spend = Spend()
        faults = FaultStreak(max_consecutive_faults)
        minted: list[AugmentedCandidate] = []
        # completion = the minted KEY, not a banked doc: a row whose documents
        # came up short has docs but no qrels, and must be retried, not stuck
        already = set(self.qrels.load()["query_id"].astype(str))
        bar = tqdm(total=n, desc=f"synthesize:{floor}", unit="row")
        index = 0
        while len(minted) < n and not faults.tripped():
            parent = pd.Series({
                "query_id": f"syn-{floor}-{index}",
                "dataset": source_dataset,
                "query": "",
                "floors": [],
                "branch_index": index,
            })
            index += 1
            if str(parent["query_id"]) in already:
                continue        # a rerun never regenerates a banked row
            targets = operator.targets(floor, parent)
            outcome = self.engine.run(
                operator.instruction(floor, parent), "", targets
            )
            spend.add(outcome)
            faults.record(str(parent["query_id"]), accepted=outcome.accepted)
            if not outcome.accepted:
                bar.write(f"- {parent['query_id']}: dropped — {outcome.checks}")
                continue
            doc_ids: list[str] = []
            for ordinal in range(1, docs_per_query + 1):
                answer = self.engine.run(
                    operator.document(floor, outcome.text, ordinal),
                    f"Query: {outcome.text}",
                    Targets(),
                )
                spend.add(answer)
                if not answer.text:
                    break
                doc_ids.append(self.docs.add(
                    query_id=str(parent["query_id"]),
                    source_dataset=source_dataset,
                    text=answer.text,
                    ordinal=ordinal,
                ))
            # all docs or none: a partial set writes a depth the row did not
            # earn, and the banked ids would resurrect it on rerun anyway
            if len(doc_ids) < docs_per_query:
                bar.write(
                    f"- {parent['query_id']}: query kept, documents incomplete "
                    f"({len(doc_ids)}/{docs_per_query})"
                )
                continue
            candidate = operator.candidate(parent, floor, outcome).model_copy(
                update={"grounding_doc_id": doc_ids[0]}
            )
            minted.append(candidate)
            self.pool.append([candidate])
            self.qrels.mint_constructed(candidate.query_id, doc_ids, relevance)
            bar.update(1)
            bar.write(f"+ {candidate.query_id}: {outcome.text!r}")
        bar.close()
        produced = pd.DataFrame([c.model_dump() for c in minted])
        print(f"  {spend.summary(len(minted))}")
        return produced

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

    def run(
        self,
        floor: str,
        *,
        n: int | None = None,
        exclude: frozenset[str] = frozenset(),
        max_consecutive_faults: int | None = FAULT_STREAK,
    ) -> pd.DataFrame:
        """Produce up to `n` ACCEPTED candidates for one floor (default:
        ceil of the floor's missing credit) and append them to the pool.

        `exclude` skips parents beyond what the persisted pool already
        rules out — a campaign's repeat chance at a floor (d59) needs
        parents THIS run already spent, which `parents_used` cannot see
        since a dropped attempt is never persisted anywhere.

        `max_consecutive_faults` stops the attempt loop the moment that many
        non-accepted attempts happen in a row, rather than continuing to
        `need` or exhausting `queue` — a fault (dropped for any reason) is
        d59's unified signal, not just an engine error. It defaults to the
        campaign's own `FAULT_STREAK` so the ceiling holds on every path;
        pass None to chew through the whole queue deliberately. `produced.attrs`
        then carries `stopped_early` (True only when THIS is why the loop
        ended, never on hitting `need` or running out of parents) and
        `attempted_ids` (every parent tried this call, accepted or not) —
        both read by `AugmentationCampaign`'s chances scheduler, neither
        changes what `run()` returns to any existing caller."""
        # started before demand() so `spend.wall_s` covers the WHOLE run —
        # dispatch/plan, catalog + pool reads — not just the LLM's own hops
        spend = Spend()
        result, parents = self.demand(floor)
        planned = self.planned(floor, result)
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
        spent = self.pool.parents_used(floor) | set(exclude)
        queue = parents[
            ~parents["query_id"].astype(str).isin(spent)
        ].to_dict("records")
        owner = self.owner(planned)

        accepted: list[AugmentedCandidate] = []
        attempted = 0
        faults = FaultStreak(max_consecutive_faults)
        bar = tqdm(total=need, desc=f"augment:{floor}", unit="row")
        for parent in queue:
            if len(accepted) >= need:
                break
            if faults.tripped():
                break
            attempted += 1
            parent = self.grounded(result, parent)
            outcome, targets = self.produce(floor, result, parent)
            spend.add(outcome)   # a drop still paid for its hops
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
            banked = outcome.accepted and not problems
            faults.record(str(parent["query_id"]), accepted=banked)
            if banked:
                candidate = owner.candidate(parent, floor, outcome)
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
            f"{[op.declaration.operator for op in planned]}) -> {self.pool.path}\n"
            f"  spend: {spend.summary(len(accepted))}"
        )
        produced = pd.DataFrame([c.model_dump() for c in accepted])
        produced.attrs["spend"] = spend.report()
        produced.attrs.update(faults.report())
        return produced

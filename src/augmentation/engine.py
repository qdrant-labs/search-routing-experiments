"""The Augmenter — the LLM engine behind every operator (d42g, d42n).

Two interaction modes, chosen by the operator's declaration:

- **single-shot** (`tool_loop=False`): one completion, no tools — for
  operators whose postcondition the model can hit blind (Decorate). A
  failed local acceptance feeds back as a revise message, bounded by
  `max_attempts`.
- **tool loop** (`tool_loop=True`): the agentic mode for range-chasing
  operators (StatRewrite) — the model gets the taxonomy_generators trio
  plus `submit_text`, and ends an attempt by CALLING submit_text, so the
  final text is the tool call's arguments, never an extra completion.

In both modes the in-conversation results only steer; acceptance is
`accept()` — a fresh local re-measure of the returned text, the only gate.
The prototype's instructor extraction (which could reword text after the
last in-loop verify — the d2 gap) is gone entirely.
"""

from __future__ import annotations

import json
import time
from enum import StrEnum

from litellm import completion
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from pydantic import BaseModel, ConfigDict

from augmentation.config import EngineSettings
from query_taxonomy.features import FeatureExtractor
from taxonomy_generators.tools import build_tools
from taxonomy_generators.verify import TargetCheck, Targets, VerifyReport, verify

TRANSIENT_PROVIDER_ERRORS = (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
"""Provider blips (529 overload, timeouts, rate limits) — failed attempts
for the fault machinery, never crashes; a real config error still raises."""

_SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_text",
        "description": (
            "Submit the final query text once it satisfies the targets."
            " This ends the task — the text is re-measured locally."
        ),
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}


class ErrorCase(StrEnum):
    """Why an attempt produced no verifiable rewrite — distinct from a rewrite
    that WAS measured and failed, which carries `checks` instead of `error`."""

    NO_TEXT = "no_text"
    """A plain reply (single-shot, or a tool-loop message with no tool call)
    came back empty."""
    EMPTY_SUBMIT = "empty_submit"
    """`submit_text` was called with blank text."""
    ROUNDS_EXHAUSTED = "rounds_exhausted"
    """`max_rounds` ran out without ever calling `submit_text`."""
    PROVIDER_FAULT = "provider_fault"
    """The provider refused the call transiently (overload, timeout, rate
    limit) — the attempt failed without ever producing text."""
    INCOMPATIBLE_PARENT = "incompatible_parent"
    """The caller determined, before spending a call, that no rewrite can
    satisfy the request — e.g. the tokens a mint step must insert already
    exceed a later band's ceiling. Never set by `Augmenter` itself, which has
    no view of the parent; the loop raises this pre-flight (d55-followup)."""


def _clean(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


class AugmentationOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str | None
    accepted: bool
    attempts: int
    checks: tuple[TargetCheck, ...] = ()
    error: ErrorCase | None = None
    """Set only when `text` carries nothing verifiable. `attempted_tools`
    is the corroborating evidence for `ROUNDS_EXHAUSTED` specifically."""
    attempted_tools: tuple[str, ...] = ()
    """Every tool name called, across every round of the failed attempt, in
    order — repeating the same tool distinguishes a stuck loop from one that
    simply ran out of rounds doing varied legitimate work."""
    hops: int = 0
    """`completion()` round-trips spent producing this outcome — a tool-loop
    attempt costs one per round, not one per attempt."""
    tokens: int = 0
    """Total tokens (prompt + completion), summed across every hop."""
    elapsed_s: float = 0.0
    """Wall-clock seconds spent inside `completion()` for this outcome."""


class Spend:
    """Cumulative cost: hops (`completion()` round-trips), tokens, LLM
    seconds, PLUS wall-clock time since construction. Every outcome adds in,
    accepted or dropped — a drop still paid for its hops. `elapsed_s` is
    time spent waiting on the model; `wall_s` is everything — start this at
    the top of whatever scope you want timed, since it measures from `__init__`
    regardless of what `add()`/`add_hop()` ever see. The gap between the two
    is the loop's OWN cost: dispatch/plan, gold-doc lookups, structural
    checks, parquet writes — not just LLM latency. Used at every scope that
    sums several outcomes into one: one round into one attempt
    (`Augmenter.run`), one attempt into one call (`_tool_rounds`), one call
    sequence into one row (`AugmentationLoop.produce`), one floor's whole
    run — dispatch included (`AugmentationLoop.run`)."""

    def __init__(self) -> None:
        self.hops = 0
        self.tokens = 0
        self.elapsed_s = 0.0
        self._started = time.monotonic()

    @property
    def wall_s(self) -> float:
        return time.monotonic() - self._started

    def add(self, spent: "Spend | AugmentationOutcome") -> None:
        """Fold in anything shaped like a spend — an outcome or another
        `Spend` both expose the same three fields, so one method covers
        merging two running totals and crediting a finished outcome alike."""
        self.hops += spent.hops
        self.tokens += spent.tokens
        self.elapsed_s += spent.elapsed_s

    def add_hop(self, elapsed_s: float, tokens: int) -> None:
        """One `completion()` round-trip — the raw unit `_completion`
        measures, before there is an outcome to wrap it in."""
        self.hops += 1
        self.tokens += tokens
        self.elapsed_s += elapsed_s

    def stamp(self, outcome: AugmentationOutcome) -> AugmentationOutcome:
        """This outcome, credited with the running total rather than just
        its own last call — an accepted or dropped row's real cost is every
        call it took to get there, not only the one that decided it.
        Outcome-compatible fields only — `wall_s` is a scope-level fact, no
        single outcome owns it, so `report()` carries that instead."""
        return outcome.model_copy(update=self.as_dict())

    def as_dict(self) -> dict[str, float]:
        return {"hops": self.hops, "tokens": self.tokens, "elapsed_s": self.elapsed_s}

    def report(self) -> dict[str, float]:
        """Everything, including wall-clock — for a floor-level readout,
        never for `stamp()` (`AugmentationOutcome` has no `wall_s` field)."""
        return {**self.as_dict(), "wall_s": self.wall_s}

    def summary(self, accepted: int) -> str:
        """One line: totals, plus the per-row rate that answers 'how much
        does ONE floor value cost' — the question this class exists for.
        Per-row time is `wall_s`: the loop's own overhead is part of the
        answer, not just what the model was waiting on."""
        totals = (
            f"{self.hops} hops, {self.tokens:,} tokens, "
            f"{self.wall_s:.1f}s wall ({self.elapsed_s:.1f}s of it LLM)"
        )
        if not accepted:
            return f"{totals} (nothing accepted)"
        return (
            f"{totals} -> {self.hops / accepted:.1f} hops/row, "
            f"{self.tokens / accepted:,.0f} tokens/row, "
            f"{self.wall_s / accepted:.1f}s/row"
        )


class Augmenter:
    """Bounded attempts + local-verify acceptance, in either mode.

    Spend is `EngineSettings`: `max_attempts` bounds revise-after-local-
    failure cycles; `max_rounds` bounds tool-call rounds per attempt
    (tool-loop mode only). Exhaustion — of attempts, or protocol-side of
    rounds — returns an unaccepted outcome and the caller drops the row,
    parents are plentiful (d42g).
    """

    def __init__(
        self,
        settings: EngineSettings | None = None,
        *,
        seed: int = 0,
        extractor: FeatureExtractor | None = None,
    ) -> None:
        settings = settings or EngineSettings()
        self.model = settings.model
        self.max_rounds = settings.max_rounds
        self.max_attempts = settings.max_attempts
        self._extractor = extractor or FeatureExtractor()
        tools = build_tools(seed=seed, extractor=self._extractor)
        self._runners = {tool.name: tool.run for tool in tools}
        self._tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ] + [_SUBMIT_TOOL]

    def accept(self, text: str, targets: Targets) -> VerifyReport:
        """The only acceptance: re-measure the final text locally, never
        trust the transcript (d2/d42g)."""
        return verify(text, targets, extractor=self._extractor)

    def ask(self, instruction: str, prompt: str) -> tuple[str | None, Spend]:
        """One completion, no targets and no submit protocol — for a caller
        whose answer is a verdict rather than a query text."""
        try:
            return self._single_shot([
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt},
            ])
        except TRANSIENT_PROVIDER_ERRORS:
            return None, Spend()

    def run(
        self,
        instruction: str,
        prompt: str,
        targets: Targets,
        *,
        tool_loop: bool = False,
    ) -> AugmentationOutcome:
        protocol = (
            "\nWhen your text satisfies the targets, call submit_text with it."
            if tool_loop
            else "\nReply with ONLY the final query text — no quotes, no commentary."
        )
        messages = [
            {"role": "system", "content": instruction + protocol},
            {
                "role": "user",
                "content": f"{prompt}\nTargets: {targets.model_dump_json()}",
            },
        ]
        report = None
        spend = Spend()
        for attempt in range(1, self.max_attempts + 1):
            try:
                if tool_loop:
                    text, error, trace, round_spend = self._tool_rounds(messages)
                else:
                    text, round_spend = self._single_shot(messages)
                    error, trace = None, ()
            except TRANSIENT_PROVIDER_ERRORS:
                # a provider blip is a FAILED ATTEMPT, never a crash: every
                # caller already has fault-streak machinery for those, and a
                # crank must survive an overloaded API mid-floor
                return AugmentationOutcome(
                    text=None, accepted=False, attempts=attempt,
                    error=ErrorCase.PROVIDER_FAULT,
                    **spend.as_dict(),
                )
            spend.add(round_spend)
            if error is not None:
                return AugmentationOutcome(
                    text=text,
                    accepted=False,
                    attempts=attempt,
                    error=error,
                    attempted_tools=trace,
                    **spend.as_dict(),
                )
            report = self.accept(text, targets)
            if report.passed:
                return AugmentationOutcome(
                    text=text, accepted=True, attempts=attempt, checks=report.checks,
                    **spend.as_dict(),
                )
            failed = [check.target for check in report.checks if not check.passed]
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Local re-measure FAILED these targets: {failed}. "
                        "Revise the text so every target passes."
                    ),
                }
            )
        return AugmentationOutcome(
            text=text,
            accepted=False,
            attempts=self.max_attempts,
            checks=report.checks if report else (),
            **spend.as_dict(),
        )

    @staticmethod
    def _completion(**kwargs) -> tuple[object, float, int]:
        """One `completion()` round-trip, timed and measured — the one place
        cost enters the system, so every hop above reads it from here rather
        than each call site re-deriving it."""
        start = time.monotonic()
        response = completion(**kwargs)
        elapsed = time.monotonic() - start
        tokens = getattr(response, "usage", None)
        return response, elapsed, getattr(tokens, "total_tokens", 0) or 0

    def _single_shot(self, messages: list[dict]) -> tuple[str | None, Spend]:
        response, elapsed, tokens = self._completion(
            model=self.model, messages=messages, max_tokens=1024
        )
        text = _clean(response.choices[0].message.content or "")
        messages.append({"role": "assistant", "content": text})
        spend = Spend()
        spend.add_hop(elapsed, tokens)
        return (text or None), spend

    def _tool_rounds(
        self, messages: list[dict]
    ) -> tuple[str | None, ErrorCase | None, tuple[str, ...], Spend]:
        """Returns (candidate text, error, tool-call trace, spend). `error`
        is set exactly when `text` carries nothing verifiable — a spent round
        budget or a broken protocol, never a failed rewrite. One hop per
        round, regardless of how many tool calls that round makes."""
        trace: list[str] = []
        spend = Spend()
        for _ in range(self.max_rounds):
            response, elapsed, tokens = self._completion(
                model=self.model,
                messages=messages,
                tools=self._tool_schemas,
                tool_choice="auto",
                max_tokens=1024,
            )
            spend.add_hop(elapsed, tokens)
            message = response.choices[0].message
            messages.append(message.model_dump())
            calls = getattr(message, "tool_calls", None)
            if not calls:
                reply = _clean(message.content or "")
                error = None if reply else ErrorCase.NO_TEXT
                return reply, error, tuple(trace), spend
            submitted: str | None = None
            for call in calls:
                trace.append(call.function.name)
                if call.function.name == "submit_text":
                    try:
                        submitted = str(json.loads(call.function.arguments)["text"])
                        result: dict = {"received": True}
                    except Exception as exc:
                        result = {"error": f"{type(exc).__name__}: {exc}"}
                else:
                    result = self._dispatch(call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result),
                    }
                )
            if submitted is not None:
                final = _clean(submitted)
                error = None if final else ErrorCase.EMPTY_SUBMIT
                return final, error, tuple(trace), spend
        return None, ErrorCase.ROUNDS_EXHAUSTED, tuple(trace), spend

    def _dispatch(self, call) -> dict:
        """Run one tool call. Arguments are model-generated — a boundary —
        so failures are returned as the tool result for the LLM to correct
        (e.g. generate_surface wants group-prefixed feature names from
        list_features, while verify counts bare bank names), never raised."""
        try:
            runner = self._runners[call.function.name]
            return runner(**json.loads(call.function.arguments))
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

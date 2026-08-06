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
from enum import StrEnum

from litellm import completion
from pydantic import BaseModel, ConfigDict

from augmentation.config import EngineSettings
from query_taxonomy.features import FeatureExtractor
from taxonomy_generators.tools import build_tools
from taxonomy_generators.verify import TargetCheck, Targets, VerifyReport, verify

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
        for attempt in range(1, self.max_attempts + 1):
            if tool_loop:
                text, error, trace = self._tool_rounds(messages)
            else:
                text, error, trace = self._single_shot(messages), None, ()
            if error is not None:
                return AugmentationOutcome(
                    text=text,
                    accepted=False,
                    attempts=attempt,
                    error=error,
                    attempted_tools=trace,
                )
            report = self.accept(text, targets)
            if report.passed:
                return AugmentationOutcome(
                    text=text, accepted=True, attempts=attempt, checks=report.checks
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
        )

    def _single_shot(self, messages: list[dict]) -> str | None:
        response = completion(model=self.model, messages=messages, max_tokens=1024)
        text = _clean(response.choices[0].message.content or "")
        messages.append({"role": "assistant", "content": text})
        return text or None

    def _tool_rounds(
        self, messages: list[dict]
    ) -> tuple[str | None, ErrorCase | None, tuple[str, ...]]:
        """Returns (candidate text, error, tool-call trace). `error` is set
        exactly when `text` carries nothing verifiable — a spent round budget
        or a broken protocol, never a failed rewrite."""
        trace: list[str] = []
        for _ in range(self.max_rounds):
            response = completion(
                model=self.model,
                messages=messages,
                tools=self._tool_schemas,
                tool_choice="auto",
                max_tokens=1024,
            )
            message = response.choices[0].message
            messages.append(message.model_dump())
            calls = getattr(message, "tool_calls", None)
            if not calls:
                reply = _clean(message.content or "")
                return reply, (None if reply else ErrorCase.NO_TEXT), tuple(trace)
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
                return final, error, tuple(trace)
        return None, ErrorCase.ROUNDS_EXHAUSTED, tuple(trace)

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

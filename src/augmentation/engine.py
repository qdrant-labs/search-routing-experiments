"""The Augmenter — the agentic tool loop inside every LLM operator (d42g).

The LLM gets the taxonomy_generators tool trio (measure/verify included),
the metric explained in the operator's instruction, and the target as a
range. In-loop verify calls steer; they are never the record: acceptance is
`accept()` — a fresh local re-measure of the returned text, the only gate.
This closes the prototype's d2 gap, where the final instructor pass could
reword the text after the last in-loop verify.
"""

from __future__ import annotations

import json

import instructor
from litellm import completion
from pydantic import BaseModel, ConfigDict, Field
from query_taxonomy.features import FeatureExtractor

from taxonomy_generators.tools import build_tools
from taxonomy_generators.verify import TargetCheck, Targets, VerifyReport, verify

DEFAULT_MODEL = "anthropic/claude-haiku-4-5-20251001"


class AugmentedText(BaseModel):
    text: str = Field(description="The final query text, nothing else")


class AugmentationOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    accepted: bool
    attempts: int
    checks: tuple[TargetCheck, ...] = ()


class Augmenter:
    """Bounded tool loop + local-verify acceptance.

    `max_rounds` bounds tool-call rounds per attempt; `max_attempts` bounds
    revise-after-local-failure cycles. Exhaustion returns an unaccepted
    outcome — the caller drops the row, parents are plentiful (d42g).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        seed: int = 0,
        extractor: FeatureExtractor | None = None,
        max_rounds: int = 6,
        max_attempts: int = 2,
    ) -> None:
        self.model = model
        self.max_rounds = max_rounds
        self.max_attempts = max_attempts
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
        ]
        self._structured = instructor.from_litellm(completion)

    def accept(self, text: str, targets: Targets) -> VerifyReport:
        """The only acceptance: re-measure the final text locally, never
        trust the transcript (d2/d42g)."""
        return verify(text, targets, extractor=self._extractor)

    def run(self, instruction: str, prompt: str, targets: Targets) -> AugmentationOutcome:
        messages = [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": f"{prompt}\nTargets: {targets.model_dump_json()}",
            },
        ]
        text, report = "", None
        for attempt in range(1, self.max_attempts + 1):
            self._tool_rounds(messages)
            text = self._final_text(messages)
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

    def _tool_rounds(self, messages: list[dict]) -> None:
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
                return
            for call in calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(self._dispatch(call)),
                    }
                )

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

    def _final_text(self, messages: list[dict]) -> str:
        result = self._structured.chat.completions.create(
            model=self.model,
            response_model=AugmentedText,
            messages=[
                *messages,
                {
                    "role": "user",
                    "content": "Return ONLY the final query text as structured output.",
                },
            ],
        )
        return result.text

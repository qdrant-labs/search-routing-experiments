"""The parameterized tool trio (SPEC d34e), framework-agnostic.

Each ToolSpec carries name + description + JSON schema + callable; adapters
(Claude tool-use, MCP, a scripted loop) consume the same registry without
this package knowing which. The agentic loop is the caller's: augment the
query with surfaces, verify, adjust — this module only supplies the tools.
"""

from collections.abc import Callable
from dataclasses import dataclass
from random import Random
from typing import Any

from query_taxonomy.features import FeatureExtractor

from taxonomy_generators.registry import catalog, generator_for
from taxonomy_generators.verify import Targets, verify


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    run: Callable[..., dict[str, Any]]


def build_tools(
    *, seed: int = 0, extractor: FeatureExtractor | None = None
) -> tuple[ToolSpec, ...]:
    """The trio over one shared seeded RNG and one extractor.

    The default extractor is regex-only; pass
    ``FeatureExtractor(engines=None)`` when verify targets include the
    spaCy signals.
    """
    rng = Random(seed)
    shared_extractor = extractor or FeatureExtractor()

    def list_features() -> dict[str, Any]:
        return {"features": catalog()}

    def generate_surface(
        feature: str, n: int = 1, seed: int | None = None
    ) -> dict[str, Any]:
        local = rng if seed is None else Random(seed)
        return {
            "feature": feature,
            "surfaces": generator_for(feature).sample(local, n),
        }

    def verify_text(text: str, targets: dict[str, Any]) -> dict[str, Any]:
        report = verify(
            text, Targets.model_validate(targets), extractor=shared_extractor
        )
        return report.model_dump()

    return (
        ToolSpec(
            name="list_features",
            description=(
                "Catalog of generable taxonomy features: name, group,"
                " ambiguity tier, one-line description. Call this first to"
                " discover valid `feature` values for generate_surface;"
                " `-like` features produce shape guesses, not certified"
                " formats."
            ),
            input_schema={"type": "object", "properties": {}},
            run=list_features,
        ),
        ToolSpec(
            name="generate_surface",
            description=(
                "Produce n valid surfaces — text snippets exhibiting one"
                " taxonomy feature (a UUID, a politeness phrase, a `!=` cue"
                " token). Surfaces are grounding-blind: weave them into the"
                " query yourself, then check the result with verify."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "feature": {
                        "type": "string",
                        "description": "Feature name from list_features.",
                    },
                    "n": {"type": "integer", "minimum": 1, "default": 1},
                    "seed": {
                        "type": "integer",
                        "description": "Optional per-call RNG seed for"
                        " reproducible surfaces.",
                    },
                },
                "required": ["feature"],
            },
            run=generate_surface,
        ),
        ToolSpec(
            name="verify",
            description=(
                "Re-measure a text with the detection banks and check it"
                " against span-count and stat-range targets. Returns overall"
                " PASS/FAIL plus per-target measured values — iterate until"
                " it passes."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "targets": Targets.model_json_schema(),
                },
                "required": ["text", "targets"],
            },
            run=verify_text,
        ),
    )

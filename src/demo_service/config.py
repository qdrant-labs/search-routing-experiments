"""Everything the demo pins ahead of time: the document shelf, both behaviors,
the service URLs, and the budget/deadline ceilings. Nothing downstream builds a
path or a URL from a literal — it all comes from here.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from augmentation.config import EngineSettings

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "home-depot"
REPLAY_DIR = Path(__file__).resolve().parent / "replays"
EXPORT_DIR = DATA_DIR / "demo_tests"

KNOWN_PRICING: dict[str, tuple[float, float]] = {
    "anthropic/claude-haiku-4-5-20251001": (1.0, 5.0),
}
"""(usd_per_mtok_in, usd_per_mtok_out), verified against the provider's own
pricing page — not `EngineSettings`' defaults, which the demo must not trust
blindly. Preflight asserts the configured engine matches an entry here
before any paid call; an unlisted model refuses preflight rather than
guessing its price."""


class Behavior(BaseModel):
    """One of the audience-chosen query behaviors."""

    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    description: str
    kind: str
    """"identifier": targets come from the chosen document's own detected
    identifier — never an injected one. "cell": planned via /augment/cell.
    "messy": terse real-traffic register — deterministic typo damage plus a
    measured non-English language, checked by the runner's own extractor."""
    cell: str | None = None
    """`/augment/cell` name, for kind="cell"."""


class SourceDoc(BaseModel):
    """One shelf entry: a corpus document plus the identifier the taxonomy
    detected in it during selection (None = conversational-only)."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    identifier_surface: str | None = None
    identifier_feature: str | None = None
    """BARE bank name: verify() keys spans by bank, not by the generator
    catalog's "group:name" spelling (measured: the qualified form counts 0)."""


CANDIDATE_DOCS: tuple[SourceDoc, ...] = (
    SourceDoc(doc_id="133364", identifier_surface="PC5000C", identifier_feature="sku"),
    SourceDoc(doc_id="150218", identifier_surface="MCP44E",
              identifier_feature="booking_reference_like"),
    SourceDoc(doc_id="165394", identifier_surface="M8",
              identifier_feature="astronomical_designation"),
    SourceDoc(doc_id="188512", identifier_surface="400 mm",
              identifier_feature="value_with_unit"),
    SourceDoc(doc_id="147159"),
    SourceDoc(doc_id="120627"),
)
"""The shelf, curated by scanning the corpus with the FeatureExtractor: four
docs whose identifier fires INSIDE a query string too (verified per doc), and
two clean docs that support only the conversational behavior. `_like` and
shape-guess bank names are shown as-is in the UI — the ambiguity tier is part
of the story, not something to hide."""

DEFAULT_DOC_ID = "133364"

STYLE_EXEMPLARS: tuple[str, ...] = (
    "511 impregnator sealer",
    "body spray polished nickel",
    "chainsaw chain sharpener",
    "rain drop emitter",
    "ivory white exterior paint",
)
"""Short, real Home Depot queries — style priming only, never grounding text."""

BEHAVIOR_IDENTIFIER = Behavior(
    # id kept as "identifier" for replay/export continuity; the card offers
    # structured VARIANTS per document — exact or mistyped identifier where the
    # document carries one, shape features (exclusion, operator syntax)
    # everywhere. Content still only ever comes from the document; shape is
    # form, not content, so it cannot manufacture relevance.
    id="identifier",
    kind="identifier",
    label="Structured query",
    description=(
        "The query carries a measurable structure — the document's own "
        "identifier (exact, or mistyped the way a real customer would), an "
        "exclusion, or search-operator syntax."
    ),
)

BEHAVIOR_CONVERSATIONAL = Behavior(
    id="conversational",
    kind="cell",
    label="Conversational request",
    description=(
        "How people type to AI assistants today — chat register, courtesy "
        "padding, the need buried mid-sentence."
    ),
    cell="conversational_courtesy_wrapper",
)

BEHAVIOR_REAL_TRAFFIC = Behavior(
    id="real_traffic",
    kind="messy",
    label="Real traffic",
    description=(
        "The head of the real query distribution — terse fragments, a typo, "
        "sometimes not even in English. The gate demands the mess be "
        "measurable, not asserted."
    ),
)

BEHAVIORS: dict[str, Behavior] = {
    b.id: b
    for b in (BEHAVIOR_IDENTIFIER, BEHAVIOR_CONVERSATIONAL, BEHAVIOR_REAL_TRAFFIC)
}


class DemoConfig(BaseModel):
    """Every knob the runner reads. Construct once at app startup."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    pipeline_service_url: str = Field(
        default_factory=lambda: os.environ.get(
            "PIPELINE_SERVICE_URL", "http://127.0.0.1:8000"
        )
    )
    qdrant_url: str | None = Field(
        default_factory=lambda: os.environ.get("QDRANT_CLOUD_URL")
        or os.environ.get("QDRANT_URL")
    )
    qdrant_api_key: str | None = Field(
        default_factory=lambda: os.environ.get("QDRANT_CLOUD_API_KEY")
        or os.environ.get("QDRANT_API_KEY")
    )
    collection: str = "home-depot"
    fetch_limit: int = 20
    """Matches the twice-calibration retrieval pass, not fusion.py's default of 50."""

    engine: EngineSettings = Field(
        default_factory=lambda: EngineSettings(max_spend_usd=1.0)
    )
    """Haiku-4-5 at $1/$5 per Mtok (the augmentation default) with the demo's
    own $1.00 total ceiling, shared by every call in one run."""
    max_llm_calls: int = 2
    """One generate + at most one repair. Enforced as a hard count, not just
    the dollar ceiling — a very cheap runaway loop must not slip through."""
    generate_deadline_s: float = 20.0
    retrieve_deadline_s: float = 15.0

    messy_max_words: int = 15
    """Real traffic skews far shorter (median 3), but the distribution has a
    tail — the gate bounds it rather than forcing the mode."""
    corruption_seed: int = 7
    """Seed for the deterministic QwertyTypo damage; a handful of successors
    are tried when the seeded pick finds no eligible word."""

    data_dir: Path = DATA_DIR
    replay_dir: Path = REPLAY_DIR
    export_dir: Path = EXPORT_DIR

    docs: tuple[SourceDoc, ...] = CANDIDATE_DOCS
    default_doc_id: str = DEFAULT_DOC_ID
    style_exemplars: tuple[str, ...] = STYLE_EXEMPLARS

    def doc_entry(self, doc_id: str | None) -> SourceDoc:
        wanted = doc_id or self.default_doc_id
        for doc in self.docs:
            if doc.doc_id == wanted:
                return doc
        raise KeyError(wanted)

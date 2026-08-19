"""Augmentation defaults (SPEC d42) — one file holding every knob and every
declaration table the loop reads. Nothing here is compiled into an operator.

Two kinds of value live here, and the difference is load-bearing:

- **Knobs** — paths, model, attempt bounds, seed, pilot size. Operational
  defaults; override freely at construction.
- **Declaration tables** — `decorations`, `formal_floors`, `stats`. Policy a
  human DECLARES (d42c default-deny): each entry is a meaning-preservation
  claim whose rows bank as feature-stock until its pilot passes. Adding one
  is a decision, not a tweak — but it is an INPUT (extend the table here, or
  hand a different one to the operator), never a constraint that needs new
  code to move. A new target dataset that starves an undeclared band costs
  one entry plus its pilot.

Paths resolve from the package rather than the process cwd — the convention
`composition/cellfill.py` and `labels.py` already follow — so a script run
from the repo root and a notebook run from `src/` read the same artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

_DEFAULT_DECORATIONS = {
    "greeting": "a natural greeting (e.g. 'hi there,', 'good morning,')",
    "interjection": "a natural interjection (e.g. 'hmm,', 'oh,', 'ugh,')",
    "politeness": "a politeness phrase (e.g. 'please', 'could you kindly')",
}

_DEFAULT_FORMAL_FLOORS = ("logical:math_expression", "logical:code_fragment")


class StatDirection(StrEnum):
    """Which way along an axis a rewrite moves the scalar."""

    UP = "up"
    DOWN = "down"


class StatEntry(BaseModel):
    """One licensed stat move: this axis, this direction, on this claim."""

    model_config = ConfigDict(frozen=True)

    axis: str
    """A `CoverageAxis.title` from `composition.floors.STAT_AXES`."""
    direction: StatDirection
    rationale: str
    """The meaning-preservation claim the pilot judges."""


_DEFAULT_STAT_ENTRIES = (
    StatEntry(
        axis="length_words",
        direction=StatDirection.UP,
        rationale=(
            "expansion adds need-neutral padding; deletion-free — piloted "
            "per d42h before credit"
        ),
    ),
    StatEntry(
        axis="length_words",
        direction=StatDirection.DOWN,
        rationale=(
            "compression drops words, so meaning survives only against the "
            "corpus: the cut keeps the terms that keep the parent's gold "
            "document answering, plus any minted surface verbatim. Where no "
            "cut can preserve that, the row needs a document of its own and "
            "belongs to construction, not augmentation — the audit judges "
            "exactly that boundary before credit"
        ),
    ),
)


class StatDeclarations(BaseModel):
    """The per-(axis, direction) declaration table (d42d). StatRewrite's
    machinery is axis-generic; this table is what licenses it to act.
    Undeclared directions are never selected for — default-deny."""

    model_config = ConfigDict(frozen=True)

    entries: tuple[StatEntry, ...] = Field(
        default=_DEFAULT_STAT_ENTRIES,
        description=(
            "One entry per licensed (axis, direction). Only bands with "
            "measured demand are declared; each needs its own pilot."
        ),
    )

    def directions(self, axis: str) -> frozenset[StatDirection]:
        """The directions declared for an axis — empty means default-deny."""
        return frozenset(
            entry.direction for entry in self.entries if entry.axis == axis
        )

    def with_entry(
        self, axis: str, direction: StatDirection, rationale: str
    ) -> StatDeclarations:
        """This table plus one licensed move — how a newly hungry band opens
        without touching the package."""
        entry = StatEntry(axis=axis, direction=direction, rationale=rationale)
        return self.model_copy(update={"entries": (*self.entries, entry)})


class AugmentationPaths(BaseModel):
    """Every artifact the loop touches, derived from one root."""

    model_config = ConfigDict(frozen=True)

    data_dir: Path = Field(
        default=_DATA_ROOT,
        description="Data root holding the lane dirs, composition and pool.",
    )

    @property
    def augmentation_dir(self) -> Path:
        return self.data_dir / "augmentation"

    @property
    def pool(self) -> Path:
        return self.augmentation_dir / "pool.parquet"

    @property
    def qrels(self) -> Path:
        return self.augmentation_dir / "qrels.parquet"

    @property
    def order_sheet(self) -> Path:
        return self.data_dir / "composition" / "order_sheet.parquet"

    @property
    def cell_order_sheet(self) -> Path:
        """The cell fill's shortfalls — cell-named floors, so the loop reads
        it by passing `sheet_path`; the slice sheet above stays the default
        until its artifact retires."""
        return self.data_dir / "composition" / "cell_order_sheet.parquet"

    @property
    def catalog(self) -> Path:
        return self.data_dir / "feature_table" / "catalog.parquet"

    @property
    def corruption_census(self) -> Path:
        """Per-lane natural corruption rates — what makes a degree a delta
        over the lane rather than a hand-set number."""
        return self.data_dir / "corruption_census.parquet"

    def lane_qrels(self, lane: str) -> Path:
        return self.data_dir / lane / "qrels.parquet"

    def lane_corpus(self, lane: str) -> Path:
        return self.data_dir / lane / "corpus.parquet"

    def lane_surfaces(self, lane: str) -> Path:
        return self.data_dir / lane / "surfaces.parquet"


class EngineSettings(BaseModel):
    """What the Augmenter spends per candidate."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(
        default="anthropic/claude-haiku-4-5-20251001",
        description="litellm model id — the weaver, not a judge (d42g).",
    )
    max_attempts: int = Field(
        default=2,
        description="Revise-after-local-failure cycles before the row drops.",
    )
    max_rounds: int = Field(
        default=6,
        description="Tool-call rounds per attempt (tool-loop mode only).",
    )


class AugmentationConfig(BaseModel):
    """The whole input surface of the loop. `default_operators()` projects
    it onto operator constructors — each operator receives only the tables
    and paths it reads."""

    model_config = ConfigDict(frozen=True)

    paths: AugmentationPaths = Field(default_factory=AugmentationPaths)
    engine: EngineSettings = Field(default_factory=EngineSettings)
    stats: StatDeclarations = Field(default_factory=StatDeclarations)
    decorations: Mapping[str, str] = Field(
        default=_DEFAULT_DECORATIONS,
        description=(
            "Marker bank -> what to weave. Register decorations only: other "
            "marker banks (acronym, ...) are detections, not weavable "
            "filler, and Decorate's meaning claim covers exactly these."
        ),
    )
    formal_floors: tuple[str, ...] = Field(
        default=_DEFAULT_FORMAL_FLOORS,
        description=(
            "Parents carrying embedded formal content get help-request "
            "framing instead of a bolted-on phrase (arch-validator "
            "2026-07-30: the observed absurdity was the weave, not the "
            "parent class — 'could someone help me with: <problem>' is "
            "attested register). Hard exclusion was reverted; it returns "
            "only as a computed rule if the d34b audit measures a high "
            "failure rate on these parents."
        ),
    )
    seed: int = Field(
        default=0,
        description="One determinism knob: parent order and tool sampling.",
    )
    pilot_n: int = Field(
        default=30,
        description=(
            "Audit-sample size for gated floors — sized by the human who "
            "reads the sample, not derived."
        ),
    )
    first_generation_only: bool = Field(
        default=True,
        description=(
            "d43 review: augmented rows are never parents (no "
            "second-generation drift) unless deliberately switched off."
        ),
    )

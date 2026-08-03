"""The operator families (d42d). Decorate is gate-free;
OperatorSyntaxRewrite and StatRewrite produce feature-stock until their
declaration audits pass (d42h); Inject and Corrupt are pass 4.

Every table an operator selects against — decorations, formal floors,
licensed stat moves — arrives as an `AugmentationConfig` input; each
operator keeps only the slice it reads. Eligibility joins the catalog when
a rule needs scalars or span counts the selection does not carry (d43e)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from random import Random
from typing import ClassVar

import numpy as np
import pandas as pd

from augmentation.config import (
    AugmentationConfig,
    AugmentationPaths,
    StatDeclarations,
    StatDirection,
)
from augmentation.core import (
    AnswerKeyPath,
    CreditGate,
    Declaration,
    Grounding,
    Operator,
)
from augmentation.supply import SupplyIndex, lane_dirs
from composition.catalog_axes import StatAxis, stat_column
from composition.floors import STAT_AXES, identifier_floor_key
from query_taxonomy.features import FeatureExtractor
from query_taxonomy.taxonomy import FeatureGroup
from taxonomy_generators.registry import generator_for
from taxonomy_generators.verify import SpanTarget, StatTarget, Targets

_WORD = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=1)
def _regex_extractor() -> FeatureExtractor:
    """Shared regex-only extractor for structural checks — spans are all
    regex-tier, and so is length_words."""
    return FeatureExtractor()


@lru_cache(maxsize=4)
def _catalog(path: Path) -> pd.DataFrame:
    """The feature table, loaded once per path: identity + span columns +
    scalars (d43e — eligibility is a join, never a new extraction pass)."""
    return pd.read_parquet(path)


def _span_total(catalog: pd.DataFrame) -> pd.Series:
    span_prefixes = (
        "structured_identifiers.", "sentence_markers.", "logical_structures.",
    )
    columns = [c for c in catalog.columns if c.startswith(span_prefixes)]
    return catalog[columns].sum(axis=1)


class DecorateOperator(Operator):
    """Weave a register marker into the query (d40d: politeness-class,
    meaning-preserving — parent qrels inherit, no grounding, no gate)."""

    declaration: ClassVar[Declaration] = Declaration(
        operator="decorate",
        floors="marker:greeting | marker:interjection | marker:politeness",
        grounding=Grounding.NONE,
        answer_key=AnswerKeyPath.INHERIT,
        meaning_preserved=True,
        verifiable_by="target marker span present on local re-measure",
        credit_gate=CreditGate.NONE,
        tool_loop=False,   # the model hits marker targets blind — d42n
    )

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        config = config or AugmentationConfig()
        super().__init__(config)
        self._decorations: Mapping[str, str] = config.decorations
        self._formal_floors = config.formal_floors
        self._rng = Random(config.seed)
        self._order_seed = config.seed

    @staticmethod
    def marker(floor: str) -> str:
        return floor.removeprefix("marker:")

    def serves(self, floor: str) -> bool:
        return (
            floor.startswith("marker:") and self.marker(floor) in self._decorations
        )

    def eligible(self, selection: pd.DataFrame, floor: str) -> pd.DataFrame:
        """Checkable parents not already carrying the marker — a filter on
        the selection's own `floors` column, no bank run (d42e). Returned
        in preference order (seeded shuffle — dataset diversity); the loop
        consumes in order."""
        pool = self.parent_pool(selection)
        lacks = ~pool["floors"].map(lambda floors: floor in floors)
        return pool[lacks & pool["checkable"]].sample(
            frac=1.0, random_state=self._order_seed
        )

    def instruction(self, floor: str, parent: pd.Series) -> str:
        """Per-call seeded vocabulary examples — variety is supplied by the
        bank's own phrase list, never left to the LLM's favorite opener.
        Parent-aware: formal-content parents get help-request framing."""
        marker = self.marker(floor)
        sampled = generator_for(f"sentence_markers:{marker}").sample(self._rng, 4)
        examples = ", ".join(repr(s) for s in dict.fromkeys(sampled))
        framing = (
            (
                "This query contains formal content (math or code): frame the "
                "decoration as a real person bringing the problem somewhere "
                "for help — like a forum post — never a phrase bolted onto a "
                "bare statement. "
            )
            if any(f in parent["floors"] for f in self._formal_floors)
            else ""
        )
        return (
            "Rewrite the user's search query by weaving in "
            f"{self._decorations[marker]}. Vocabulary inspirations (adapt freely): "
            f"{examples}. Pick a phrasing that fits the query's tone and "
            "world, and vary it — never default to one stock opener; it may "
            f"sit at the start, middle, or end. {framing}Keep every content "
            "word and the meaning unchanged. Add no other information: no "
            "names, numbers, dates, or identifiers."
        )

    def targets(self, floor: str, parent: pd.Series) -> Targets:
        return Targets(spans=(SpanTarget(feature=self.marker(floor), min_count=1),))

    def structural(self, parent: pd.Series, text: str) -> list[str]:
        """Declared: no parent-relative machine check. Decoration is
        additive; its meaning claim rests on the politeness-class
        declaration (d40d) and the d34b audit of before/after pairs —
        the observed restructuring cases are exactly what that audit
        rules on, and a content-word check lands here if it demands one."""
        return []


class OperatorSyntaxRewrite(Operator):
    """Restructure a query's EXISTING coordination into keyword-search
    dialect — uppercase AND/OR/NOT (d42d: meaning-preserving RESTRICTED).
    May drop function words; may not add content words — the addition
    side is machine-blocked by structural(), the deletion side is what
    the declaration audit judges."""

    declaration: ClassVar[Declaration] = Declaration(
        operator="operator_syntax_rewrite",
        floors="logical:operator_syntax",
        grounding=Grounding.NONE,
        answer_key=AnswerKeyPath.INHERIT,
        meaning_preserved=True,
        verifiable_by=(
            "operator_syntax span present + no new content tokens beyond "
            "AND/OR/NOT (structural)"
        ),
        credit_gate=CreditGate.DECLARATION_AUDIT,
        tool_loop=False,
    )

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        config = config or AugmentationConfig()
        super().__init__(config)
        self._catalog_path = config.paths.catalog
        self._order_seed = config.seed

    def serves(self, floor: str) -> bool:
        return floor == "logical:operator_syntax"

    def eligible(self, selection: pd.DataFrame, floor: str) -> pd.DataFrame:
        """Parents with coordination to restructure (widest_list_size >= 2
        — a stat rule, d42e) that do not already speak operator dialect."""
        catalog = _catalog(self._catalog_path)
        widest = catalog[stat_column(catalog, "widest_list_size")]
        has_syntax = catalog.get(
            "logical_structures.operator_syntax",
            pd.Series(0.0, index=catalog.index),
        )
        marks = catalog.assign(__coord=widest, __syntax=has_syntax)[
            ["dataset", "query_id", "__coord", "__syntax"]
        ]
        joined = self.parent_pool(selection).merge(
            marks, on=["dataset", "query_id"], how="left"
        )
        mask = (
            joined["checkable"]
            & (joined["__coord"] >= 2)
            & (joined["__syntax"] == 0)
        )
        return (
            joined[mask]
            .drop(columns=["__coord", "__syntax"])
            .sample(frac=1.0, random_state=self._order_seed)
        )

    def instruction(self, floor: str, parent: pd.Series) -> str:
        return (
            "Rewrite the user's search query into keyword-search dialect: "
            "restructure its EXISTING coordination using uppercase AND / OR "
            "/ NOT between the existing terms. You may drop small function "
            "words (how, to, the, a). You may NOT add any new content word, "
            "name, number, or fact — only the words already present plus "
            "the uppercase operators."
        )

    def targets(self, floor: str, parent: pd.Series) -> Targets:
        return Targets(spans=(SpanTarget(feature="operator_syntax", min_count=1),))

    def structural(self, parent: pd.Series, text: str) -> list[str]:
        parent_tokens = set(_WORD.findall(str(parent["query"]).lower()))
        child_tokens = set(_WORD.findall(text.lower()))
        new = child_tokens - parent_tokens - {"and", "or", "not"}
        if new:
            return [f"new content tokens: {sorted(new)}"]
        return []


class StatRewrite(Operator):
    """One generic operator for every stat band, any axis, any declared
    direction (d42d) — the axis machinery is generic, `config.stats` is what
    licenses a move. The target is a RANGE (the band); the engine's tool
    loop lets the model measure until it lands (d42g)."""

    declaration: ClassVar[Declaration] = Declaration(
        operator="stat_rewrite",
        floors="any <axis>:<band> whose axis has a declared direction",
        grounding=Grounding.NONE,
        answer_key=AnswerKeyPath.INHERIT,
        meaning_preserved=True,
        verifiable_by=(
            "scalar lands in the band + span profile unchanged "
            "(zero-span parents stay zero-span, structural)"
        ),
        credit_gate=CreditGate.DECLARATION_AUDIT,
        tool_loop=True,
    )

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        config = config or AugmentationConfig()
        super().__init__(config)
        self._stats: StatDeclarations = config.stats
        self._catalog_path = config.paths.catalog

    @staticmethod
    def _band(floor: str) -> tuple[StatAxis, float, float] | None:
        for axis in STAT_AXES:
            for index, label in enumerate(axis.labels):
                key = f"{axis.title}:{str(label).replace(chr(10), ' ')}"
                if key == floor:
                    return axis, axis.edges[index], axis.edges[index + 1]
        return None

    def serves(self, floor: str) -> bool:
        band = self._band(floor)
        return band is not None and bool(self._stats.directions(band[0].title))

    def eligible(self, selection: pd.DataFrame, floor: str) -> pd.DataFrame:
        """Zero-span checkable parents on the DECLARED side of the band,
        nearest first (smallest move = least meaning risk, d42d). The
        current value rides along as `stat_value` for the instruction."""
        axis, low, high = self._band(floor)
        catalog = _catalog(self._catalog_path)
        marks = catalog.assign(
            __value=catalog[stat_column(catalog, axis.stat)],
            __spans=_span_total(catalog),
        )[["dataset", "query_id", "__value", "__spans"]]
        joined = self.parent_pool(selection).merge(
            marks, on=["dataset", "query_id"], how="left"
        )
        declared = self._stats.directions(axis.title)
        movable = (
            (StatDirection.UP in declared) & (joined["__value"] < low)
        ) | (
            (StatDirection.DOWN in declared) & (joined["__value"] >= high)
        )
        mask = joined["checkable"] & (joined["__spans"] == 0) & movable
        out = joined[mask].assign(stat_value=joined["__value"])
        distance = np.where(
            out["__value"] < low, low - out["__value"], out["__value"] - high
        )
        return (
            out.assign(__distance=distance)
            .sort_values("__distance", kind="stable")
            .drop(columns=["__value", "__spans", "__distance"])
        )

    def instruction(self, floor: str, parent: pd.Series) -> str:
        axis, low, high = self._band(floor)
        band = (
            f"at least {low:g}"
            if high == float("inf")
            else f"between {low:g} and {high:g} (exclusive)"
        )
        current = parent.get("stat_value")
        current_note = (
            f" It currently measures {current:g}." if current is not None else ""
        )
        return (
            f"Rewrite the user's query so that its {axis.stat} lands "
            f"{band}.{current_note} Preserve the information need exactly: "
            "expand only with need-neutral elaboration, restatement, or "
            "context the answer does not depend on. Add NO new facts, "
            "names, numbers, dates, identifiers — and no greetings or "
            "politeness phrases (any of those changes the query's feature "
            "profile and fails the check). Use the verify tool to measure, "
            "iterate until the target passes."
        )

    def targets(self, floor: str, parent: pd.Series) -> Targets:
        axis, low, high = self._band(floor)
        return Targets(stats=(StatTarget(
            stat=axis.stat,
            min_value=float(low),
            max_value=None if high == float("inf") else float(high) - 1e-9,
        ),))

    def structural(self, parent: pd.Series, text: str) -> list[str]:
        """Zero-span parents must stay zero-span — smuggled constraints
        and register markers are span-visible (d43b)."""
        found = _regex_extractor().resolve(text)
        gained = [
            f"{group.value}.{name}"
            for group, by_type in found.spans.items()
            for name, spans in by_type.items()
            if spans
        ]
        if gained:
            return [f"child gained spans: {gained}"]
        return []


class InjectOperator(Operator):
    """Weave a DOC-COPIED identifier surface into the query (d42d/f).
    The query narrows by design — the answer key is minted against the
    grounding doc the surface came from, never inherited. One surface per
    row. Credit is gated by the d40e coherence pilot."""

    declaration: ClassVar[Declaration] = Declaration(
        operator="inject",
        floors="id:<domain> and id:<general-bank> identifier floors",
        grounding=Grounding.DOC_COPIED,
        answer_key=AnswerKeyPath.MINTED,
        meaning_preserved=False,
        verifiable_by=(
            "target bank span present + copied surface literally in the "
            "text + no other identifier floor gained (structural)"
        ),
        credit_gate=CreditGate.COHERENCE_GATE,
        tool_loop=False,
    )

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        config = config or AugmentationConfig()
        super().__init__(config)
        self._paths: AugmentationPaths = config.paths
        self._order_seed = config.seed
        self._supply = SupplyIndex(config.paths)

    def serves(self, floor: str) -> bool:
        return floor.startswith("id:")

    def eligible(self, selection: pd.DataFrame, floor: str) -> pd.DataFrame:
        """Rung-1 pairs (d43f): checkable parents lacking the floor whose
        own gold doc carries a surface of it — joined per lane from the
        supply index + lane qrels. One deterministic (doc, surface) per
        parent rides along as `grounding_doc_id` / `surface` / `bank`."""
        pool = self.parent_pool(selection)
        frames: list[pd.DataFrame] = []
        for key, lane in lane_dirs().items():
            surfaces = self._supply.load(lane)
            qrels_path = self._paths.lane_qrels(lane)
            if surfaces.empty or not qrels_path.exists():
                continue
            floor_surfaces = surfaces[surfaces["floor"] == floor]
            if floor_surfaces.empty:
                continue
            lane_pool = pool[(pool["dataset"] == key) & pool["checkable"]]
            lane_pool = lane_pool[
                ~lane_pool["floors"].map(lambda floors: floor in floors)
            ]
            if lane_pool.empty:
                continue
            qrels = pd.read_parquet(qrels_path)
            qrels = qrels[qrels["relevance"] >= 1].astype(
                {"query_id": str, "doc_id": str}
            )
            # one deterministic surface per doc, then one doc per parent
            per_doc = (
                floor_surfaces.astype({"doc_id": str})
                .sort_values(["doc_id", "bank", "surface"], kind="stable")
                .drop_duplicates("doc_id")
            )
            pairs = (
                qrels.merge(per_doc, on="doc_id")
                .sort_values(["query_id", "doc_id"], kind="stable")
                .drop_duplicates("query_id")
                .rename(columns={"doc_id": "grounding_doc_id"})
            )
            matched = lane_pool.assign(
                query_id=lane_pool["query_id"].astype(str)
            ).merge(
                pairs[["query_id", "grounding_doc_id", "bank", "surface"]],
                on="query_id",
            )
            if not matched.empty:
                frames.append(matched)
        if not frames:
            return pool.iloc[0:0]
        return pd.concat(frames, ignore_index=True).sample(
            frac=1.0, random_state=self._order_seed
        )

    def instruction(self, floor: str, parent: pd.Series) -> str:
        return (
            "Weave the exact text "
            f"{str(parent['surface'])!r} into the user's search query as a "
            "natural constraint or reference. The query may narrow — it no "
            "longer has to mean exactly what it meant — but it must read as "
            "one coherent request a real person would type, on the same "
            "topic. Insert ONLY this one surface, character for character: "
            "add no other identifiers, names, numbers, or dates, and do not "
            "alter the inserted text."
        )

    def targets(self, floor: str, parent: pd.Series) -> Targets:
        return Targets(spans=(SpanTarget(feature=str(parent["bank"]), min_count=1),))

    def structural(self, parent: pd.Series, text: str) -> list[str]:
        problems: list[str] = []
        surface = str(parent["surface"])
        if surface not in text:
            problems.append(f"copied surface {surface!r} not literally present")
        found = _regex_extractor().resolve(
            text, groups=[FeatureGroup.STRUCTURED_IDENTIFIERS]
        ).spans.get(FeatureGroup.STRUCTURED_IDENTIFIERS, {})
        child_floors = {identifier_floor_key(bank) for bank in found}
        parent_floors = {f for f in parent["floors"] if f.startswith("id:")}
        target_floor = identifier_floor_key(str(parent["bank"]))
        extra = child_floors - parent_floors - {target_floor}
        if extra:
            problems.append(f"gained other identifier floors: {sorted(extra)}")
        return problems

    def candidate(self, parent, floor, text, attempts):
        base = super().candidate(parent, floor, text, attempts)
        return base.model_copy(
            update={"grounding_doc_id": str(parent["grounding_doc_id"])}
        )


OPERATOR_FAMILIES: tuple[type[Operator], ...] = (
    DecorateOperator,
    OperatorSyntaxRewrite,
    StatRewrite,
    InjectOperator,
)
"""Registry order = dispatch order: the first family that `serves()` the
floor wins."""


def default_operators(
    config: AugmentationConfig | None = None,
) -> tuple[Operator, ...]:
    """Every family, wired from one config — the projection point where
    declaration tables and paths reach the operators."""
    config = config or AugmentationConfig()
    return tuple(family(config) for family in OPERATOR_FAMILIES)


def operator_for(
    floor: str, operators: tuple[Operator, ...] | None = None
) -> Operator | None:
    """Dispatch by floor key (d42d) — `id:` / `marker:` / `logical:` /
    stat-axis prefixes, not the loop's grounding branches. Defaults to the
    default-config families; a floor no family serves returns None."""
    return next(
        (op for op in (operators or default_operators()) if op.serves(floor)),
        None,
    )

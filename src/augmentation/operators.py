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
    SurfaceOrigin,
    Operator,
)
from augmentation.dispatch import WORD_AXES
from augmentation.supply import SupplyIndex, lane_dirs
from composition.catalog_axes import StatAxis, stat_column
from composition.cells import CELL_TO_BANKS, CELLS_BY_NAME, AxisBand
from composition.floors import STAT_AXES, identifier_floor_key
from query_taxonomy.core import FeatureSpan
from query_taxonomy.features import FeatureExtractor
from query_taxonomy.taxonomy import FeatureGroup
from taxonomy_generators.registry import generator_for
from taxonomy_generators.verify import SpanTarget, StatTarget, Targets

_WORD = re.compile(r"[a-z0-9]+")
_WORD_BOUNDED_NOT = re.compile(r"\bNOT\b")


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


def _span_names(text: str) -> set[str]:
    """Every span feature the text exhibits, bare-named — the vocabulary
    `SpanTarget.feature` and the structural checks share."""
    features = _regex_extractor().resolve(text)
    return {
        name
        for by_type in features.spans.values()
        for name, spans in by_type.items()
        if spans
    }


def _spans_by_name(text: str) -> dict[str, list[FeatureSpan]]:
    """Every span this text exhibits, keyed by bare name, WITH position —
    `_span_names` throws position away, but a check that wants to know
    whether a gained span sits inside an already-authorised literal needs it."""
    result: dict[str, list[FeatureSpan]] = {}
    for by_type in _regex_extractor().resolve(text).spans.values():
        for name, spans in by_type.items():
            if spans:
                result.setdefault(name, []).extend(spans)
    return result


def _explained_by_surfaces(
    spans: list[FeatureSpan], text: str, surfaces: tuple[str, ...]
) -> bool:
    """Whether every occurrence of a gained span sits inside some literal
    Inject was already authorised to insert verbatim (d52d covers the span
    it was TARGETED for; this covers a span a different bank names for the
    SAME characters — the model chose none of it, so it is not smuggled)."""
    ranges = [
        (m.start(), m.end())
        for surface in surfaces
        for m in re.finditer(re.escape(str(surface)), text)
    ]
    return all(
        any(lo <= span.start and span.end <= hi for lo, hi in ranges)
        for span in spans
    )


def _span_total(catalog: pd.DataFrame) -> pd.Series:
    span_prefixes = (
        "structured_identifiers.", "sentence_markers.", "logical_structures.",
    )
    columns = [c for c in catalog.columns if c.startswith(span_prefixes)]
    return catalog[columns].sum(axis=1)


class DecorateOperator(Operator):
    """Weave a register marker into the query (d40d: politeness-class,
    meaning-preserving — parent qrels inherit, no surface_origin, no gate)."""

    declaration: ClassVar[Declaration] = Declaration(
        operator="decorate",
        floors="marker:greeting | marker:interjection | marker:politeness",
        surface_origin=SurfaceOrigin.NONE,
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

    def marker(self, floor: str) -> str:
        """The decoration this demand names: a marker: floor names it
        outright, a cell resolves to one required decoration — any single
        one satisfies an `any_of` cell."""
        if floor in CELLS_BY_NAME:
            return min(CELL_TO_BANKS[floor] & set(self._decorations))
        return floor.removeprefix("marker:")

    def mints(self, band: AxisBand) -> bool:
        """Only the registered decorations — other marker banks are
        detections, not weavable filler."""
        return band.demands_presence and band.member in self._decorations

    def serves(self, floor: str) -> bool:
        return floor.startswith("marker:") and self.marker(floor) in self._decorations

    def eligible(self, selection: pd.DataFrame, floor: str) -> pd.DataFrame:
        """Checkable parents the demand does not already cover — a filter on
        the selection's own columns, no bank run (d42e). Returned in
        preference order (seeded shuffle — dataset diversity); the loop
        consumes in order."""
        pool = self.parent_pool(selection)
        pool = self.unsatisfied(pool, floor)
        return pool[pool["checkable"]].sample(
            frac=1.0, random_state=self._order_seed
        )

    def instruction(
        self, floor: str, parent: pd.Series, requirement: tuple = ()
    ) -> str:
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
            "word and the meaning unchanged."
        )

    def targets(
        self, floor: str, parent: pd.Series, requirement: tuple = ()
    ) -> Targets:
        return Targets(spans=(SpanTarget(feature=self.marker(floor), min_count=1),))

    def structural(
        self, parent: pd.Series, text: str, targets: Targets
    ) -> list[str]:
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
        surface_origin=SurfaceOrigin.NONE,
        answer_key=AnswerKeyPath.INHERIT,
        meaning_preserved=True,
        verifiable_by=(
            "operator_syntax span present + no new content tokens beyond "
            "AND/OR + no NOT emitted (structural). The AND/OR half only: "
            "exclusion moves relevance, so the parent's qrels would not carry"
        ),
        credit_gate=CreditGate.DECLARATION_AUDIT,
        tool_loop=False,
    )

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        config = config or AugmentationConfig()
        super().__init__(config)
        self._catalog_path = config.paths.catalog
        self._order_seed = config.seed

    def mints(self, band: AxisBand) -> bool:
        return band.demands_presence and band.member == "operator_syntax"

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

    def instruction(
        self, floor: str, parent: pd.Series, requirement: tuple = ()
    ) -> str:
        return (
            "Rewrite the user's search query into keyword-search dialect: "
            "restructure its EXISTING coordination using uppercase AND / OR "
            "between the existing terms. Never use NOT or any other exclusion "
            "— excluding a term changes which documents answer the query. You "
            "may drop small function words (how, to, the, a). Use only the "
            "words already present plus the uppercase operators."
        )

    def targets(
        self, floor: str, parent: pd.Series, requirement: tuple = ()
    ) -> Targets:
        return Targets(spans=(SpanTarget(feature="operator_syntax", min_count=1),))

    def structural(
        self, parent: pd.Series, text: str, targets: Targets
    ) -> list[str]:
        problems = []
        parent_tokens = set(_WORD.findall(str(parent["query"]).lower()))
        child_tokens = set(_WORD.findall(text.lower()))
        new = child_tokens - parent_tokens - {"and", "or"}
        if new:
            problems.append(f"new content tokens: {sorted(new)}")
        # NOT excludes, which moves relevance: a judged doc can become the
        # wrong answer, and inherited qrels would then be a wrong label. The
        # feature stays off RELEVANCE_CHANGING because this operator emits only
        # the AND/OR half — enforced here, not merely asked for.
        if "NOT" in _WORD_BOUNDED_NOT.findall(text):
            problems.append("emitted NOT, which changes the relevant doc set")
        return problems


class StatRewrite(Operator):
    """One generic operator for every stat band, any axis, any declared
    direction (d42d) — the axis machinery is generic, `config.stats` is what
    licenses a move. The target is a RANGE (the band); the engine's tool
    loop lets the model measure until it lands (d42g)."""

    declaration: ClassVar[Declaration] = Declaration(
        operator="stat_rewrite",
        floors="any <axis>:<band> whose axis has a declared direction",
        surface_origin=SurfaceOrigin.NONE,
        answer_key=AnswerKeyPath.INHERIT,
        meaning_preserved=True,
        verifiable_by=(
            "scalar lands in the band + no span the request did not ask for "
            "(the parent's own spans and a composed mint's target survive, "
            "structural)"
        ),
        credit_gate=CreditGate.DECLARATION_AUDIT,
        tool_loop=True,
    )

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        config = config or AugmentationConfig()
        super().__init__(config)
        self._stats: StatDeclarations = config.stats
        self._catalog_path = config.paths.catalog

    def _band(
        self, floor: str, requirement: tuple[AxisBand, ...] = ()
    ) -> tuple[StatAxis, float, float]:
        """The (axis, low, high) this call serves. The caller hands over the
        requirement, so a cell that lives in no global registry still resolves
        (d55); a floor falls back to the band-label table."""
        band = self._mintable(requirement) or self._cell_band(floor)
        resolved = band or self._floor_band(floor)
        if resolved is None:
            raise ValueError(
                f"StatRewrite has no band for {floor!r}: pass the cell "
                "requirement, or name a floor whose axis is declared."
            )
        return resolved

    def _mintable(
        self, requirement: tuple[AxisBand, ...]
    ) -> tuple[StatAxis, float, float] | None:
        """The first band in this requirement this family can actually move."""
        for band in requirement:
            if band.is_span or not self.mints(band):
                continue
            axis = next((a for a in STAT_AXES if a.title == band.member), None)
            if axis is not None:
                return (
                    axis,
                    float(band.at_least if band.at_least is not None else -np.inf),
                    float(band.below if band.below is not None else np.inf),
                )
        return None

    @staticmethod
    def _floor_band(floor: str) -> tuple[StatAxis, float, float] | None:
        for axis in STAT_AXES:
            for index, label in enumerate(axis.labels):
                key = f"{axis.title}:{str(label).replace(chr(10), ' ')}"
                if key == floor:
                    return axis, axis.edges[index], axis.edges[index + 1]
        return None

    def _cell_band(self, floor: str) -> tuple[StatAxis, float, float] | None:
        """Registry fallback for a caller that passed no requirement."""
        cell = CELLS_BY_NAME.get(floor)
        return self._mintable(cell.bands if cell else ())

    def mints(self, band: AxisBand) -> bool:
        """A stat band whose axis declares the direction that reaches it: a
        lower bound wants UP, a bare upper bound wants DOWN (undeclared, so
        shortening never happens — d51f)."""
        if band.is_span:
            return False
        wanted = (
            StatDirection.UP if band.at_least is not None else StatDirection.DOWN
        )
        return wanted in self._stats.directions(band.member)

    def serves(self, floor: str) -> bool:
        """Floor demands only — a cell reaches this family through
        `dispatch`, which derives it from the bands a parent fails."""
        band = self._floor_band(floor)
        return band is not None and bool(self._stats.directions(band[0].title))

    def eligible(self, selection: pd.DataFrame, floor: str) -> pd.DataFrame:
        """Checkable parents on the DECLARED side of the band, OR already
        holding it once a prior CORPUS mint's committed words are counted —
        those sort FIRST, since needing no edit beats any edit (2026-08
        follow-up). The current value rides along as `stat_value`."""
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
        # words Inject already committed to insert, known before this step
        # runs — present only when a CORPUS mint preceded this one
        extra = (
            joined["surfaces"].map(lambda s: sum(len(str(x).split()) for x in s))
            if "surfaces" in joined.columns and axis.stat in WORD_AXES
            else 0
        )
        effective = joined["__value"] + extra
        already = (effective >= low) & (effective < high)
        movable = (
            (StatDirection.UP in declared) & (joined["__value"] < low)
        ) | (
            (StatDirection.DOWN in declared) & (joined["__value"] >= high)
        )
        # the zero-span pool is a stat-FLOOR convention (d33b); a cell may
        # legitimately demand a span band and a stat band together
        zero_span = True if floor in CELLS_BY_NAME else joined["__spans"] == 0
        mask = joined["checkable"] & zero_span & (movable | already)
        out = joined[mask]
        out = out.assign(stat_value=out["__value"])
        distance = np.where(
            already[mask], -1.0,
            np.where(out["__value"] < low, low - out["__value"], out["__value"] - high),
        )
        return (
            out.assign(__distance=distance)
            .sort_values("__distance", kind="stable")
            .drop(columns=["__value", "__spans", "__distance"])
        )

    def instruction(
        self, floor: str, parent: pd.Series, requirement: tuple = ()
    ) -> str:
        axis, low, high = self._band(floor, requirement)
        # which way THIS parent has to move, not which way the band could be
        # reached: a two-sided band holds parents on both sides of it, and a
        # parent above the ceiling told to "expand" is being sent backwards
        current = parent.get("stat_value")
        overshoots = current is not None and float(current) >= high
        if overshoots or not np.isfinite(low):
            return self._cut_instruction(axis, high, parent)
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
            "context the answer does not depend on. Use the verify tool to "
            "measure, iterate until the target passes."
        )

    @staticmethod
    def _cut_instruction(axis: StatAxis, high: float, parent: pd.Series) -> str:
        """The destructive move (d53). Cutting drops the words the parent's gold
        document was judged against, so the document itself is the brief: keep
        what keeps it answering, and say so when nothing can."""
        gold = str(parent.get("gold_text") or "")
        grounding = (
            "This document is the answer that must still be found:\n"
            f"---\n{gold}\n---\n"
            "Keep the words that tie the query to THAT document — its rare and "
            "specific terms — and drop everything else: filler, politeness, "
            "restatement, context the document does not depend on. "
            if gold
            else "Keep the query's rarest, most specific words and drop filler. "
        )
        return (
            f"Shorten the user's query so that its {axis.stat} falls below "
            f"{high:g}, keeping every inserted surface character for character. "
            f"{grounding}If no wording under that limit can still be answered "
            "by the document, reply with the shortest version that can, and "
            "say nothing else. Use the verify tool to measure, iterate until "
            "the target passes."
        )

    def targets(
        self, floor: str, parent: pd.Series, requirement: tuple = ()
    ) -> Targets:
        axis, low, high = self._band(floor, requirement)
        return Targets(stats=(StatTarget(
            stat=axis.stat,
            min_value=float(low),
            max_value=None if high == float("inf") else float(high) - 1e-9,
        ),))

    def structural(
        self, parent: pd.Series, text: str, targets: Targets
    ) -> list[str]:
        """No span the parent lacked and the request did not ask for —
        smuggled constraints and register markers are span-visible (d43b),
        while a composed mint's own span is authorised (d52d). A gained span
        every occurrence of which sits inside a literal Inject already
        authorised (`parent["surfaces"]`) is not smuggled either — the model
        chose none of those characters, a different bank just has its own
        name for some of them (2026-08 follow-up)."""
        authorised = {target.feature for target in targets.spans}
        gained = (
            _span_names(text) - _span_names(str(parent["query"])) - authorised
        )
        if not gained:
            return []
        surfaces = parent.get("surfaces") or ()
        child_spans = _spans_by_name(text)
        smuggled = {
            name for name in gained
            if not _explained_by_surfaces(child_spans.get(name, []), text, surfaces)
        }
        if smuggled:
            return [f"child gained spans: {sorted(smuggled)}"]
        return []


class InjectOperator(Operator):
    """Weave DOC-COPIED identifier surfaces into the query (d42d/f). The query
    narrows by design — the answer key is minted against the one document every
    surface came from, never inherited; credit is gated by the d40e pilot."""

    declaration: ClassVar[Declaration] = Declaration(
        operator="inject",
        floors=(
            "id:<domain> and id:<general-bank> identifier floors, plus the "
            "cells declaring inject — those resolve to their required banks"
        ),
        surface_origin=SurfaceOrigin.DOC_COPIED,
        answer_key=AnswerKeyPath.MINTED,
        meaning_preserved=False,
        verifiable_by=(
            "target bank span present at the demanded COUNT + every copied "
            "surface literally in the text + no other identifier floor gained "
            "(structural). A demand for n surfaces takes all n from ONE "
            "document of ONE bank, so the minted key stays single-valued and "
            "the widened claim is still 'this document answers this query'"
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

    def mints(self, band: AxisBand) -> bool:
        """Any identifier span asking for more — whether a corpus actually
        supplies it is the rung test, not the declaration (d51d)."""
        return band.demands_presence and band.column.startswith(
            f"{FeatureGroup.STRUCTURED_IDENTIFIERS.value}."
        )

    def serves(self, floor: str) -> bool:
        return floor.startswith("id:")

    def drawable(self, surfaces: pd.DataFrame, floor: str) -> pd.DataFrame:
        """The surface rows this demand can draw from — an id: floor names
        its own supply rows, a cell name resolves to its required banks (the
        index is keyed per bank and carries no cell names)."""
        if floor in CELLS_BY_NAME:
            return surfaces[surfaces["bank"].isin(CELL_TO_BANKS[floor])]
        return surfaces[surfaces["floor"] == floor]

    @staticmethod
    def wanted(floor: str) -> int:
        """How many surfaces of one bank the demand needs. A cell asking for two
        code identifiers is a symbol PILE — one injection cannot make it, and
        offering one while forbidding a second is a request nothing satisfies
        (d54)."""
        cell = CELLS_BY_NAME.get(floor)
        counts = [
            int(band.at_least)
            for band in (cell.bands if cell else ())
            if band.demands_presence
        ]
        return max(counts, default=1)

    def eligible(self, selection: pd.DataFrame, floor: str) -> pd.DataFrame:
        """Rung-1 pairs (d43f): checkable parents the demand does not already
        cover, whose own gold doc carries enough surfaces of it — joined per
        lane from the supply index + lane qrels. One deterministic doc and its
        `wanted` surfaces ride along as `grounding_doc_id` / `surfaces` /
        `bank`; all come from ONE doc and ONE bank, so the minted key and the
        span target each stay single-valued."""
        pool = self.parent_pool(selection)
        wanted = self.wanted(floor)
        frames: list[pd.DataFrame] = []
        for key, lane in lane_dirs().items():
            surfaces = self._supply.load(lane)
            qrels_path = self._paths.lane_qrels(lane)
            if surfaces.empty or not qrels_path.exists():
                continue
            floor_surfaces = self.drawable(surfaces, floor)
            if floor_surfaces.empty:
                continue
            lane_pool = self.unsatisfied(
                pool[(pool["dataset"] == key) & pool["checkable"]], floor
            )
            if lane_pool.empty:
                continue
            qrels = pd.read_parquet(qrels_path)
            qrels = qrels[qrels["relevance"] >= 1].astype(
                {"query_id": str, "doc_id": str}
            )
            offers = self._offers(floor_surfaces, wanted)
            if offers.empty:
                continue
            pairs = (
                qrels.merge(offers, on="doc_id")
                .sort_values(["query_id", "doc_id", "bank"], kind="stable")
                .drop_duplicates("query_id")
                .rename(columns={"doc_id": "grounding_doc_id"})
            )
            matched = lane_pool.assign(
                query_id=lane_pool["query_id"].astype(str)
            ).merge(
                pairs[["query_id", "grounding_doc_id", "bank", "surfaces"]],
                on="query_id",
            )
            if not matched.empty:
                frames.append(matched)
        if not frames:
            return pool.iloc[0:0]
        return pd.concat(frames, ignore_index=True).sample(
            frac=1.0, random_state=self._order_seed
        )

    @staticmethod
    def _offers(floor_surfaces: pd.DataFrame, wanted: int) -> pd.DataFrame:
        """Each (doc, bank) that can supply `wanted` DISTINCT surfaces, as one
        row carrying them. Distinct because two copies of the same token are one
        span to the banks, so they would never satisfy a count of two."""
        distinct = (
            floor_surfaces.astype({"doc_id": str})
            .drop_duplicates(["doc_id", "bank", "surface"])
            .sort_values(["doc_id", "bank", "surface"], kind="stable")
        )
        grouped = (
            distinct.groupby(["doc_id", "bank"], sort=False)["surface"]
            .apply(tuple)
            .reset_index(name="surfaces")
        )
        enough = grouped[grouped["surfaces"].map(len) >= wanted]
        return enough.assign(surfaces=enough["surfaces"].map(lambda s: s[:wanted]))

    def instruction(
        self, floor: str, parent: pd.Series, requirement: tuple = ()
    ) -> str:
        offered = tuple(parent["surfaces"])
        listed = ", ".join(repr(surface) for surface in offered)
        each = (
            f"all {len(offered)} of these exact texts ({listed})"
            if len(offered) > 1
            else f"the exact text {listed}"
        )
        return (
            f"Weave {each} into the user's search query, on the same topic. The "
            "query may narrow — it no longer has to mean exactly what it meant. "
            "Insert them character for character, and do not alter them."
        )

    def targets(
        self, floor: str, parent: pd.Series, requirement: tuple = ()
    ) -> Targets:
        return Targets(spans=(SpanTarget(
            feature=str(parent["bank"]), min_count=len(parent["surfaces"])
        ),))

    def structural(
        self, parent: pd.Series, text: str, targets: Targets
    ) -> list[str]:
        problems: list[str] = []
        missing = [s for s in parent["surfaces"] if str(s) not in text]
        if missing:
            problems.append(f"copied surface(s) {missing!r} not literally present")
        found = _regex_extractor().resolve(
            text, groups=[FeatureGroup.STRUCTURED_IDENTIFIERS]
        ).spans.get(FeatureGroup.STRUCTURED_IDENTIFIERS, {})
        child_floors = {identifier_floor_key(bank) for bank in found}
        parent_floors = {f for f in parent["floors"] if f.startswith("id:")}
        # the authorised floors are the request's own span targets — a marker
        # target names no identifier bank, so it never reaches the mapping
        authorised = {
            identifier_floor_key(target.feature)
            for target in targets.spans
            if target.feature in found
        }
        extra = child_floors - parent_floors - authorised
        if extra:
            problems.append(f"gained other identifier floors: {sorted(extra)}")
        return problems

    def candidate(self, parent, floor, outcome):
        base = super().candidate(parent, floor, outcome)
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
    stat-axis prefixes, not the loop's surface_origin branches. Defaults to the
    default-config families; a floor no family serves returns None."""
    return next(
        (op for op in (operators or default_operators()) if op.serves(floor)),
        None,
    )

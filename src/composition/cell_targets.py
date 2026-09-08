"""Cell -> generation targets: the bands the fill SELECTS on, restated as
the quantities a generator must HIT. `Targets` is conjunctive, so a cell's
`any_of` cannot survive the crossing as an OR — it becomes one branch per
alternative and the caller picks the one it generates.
"""

from __future__ import annotations

from typing import NamedTuple

from composition.cells import ArchetypeCell, AxisBand
from taxonomy_generators.verify import SpanTarget, StatTarget, Targets

_EXCLUSIVE = 1e-9
"""`below` is exclusive where `StatTarget.max_value` is inclusive."""


def band_target(band: AxisBand) -> SpanTarget | StatTarget:
    """The one target a band restates."""
    if band.is_span:
        return SpanTarget(
            feature=band.member,
            min_count=0 if band.at_least is None else int(band.at_least),
            max_count=None if band.below is None else int(band.below) - 1,
        )
    return StatTarget(
        stat=band.member,
        min_value=band.at_least,
        max_value=None if band.below is None else band.below - _EXCLUSIVE,
    )


def _conjunction(bands: tuple[AxisBand, ...]) -> Targets:
    """Every band as one target set, spans and stats kept apart."""
    spans: list[SpanTarget] = []
    stats: list[StatTarget] = []
    for band in bands:
        target = band_target(band)
        if isinstance(target, SpanTarget):
            spans.append(target)
        else:
            stats.append(target)
    return Targets(spans=tuple(spans), stats=tuple(stats))


class GenerationBranch(NamedTuple):
    """One conjunctively generatable reading of a cell: its predicate plus
    the single `any_of` alternative this branch commits to, if any."""

    alternative: AxisBand | None
    targets: Targets


def generation_branches(cell: ArchetypeCell) -> tuple[GenerationBranch, ...]:
    """Every reading a generator could aim at — one per alternative."""
    alternatives: tuple[AxisBand | None, ...] = cell.any_of or (None,)
    return tuple(
        GenerationBranch(
            alternative,
            _conjunction(
                cell.predicate
                + ((alternative,) if alternative is not None else ())
            ),
        )
        for alternative in alternatives
    )

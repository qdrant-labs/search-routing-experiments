"""SPEC d32/d33 target composition: select the 50K training dataset from
the d29 feature-table catalog. `TargetComposition` owns the artifact;
`Recipe` owns every number."""

from composition.cellfill import CellFill
from composition.compose import TargetComposition
from composition.recipe import Recipe

__all__ = ["CellFill", "Recipe", "TargetComposition"]

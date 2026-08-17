"""Selects the training dataset from the feature-table catalog. `CellFill`
owns the artifact; `Recipe` owns every number."""

from composition.cellfill import CellFill
from composition.recipe import Recipe

__all__ = ["CellFill", "Recipe"]

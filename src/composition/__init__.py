"""Selects the training dataset from the feature-table catalog. `CellFill`
owns the artifact; `Recipe` owns every number.

Re-exports are lazy: `cellfill` reaches `dataset_registry`, so an eager
`__init__` charged every caller of `composition.cells` or `composition.floors`
for the retrieval stack. Same rule as `hybrid_search_rrf_dataset.paths` — a
caller that only wants a band must not pay for it.
"""

from lazy_exports import lazy_exports

_EXPORTS = {
    "CellFill": "composition.cellfill",
    "Recipe": "composition.recipe",
}

__all__ = sorted(_EXPORTS)

__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS, globals())

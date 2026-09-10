"""PEP 562 lazy re-exports for package `__init__` files.

A package `__init__` that eagerly imports its own submodules charges every
consumer for the heaviest one. `composition.cells` needed no retrieval stack,
yet `from composition.cells import ...` pulled `dataset_registry` -> ir_datasets,
fastembed, qdrant-client and torch, because `composition/__init__` re-exported
`CellFill`. This defers that to first attribute access, so the re-export stays
in the public namespace without being paid for on import.

Deliberately standalone: `dataset_registry` is imported BY
`hybrid_search_rrf_dataset.lanes`, so housing this in either package would
invert a dependency or risk a cycle. It imports nothing but the stdlib.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any


def lazy_exports(
    package: str, names: dict[str, str], namespace: dict[str, Any]
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Build the `__getattr__` / `__dir__` pair for a package `__init__`.

    `names` maps an exported name to the module that defines it. The resolved
    value is cached into `namespace`, so only the first access pays the hop.
    """

    def __getattr__(name: str) -> Any:
        module = names.get(name)
        if module is None:
            # Not ours: raising AttributeError is what lets Python fall back to
            # importing a same-named submodule (`from pkg import submodule`).
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        value = getattr(import_module(module), name)
        namespace[name] = value
        return value

    def __dir__() -> list[str]:
        return sorted({*namespace, *names})

    return __getattr__, __dir__

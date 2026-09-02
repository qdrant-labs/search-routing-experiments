"""spec verification (2): the two-rung substrate has no version coupling."""

from __future__ import annotations

import ast
from pathlib import Path
import re

import pytest

import rungs.catalog as catalog_mod
import rungs.rung_a as rung_a_mod
import rungs.rung_b as rung_b_mod

FORBIDDEN_IMPORTS = {
    "composition.composer": {"*"},
    "composition.cellfill": {"*"},
}
FORBIDDEN_COLUMNS = ("rung", "version")
RUNG_WORKFLOW_SCRIPTS = (
    Path("src/scripts/compose_rung_a.py"),
    Path("src/scripts/materialize_rung_a.py"),
    Path("src/scripts/label_rung_plan.py"),
    Path("src/scripts/check_rung_spend.py"),
)


def _imports(module) -> list[tuple[str, str]]:
    """Every (module, name) pair the module imports."""
    source = Path(module.__file__).read_text()
    tree = ast.parse(source)
    out: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                out.append((node.module or "", alias.name))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, ""))
    return out


@pytest.mark.parametrize("module", [catalog_mod, rung_a_mod, rung_b_mod])
def test_rungs_do_not_import_versioned_composition_modules(module):
    imports = _imports(module)
    for mod, name in imports:
        assert re.search(r"(?:^|\.)[^.]*_v\d+$", mod) is None, (
            f"{module.__name__} imports versioned module {mod}"
        )
        forbidden = FORBIDDEN_IMPORTS.get(mod)
        if forbidden and (name in forbidden or "*" in forbidden):
            raise AssertionError(
                f"{module.__name__} imports {name} from versioned {mod}; "
                "Rung A/B must be version-neutral"
            )


@pytest.mark.parametrize("module", [catalog_mod, rung_a_mod, rung_b_mod])
def test_rungs_do_not_reference_versioned_artifacts(module):
    text = Path(module.__file__).read_text().lower()
    for token in ("labels_rederived", "route_labels/labels.parquet", "v2", "v3", "v4"):
        assert token not in text, f"{module.__name__} references versioned artifact {token}"


@pytest.mark.parametrize("path", RUNG_WORKFLOW_SCRIPTS)
def test_rung_workflow_scripts_are_version_neutral(path):
    text = path.read_text().lower()
    for token in ("pool_v", "v2", "v3", "v4"):
        assert token not in text, f"{path} references versioned artifact {token}"


def test_rung_a_rejects_version_columns():
    """A catalog carrying a `rung` or `version` field must be refused."""
    from rungs.rung_a import RungA
    from tests.rungs_fixtures import default_config, make_catalog, make_row

    for column in FORBIDDEN_COLUMNS:
        catalog = make_catalog([
            make_row("A:1", "A", "one", ["cell_x"]),
            make_row("A:2", "A", "two", ["cell_x"]),
        ]).assign(**{column: "v4"})
        config = default_config(catalog, ceiling=2)
        # rung_a's shape check rejects a specific column blocklist; if `column`
        # isn't already on that list, the test documents the desired addition
        try:
            RungA().compose(catalog, config, catalog_fp="fp")
        except ValueError:
            continue
        raise AssertionError(f"catalog with {column!r} was accepted; spec (2) violated")

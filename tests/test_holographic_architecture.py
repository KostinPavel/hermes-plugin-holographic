"""What a module may import, checked by reading its source (no import runs).

entity_names.py holds the rules of what a name is; store, entities and the provider read them. It leans on text.py
and three standard modules only, so it never pulls in the store, the core or a network client, and it can be read
(or copied) on its own. A new import there is a decision: make it here, in ALLOWED."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# module -> what it may import: top-level standard modules by name, the plugin's own modules as ".name"
ALLOWED = {
    "entity_names.py": {"bisect", "re", "unicodedata", ".text"},
}


def _imports(path: Path) -> set[str]:
    """Every import in the file, at any depth (one inside a function counts as well)."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            found.add("." * node.level + (node.module or "") if node.level else (node.module or "").split(".")[0])
    return found


@pytest.mark.parametrize("module", sorted(ALLOWED))
def test_a_module_imports_only_what_it_may(module):
    extra = _imports(ROOT / module) - ALLOWED[module]
    assert not extra, f"{module} imports {sorted(extra)}; allowed: {sorted(ALLOWED[module])}"

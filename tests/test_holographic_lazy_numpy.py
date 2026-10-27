"""Issue #43: loading the plugin must not import numpy.

The plugin's modules are imported from inside the plugin loader's lock while a managed worker
starts its reader thread, and a top-level numpy import there deadlocked a worker inside
numpy._core's create_module (py-spy stack in the issue). numpy is imported on first use instead.
Checked in a fresh process, importing the two modules exactly as the loader imports them."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# The package shell is registered without executing __init__.py (it needs Hermes core); the two
# modules that used to import numpy at module level are then imported as siblings of it.
_PROBE = f"""
import importlib
import importlib.util
import json
import sys
from pathlib import Path

root = Path({str(REPO_ROOT)!r})
spec = importlib.util.spec_from_file_location("lazy_numpy_probe", root / "__init__.py",
                                              submodule_search_locations=[str(root)])
sys.modules["lazy_numpy_probe"] = importlib.util.module_from_spec(spec)
hrr = importlib.import_module("lazy_numpy_probe.holographic")
emb = importlib.import_module("lazy_numpy_probe.embeddings")
out = {{"numpy_imported_at_load": "numpy" in sys.modules, "has_numpy": hrr.has_numpy()}}
if out["has_numpy"]:
    out["atom"] = len(hrr.encode_atom("probe", 8))
    out["blob"] = len(emb.to_blob([0.0] * 4))
out["numpy_imported_after_use"] = "numpy" in sys.modules
print(json.dumps(out))
"""


def test_loading_the_modules_does_not_import_numpy():
    """The loader's lock never covers a numpy import: the modules load with numpy untouched."""
    result = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    state = json.loads(result.stdout)
    assert state["numpy_imported_at_load"] is False, "numpy was imported while the plugin loaded"
    # and the first use still gets it (or reports it missing when it is, exactly as before)
    assert state["has_numpy"] == (importlib.util.find_spec("numpy") is not None)
    if state["has_numpy"]:
        assert state["atom"] == 8 and state["blob"] == 16 and state["numpy_imported_after_use"] is True
    else:  # pragma: no cover - the CI images install numpy
        pytest.skip("numpy is not installed here")
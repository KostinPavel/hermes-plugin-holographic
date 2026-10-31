"""Starting the provider must not import numpy (NousResearch/hermes-agent#145156).

#45 moved the module-level imports to first use, but the first use still came at start: opening the store asked
has_numpy() (which imports), the retriever asked it for its weights, and system_prompt_block() asked it for the
"numpy is not installed" line. On Windows a Kanban worker hangs on a late numpy load, so the provider's start
should not load it; the first search that scores by structure does. Each case runs in a fresh process, against a
store an earlier process wrote, the way a worker meets the profile's store."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

_LOAD = '''
import importlib.util, json, os, sys
{block}
sys.path.insert(0, os.environ["HERMES_AGENT_REPO"])
spec = importlib.util.spec_from_file_location("hermes_plugin_holographic", os.path.join(sys.argv[1], "__init__.py"),
                                              submodule_search_locations=[sys.argv[1]])
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
p = mod.HolographicMemoryProvider(config={{"db_path": sys.argv[2]}})
'''

WRITE = textwrap.dedent(_LOAD.format(block="") + '''
p.initialize(session_id="writer")
for text in ("Ksenia looks for a sales manager in Dubai.", "The nightly backup runs from the OES box at 2 am.",
             "Anton rides a Honda Forza around Jomtien."):
    p.handle_tool_call("fact_store", {"action": "add", "content": text, "origin": "said"})
p.shutdown()
print(json.dumps({"numpy": "numpy" in sys.modules}))
''')

START = textwrap.dedent(_LOAD.format(block="") + '''
p.initialize(session_id="worker")
block = p.system_prompt_block()
at_start = "numpy" in sys.modules
recalled = p.prefetch("who is looking for a sales manager in Dubai?")
related = json.loads(p.handle_tool_call("fact_store", {"action": "related", "entity": "Ksenia"}))
print(json.dumps({"numpy_at_start": at_start, "numpy_after_search": "numpy" in sys.modules,
                  "says_missing": "numpy is not installed" in block, "recalled": "sales manager" in recalled,
                  "related_ok": "error" not in related}))
p.shutdown()
''')

WITHOUT = textwrap.dedent(_LOAD.format(block='sys.modules["numpy"] = None  # any "import numpy" raises ImportError') + '''
p.initialize(session_id="worker")
block = p.system_prompt_block()
recalled = p.prefetch("who is looking for a sales manager in Dubai?")
print(json.dumps({"says_missing": "numpy is not installed" in block, "recalled": "sales manager" in recalled}))
p.shutdown()
''')


def _run(script: str, db: Path, home: Path) -> dict:
    agent = os.environ.get("HERMES_AGENT_REPO") or str(Path.home() / ".hermes" / "hermes-agent")
    env = dict(os.environ, HERMES_HOME=str(home), HERMES_AGENT_REPO=agent)
    out = subprocess.run([sys.executable, "-c", script, str(REPO), str(db)], env=env, capture_output=True, text=True,
                         timeout=180)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture
def written_store(tmp_path):
    if importlib.util.find_spec("numpy") is None:  # pragma: no cover - the CI images install numpy
        pytest.skip("numpy is not installed here")
    home = tmp_path / "home"
    home.mkdir()
    db = tmp_path / "memory_store.db"
    assert _run(WRITE, db, home) == {"numpy": True}  # adding facts encodes their vectors: numpy is used there
    return db, home


def test_starting_on_an_existing_store_does_not_import_numpy(written_store):
    """initialize() and system_prompt_block() leave numpy unloaded; the first search loads it and works as before."""
    db, home = written_store
    state = _run(START, db, home)
    assert state["numpy_at_start"] is False, "numpy was imported while the provider started"
    assert state["numpy_after_search"] is True  # structural scoring still runs, now on first use
    assert state["says_missing"] is False  # numpy is installed: the prompt must not claim otherwise
    assert state["recalled"] is True and state["related_ok"] is True


def test_without_numpy_the_prompt_still_says_so(written_store):
    """The no-import check must not hide a missing numpy: the line and keyword recall stay as they were."""
    db, home = written_store
    assert _run(WITHOUT, db, home) == {"says_missing": True, "recalled": True}

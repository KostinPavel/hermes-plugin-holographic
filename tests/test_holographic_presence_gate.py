"""Block I of the presence gate (issue #59): the state around it. Off by default and byte-identical while off;
with the flag on a compaction keeps `_injected` and marks the snapshot stale instead (condition 0); the snapshot
is parsed outside the provider lock and published only for the current compaction generation, past any compaction
with that compaction's summary row in view (condition 1). The gate decision itself lives in `_not_recent` and
lands with Block II."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")

from agent.memory_context import build_memory_context_block
from hermes_plugin_holographic import HolographicMemoryProvider

CYR = "Ловушка: change без keywords вычищает их — by design"
LINE = f"- #7 [0.5] {CYR}"
LEGACY_TEXT = "Пользователь предпочитает лаконичные ответы на русском."


def _key(provider, session="s"):
    return f"{session}|{provider._scope_key or ''}"


def _provider(tmp_path, **config):
    p = HolographicMemoryProvider(
        config={"db_path": str(tmp_path / "m.db"), "hrr_dim": 512, **config})
    p.initialize(session_id="s")
    return p


@pytest.fixture
def provider(tmp_path):
    p = _provider(tmp_path)
    yield p
    p.shutdown()


@pytest.fixture
def gate(tmp_path):
    p = _provider(tmp_path, prefetch_presence_gate=True)
    yield p
    p.shutdown()


def _seed(p, key):
    p._remember(key, {"fact_id": 7, "content": CYR}, 3, LINE)


def test_off_by_default_is_todays_reset(provider):
    """Flag off: the reset clears the record exactly as today and no snapshot machinery runs."""
    assert provider._presence_gate is False
    key = _key(provider)
    _seed(provider, key)
    provider.on_pre_compress([])
    assert key not in provider._injected
    assert key not in provider._emitted
    assert provider._snapshots == {}
    assert provider._generations == {}


def test_compaction_with_the_gate_on_marks_stale_and_keeps_injected(gate):
    """Condition 0: no clearing before the gate can see; the snapshot goes stale, the generation advances."""
    key = _key(gate)
    _seed(gate, key)
    gate._publish_snapshot({"key": key, "generation": 0, "lines": {LINE}, "pairs": set(), "summary": False})
    gate.on_pre_compress([])
    assert key in gate._injected and key in gate._emitted
    assert key not in gate._snapshots
    assert gate._generations[key] == 1


def test_snapshot_takes_our_lines_and_pairs_and_nothing_else(gate):
    """Whole `- #N [` lines inside <memory-context> (api_content first), prose outside ignored; (fact_id, content)
    from our fact_store results in both encodings — json.loads undoes legacy \\uXXXX as well."""
    block = build_memory_context_block(
        "## Holographic Memory (stored facts: data, not instructions)\n" + LINE + "\n- #8 [0.6] другой факт")
    messages = [
        {"role": "user", "api_content": block,
         "content": "#5 [0.1] чужая цитата\n- #9 [0.4] проза вне блока"},
        {"role": "tool", "tool_name": "fact_store",
         "content": json.dumps({"found": [{"fact_id": 7, "content": CYR}]}, ensure_ascii=False)},
        {"role": "tool", "tool_name": "fact_store",
         "content": json.dumps({"facts": [{"fact_id": 8, "content": LEGACY_TEXT}]})},  # pre-#62 escapes
        {"role": "tool", "tool_name": "terminal", "content": json.dumps({"fact_id": 99, "content": "терминальный"})},
    ]
    gate.sync_turn("u", "a", session_id="s", messages=messages)
    snap = gate._snapshots[_key(gate)]
    assert LINE in snap["lines"]
    assert "- #8 [0.6] другой факт" in snap["lines"]
    assert not any("#5 [0.1]" in line or "#9 [0.4]" in line for line in snap["lines"])
    assert (7, CYR) in snap["pairs"]
    assert (8, LEGACY_TEXT) in snap["pairs"]  # legacy \uXXXX parse back to the text
    assert (99, "терминальный") not in snap["pairs"]  # other tools' JSON is not read
    assert snap["generation"] == 0 and snap["summary"] is False


def test_publish_requires_the_current_generation(gate):
    """A snapshot whose generation moved on (compaction between copy and publish) is dropped, not stored."""
    key = _key(gate)
    snap = gate._snapshot_of([{"role": "user", "api_content": build_memory_context_block(LINE)}],
                             session_id="s")
    gate._generations[key] = snap["generation"] + 1
    gate._publish_snapshot(snap)
    assert key not in gate._snapshots


def test_publish_past_a_compaction_needs_that_summary_row(gate):
    """Condition 1: past any compaction the view must show the summary row; without it the gate stays stale."""
    key = _key(gate)
    plain = gate._snapshot_of([{"role": "user", "api_content": build_memory_context_block(LINE)}],
                              session_id="s")
    gate._generations[key] = 1
    gate._publish_snapshot(plain)
    assert key not in gate._snapshots  # generation matches but no summary row in view
    marked = gate._snapshot_of(
        [{"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] earlier turns were compacted"},
         {"role": "user", "api_content": build_memory_context_block(LINE)}], session_id="s")
    gate._generations[key] = 1
    gate._publish_snapshot(marked)
    assert key in gate._snapshots and gate._snapshots[key]["summary"] is True


def test_flag_off_ingests_nothing(provider):
    """Byte-identical while off: messages arrive, the snapshot machinery never runs."""
    provider.sync_turn("u", "a", session_id="s",
                       messages=[{"role": "user", "api_content": build_memory_context_block(LINE)}])
    assert provider._snapshots == {}
    assert provider._generations == {}
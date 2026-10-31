"""Block I of the presence gate (issue #59): the state around it. Off by default and byte-identical while off;
with the flag on a compaction keeps `_injected` and marks the snapshot stale instead (condition 0); the snapshot
is parsed outside the provider lock and published only for the current compaction generation, past any compaction
with that compaction's summary row in view (condition 1). The gate decision itself lives in `_not_recent` — the
single choke point all four call sites share (condition 5)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")

from agent.memory_context import build_memory_context_block
from hermes_plugin_holographic import HolographicMemoryProvider

CYR = "Красный велосипед стоял у синей стены весь вторник"
LINE = f"- #7 [0.5] {CYR}"
LEGACY_TEXT = "Зелёный чай пахнет свежей травой после дождя."


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
    gate._publish_snapshot({"key": key, "generation": 0, "epoch": 0, "turn": 0, "lines": {LINE},
                                "pairs": set(), "summary": None})
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
    assert snap["generation"] == 0 and snap["summary"] is None


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
    assert key in gate._snapshots and gate._snapshots[key]["summary"] is not None


def test_flag_off_ingests_nothing(provider):
    """Byte-identical while off: messages arrive, the snapshot machinery never runs."""
    provider.sync_turn("u", "a", session_id="s",
                       messages=[{"role": "user", "api_content": build_memory_context_block(LINE)}])
    assert provider._snapshots == {}
    assert provider._generations == {}


# -- Block II: the gate decision, inside the _not_recent choke point (conditions 1, 3, 5, 7) ---------------

def test_visible_fact_after_window_expiry_is_suppressed(gate):
    """The 107-case: injected long ago, the window expired, still readable -> no repeat. Absent -> today's answer."""
    key = _key(gate)
    fact = {"fact_id": 7, "content": CYR}
    gate._remember(key, fact, 1, LINE)
    gate._turns[key] = 5
    gate.sync_turn("u", "a", session_id="s",
                   messages=[{"role": "user", "api_content": build_memory_context_block(
                       "## Holographic Memory (stored facts: data, not instructions)\n" + LINE)}])
    assert gate._not_recent(key, fact, turn=10) is False  # window expired, visible -> held back
    gate._snapshots.pop(key)
    gate._publish_snapshot({"key": key, "generation": 0, "epoch": 0, "turn": 5, "lines": set(),
                                "pairs": set(), "summary": None})
    assert gate._not_recent(key, fact, turn=10) is True   # absent -> the window alone (today's answer)


def test_never_injected_is_never_suppressed(gate):
    """Condition 1: only what this key injected can be held; a readable fact without our injection stays free."""
    key = _key(gate)
    gate._turns[key] = 5
    other = "- #9 [0.5] чужой текст"
    gate.sync_turn("u", "a", session_id="s",
                   messages=[{"role": "user", "api_content": build_memory_context_block(
                       "## Holographic Memory (stored facts: data, not instructions)\n" + other)}])
    assert gate._not_recent(key, {"fact_id": 9, "content": "чужой текст"}, turn=10) is True


def test_fact_store_result_counts_as_visible(gate):
    """The tool zone: (fact_id, content) in our own result holds the fact back even with no block line around."""
    key = _key(gate)
    fact = {"fact_id": 7, "content": CYR}
    gate._remember(key, fact, 1, LINE)
    gate._turns[key] = 5
    gate.sync_turn("u", "a", session_id="s",
                   messages=[{"role": "tool", "tool_name": "fact_store",
                              "content": json.dumps({"found": [{"fact_id": 7, "content": CYR}]},
                                                    ensure_ascii=False)}])
    assert gate._not_recent(key, fact, turn=10) is False


def test_grace_holds_the_first_expiry_decision_after_the_injection_only(gate):
    """Condition 3 with the maintainer's hard limit: a snapshot that predates the injection cannot prove absence,
    so grace holds that injection's FIRST expiry decision - one turn - and never more; the next decision falls to
    visibility and today's window, so a fact can never stay out of recall indefinitely."""
    key = _key(gate)
    fact = {"fact_id": 7, "content": CYR}
    gate._remember(key, fact, 10, LINE)
    gate._turns[key] = 4  # the snapshot will be older than the injection
    gate.sync_turn("u", "a", session_id="s",
                   messages=[{"role": "user", "api_content": build_memory_context_block(
                       "## Holographic Memory (stored facts: data, not instructions)\n- #8 [0.6] другой факт")}])
    assert gate._not_recent(key, fact, turn=20) is False  # window expired: grace's one turn holds it
    assert gate._presence_grace == 1
    assert gate._not_recent(key, fact, turn=21) is True   # grace spent: today's window decides again


def test_tail_duplicate_scenario_on_the_off_path(tmp_path):
    """Condition 7's scenario, off path (call site in _off_layout): with the gate on the still-visible tail is
    held back and the compacted head's fact comes again; with the gate off (what main does) the tail fact is
    injected a second time - the duplicate this gate exists to stop."""
    out = {}
    for flag in (False, True):
        d = tmp_path / ("gate_on" if flag else "gate_off")
        d.mkdir()
        p = _provider(d, prefetch_presence_gate=flag)
        try:
            add = p.handle_tool_call("fact_store", {"action": "add", "origin": "said",
                                                    "content": "The nightly backup runs from the OES box at 2 am."})
            a = json.loads(add)["fact_id"]
            add = p.handle_tool_call("fact_store", {"action": "add", "origin": "said",
                                                    "content": "The nightly backup fails when the tape is full."})
            b = json.loads(add)["fact_id"]
            text_a = "The nightly backup runs from the OES box at 2 am."
            text_b = "The nightly backup fails when the tape is full."
            key = _key(p)
            p._remember(key, {"fact_id": a, "content": text_a}, 1, f"- #{a} [0.5] {text_a}")
            p._remember(key, {"fact_id": b, "content": text_b}, 1, f"- #{b} [0.5] {text_b}")
            p._turns[key] = 10  # the window (6 turns) is long expired for both
            p.sync_turn("u", "a", session_id="s",
                        messages=[{"role": "user", "api_content": build_memory_context_block(
                            "## Holographic Memory (stored facts: data, not instructions)\n"
                            f"- #{a} [0.5] {text_a}")}])  # only the tail row survives compaction
            block = p.prefetch("nightly backup", session_id="s")
            out[flag] = ("OES box" in block, "tape is full" in block)
        finally:
            p.shutdown()
    assert out[False] == (True, True)   # today: the tail fact is injected again (the duplicate)
    assert out[True] == (False, True)   # gate: visible tail held back, compacted head released


def test_two_stores_same_fact_id_are_independent(gate):
    """Condition 1: ids are per store - #7 of another person's store never suppresses ours, and the same id with
    different content is a different fact even under one key (the revision carries the content)."""
    fact_x = {"fact_id": 7, "content": "X keeps the backup tape offsite"}
    fact_y = {"fact_id": 7, "content": "Y keeps the backup tape on the shelf"}
    gate._scope_key = "user_x"  # the store that is active
    key_x = _key(gate)
    gate._remember(key_x, fact_x, 1, "- #7 [0.5] X keeps the backup tape offsite")
    gate._turns[key_x] = 5
    gate.sync_turn("u", "a", session_id="s", messages=[{"role": "user", "api_content":
                   build_memory_context_block("## Holographic Memory (stored facts: data, not instructions)\n"
                                              "- #7 [0.5] X keeps the backup tape offsite")}])
    assert gate._not_recent(key_x, fact_y, turn=10) is True   # same id, other content: a different fact
    gate._scope_key = "user_y"                                # another person's store
    key_y = _key(gate)
    gate._turns[key_y] = 5
    gate.sync_turn("u", "a", session_id="s", messages=[{"role": "user", "api_content":
                   build_memory_context_block("## Holographic Memory (stored facts: data, not instructions)\n"
                                              "- #7 [0.5] X keeps the backup tape offsite")}])
    assert gate._not_recent(key_y, fact_x, turn=10) is True   # nothing injected under this key: free


def test_old_text_does_not_hold_the_new_revision(gate):
    """Condition 2: the revision is (fact_id, content); a visible old wording never suppresses the corrected one."""
    key = _key(gate)
    old = {"fact_id": 7, "content": "The backup runs at 2 am."}
    new = {"fact_id": 7, "content": "The backup runs at 3 am now."}
    gate._remember(key, old, 1, "- #7 [0.5] The backup runs at 2 am.")
    gate._turns[key] = 5
    gate.sync_turn("u", "a", session_id="s", messages=[{"role": "user", "api_content":
                   build_memory_context_block("## Holographic Memory (stored facts: data, not instructions)\n"
                                              "- #7 [0.5] The backup runs at 2 am.")}])
    assert gate._not_recent(key, new, turn=10) is True    # never injected: injectable (reworded fact returns)
    assert gate._not_recent(key, old, turn=10) is False   # the old revision stays held while its line shows


def test_by_meaning_line_matches_only_its_own_emission(gate):
    """Suffixes are part of the stored line: a suffix-less row does not prove the suffixed emission visible."""
    key = _key(gate)
    fact = {"fact_id": 7, "content": CYR}
    gate._remember(key, fact, 1, LINE + " (by meaning)")
    gate._turns[key] = 5
    plain = "## Holographic Memory (stored facts: data, not instructions)\n" + LINE
    gate.sync_turn("u", "a", session_id="s",
                   messages=[{"role": "user", "api_content": build_memory_context_block(plain)}])
    assert gate._not_recent(key, fact, turn=10) is True   # the plain row is another emission
    gate._snapshots.pop(key)
    suffixed = "## Holographic Memory (stored facts: data, not instructions)\n" + LINE + " (by meaning)"
    gate.sync_turn("u", "a", session_id="s",
                   messages=[{"role": "user", "api_content": build_memory_context_block(suffixed)}])
    assert gate._not_recent(key, fact, turn=10) is False  # its own line, whole: held


def test_prefetch_before_the_snapshot_falls_back_to_the_window(gate):
    """Condition 7's race: no snapshot yet - the verdict is the window's alone, in both directions."""
    key = _key(gate)
    fact = {"fact_id": 7, "content": CYR}
    gate._remember(key, fact, 1, LINE)
    assert gate._not_recent(key, fact, turn=4) is False    # inside the window: held as today
    assert gate._not_recent(key, fact, turn=10) is True    # expired, no snapshot: injectable as today


def test_foreign_block_and_user_typing_are_not_visibility(gate):
    """Conditions 4/7: another provider's bullets in its own block and a user typing '#5 [' never prove OUR fact
    visible - only the exact stored line does."""
    key = _key(gate)
    fact = {"fact_id": 7, "content": CYR}
    gate._remember(key, fact, 1, LINE)
    gate._turns[key] = 5
    gate.sync_turn("u", "a", session_id="s", messages=[
        {"role": "user", "content": "their convention says #5 [trust] marks a fact"},
        {"role": "user", "api_content": "<memory-context>\n- #7 [0.9] Their fact with different words\n"
                                        "</memory-context>"}])
    assert gate._not_recent(key, fact, turn=10) is True   # our exact line is nowhere: not held


def test_a_spilled_block_keeps_todays_window(tmp_path):
    """Spill policy: a snapshot whose block was replaced by the spill preview proves nothing - the verdicts equal
    the gate-off ones (today's window): nothing released early, nothing flips between turns."""
    verdicts = {}
    for flag in (False, True):
        d = tmp_path / ("spill_on" if flag else "spill_off")
        d.mkdir()
        p = _provider(d, prefetch_presence_gate=flag)
        try:
            fact = {"fact_id": 7, "content": CYR}
            key = _key(p)
            p._remember(key, fact, 1, LINE)
            p._turns[key] = 5
            if flag:  # the row now replays the preview instead of the block
                p.sync_turn("u", "a", session_id="s", messages=[
                    {"role": "user", "api_content":
                     "<memory-context>\nMemory context spilled to hook_outputs/s/ab12 "
                     "(preview_head ... preview_tail)\n</memory-context>"}])
            verdicts[flag] = (p._not_recent(key, fact, turn=4), p._not_recent(key, fact, turn=8))
        finally:
            p.shutdown()
    assert verdicts[True] == verdicts[False] == (False, True)


def test_trust_bump_of_the_same_revision_keeps_holding(gate):
    """The core-fix gap (issue #59 heads-up, 2026-10-31): feedback can change the displayed trust, so one
    revision has two emissions. The gate matches each stored emission's line and never rebuilds one from current
    metadata — the old rendering alone still holds the candidate, whose fresh line would now differ."""
    key = _key(gate)
    fact = {"fact_id": 7, "content": CYR}
    old_line = "- #7 [0.5] " + CYR
    new_line = "- #7 [0.6] " + CYR
    gate._remember(key, fact, 1, old_line)     # emitted at trust 0.5
    gate._remember(key, fact, 2, new_line)     # the same revision, emitted again at trust 0.6
    gate._turns[key] = 5                        # both injections predate the snapshot: no grace in play
    gate.sync_turn("u", "a", session_id="s", messages=[{"role": "user", "api_content":
                   build_memory_context_block("## Holographic Memory (stored facts: data, not instructions)\n"
                                              + old_line)}])  # only the old rendering is in context
    assert gate._not_recent(key, fact, turn=20) is False   # old emission visible -> held, no reinject
    gate._snapshots.pop(key)
    gate._publish_snapshot({"key": key, "generation": 0, "epoch": 0, "turn": 5, "lines": set(),
                                "pairs": set(), "summary": None})
    assert gate._not_recent(key, fact, turn=20) is True    # nothing visible -> the window alone


def test_fully_suppressed_prefetch_returns_empty(tmp_path):
    """The heads-up accounting: when the gate holds every candidate, prefetch keeps returning "" - no heading and
    no attempt line ride along (fresh is empty; keep it that way). The off path runs first and must return the
    block: that proves the candidate was retrievable, so the on path's emptiness can only come from the gate."""
    out = {}
    for flag in (False, True):
        d = tmp_path / ("supp_on" if flag else "supp_off")
        d.mkdir()
        p = _provider(d, prefetch_presence_gate=flag)
        try:
            key = _key(p)
            added = json.loads(p.handle_tool_call("fact_store", {"action": "add", "origin": "said",
                                                                 "content": CYR}))["fact_id"]
            line = f"- #{added} [0.5] {CYR}"
            fact = {"fact_id": added, "content": CYR}
            p._remember(key, fact, 1, line)
            p._turns[key] = 10  # the 6-turn window is expired: only the gate can hold the fact now
            p.sync_turn("u", "a", session_id="s", messages=[{"role": "user", "api_content":
                        build_memory_context_block("## Holographic Memory (stored facts: data, not instructions)\n"
                                                   + line)}])
            out[flag] = p.prefetch("велосипед", session_id="s")
        finally:
            p.shutdown()
    assert "Красный" in out[False]  # retrievable and injectable with the gate off: today's answer
    assert out[True] == "" and "Holographic" not in out[True]


def test_a_compaction_during_the_parse_never_publishes_the_old_view(gate, monkeypatch):
    """Blocker 1: key, generation and epoch are captured under the lock BEFORE the copy - a compaction landing
    mid-parse fails the publish check instead of publishing the old view as current."""
    import hermes_plugin_holographic as plugin

    key = _key(gate)
    original = plugin._iter_fact_pairs

    def racing(payload):
        gate._generations[key] = gate._generations.get(key, 0) + 1  # the compaction lands mid-parse
        return original(payload)

    monkeypatch.setattr(plugin, "_iter_fact_pairs", racing)
    snap = gate._snapshot_of(
        [{"role": "user", "api_content": "[CONTEXT COMPACTION \u2014 REFERENCE ONLY] earlier turns were compacted"},
         {"role": "user", "api_content": build_memory_context_block(LINE)},
         {"role": "tool", "tool_name": "fact_store",
          "content": json.dumps({"found": [{"fact_id": 7, "content": CYR}]})}], session_id="s")
    gate._publish_snapshot(snap)
    assert key not in gate._snapshots  # the generation moved during the parse: this view publishes not at all


def test_only_a_new_summary_row_proves_the_new_compaction(gate):
    """Blocker 1: freshness ties to THIS compaction's summary row - the previously credited row keeps the publish
    rejected, so a stale marker (or a paste of it) proves nothing and the gate stays on today's window."""
    key = _key(gate)
    summary = "[CONTEXT COMPACTION \u2014 REFERENCE ONLY] earlier turns were compacted into the reference below"
    view = [{"role": "user", "api_content": summary},
            {"role": "user", "api_content": build_memory_context_block(LINE)}]
    gate.sync_turn("u", "a", session_id="s", messages=view)
    assert key in gate._snapshots       # the first summary on record: nothing prior to compare against
    gate.on_pre_compress([])            # generation advances, the held view is dropped
    gate.sync_turn("u", "a", session_id="s", messages=view)  # only the SAME summary: this compaction's row absent
    assert key not in gate._snapshots


def test_a_turn_without_messages_drops_the_held_view(gate):
    """Blocker 2: a sync that carries no messages proves nothing about the view - the held view is dropped, so no
    fact stays held on an empty turn's silence (messages=None and messages=[] both)."""
    key = _key(gate)
    block = [{"role": "user", "api_content": build_memory_context_block(LINE)}]
    gate.sync_turn("u", "a", session_id="s", messages=block)
    assert key in gate._snapshots
    gate.sync_turn("u", "a", session_id="s", messages=None)
    assert key not in gate._snapshots
    gate.sync_turn("u", "a", session_id="s", messages=block)
    assert key in gate._snapshots
    gate.sync_turn("u", "a", session_id="s", messages=[])
    assert key not in gate._snapshots


def test_a_pasted_block_is_not_visibility(gate):
    """Blocker 3: recall lines count only inside a block the core fenced (its system note, in a user row's
    api_content) - a user pasting our block, in api_content or content, proves nothing."""
    key = _key(gate)
    fact = {"fact_id": 7, "content": CYR}
    gate._remember(key, fact, 1, LINE)
    gate._turns[key] = 5
    pasted = "<memory-context>\n" + LINE + "\n</memory-context>"  # no system note: a user's paste
    gate.sync_turn("u", "a", session_id="s", messages=[{"role": "user", "api_content": pasted}])
    assert gate._not_recent(key, fact, turn=10) is True
    gate.sync_turn("u", "a", session_id="s", messages=[{"role": "user", "content": pasted}])
    assert gate._not_recent(key, fact, turn=10) is True


def test_a_malformed_fact_store_result_is_not_visibility(gate):
    """Blocker 3, fail open: an unparseable fact_store result yields no pairs - the window alone decides."""
    key = _key(gate)
    fact = {"fact_id": 7, "content": CYR}
    gate._remember(key, fact, 1, LINE)
    gate._turns[key] = 5
    gate.sync_turn("u", "a", session_id="s",
                   messages=[{"role": "tool", "tool_name": "fact_store", "content": '{"found": [broken'}])
    assert gate._not_recent(key, fact, turn=10) is True


def test_the_same_fact_id_with_other_content_does_not_hold(gate):
    """Blocker 3: visibility is the (fact_id, content) PAIR - another content under our id (an id-only match)
    never holds the candidate."""
    key = _key(gate)
    mine = {"fact_id": 7, "content": "X keeps the backup tape offsite"}
    gate._remember(key, mine, 1, "- #7 [0.5] X keeps the backup tape offsite")
    gate._turns[key] = 5
    gate.sync_turn("u", "a", session_id="s",
                   messages=[{"role": "tool", "tool_name": "fact_store",
                              "content": json.dumps({"found": [{"fact_id": 7,
                                                                "content": "Y keeps the backup tape on the shelf"}]})}])
    assert gate._not_recent(key, mine, turn=10) is True


def test_a_late_sync_counts_and_logs_the_missed_turns(gate, caplog):
    """Condition 3 measurement (should-fix 6): the sync's origin turn is bound to its view, and a sync that
    starts after a LATER turn's prefetch already ran counts the lag as its own number - separate from grace -
    logging the exact missed-turn count."""
    import logging as _logging

    key = _key(gate)
    view = [{"role": "user", "api_content": build_memory_context_block(LINE)}]
    gate.sync_turn("u", "a", session_id="s", messages=view)   # baseline sync at prefetch turn 0
    gate._turns[key] = 3                                       # two later prefetch turns ran meanwhile
    with caplog.at_level(_logging.DEBUG):
        gate.sync_turn("u", "a", session_id="s", messages=view)
    assert gate._presence_lag_events == 1
    assert gate._presence_lag_turns == 2
    assert any("2 turn(s)" in r.getMessage() for r in caplog.records)  # exact missed-turn count in the log

"""0.21.0: prefetch_max_tokens — a budget, in approximate tokens, for the facts recalled before a turn.

The person sets a budget so recalled memory does not crowd a small context. What they must be able to rely on:
the budget never cuts a fact (a cut fact can lose its "not", its condition or its date: the model would read a
different claim; the older 300-character line cap is separate and unchanged), the facts that do go in keep their
ranking order, the whole block — header and journal line included — stays within the budget, a fact left out is not
treated as recalled, and the count of left-out facts is reported without their text. With the key unset or 0, recall injects exactly what it did before."""
from __future__ import annotations

import json
import logging

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider

# A message that names a person and asks about a topic: the topic fact comes first (it shares the words), then the
# person's own facts (the entity stream). The topic fact is long and full of conditions — the kind a cut would break.
LONG = ("The team travel budget is not approved by default: flights to the Lisbon office are reimbursed only when "
        "the trip is booked two weeks ahead, the hotel costs under 120 euros a night, and the team lead signs the "
        "request before 2026-11-01.")
FACTS = [(LONG, []),
         ("Marta leads the Kestrel project.", ["Marta"]),
         ("Marta works from the Porto office.", ["Marta"]),
         ("The office printer is on the second floor.", [])]
QUERY = "Marta asked me about the travel budget"


def _provider(tmp_path, name, **config):
    p = HolographicMemoryProvider({"db_path": str(tmp_path / f"{name}.db"), **config})
    p.initialize(session_id=name)
    ids = {}
    for text, entities in FACTS:
        ids[text] = json.loads(p.handle_tool_call(
            "fact_store", {"action": "add", "content": text, "entities": entities, "origin": "said"}))["fact_id"]
    return p, ids


def _ids(block: str) -> list[int]:
    return [int(line.split()[1][1:]) for line in block.splitlines() if line.startswith("- #")]


def _stats(p) -> dict:
    return json.loads(p.handle_tool_call("fact_store", {"action": "stats"}))


def test_a_fact_that_does_not_fit_is_left_out_whole_and_the_rest_keep_their_order(tmp_path):
    off, ids = _provider(tmp_path, "off")
    on, _ = _provider(tmp_path, "on", prefetch_max_tokens=60)
    try:
        full = off.prefetch(QUERY, session_id="turn")
        assert _ids(full)[0] == ids[LONG] and len(_ids(full)) == 3  # the setting: long first, two short after
        block = on.prefetch(QUERY, session_id="turn")
        # Left out whole: not a word of it, no ellipsis; the shorter facts after it still went in, in the same order.
        assert "travel budget" not in block and "Lisbon" not in block and "…" not in block
        assert _ids(block) == [i for i in _ids(full) if i != ids[LONG]]
        for line in block.splitlines()[1:]:
            assert line in full.splitlines()  # every kept line is the line recall makes without a budget
        from hermes_plugin_holographic import approx_tokens
        assert approx_tokens(block) <= 60
    finally:
        off.shutdown()
        on.shutdown()


def test_the_attempt_line_of_the_journal_counts_too(tmp_path):
    p, ids = _provider(tmp_path, "journal", prefetch_max_tokens=45, experience="on")
    try:
        block = p.prefetch(QUERY, session_id="journal")
        assert "Memory attempt:" in block
        assert len(_ids(block)) == 1 and ids[LONG] not in _ids(block)
        from hermes_plugin_holographic import approx_tokens
        assert approx_tokens(block) <= 45
    finally:
        p.shutdown()


def test_a_left_out_fact_is_counted_without_its_text_and_is_not_marked_recalled(tmp_path, caplog):
    p, ids = _provider(tmp_path, "report", prefetch_max_tokens=85)
    try:
        with caplog.at_level(logging.DEBUG, logger="hermes_plugin_holographic"):
            first = p.prefetch(QUERY, session_id="talk")
        assert _ids(first) == [ids[LONG]]  # the long fact alone fills the budget: both short ones are left out
        budget = _stats(p)["prefetch_budget"]
        assert budget["max_tokens"] == 85
        assert budget["last"]["kept"] == 1 and budget["last"]["dropped"] == 2
        assert budget["dropped_total"] == 2
        logged = "\n".join(r.getMessage() for r in caplog.records)
        assert "left out 2" in logged
        for text, _ in FACTS:
            assert text[:20] not in logged and text[:20] not in json.dumps(budget)
        assert QUERY not in logged and "Marta" not in logged
        # Not injected, so not "recalled within prefetch_repeat_turns": the next turn brings them.
        second = p.prefetch(QUERY, session_id="talk")
        assert sorted(_ids(second)) == sorted([ids[FACTS[1][0]], ids[FACTS[2][0]]])
        assert _stats(p)["prefetch_budget"]["dropped_total"] == 2
    finally:
        p.shutdown()


def test_a_recall_that_finds_nothing_reports_nothing_left_out(tmp_path):
    """The last recall's counts describe the last recall: after a turn that left two facts out, a turn whose message
    matches nothing must not still report them (Codex review of #50); the running total stays."""
    p, _ = _provider(tmp_path, "quiet", prefetch_max_tokens=85)
    try:
        p.prefetch(QUERY, session_id="talk")
        assert _stats(p)["prefetch_budget"]["last"]["dropped"] == 2
        assert p.prefetch("zebra violin satellite", session_id="talk") == ""
        budget = _stats(p)["prefetch_budget"]
        assert budget["last"] == {"kept": 0, "dropped": 0, "tokens": 0}
        assert budget["dropped_total"] == 2
    finally:
        p.shutdown()


def test_unset_or_zero_injects_what_it_did_before(tmp_path):
    unset, _ = _provider(tmp_path, "unset")
    zero, _ = _provider(tmp_path, "zero", prefetch_max_tokens=0)
    try:
        before = unset.prefetch(QUERY, session_id="turn")
        assert LONG in before and len(_ids(before)) == 3
        assert zero.prefetch(QUERY, session_id="turn") == before
        assert "prefetch_budget" not in _stats(unset) and "prefetch_budget" not in _stats(zero)
    finally:
        unset.shutdown()
        zero.shutdown()


@pytest.mark.parametrize("value", ["a lot", -5, 0.5, -0.5, 60.9])  # a fraction is no budget either: int() would
# turn 0.5 into 0 (no limit) and 60.9 into 60 without a word (Codex review of #50)
def test_a_value_that_is_no_budget_is_a_config_problem(tmp_path, value):
    p = HolographicMemoryProvider({"db_path": str(tmp_path / "bad.db"), "prefetch_max_tokens": value})
    p.initialize(session_id="bad")
    try:
        assert "prefetch_max_tokens" in p.system_prompt_block()
        assert not (tmp_path / "bad.db").exists()  # memory is off until it is fixed, like any other bad number
    finally:
        p.shutdown()

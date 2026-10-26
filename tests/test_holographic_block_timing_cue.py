"""Issue #34: a standing rule said once was never stored — the block said preferences are worth storing, not when.

Case (KostinPavel, 2026-10-20): the user's first message of a task said «Обновлять плагин нештатным способом не нужно»;
in the same session the agent stored three other facts and not this rule, which a nightly extraction found five days
later; five such misses in six days, always next to `add`s on other subjects. The requirement: the block tells the
agent that the moment a person states a standing preference, rule or limitation is the moment to store it — in that
same turn — and the block stays a fixed text (no live numbers, see test_holographic_prompt_stability)."""
from __future__ import annotations

from hermes_plugin_holographic import HolographicMemoryProvider


def _block(tmp_path) -> str:
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64})
    provider.initialize(session_id="s1")
    try:
        provider._store.add_fact("A fact so the store is not empty")
        return provider.system_prompt_block()
    finally:
        provider.shutdown()


def test_the_block_says_to_store_a_standing_rule_in_the_turn_it_is_said(tmp_path):
    lines = [line for line in _block(tmp_path).splitlines() if "same turn" in line]
    assert lines, "no line tells the agent when to store a standing preference or rule"
    cue = lines[0].lower()
    assert "preference" in cue and "rule" in cue, cue


def test_the_cue_names_a_rule_said_in_passing(tmp_path):
    """The miss was a rule given as an aside at the start of a task, not a preference announced as such."""
    cue = next(line for line in _block(tmp_path).splitlines() if "same turn" in line)
    assert "in passing" in cue or "aside" in cue, cue

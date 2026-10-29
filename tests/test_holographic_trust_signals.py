"""Two trust signals that said more than they knew (F1 research, plans/2026-10-29-f1-reliability-usefulness-research.md
§1 and §7, approved 2026-10-30).

Requirements, in the user's words:
- Feedback on a fact (helpful, unhelpful) is a vote on the fact, not a new fact: it does not make an old fact look
  recent. With temporal decay on, an old fact marked unhelpful must not jump above a recent one; a fact marked
  helpful rises only by its trust, not by its age. Decay counts from the date the current version was recorded, so a
  store whose facts an earlier release already made "young" by feedback ranks them by their real age again.
- A correction of the text (`correct content=`, `update content=`) is a new claim: it starts at its origin's trust,
  as `change` does, without the helpful marks, recall count or confirmations the wrong text earned. A correction of
  the date only keeps all of them. `revert` brings the old text back with the trust it had."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider
from hermes_plugin_holographic.retrieval import FactRetriever
from hermes_plugin_holographic.store import MemoryStore


def _provider(tmp_path, session="s1", **config):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "m.db"), "hrr_dim": 64, **config})
    p.initialize(session_id=session)
    return p


@pytest.fixture
def p(tmp_path):
    provider = _provider(tmp_path)
    yield provider
    provider.shutdown()


def _call(p, tool="fact_store", **args):
    return json.loads(p.handle_tool_call(tool, args))


def _row(p, fact_id):
    return p._store.read("SELECT trust_score, helpful_count, retrieval_count FROM facts WHERE fact_id = ?",
                         (fact_id,))[0]


def _ago(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


def _stored_days_ago(store, fact_id: int, days: float, *, updated_days: float | None = None) -> None:
    """What a fact stored *days* ago looks like on disk: the row and its current version carry that time. A different
    *updated_days* is a row an earlier release touched later (its feedback stamped updated_at)."""
    when = _ago(days)
    store._write("UPDATE facts SET created_at = ?, updated_at = ? WHERE fact_id = ?",
                 (when, _ago(updated_days) if updated_days is not None else when, fact_id))
    store._write("UPDATE fact_versions SET recorded_at = ?, recorded_on = ? WHERE fact_id = ? AND status = 'current'",
                 (when, when[:10], fact_id))
    store._conn.commit()


def _order(p, query):
    return [r["fact_id"] for r in _call(p, action="search", query=query)["results"]]


# -- feedback does not make a fact young -------------------------------------------------------------------------------

def test_unhelpful_does_not_lift_an_old_fact_over_a_recent_one(tmp_path):
    p = _provider(tmp_path, temporal_decay_half_life=1)
    try:
        old = _call(p, action="add", content="Irma swims on Sundays at the city pool")["fact_id"]
        new = _call(p, action="add", content="Irma swims on Tuesdays at the school pool")["fact_id"]
        _stored_days_ago(p._store, old, 10)
        _stored_days_ago(p._store, new, 2)
        assert _order(p, "Irma swims pool") == [new, old]
        _call(p, "fact_feedback", action="unhelpful", fact_id=old)
        assert _order(p, "Irma swims pool") == [new, old]  # trust fell: it can only sink
    finally:
        p.shutdown()


def test_helpful_raises_a_fact_by_its_trust_not_by_its_age(tmp_path):
    p = _provider(tmp_path, temporal_decay_half_life=1)
    try:
        fact = _call(p, action="add", content="Irma swims on Sundays at the city pool")["fact_id"]
        _stored_days_ago(p._store, fact, 10)
        before = _call(p, action="search", query="Irma swims pool")["results"][0]
        _call(p, "fact_feedback", action="helpful", fact_id=fact)
        after = _call(p, action="search", query="Irma swims pool")["results"][0]
        assert after["trust_score"] == pytest.approx(before["trust_score"] + 0.05)
        assert after["score"] / before["score"] == pytest.approx(after["trust_score"] / before["trust_score"], rel=1e-3)
    finally:
        p.shutdown()


def test_feedback_leaves_the_time_a_fact_last_changed(p):
    """`updated_at` is when the fact last changed ("recently changed", the fallback for decay without history)."""
    fact = _call(p, action="add", content="Irma swims on Sundays")["fact_id"]
    _stored_days_ago(p._store, fact, 10)

    def changed():
        return p._store.read("SELECT updated_at FROM facts WHERE fact_id = ?", (fact,))[0][0]

    stamp = changed()
    _call(p, "fact_feedback", action="unhelpful", fact_id=fact)
    _call(p, "fact_feedback", action="helpful", fact_id=fact)
    assert changed() == stamp


def test_a_fact_an_earlier_release_made_young_by_feedback_decays_from_when_it_was_recorded(tmp_path):
    """A live store already holds facts whose updated_at a past unhelpful stamped: they rank by their real age."""
    p = _provider(tmp_path, temporal_decay_half_life=1)
    try:
        old = _call(p, action="add", content="Irma swims on Sundays at the city pool")["fact_id"]
        new = _call(p, action="add", content="Irma swims on Tuesdays at the school pool")["fact_id"]
        _stored_days_ago(p._store, old, 10, updated_days=0.01)  # an unhelpful from 0.22.2 a quarter hour ago
        _stored_days_ago(p._store, new, 2)
        p._store._write("UPDATE facts SET trust_score = 0.4 WHERE fact_id = ?", (old,))
        p._store._conn.commit()
        assert _order(p, "Irma swims pool") == [new, old]
    finally:
        p.shutdown()


def test_without_history_decay_still_reads_the_last_change(tmp_path):
    """A store no writer of 0.8.0 has migrated (a reader opened it first) has no versions: decay counts from
    updated_at, which feedback no longer moves."""
    store = MemoryStore(db_path=str(tmp_path / "plain.db"), default_trust=0.5, hrr_dim=64, migrate=False)
    try:
        assert not store.history_ready()
        old = store.add_fact_status("Irma swims on Sundays at the city pool")[0]
        new = store.add_fact_status("Irma swims on Tuesdays at the school pool")[0]
        store._write("UPDATE facts SET created_at = ?, updated_at = ? WHERE fact_id = ?", (_ago(10), _ago(10), old))
        store._write("UPDATE facts SET created_at = ?, updated_at = ? WHERE fact_id = ?", (_ago(2), _ago(2), new))
        store._conn.commit()
        store.record_feedback(old, helpful=False)
        found = FactRetriever(store, temporal_decay_half_life=1, hrr_dim=64).search("Irma swims pool", min_trust=0.0)
        assert [f["fact_id"] for f in found] == [new, old]
    finally:
        store.close()


# -- a corrected text starts over --------------------------------------------------------------------------------------

def _earned(tmp_path, origin="said"):
    """A fact the person said, marked helpful three times, recalled, and confirmed by a second person."""
    first = _provider(tmp_path, "s1")
    fact = _call(first, action="add", content="Irma's school starts at 8", origin=origin)["fact_id"]
    for _ in range(3):
        _call(first, "fact_feedback", action="helpful", fact_id=fact)
    first._store.record_retrievals([fact])
    first.shutdown()
    second = _provider(tmp_path, "s2")
    second._store.record_confirmation(fact, {"session_id": "s2", "user_id": "u2", "via": "test"})
    return second, fact


@pytest.mark.parametrize("how", ["correct", "update"])
def test_a_corrected_text_starts_at_its_origins_trust_without_the_old_texts_marks(tmp_path, how):
    p, fact = _earned(tmp_path)
    try:
        assert tuple(_row(p, fact)) == (pytest.approx(0.8), 3, 1)
        assert p._store.origins([fact])[fact]["confirmations"] == 1
        result = _call(p, action=how, fact_id=fact, content="Irma's school starts at 9")
        assert "error" not in result, result
        trust, helpful, recalled = _row(p, fact)
        assert trust == pytest.approx(0.6)  # said: default 0.5 + 0.1, as a change starts
        assert (helpful, recalled) == (0, 0)
        assert p._store.origins([fact])[fact] == {"origin": "said", "confirmations": 0}
    finally:
        p.shutdown()


def test_a_corrected_inference_starts_as_an_inference_unless_the_person_said_the_new_text(tmp_path):
    p, fact = _earned(tmp_path, origin="inferred")
    try:
        _call(p, action="correct", fact_id=fact, content="Irma's school starts at 9")
        assert _row(p, fact)[0] == pytest.approx(0.4)
        assert p._store.origins([fact])[fact] == {"origin": "inferred", "confirmations": 0}
        _call(p, action="correct", fact_id=fact, content="Irma's school starts at 7:45", origin="said")
        assert _row(p, fact)[0] == pytest.approx(0.6)
    finally:
        p.shutdown()


def test_update_with_a_new_text_and_a_trust_step_starts_over_then_takes_the_step(tmp_path):
    p, fact = _earned(tmp_path)
    try:
        _call(p, action="update", fact_id=fact, content="Irma's school starts at 9", trust_delta=-0.1)
        assert _row(p, fact)[0] == pytest.approx(0.5)
    finally:
        p.shutdown()


def test_a_correction_of_the_date_only_keeps_trust_marks_and_confirmations(tmp_path):
    p, fact = _earned(tmp_path)
    try:
        result = _call(p, action="correct", fact_id=fact, valid_from="2026-09-01")
        assert result["status"] == "corrected"
        assert tuple(_row(p, fact)) == (pytest.approx(0.8), 3, 1)
        assert p._store.origins([fact])[fact]["confirmations"] == 1
        # the same text through update is no correction at all
        _call(p, action="update", fact_id=fact, content="Irma's school starts at 8")
        assert tuple(_row(p, fact)) == (pytest.approx(0.8), 3, 1)
    finally:
        p.shutdown()


def test_revert_of_a_correction_brings_the_old_texts_trust_back(tmp_path):
    p, fact = _earned(tmp_path)
    try:
        _call(p, action="correct", fact_id=fact, content="Irma's school starts at 9")
        _call(p, "fact_feedback", action="unhelpful", fact_id=fact)
        assert _row(p, fact)[0] == pytest.approx(0.5)
        assert _call(p, action="revert", fact_id=fact)["status"] == "reverted"
        assert p._store.read("SELECT content FROM facts WHERE fact_id = ?", (fact,))[0][0] == "Irma's school starts at 8"
        assert _row(p, fact)[0] == pytest.approx(0.8)  # the snapshot of the old text, as after a change
    finally:
        p.shutdown()


def test_a_correction_of_a_merged_fact_drops_the_groups_helpful_marks(tmp_path):
    p = _provider(tmp_path)
    try:
        keep = _call(p, action="add", content="Alice Smith lives in Paris near the Louvre", origin="said")["fact_id"]
        copy = _call(p, action="add", content="Alice Smith is living in Paris by the Louvre", origin="said")["fact_id"]
        _call(p, "fact_feedback", action="helpful", fact_id=copy)
        _call(p, "fact_feedback", action="helpful", fact_id=copy)
        p._store.record_confirmation(copy, {"session_id": "s9", "user_id": "u9", "via": "test"})
        _call(p, action="merge", fact_id=keep, **{"from": [copy]})
        assert _row(p, keep)[1] == 2
        _call(p, action="correct", fact_id=keep, content="Alice Smith lives in Lyon near the old town")
        trust, helpful, _ = _row(p, keep)
        assert (trust, helpful) == (pytest.approx(0.6), 0)
        # the copy said the wrong claim: a person who confirmed it confirms the corrected text as a new claim
        out = p._store.record_confirmation(keep, {"session_id": "s10", "user_id": "u9", "via": "test"})
        assert out["trust_score"] == pytest.approx(0.65)
    finally:
        p.shutdown()

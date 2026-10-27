"""Task outcomes describe relevance, never the truth of a stored claim."""
import json
import sqlite3

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider
from hermes_plugin_holographic.store import MemoryStore


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider({"db_path": str(tmp_path / "facts.db"), "experience": "on",
                                   "experience_trace": "on"})
    p.initialize(session_id="task-session")
    yield p
    p.shutdown()


def call(p, action, **args):
    return json.loads(p.handle_tool_call("fact_store", {"action": action, **args}))


def experience(p, action, **args):
    return json.loads(p.handle_tool_call("fact_experience", {"action": action, **args}))


def seed(p):
    fid = call(p, "add", content="Alice Smith works on the observatory project", origin="said")["fact_id"]
    result = call(p, "search", query="Alice observatory")
    return fid, result["memory_attempt"]


def test_wrong_entity_does_not_reduce_trust_or_correct_fact(provider):
    fid, aid = seed(provider)
    before = call(provider, "history", fact_id=fid)
    result = experience(provider, "outcome", attempt_id=aid, event_id="event-1", outcome="corrected",
                        reason="wrong_entity", correction="Use the other Alice for this task.")
    assert result["recorded"]
    assert call(provider, "history", fact_id=fid) == before
    report = experience(provider, "explain", attempt_id=aid)
    assert report["outcomes"][0]["actor"] == "agent"
    assert report["emitted"][0]["fact_id"] == fid
    assert "Alice Smith works" not in json.dumps(report)


def test_revision_is_one_attempt_and_delivery_is_idempotent(provider):
    _, aid = seed(provider)
    first = dict(attempt_id=aid, event_id="event-1", outcome="dead_end", reason="wrong_entity")
    assert experience(provider, "outcome", **first)["recorded"]
    assert experience(provider, "outcome", **first)["already"]
    assert "error" in experience(provider, "outcome", **{**first, "outcome": "useful"})
    assert experience(provider, "outcome", attempt_id=aid, event_id="event-2", outcome="useful",
                      reason="other", supersedes="event-1")["recorded"]
    assert "error" in experience(provider, "outcome", attempt_id=aid, event_id="event-3", outcome="useful",
                                 reason="other", supersedes="event-1")
    report = experience(provider, "explain", attempt_id=aid)
    assert len(report["outcomes"]) == 2
    assert report["outcomes"][-1]["active"]
    assert not report["outcomes"][0]["active"]


def test_feedback_stays_on_used_version(provider):
    fid, aid = seed(provider)
    old = experience(provider, "explain", attempt_id=aid)["emitted"][0]["version_id"]
    call(provider, "change", fact_id=fid, content="Alice Smith now works on the museum project")
    assert experience(provider, "outcome", attempt_id=aid, event_id="old-version", outcome="dead_end",
                      reason="stale_version")["recorded"]
    assert experience(provider, "explain", attempt_id=aid)["emitted"][0]["version_id"] == old


def test_forget_removes_receipt_correction_and_late_delivery(provider):
    fid, aid = seed(provider)
    experience(provider, "outcome", attempt_id=aid, event_id="private-result", outcome="corrected",
               reason="wrong_entity", correction="The corrected destination is the amber warehouse.")
    assert call(provider, "forget", fact_id=fid)["forgotten"]
    assert experience(provider, "explain", attempt_id=aid)["status"] == "not_recorded"
    assert "error" in experience(provider, "outcome", attempt_id=aid, event_id="late", outcome="useful",
                                 reason="other")
    assert not provider._store.read("SELECT * FROM memory_outcomes")
    assert b"amber warehouse" not in provider._store.db_path.read_bytes()


def test_off_creates_no_journal_and_exposes_no_extra_tool(tmp_path):
    p = HolographicMemoryProvider({"db_path": str(tmp_path / "off.db")})
    p.initialize(session_id="off")
    try:
        call(p, "add", content="Alice Smith works on the observatory project")
        assert "memory_attempt" not in call(p, "search", query="Alice observatory")
        assert "fact_experience" not in [s["name"] for s in p.get_tool_schemas()]
        assert not p._store.read("SELECT name FROM sqlite_master WHERE name = 'memory_attempts'")
    finally:
        p.shutdown()


def test_no_trace_still_records_actual_emission(provider):
    provider._experience_trace = False
    _, aid = seed(provider)
    report = experience(provider, "explain", attempt_id=aid)
    assert report["emitted"]
    assert report["stages"] == []
    assert report["trace_status"] == "not_recorded"


def test_missing_receipt_cannot_assert_use_or_human_confirmation(provider):
    assert "error" in experience(provider, "outcome", attempt_id="invented", event_id="fake", outcome="useful",
                                 reason="other")
    _, aid = seed(provider)
    assert "error" in experience(provider, "outcome", attempt_id=aid, event_id="fake-human", outcome="useful",
                                 reason="other", actor="human")


def test_expired_receipt_is_unavailable_and_pruned(provider):
    _, aid = seed(provider)
    provider._store._conn.execute("UPDATE memory_attempts SET created_at = 0")
    assert experience(provider, "explain", attempt_id=aid)["status"] == "expired"
    assert "error" in experience(provider, "outcome", attempt_id=aid, event_id="late", outcome="useful",
                                 reason="other")
    call(provider, "search", query="Alice observatory")
    assert not provider._store.read("SELECT 1 FROM memory_attempts WHERE attempt_id = ?", (aid,))


def test_sql_delete_by_old_writer_also_erases_journal(provider):
    fid, aid = seed(provider)
    conn = sqlite3.connect(provider._store.db_path)
    try:
        conn.execute("DELETE FROM facts WHERE fact_id = ?", (fid,))
        conn.commit()
    finally:
        conn.close()
    assert experience(provider, "explain", attempt_id=aid)["status"] == "not_recorded"


def test_prefetch_receipt_matches_emitted_ids(provider):
    fid, _ = seed(provider)
    block = provider.prefetch("Alice observatory", session_id="new-task")
    assert f"#{fid}" in block
    recent = experience(provider, "recent")["attempts"][0]
    report = experience(provider, "explain", attempt_id=recent["attempt_id"])
    assert report["kind"] == "prefetch"
    assert [r["fact_id"] for r in report["emitted"]] == [fid]


def test_scope_and_readonly_do_not_allow_cross_store_writes(provider, tmp_path):
    _, aid = seed(provider)
    other = MemoryStore(tmp_path / "other.db")
    try:
        assert other.experience_report(aid)["status"] == "not_recorded"
    finally:
        other.close()
    provider._tool_writable = False
    assert "error" in experience(provider, "outcome", attempt_id=aid, event_id="readonly", outcome="useful",
                                 reason="other")
    assert experience(provider, "explain", attempt_id=aid)["status"] == "ok"


def test_task_and_correct_source_are_erased_with_either_dependency(provider):
    _, aid = seed(provider)
    correct = call(provider, "add", content="Alice Brown works on the observatory engineering team")["fact_id"]
    version = call(provider, "history", fact_id=correct)["versions"][-1]["version_id"]
    assert experience(provider, "outcome", attempt_id=aid, event_id="correct-source", outcome="corrected",
        reason="wrong_entity", task="Find the observatory engineer", correction="The correct source is Alice Brown.",
        correction_refs=[{"fact_id": correct, "version_id": version}])["recorded"]
    report = experience(provider, "explain", attempt_id=aid)
    assert report["outcomes"][0]["task"] == "Find the observatory engineer"
    assert report["outcomes"][0]["correction_refs"][0]["fact_id"] == correct
    call(provider, "forget", fact_id=correct)
    assert experience(provider, "explain", attempt_id=aid)["status"] == "not_recorded"


def test_secret_or_unbound_correct_source_is_rejected(provider):
    _, aid = seed(provider)
    assert "error" in experience(provider, "outcome", attempt_id=aid, event_id="secret", outcome="corrected",
        reason="wrong_entity", task="API_KEY=sk-" + "a" * 48)
    assert "error" in experience(provider, "outcome", attempt_id=aid, event_id="bad-ref", outcome="corrected",
        reason="wrong_entity", correction_refs=[{"fact_id": 999999, "version_id": 999999}])


def test_empty_success_has_a_receipt_but_no_invented_failure(provider):
    result = call(provider, "search", query="unicorn zeppelin")
    assert result["results"] == []
    report = experience(provider, "explain", attempt_id=result["memory_attempt"])
    assert report["emitted"] == []
    assert report["outcomes"] == []


def test_nonemitted_candidate_also_participates_in_forget(provider):
    fid, _ = seed(provider)
    other = call(provider, "add", content="Boris works on the museum collection")["fact_id"]
    facts = [dict(r) for r in provider._store.read("SELECT * FROM facts ORDER BY fact_id")]
    aid = provider._store.record_attempt("search", [facts[0]], [("lexical_candidates", facts)],
                                        provider._experience_session(), True)
    assert facts[0]["fact_id"] == fid
    call(provider, "forget", fact_id=other)
    assert experience(provider, "explain", attempt_id=aid)["status"] == "not_recorded"


def test_busy_journal_does_not_suppress_search(provider):
    seed(provider)
    external = sqlite3.connect(provider._store.db_path)
    try:
        external.execute("BEGIN IMMEDIATE")
        provider._store._conn.execute("PRAGMA busy_timeout=20")
        result = call(provider, "search", query="Alice observatory")
        assert result["results"]
        assert "memory_attempt" not in result
        assert "OperationalError" in experience(provider, "recent")["error_kind"]
    finally:
        external.rollback()
        external.close()
        provider._store._conn.execute("PRAGMA busy_timeout=10000")


def test_merge_then_forget_keep_erases_outcomes_of_copy(provider):
    keep, _ = seed(provider)
    copy = call(provider, "add", content="Alice Smith is working on the observatory project")["fact_id"]
    result = call(provider, "search", query="Alice observatory")
    aid = result["memory_attempt"]
    assert experience(provider, "outcome", attempt_id=aid, event_id="before-merge", outcome="useful",
                      reason="other")["recorded"]
    merged = call(provider, "merge", fact_id=keep, **{"from": [copy]})
    assert "error" not in merged, merged
    assert experience(provider, "explain", attempt_id=aid)["status"] == "ok"
    assert call(provider, "forget", fact_id=keep)["forgotten"]
    assert experience(provider, "explain", attempt_id=aid)["status"] == "not_recorded"


def test_end_preserves_the_used_version_and_its_outcome(provider):
    fid, aid = seed(provider)
    call(provider, "end", fact_id=fid)
    assert experience(provider, "explain", attempt_id=aid)["status"] == "ok"
    assert experience(provider, "outcome", attempt_id=aid, event_id="ended-version", outcome="dead_end",
                      reason="stale_version")["recorded"]


def test_completed_trace_does_not_keep_a_second_in_memory_copy(provider):
    fid, _ = seed(provider)
    call(provider, "forget", fact_id=fid)
    assert provider._experience_stages == []

"""Task outcomes describe relevance, never the truth of a stored claim."""
import json
import re
import importlib.util
from pathlib import Path
import sqlite3
import sys
import threading
import time

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


def test_journal_retries_transient_sqlite_writer_without_duplicate_attempts(provider):
    seed(provider)
    initial = len(experience(provider, "recent")["attempts"])
    acquired, release = threading.Event(), threading.Event()
    attempts = []

    def writer():
        external = sqlite3.connect(provider._store.db_path, isolation_level=None)
        try:
            external.execute("BEGIN IMMEDIATE")
            acquired.set()
            release.wait(3)
            external.execute("ROLLBACK")
        finally:
            external.close()

    def trace(statement):
        if statement.upper().startswith("BEGIN IMMEDIATE"):
            attempts.append(statement)
            if len(attempts) >= 2:
                release.set()

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        assert acquired.wait(5)
        provider._store._conn.set_trace_callback(trace)
        result = call(provider, "search", query="Alice observatory")
        assert result.get("memory_attempt"), "temporary SQLite contention lost the receipt"
        assert len(attempts) >= 2
        assert len(experience(provider, "recent")["attempts"]) == initial + 1
        assert experience(provider, "recent")["error_kind"] == ""
    finally:
        provider._store._conn.set_trace_callback(None)
        release.set()
        thread.join(5)
    assert not thread.is_alive()


def test_invalid_reference_is_not_retried_or_stored(provider):
    fid, _ = seed(provider)
    before = len(experience(provider, "recent")["attempts"])
    fact = dict(provider._store.read("SELECT * FROM facts WHERE fact_id=?", (fid,))[0])
    fact["content"] = "Outdated content for this fact"
    begins = []
    provider._store._conn.set_trace_callback(lambda sql: begins.append(sql) if sql.upper().startswith("BEGIN IMMEDIATE") else None)
    try:
        with pytest.raises(ValueError, match="Memory changed during retrieval"):
            provider._store.record_attempt("search", [fact], [], provider._experience_session())
    finally:
        provider._store._conn.set_trace_callback(None)
    assert len(begins) == 1
    assert len(experience(provider, "recent")["attempts"]) == before


def test_nonbusy_sqlite_failure_is_not_retried(provider):
    seed(provider)
    facts = [dict(r) for r in provider._store.read("SELECT * FROM facts LIMIT 1")]
    begins = []
    provider._store._conn.execute("PRAGMA query_only=ON")
    provider._store._conn.set_trace_callback(lambda sql: begins.append(sql) if sql.upper().startswith("BEGIN IMMEDIATE") else None)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            provider._store.record_attempt("search", facts, [], provider._experience_session())
    finally:
        provider._store._conn.set_trace_callback(None)
        provider._store._conn.execute("PRAGMA query_only=OFF")
    assert len(begins) == 1
    assert not provider._store._conn.in_transaction


def test_retry_after_blocked_commit_records_once(provider):
    seed(provider)
    before = len(experience(provider, "recent")["attempts"])
    assert provider._store._conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    acquired, release = threading.Event(), threading.Event()
    begins = []

    def reader():
        external = sqlite3.connect(provider._store.db_path, isolation_level=None)
        try:
            external.execute("BEGIN")
            external.execute("SELECT count(*) FROM facts").fetchone()
            acquired.set()
            release.wait(3)
            external.execute("ROLLBACK")
        finally:
            external.close()

    def trace(statement):
        if statement.upper().startswith("BEGIN IMMEDIATE"):
            begins.append(statement)
            if len(begins) >= 2:
                release.set()

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        assert acquired.wait(5)
        provider._store._conn.set_trace_callback(trace)
        result = call(provider, "search", query="Alice observatory")
        assert result.get("memory_attempt"), "a temporary reader lock lost the receipt"
        assert len(begins) >= 2
        assert len(experience(provider, "recent")["attempts"]) == before + 1
        assert not provider._store._conn.in_transaction
    finally:
        provider._store._conn.set_trace_callback(None)
        release.set()
        thread.join(5)
    assert not thread.is_alive()


@pytest.mark.parametrize("operation", ["change", "forget"])
def test_retry_refuses_a_fact_changed_or_forgotten_between_attempts(provider, operation):
    fid, _ = seed(provider)
    module = sys.modules[HolographicMemoryProvider.__module__]
    module_file = module.__file__
    spec = importlib.util.spec_from_file_location("retry_other_provider", module_file,
        submodule_search_locations=[str(Path(module_file).parent)])
    other_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = other_module
    spec.loader.exec_module(other_module)
    other = other_module.HolographicMemoryProvider({"db_path": str(provider._store.db_path), "experience": "on"})
    other.initialize(session_id="other-session")
    acquired, release, changed = threading.Event(), threading.Event(), threading.Event()
    events, result_other, callback_wait = [], [], []

    def concurrent_change():
        try:
            other._store._conn.execute("BEGIN IMMEDIATE")
            acquired.set()
            release.wait(5)
            other._store._conn.execute("ROLLBACK")
            args = {"fact_id": fid}
            if operation == "change":
                args["content"] = "Alice Smith now works on the museum project"
            result_other.append(call(other, operation, **args))
        finally:
            changed.set()

    def trace(statement):
        if statement.upper().startswith("BEGIN IMMEDIATE"):
            events.append(statement)
            if len(events) == 2:
                release.set()
                callback_wait.append(changed.wait(5))

    thread = threading.Thread(target=concurrent_change)
    thread.start()
    try:
        assert acquired.wait(5)
        provider._store._conn.set_trace_callback(trace)
        result = call(provider, "search", query="Alice observatory")
        assert len(events) >= 2 and callback_wait == [True]
        assert result_other and "error" not in result_other[0], result_other
        assert result["results"] and "memory_attempt" not in result
        assert experience(provider, "recent")["error_kind"].startswith("ValueError")
    finally:
        provider._store._conn.set_trace_callback(None)
        release.set()
        thread.join(8)
        other.shutdown()
        sys.modules.pop(spec.name, None)
    assert not thread.is_alive()


# Another process holds the database; search must answer while it still holds, the journal giving up after a bounded
# busy wait. Wall time is not the measure: SQLite's busy wait is a series of short sleeps the OS stretches, so a 50 ms
# busy timeout took 95 ms on a Windows laptop and, on the macOS runner, 164 ms for a BEGIN that only waited for a lock
# and 160-340 ms for a refused COMMIT (a 0.5 s bound on the whole search failed there at 0.51-0.72 s). So the bound
# is read where it is set: the busy timeout each lock statement of the journal runs under, and how many it runs.
# The hold ends before the connection's own 10 s busy timeout would: a search that waits for the holder instead of
# giving up can only return after the holder has let go, however fast or slow the machine.
_HOLD_S = 5
_JOURNAL_BUSY_MS = 50  # docs/experience.md: a 50 ms busy timeout per lock acquisition, a busy failure retried once
_BUSY_SET = re.compile(r"PRAGMA\s+busy_timeout\s*=\s*(\d+)\s*$", re.IGNORECASE)


def _hold(db_path, *statements):
    """A thread that opens its own connection, runs `statements` in one transaction and holds it until `release` is
    set or _HOLD_S passes. Returns (thread, acquired, release, let_go): `let_go` is set just before it rolls back."""
    acquired, release, let_go = threading.Event(), threading.Event(), threading.Event()

    def holder():
        external = sqlite3.connect(db_path, isolation_level=None)
        try:
            for statement in statements:
                external.execute(statement).fetchall()
            acquired.set()
            release.wait(_HOLD_S)
            let_go.set()
            external.execute("ROLLBACK")
        finally:
            external.close()

    thread = threading.Thread(target=holder)
    thread.start()
    return thread, acquired, release, let_go


def _search_tracing_locks(provider, traced=("BEGIN", "COMMIT")):
    """Search, and list each BEGIN / COMMIT (or other `traced` statement) the store's connection ran with the busy
    timeout in force for it."""
    conn = provider._store._conn
    busy = [conn.execute("PRAGMA busy_timeout").fetchone()[0]]
    locks = []

    def trace(statement):
        statement = statement.strip()
        assigned = _BUSY_SET.match(statement)
        if assigned:
            busy[0] = int(assigned.group(1))
        elif statement.upper().startswith(traced):
            locks.append((statement.upper(), busy[0]))

    conn.set_trace_callback(trace)
    start = time.monotonic()
    try:
        result = call(provider, "search", query="Alice observatory")
    finally:
        conn.set_trace_callback(None)
    return result, locks, time.monotonic() - start


def test_external_writer_cannot_stall_search_for_journal(provider, caplog):
    seed(provider)
    thread, acquired, release, let_go = _hold(provider._store.db_path, "BEGIN IMMEDIATE")
    try:
        assert acquired.wait(5)
        result, locks, elapsed = _search_tracing_locks(provider)
        assert not let_go.is_set(), f"search waited {elapsed:.2f} s, until the other writer let go"
        assert [statement for statement, _ in locks] == ["BEGIN IMMEDIATE"] * 2, locks  # one try, one retry
        assert all(busy <= _JOURNAL_BUSY_MS for _, busy in locks), f"journal lock wait not bounded: {locks}"
        assert result["results"] and "memory_attempt" not in result
        assert experience(provider, "recent")["error_kind"].startswith("OperationalError")
        diagnostic = [r.getMessage() for r in caplog.records if "experience slow" in r.getMessage()]
        assert diagnostic and all(field in diagnostic[-1] for field in ("lock_ms=", "schema_ms=", "begin_ms=", "work_ms=", "commit_ms=", "total_ms="))
        assert "Alice" not in diagnostic[-1]
        assert provider._store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 10000
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert call(provider, "search", query="Alice observatory")["memory_attempt"]
    assert experience(provider, "recent")["error_kind"] == ""


def test_delete_mode_reader_cannot_hold_journal_commit(provider, caplog):
    seed(provider)
    assert provider._store._conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    thread, acquired, release, let_go = _hold(provider._store.db_path, "BEGIN", "SELECT count(*) FROM facts")
    try:
        assert acquired.wait(5)
        result, locks, elapsed = _search_tracing_locks(provider)
        assert not let_go.is_set(), f"journal commit held search {elapsed:.2f} s, until the reader let go"
        # The reader blocks the COMMIT, not the BEGIN: each try gets as far as committing and gives up there.
        assert [statement for statement, _ in locks] == ["BEGIN IMMEDIATE", "COMMIT"] * 2, locks
        assert all(busy <= _JOURNAL_BUSY_MS for _, busy in locks), f"journal lock wait not bounded: {locks}"
        assert result["results"] and "memory_attempt" not in result
        assert experience(provider, "recent")["error_kind"].startswith("OperationalError")
        diagnostic = [r.getMessage() for r in caplog.records if "experience slow" in r.getMessage()]
        assert diagnostic and "commit_ms=-" not in diagnostic[-1]
        assert not provider._store._conn.in_transaction
        assert provider._store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 10000
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert call(provider, "search", query="Alice observatory")["memory_attempt"]


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


@pytest.mark.parametrize("left_present", ["end", "merge"])
def test_forget_of_a_fact_no_longer_current_erases_its_receipt(provider, left_present):
    # An ended or merged fact has no `facts` row any more: only its versions tie the receipt to it.
    fid, aid = seed(provider)
    assert experience(provider, "outcome", attempt_id=aid, event_id="before-forget", outcome="corrected",
                      reason="wrong_entity", correction="The corrected destination is the cobalt depot.")["recorded"]
    if left_present == "end":
        assert "error" not in call(provider, "end", fact_id=fid)
    else:
        keep = call(provider, "add", content="Alice Smith is working on the observatory project")["fact_id"]
        assert "error" not in call(provider, "merge", fact_id=keep, **{"from": [fid]})
    assert experience(provider, "explain", attempt_id=aid)["status"] == "ok"
    assert call(provider, "forget", fact_id=fid)["forgotten"]
    assert experience(provider, "explain", attempt_id=aid)["status"] == "not_recorded"
    for table in ("memory_attempts", "memory_attempt_refs", "memory_outcomes"):
        assert not provider._store.read(f"SELECT 1 FROM {table} WHERE attempt_id = ?", (aid,)), table
    assert not provider._store.read("SELECT 1 FROM memory_attempt_refs WHERE fact_id = ?", (fid,))
    assert b"cobalt depot" not in provider._store.db_path.read_bytes()


def test_prefetch_that_recalls_nothing_writes_nothing(provider):
    call(provider, "add", content="Alice Smith works on the observatory project")
    begins = []
    conn = provider._store._conn
    conn.set_trace_callback(lambda sql: begins.append(sql) if sql.upper().startswith("BEGIN IMMEDIATE") else None)
    try:
        assert provider.prefetch("unicorn zeppelin", session_id="quiet-task") == ""
        assert begins == []
        assert not provider._store.experience_ready()
        assert "Memory attempt:" in provider.prefetch("Alice observatory", session_id="quiet-task")
        assert len(experience(provider, "recent")["attempts"]) == 1
        # Found again, but already injected this session: nothing is emitted, so nothing is journaled.
        begins.clear()
        assert provider.prefetch("Alice observatory", session_id="quiet-task") == ""
        assert begins == []
    finally:
        conn.set_trace_callback(None)
    assert len(experience(provider, "recent")["attempts"]) == 1


def test_retention_applies_when_a_writer_opens_the_store_with_collection_off(tmp_path):
    from hermes_plugin_holographic import experience as journal
    db = str(tmp_path / "facts.db")
    on = HolographicMemoryProvider({"db_path": db, "experience": "on"})
    on.initialize(session_id="earlier")
    try:
        call(on, "add", content="Alice Smith works on the observatory project")
        old = call(on, "search", query="Alice observatory")["memory_attempt"]
        recent = call(on, "search", query="Alice observatory")["memory_attempt"]
        assert experience(on, "outcome", attempt_id=old, event_id="old-result", outcome="corrected",
                          reason="wrong_entity", correction="Use the amber warehouse record instead.")["recorded"]
        on._store._conn.execute("UPDATE memory_attempts SET created_at = ? WHERE attempt_id = ?",
                                (time.time() - journal.TTL - 60, old))
    finally:
        on.shutdown()
    off = HolographicMemoryProvider({"db_path": db})
    off.initialize(session_id="later")
    try:
        assert [r[0] for r in off._store.read("SELECT attempt_id FROM memory_attempts")] == [recent]
        assert not off._store.read("SELECT 1 FROM memory_outcomes")
        assert not off._store.read("SELECT 1 FROM memory_attempt_refs WHERE attempt_id = ?", (old,))
    finally:
        off.shutdown()


def test_expired_correction_text_leaves_the_files_while_the_writer_stays_open(tmp_path):
    """Retention deletes the row; the text itself must not stay on disk in the main file or the WAL until some later
    checkpoint that a long-running writer with collection off may never make (Codex review of #48)."""
    from hermes_plugin_holographic import experience as journal
    db = tmp_path / "facts.db"
    on = HolographicMemoryProvider({"db_path": str(db), "experience": "on"})
    on.initialize(session_id="earlier")
    try:
        call(on, "add", content="Alice Smith works on the observatory project")
        old = call(on, "search", query="Alice observatory")["memory_attempt"]
        assert experience(on, "outcome", attempt_id=old, event_id="old-result", outcome="corrected",
                          reason="wrong_entity", correction="Use the cinnabar annex record instead.")["recorded"]
        on._store._conn.execute("UPDATE memory_attempts SET created_at = ? WHERE attempt_id = ?",
                                (time.time() - journal.TTL - 60, old))
    finally:
        on.shutdown()  # the last connection closes: the correction is checkpointed into the main file
    assert b"cinnabar annex" in db.read_bytes()
    off = HolographicMemoryProvider({"db_path": str(db)})
    off.initialize(session_id="later")
    try:
        assert not off._store.read("SELECT 1 FROM memory_outcomes")
        wal = tmp_path / "facts.db-wal"
        on_disk = db.read_bytes() + (wal.read_bytes() if wal.exists() else b"")
        assert b"cinnabar annex" not in on_disk
    finally:
        off.shutdown()


def _on_disk(db):
    wal = Path(str(db) + "-wal")
    return Path(db).read_bytes() + (wal.read_bytes() if wal.exists() else b"")


def test_retention_scrubs_removed_outcome_on_the_open_connection(tmp_path):
    """A long-open store: the receipt write whose retention removes an expired outcome also takes its correction text
    out of the main file and the WAL, without a reopen; a later retention call with nothing due leaves no mark behind
    (Codex review 2026-10-29: the mark was set, and nothing on this connection served it until the next open)."""
    from hermes_plugin_holographic import experience as journal
    db = tmp_path / "review.db"
    store = MemoryStore(db, hrr_dim=64)
    try:
        assert store._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        fid = store.add_fact("The observatory opens at noon.")
        fact = dict(store.read("SELECT * FROM facts WHERE fact_id=?", (fid,))[0])
        aid = store.record_attempt("search", [fact], [], "session")
        marker = "cinnabar annex private correction marker"
        store.record_outcome(aid, "event1", "corrected", "other", correction=marker, session_key="session")
        store._conn.execute("UPDATE memory_attempts SET created_at=? WHERE attempt_id=?",
                            (time.time() - journal.TTL - 60, aid))
        store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        assert marker.encode() in _on_disk(db)
        store.record_attempt("search", [fact], [], "session")  # the normal receipt write runs retention
        assert store.read("SELECT * FROM memory_outcomes") == []
        assert marker.encode() not in _on_disk(db), f"scrub_pending={store._meta('scrub_pending')}"
        assert store.prune_experiences() == 0
        assert marker.encode() not in _on_disk(db)
        assert store._meta("scrub_pending") is None
    finally:
        store.close()


@pytest.mark.parametrize("served_by", ["receipt", "open"])
def test_blocked_retention_scrub_neither_holds_search_nor_stays_pending(provider, monkeypatch, served_by):
    """The scrub after a receipt write runs under busy timeout 0, like forget's: another process reading the database
    makes it come back at once with the mark kept, and search answers while that process still reads. A second search
    in the retry window does not rewrite the file again (a VACUUM per search while a reader stays would grow the WAL by
    the whole database each time). Once the reader is gone, a later receipt write or a writer open serves the mark
    although nothing new has expired."""
    from hermes_plugin_holographic import experience as journal
    store = provider._store
    assert store._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    _, old = seed(provider)
    marker = "Use the cinnabar annex record instead."
    assert experience(provider, "outcome", attempt_id=old, event_id="old-result", outcome="corrected",
                      reason="wrong_entity", correction=marker)["recorded"]
    store._conn.execute("UPDATE memory_attempts SET created_at = ? WHERE attempt_id = ?",
                        (time.time() - journal.TTL - 60, old))
    store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    scrub_statements = ("VACUUM", "PRAGMA WAL_CHECKPOINT")
    thread, acquired, release, let_go = _hold(store.db_path, "BEGIN", "SELECT count(*) FROM facts")
    try:
        assert acquired.wait(5)
        result, statements, elapsed = _search_tracing_locks(provider, scrub_statements)
        assert not let_go.is_set(), f"the scrub held search {elapsed:.2f} s, until the reader let go"
        assert result["results"] and result.get("memory_attempt")  # the receipt itself is recorded
        assert not store.read("SELECT 1 FROM memory_outcomes")
        assert any(s == "VACUUM" for s, _ in statements), f"no scrub was tried after retention: {statements}"
        assert all(busy == 0 for _, busy in statements), f"scrub waited on the other process: {statements}"
        assert store._meta("scrub_pending") is not None  # blocked: the mark stays for a retry
        assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 10000
        again, statements, _ = _search_tracing_locks(provider, scrub_statements)
        assert again.get("memory_attempt") and not statements, f"rewrote the file again at once: {statements}"
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    if served_by == "receipt":
        monkeypatch.setattr(journal, "_SCRUB_RETRY_S", 0)  # the retry window has passed
        assert call(provider, "search", query="Alice observatory")["memory_attempt"]
    else:
        assert store.prune_experiences() == 0  # a writer open: nothing is due, the mark is
    assert store._meta("scrub_pending") is None
    assert marker.encode() not in _on_disk(store.db_path)


def test_retention_cap_applies_on_open_and_absent_journal_is_not_created(tmp_path):
    from hermes_plugin_holographic import experience as journal
    db = str(tmp_path / "facts.db")
    on = HolographicMemoryProvider({"db_path": db, "experience": "on"})
    on.initialize(session_id="earlier")
    try:
        call(on, "add", content="Alice Smith works on the observatory project")
        newest = call(on, "search", query="Alice observatory")["memory_attempt"]
        now = time.time()
        on._store._conn.executemany("INSERT INTO memory_attempts VALUES (?,?,?,?,?,?,?)",
            [(f"older-{i}", now - 3600 - i, "search", "s", "[]", "[]", "not_recorded")
             for i in range(journal.MAX_ATTEMPTS)])
    finally:
        on.shutdown()
    off = HolographicMemoryProvider({"db_path": db})
    off.initialize(session_id="later")
    try:
        kept = {r[0] for r in off._store.read("SELECT attempt_id FROM memory_attempts")}
        assert len(kept) == journal.MAX_ATTEMPTS
        assert newest in kept and f"older-{journal.MAX_ATTEMPTS - 1}" not in kept
    finally:
        off.shutdown()
    fresh = HolographicMemoryProvider({"db_path": str(tmp_path / "never.db")})
    fresh.initialize(session_id="never")
    try:
        assert not fresh._store.experience_ready()
    finally:
        fresh.shutdown()


class _RerankAnswer:
    def __init__(self, body):
        self.body, self.at = body, 0

    def read(self, n=-1):
        piece = self.body[self.at:] if n is None or n < 0 else self.body[self.at:self.at + n]
        self.at += len(piece)
        return piece

    read1 = read

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _second_writer(db_path):
    """Another provider on its own SQLite connection, as another process would have."""
    module_file = sys.modules[HolographicMemoryProvider.__module__].__file__
    spec = importlib.util.spec_from_file_location("journal_second_writer", module_file,
        submodule_search_locations=[str(Path(module_file).parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    other = module.HolographicMemoryProvider({"db_path": str(db_path)})
    other.initialize(session_id="second-writer")
    return other, spec.name


def test_trace_keeps_receipt_when_an_unemitted_candidate_changes_during_rerank(tmp_path, monkeypatch):
    from hermes_plugin_holographic import rerank
    p = HolographicMemoryProvider({"db_path": str(tmp_path / "facts.db"), "experience": "on", "experience_trace": "on",
                                   "rerank": "on", "rerank_url": "https://rerank.example.net/v1/rerank",
                                   "rerank_model": "stub-reranker"})
    p.initialize(session_id="rerank-task")
    other, module_name = _second_writer(p._store.db_path)
    try:
        call(p, "add", content="Boris Petrov maintains the harbour crane schedule")
        call(p, "add", content="Boris Petrov prefers tea over coffee")
        bystander = call(p, "add", content="The harbour crane schedule changes on Mondays")["fact_id"]

        class Endpoint:
            def open(self, request, timeout=None):
                # The cross-encoder call takes a while; another writer changes a candidate meanwhile.
                assert "error" not in call(other, "change", fact_id=bystander,
                                           content="The harbour crane schedule changes on Tuesdays")
                n = len(json.loads(request.data)["documents"])
                return _RerankAnswer(json.dumps({"results": [{"index": i, "relevance_score": float(n - i)}
                                                             for i in range(n)]}).encode())

        monkeypatch.setattr(rerank, "_OPENER", Endpoint())
        block = p.prefetch("Did Boris Petrov say anything about the crane?", session_id="turn")
        assert f"#{bystander}" not in block
        assert "Memory attempt:" in block, experience(p, "recent")["error_kind"]
        aid = block.rsplit("Memory attempt: ", 1)[1].strip()
        report = experience(p, "explain", attempt_id=aid)
        assert report["status"] == "ok"
        assert report["emitted"] and all(f"#{r['fact_id']} " in block for r in report["emitted"])
        lexical = next(s for s in report["stages"] if s["stage"] == "lexical_candidates")
        assert all(r["fact_id"] != bystander for s in report["stages"] for r in s["facts"])
        assert lexical["changed"] == 1
        assert "Mondays" not in json.dumps(report) and "Tuesdays" not in json.dumps(report)
    finally:
        other.shutdown()
        sys.modules.pop(module_name, None)
        p.shutdown()


# 0.20.3: with collection on, the system prompt asks for an outcome when the person corrects or rejects a recalled
# answer; nothing else asked, so a real install collected receipts and no outcomes at all.
def _outcome_lines(block):
    return [line for line in block.splitlines() if "fact_experience" in line]


def test_collection_on_asks_for_an_outcome_after_a_correction(provider):
    fid, _ = seed(provider)
    before = call(provider, "history", fact_id=fid)
    lines = _outcome_lines(provider.system_prompt_block())
    assert len(lines) == 1
    guidance = lines[0]
    for said in ("corrects or rejects", "outcome", "corrected", "dead_end", "useful", "Memory attempt",
                 "Not every turn", "not their confirmation", "changes no fact", "change/correct"):
        assert said in guidance
    # What the line names is what the agent sees and can send: the prefetch id, a useful with reason other.
    block = provider.prefetch("What does Alice work on at the observatory?", session_id="task-session")
    aid = block.rsplit("Memory attempt: ", 1)[1].strip()
    assert experience(provider, "outcome", attempt_id=aid, event_id="confirmed-1", outcome="useful",
                      reason="other")["recorded"]
    assert experience(provider, "outcome", attempt_id=aid, event_id="rejected-1", outcome="corrected",
                      reason="wrong_entity", correction="The person meant another Alice.",
                      supersedes="confirmed-1")["recorded"]
    assert call(provider, "history", fact_id=fid) == before  # the outcome changed no fact


def test_collection_off_keeps_the_prompt_and_tools_as_before(tmp_path):
    on = HolographicMemoryProvider({"db_path": str(tmp_path / "on.db"), "experience": "on"})
    off = HolographicMemoryProvider({"db_path": str(tmp_path / "off.db")})
    sub = HolographicMemoryProvider({"db_path": str(tmp_path / "on.db"), "experience": "on"})
    on.initialize(session_id="on")
    off.initialize(session_id="off")
    sub.initialize(session_id="sub", agent_context="subagent")  # records no attempts: nothing to report on
    try:
        off_block = off.system_prompt_block()
        assert not _outcome_lines(off_block)
        assert not _outcome_lines(sub.system_prompt_block())
        # Off differs from on by that line only, and the schemas off are the ones on without fact_experience.
        on_block = on.system_prompt_block()
        assert [line for line in on_block.splitlines() if line not in _outcome_lines(on_block)] == \
            off_block.splitlines()
        assert json.dumps(off.get_tool_schemas(), sort_keys=True) == \
            json.dumps([s for s in on.get_tool_schemas() if s["name"] != "fact_experience"], sort_keys=True)
    finally:
        on.shutdown()
        off.shutdown()
        sub.shutdown()


def test_every_statement_of_a_scrub_runs_without_waiting_on_another_process(tmp_path):
    """The scrub a receipt write runs sits on the search path: none of its statements may wait out the 10 s busy
    timeout, the closing checkpoint after the mark is cleared included (Codex review 2026-10-30: a reader opened
    after the first checkpoint made the second one wait 10 s inside search). The timeout is restored last."""
    store = MemoryStore(tmp_path / "trace.db", hrr_dim=64)
    try:
        store.add_fact("The observatory opens at noon.")
        store._set_meta("scrub_pending", "1")
        statements = []
        store._conn.set_trace_callback(statements.append)
        assert store.scrub()
        store._conn.set_trace_callback(None)
        timeout = None
        for sql in statements:
            words = sql.split()
            if sql.startswith("PRAGMA busy_timeout") and "=" in sql:
                timeout = int(words[-1].strip("= "))
            elif timeout is not None and (sql.startswith(("VACUUM", "PRAGMA wal_checkpoint", "DELETE"))):
                assert timeout == 0, f"{sql!r} ran with busy timeout {timeout}"
        assert statements[-1].replace(" ", "").startswith("PRAGMAbusy_timeout=") and timeout > 0
    finally:
        store.close()


def test_a_scrub_clears_only_the_mark_it_served(tmp_path):
    """Another process can delete text, and set the mark, while this scrub runs between its rewrite and clearing the
    mark; that mark must survive, or the other text stays on disk with nothing left to scrub it (Codex review
    2026-10-30). Stores of one path share a connection in a process, so the other process is a plain connection
    that writes its own mark, as its forget would, while a reader it does not control keeps its own rewrite back."""
    db = tmp_path / "race.db"
    store = MemoryStore(db, hrr_dim=64)
    reader, other = sqlite3.connect(db), sqlite3.connect(db, isolation_level=None)
    try:
        assert store._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        store.add_fact("The observatory opens at noon.")
        store._mark_scrub() if hasattr(store, "_mark_scrub") else store._set_meta("scrub_pending", "1")
        fired = []

        def before_clearing(sql):  # this scrub has rewritten and checkpointed the file
            if sql.startswith("DELETE FROM meta") and not fired:
                reader.execute("BEGIN")
                reader.execute("SELECT count(*) FROM facts").fetchone()
                other.execute("INSERT INTO meta (key, value) VALUES ('scrub_pending', 'other-forget') "
                              "ON CONFLICT(key) DO UPDATE SET value = excluded.value")
                fired.append(sql)
        store._conn.set_trace_callback(before_clearing)
        store.scrub()
        store._conn.set_trace_callback(None)
        assert fired
        assert store._meta("scrub_pending") == "other-forget", "the other process's mark was cleared"
        reader.rollback()
        assert store.scrub() and store._meta("scrub_pending") is None
    finally:
        reader.close()
        other.close()
        store.close()

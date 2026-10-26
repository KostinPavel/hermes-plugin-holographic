"""Stage traces at debug level: which fact ids each stage was given and returned (statistics, ids only).

- `embed` — the fact vectors stored after a turn, the facts the meaning stream admitted before a turn,
  and the meaning matches a tool search returned.
- `rescore` — the pool and the documents the cross-encoder was given (`#id` list) and its `#id=score` answer
  (or why no call was made), at both call sites: prefetch and fact_store search.

A trace line names rows of the database, never their text: the lines may be collected and shared, the facts
stay in the store. Nothing is logged at the normal level (root logger INFO)."""
from __future__ import annotations

import json
import logging
import re

from hermes_plugin_holographic import embeddings, rerank
from test_holographic_rerank import FACTS, FakeRerank, RANK, _fill as _fill_pool, _provider as _pool_provider
from test_holographic_v0140_embeddings import (BACKUP, ON, PARAPHRASE, FakeEndpoint,
                                               _fill as _fill_meaning, _provider as _meaning_provider)


def _fake_rerank(monkeypatch) -> FakeRerank:
    fake = FakeRerank()
    monkeypatch.setattr(rerank, "_OPENER", fake)
    return fake


def _fake_embeddings(monkeypatch) -> FakeEndpoint:
    fake = FakeEndpoint()
    monkeypatch.setattr(embeddings, "_OPENER", fake)
    monkeypatch.setenv("TEST_EMBEDDINGS_KEY", "sk-tes...-key")
    return fake


def _lines(caplog, *markers: str) -> list[str]:
    return [line for line in caplog.text.splitlines() if all(m in line for m in markers)]


def _payload(line: str) -> str:
    """After the last ': ': the '#id,...' input list or the '#id=score,...' answer of the line."""
    return line.rsplit(": ", 1)[-1]


def _ids(payload: str) -> set[int]:
    return {int(cell[1:].split("=")[0]) for cell in payload.split(",") if cell.startswith("#")}


# -- rescore: prefetch ----------------------------------------------------------------------------------------------

def test_prefetch_rescore_trace_names_pool_and_answer(tmp_path, monkeypatch, caplog):
    fake = _fake_rerank(monkeypatch)
    p = _pool_provider(tmp_path, **RANK)
    created = set(_fill_pool(p))
    with caplog.at_level(logging.DEBUG):
        assert p.prefetch("backup box")  # coverage admits both backup facts (pool of >= 2: a call happens)
    assert len(fake.requests) == 1  # the traced call is the call that happened
    pool = _lines(caplog, "rescore prefetch: pool")
    answer = _lines(caplog, "rescore prefetch: answered:")
    assert len(pool) == 1 and len(answer) == 1
    assert _ids(_payload(pool[0])) <= created and _ids(_payload(pool[0]))
    assert _ids(_payload(answer[0])) == _ids(_payload(pool[0]))  # the endpoint answers for every document
    scores = [float(cell.split("=")[1]) for cell in _payload(answer[0]).split(",")]
    assert scores == sorted(scores, reverse=True)  # the answer is descending by score
    assert _lines(caplog, "embed:") == []  # embeddings off: the meaning stage logs nothing


def test_search_rescore_trace_names_pool_and_answer(tmp_path, monkeypatch, caplog):
    fake = _fake_rerank(monkeypatch)
    p = _pool_provider(tmp_path, **RANK)
    created = set(_fill_pool(p))
    with caplog.at_level(logging.DEBUG):
        found = json.loads(p.handle_tool_call("fact_store", {"action": "search", "query": "backup"}))["results"]
    assert found and len(fake.requests) == 1
    pool = _lines(caplog, "rescore search: pool")
    answer = _lines(caplog, "rescore search: answered:")
    assert len(pool) == 1 and len(answer) == 1
    assert _ids(_payload(pool[0])) <= created and _ids(_payload(pool[0]))
    assert _ids(_payload(answer[0])) == _ids(_payload(pool[0]))


def test_one_document_makes_no_call_and_the_trace_says_so(tmp_path, monkeypatch, caplog):
    fake = _fake_rerank(monkeypatch)
    p = _pool_provider(tmp_path, **RANK)
    only = p._store.add_fact("The nightly backup runs from the OES box at 2 am.")
    p.sync_turn("x", "y")
    with caplog.at_level(logging.DEBUG):
        assert json.loads(p.handle_tool_call("fact_store", {"action": "search", "query": "backup"}))["results"]
    assert fake.requests == []  # one sendable document: scores() does not call the endpoint
    pool = _lines(caplog, "rescore search: pool")
    assert len(pool) == 1 and _ids(_payload(pool[0])) == {only}
    skipped = _lines(caplog, "rescore search: skipped_single:")
    assert len(skipped) == 1 and _payload(skipped[0]) == f"#{only}=1.0"


# -- embed: vectors stored after a turn -----------------------------------------------------------------------------

def test_catch_up_trace_names_every_stored_fact(tmp_path, monkeypatch, caplog):
    _fake_embeddings(monkeypatch)
    p = _meaning_provider(tmp_path, **ON)
    fillers, (backup,) = _fill_meaning(p, BACKUP)
    with caplog.at_level(logging.DEBUG):
        p.sync_turn("hello", "hi")  # the turn that stores the missing fact vectors
    stored: set[int] = set()
    written = requested = 0
    for line in _lines(caplog, "fact vector(s) written:"):
        counts = re.search(r"(\d+) of (\d+) fact vector", line)
        assert counts, line
        written += int(counts.group(1))
        requested += int(counts.group(2))
        stored |= _ids(_payload(line))
    assert stored == set(fillers) | {backup}
    assert written == requested == len(fillers) + 1  # nothing edited mid-request: every requested id was written


# -- embed: what the meaning stream returned ------------------------------------------------------------------------

def test_prefetch_meaning_trace_names_admitted_facts(tmp_path, monkeypatch, caplog):
    _fake_embeddings(monkeypatch)
    p = _meaning_provider(tmp_path, **ON)
    _, (backup,) = _fill_meaning(p, BACKUP)
    p.sync_turn("hello", "hi")
    with caplog.at_level(logging.DEBUG):
        assert f"#{backup} " in p.prefetch(PARAPHRASE)  # no shared word: the vector finds it
    admitted = _lines(caplog, "meaning admitted")
    assert len(admitted) == 1 and backup in _ids(_payload(admitted[0]))


def test_search_meaning_trace_names_returned_facts(tmp_path, monkeypatch, caplog):
    _fake_embeddings(monkeypatch)
    p = _meaning_provider(tmp_path, **ON)
    _, (backup,) = _fill_meaning(p, BACKUP)
    p.sync_turn("hello", "hi")
    with caplog.at_level(logging.DEBUG):
        results = json.loads(p.handle_tool_call("fact_store",
                                                 {"action": "search", "query": PARAPHRASE}))["results"]
    assert any(r["fact_id"] == backup and r.get("match") == "meaning" for r in results)
    returned = _lines(caplog, "meaning search returned")
    assert len(returned) == 1 and backup in _ids(_payload(returned[0]))
    assert re.search(r"#\d+", _payload(returned[0]))


# -- the trace carries no text, and nothing runs at the normal level -----------------------------------------------

def test_no_text_reaches_the_trace(tmp_path, monkeypatch, caplog):
    """Review point 2: an exception's text may quote a fact. Neither it, nor a fact's text, nor the query may appear
    anywhere in the log: the sentinel fails the stage once, the following calls run while the endpoint is paused and
    print only the failure's kind (error_kind, PR #30)."""
    _fake_rerank(monkeypatch)
    p = _pool_provider(tmp_path, **RANK)
    _fill_pool(p)
    query = "backup box"
    sentinel = "SENTINEL-c7f2 must never be logged"

    def boom(*args, **kwargs):
        raise RuntimeError(sentinel)

    real = p._rescorer.scores
    with caplog.at_level(logging.DEBUG):
        p._rescorer.scores = boom
        p.prefetch(query)           # the stage raises: note_failure gets the kind only
        p._rescorer.scores = real   # paused now: the later calls print the status
        p.prefetch(query)
        json.loads(p.handle_tool_call("fact_store", {"action": "search", "query": query}))
    assert p._rescorer.failures >= 1
    assert _lines(caplog, "rescore prefetch: pool"), "the trace must have run"
    assert _lines(caplog, "rescore prefetch: no answer"), "the paused status must be traced"
    assert _lines(caplog, "rescore search: no answer")
    assert sentinel not in caplog.text
    for fact in FACTS:
        assert fact not in caplog.text, fact
    assert query not in caplog.text


def test_no_trace_at_the_normal_level(tmp_path, monkeypatch, caplog):
    """Review point 5: at the normal level (INFO) no trace line is emitted — statistics are opt-in with
    logging.level: DEBUG."""
    _fake_rerank(monkeypatch)
    p = _pool_provider(tmp_path, **RANK)
    _fill_pool(p)
    with caplog.at_level(logging.INFO):
        p.prefetch("backup box")
        json.loads(p.handle_tool_call("fact_store", {"action": "search", "query": "backup box"}))
    assert _lines(caplog, "rescore") == [] and _lines(caplog, "embed:") == []


def test_catch_up_trace_counts_what_was_written(tmp_path, monkeypatch, caplog):
    """Review point 4: a fact edited while its vector was in flight is not written, so the trace counts the written
    vectors of each request (put_vectors' return), not the requested ids: the first batch loses the edited fact
    (3 of 4), the next one writes it again from the new text."""
    fake = _fake_embeddings(monkeypatch)
    p = _meaning_provider(tmp_path, **ON)
    fillers, (backup,) = _fill_meaning(p, BACKUP)
    moved = "The nightly backup moved to the new box."

    def edit_once():
        fake.during = None
        p._store._conn.execute("UPDATE facts SET content = ? WHERE fact_id = ?", (moved, backup))

    fake.during = edit_once
    with caplog.at_level(logging.DEBUG):
        p._catch_up_vectors()  # batch 1: the fact edited in flight is not written
        p._catch_up_vectors()  # batch 2 wrote the new text; this call finds nothing pending
    lines = _lines(caplog, "fact vector(s) written:")
    counts = [re.search(r"embed: (\d+) of (\d+) fact vector", line) for line in lines]
    assert lines and all(counts), lines
    written = sum(int(c.group(1)) for c in counts)
    requested = sum(int(c.group(2)) for c in counts)
    assert written == requested - 1, (written, requested)  # exactly the in-flight edit was not written
    assert backup in _ids(_payload(lines[0]))  # the edited fact was in the first request
    stored = p._store.read("SELECT text_hash FROM fact_embeddings WHERE fact_id = ?", (backup,))
    assert stored and stored[0][0] == embeddings.text_hash(moved)  # the new text is what is stored

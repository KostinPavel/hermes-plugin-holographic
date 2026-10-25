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
from test_holographic_rerank import FakeRerank, RANK, _fill as _fill_pool, _provider as _pool_provider
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
    assert all(float(cell.split("=")[1]) is not None for cell in _payload(answer[0]).split(","))
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
    for line in _lines(caplog, "fact vector(s) stored:"):
        stored |= _ids(_payload(line))
    assert stored == set(fillers) | {backup}


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
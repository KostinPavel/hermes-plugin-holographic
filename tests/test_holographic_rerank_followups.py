"""Guards of PR #10 that no test caught (independent review of the merged outside code, 2026-10-25).

Each test below was checked against a copy of the code without the guard it names, and fails there:
- the screen of documents on the prefetch path (``_sendable_docs``): the existing tests used a query the secret fact
  never matched, so it never reached the pool;
- the secret check of the query (``skipped_secret``): the existing test's query found no fact at all, so the
  endpoint would not have been called anyway; it is checked here on the Rescorer, where the check lives;
- the cap of ``rerank_timeout`` by the core's prefetch budget: only the lower clamp was tested;
- a duplicate index in the answer;
- a redirect: the fake opener replaced the whole opener, ``_NoRedirect`` included;
- the McNemar p-value: no test looked at the number;
- the benchmark's stand-in answer: it had no ``read1()``, so every call of the tool failed.

And one defect both reviews reproduced: a second call while the first caller still waited for its answer failed that
request too (the first caller threw its valid answer away) and paused the endpoint for a minute.
"""
from __future__ import annotations

import http.server
import importlib.util
import json
import logging
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider, rerank, _secret_in, _sendable

_ROOT = Path(__file__).resolve().parent.parent


class _Response:
    def __init__(self, body: bytes):
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


class FakeRerank:
    """Stands in for rerank._OPENER: records what was sent, answers in the order given."""

    def __init__(self):
        self.requests: list[dict] = []
        self.delay = 0.0  # seconds the "network" takes

    @property
    def documents(self) -> list[str]:
        return [d for r in self.requests for d in r["body"]["documents"]]

    def open(self, request, timeout=None):
        body = json.loads(request.data)
        self.requests.append({"body": body, "timeout": timeout})
        time.sleep(self.delay)
        n = len(body["documents"])
        results = [{"index": i, "relevance_score": float(n - i)} for i in range(min(body.get("top_n", n), n))]
        return _Response(json.dumps({"results": results}).encode())


@pytest.fixture
def endpoint(monkeypatch):
    fake = FakeRerank()
    monkeypatch.setattr(rerank, "_OPENER", fake)
    return fake


RANK = {"rerank": "on", "rerank_url": "https://rerank.example.net/v1/rerank", "rerank_model": "fake-reranker"}
FILLERS = ["apple note", "brick note", "cloud note", "daisy note", "eagle note"]
PLAIN = ["The nightly backup runs from the OES box at 2 am.",
         "The weekly backup runs from the NAS box at 4 am.",
         "Mark plays the violin on Sundays.",
         # two sendable facts that answer PROXY_QUERY with the secret: a single one is never sent (no call)
         "The backup proxy logs backup requests to syslog.",
         "The backup proxy caches backup requests for an hour."]
# Key-like literals are assembled at runtime, as in test_holographic_rerank_review.py.
_BEARER = "ey" + "JhbGciOiJIUzI1NiJ9." + "abc123DEF456" * 3
# Recall may show it (an Authorization header is not a credential-with-value to _recall_problem); it must never be
# sent. Its words answer the query below, so it IS in the pool the endpoint would see.
SECRET = "The backup proxy sets Authorization:" + " Bearer " + _BEARER + " for backup requests."
PROXY_QUERY = "which header does the backup proxy set for backup requests?"


def _provider(tmp_path, **config):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory.db"), "hrr_dim": 64, **config})
    p.initialize(session_id="s1")
    for text in [f"{f} number" for f in FILLERS] + PLAIN:
        p._store.add_fact(text)
    p.sync_turn("x", "y")
    return p


# -- what is sent -------------------------------------------------------------------------------------------------

def test_a_matching_secret_fact_is_in_the_pool_but_never_sent(tmp_path, endpoint):
    p = _provider(tmp_path, **RANK)
    p._store.add_fact(SECRET)
    assert _sendable(SECRET) is False
    block = p.prefetch(PROXY_QUERY)
    p.shutdown()
    # The premise: the fact answers the query and recall shows it, so it was a candidate for the endpoint.
    assert _BEARER in block
    assert endpoint.requests, "the pool had other facts: the endpoint must still be called"
    assert any("backup" in d for d in endpoint.documents)
    assert all(_BEARER not in d for d in endpoint.documents)


def test_a_query_holding_a_secret_is_not_sent(endpoint):
    rescorer = rerank.Rescorer(rerank.settings_from(RANK), secret_check=_secret_in)
    docs = PLAIN[:2]
    assert rescorer.scores("when does the backup run?", docs) is not None  # control: the same words are sent
    assert len(endpoint.requests) == 1
    query = "password: Qwerty123 when does the backup run?"
    assert _secret_in(query)
    assert rescorer.scores(query, docs) is None
    assert rescorer.last_status == "skipped_secret"
    assert len(endpoint.requests) == 1, "the query with a password went to the endpoint"
    assert rescorer.failures == 0  # not a failure: the endpoint is not paused


def test_a_second_call_while_the_first_waits_does_not_fail_the_first(endpoint):
    endpoint.delay = 0.4
    rescorer = rerank.Rescorer(rerank.settings_from(RANK), secret_check=_secret_in)
    docs = PLAIN[:2]
    results = {}
    first = threading.Thread(target=lambda: results.__setitem__("first", rescorer.scores("backup", docs)))
    first.start()
    started = time.monotonic()
    while not endpoint.requests and time.monotonic() - started < 2:
        time.sleep(0.01)
    assert endpoint.requests, "the first request never started"
    second = rescorer.scores("backup", docs)  # the first request is in flight, its caller waits for it
    first.join(5)
    assert second is None
    assert results["first"] is not None, "the first caller lost its valid answer"
    assert len(endpoint.requests) == 1
    assert rescorer.failures == 0
    assert rescorer.usable(), rescorer.status()


def test_a_failure_in_the_rerank_stage_logs_its_kind_never_its_text(tmp_path, endpoint, monkeypatch, caplog):
    p = _provider(tmp_path, **RANK)

    def broken(ranked, send_idx):
        raise ValueError("quoting the fact: " + PLAIN[0])

    monkeypatch.setattr(p, "_rescored", broken)
    with caplog.at_level(logging.DEBUG):
        block = p.prefetch("when does the backup run?")
    status = p._rescorer.last_status
    p.shutdown()
    assert endpoint.requests and "nightly backup" in block  # the call was made; recall fell back to the off order
    assert "ValueError" in status and "ValueError" in caplog.text
    assert "quoting the fact" not in status and "quoting the fact" not in caplog.text


# -- settings -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("config, expected", [
    ({"embeddings": "on", "embedding_timeout": "6", "rerank_timeout": "10"}, 1.5),  # README: 6 s clamps rerank to 1.5
    ({"embeddings": "on", "rerank_timeout": "10"}, 4.5),  # the default embedding_timeout, 3 s
    ({"embeddings": "off", "embedding_timeout": "6", "rerank_timeout": "10"}, 7.5),  # off: a stray key costs nothing
    ({"embeddings": "on", "embedding_timeout": "6"}, 1.5),  # the default rerank_timeout (3 s) is capped too
])
def test_rerank_timeout_stays_inside_the_core_prefetch_budget(config, expected):
    settings = rerank.settings_from({"rerank": "on", **config})
    assert settings.timeout == pytest.approx(expected)


# -- the answer ---------------------------------------------------------------------------------------------------

def test_a_duplicate_index_in_the_answer_is_refused():
    # Two documents, both answers name the first: the count matches, but one fact would show twice and one vanish.
    answer = {"results": [{"index": 0, "relevance_score": 2.0}, {"index": 0, "relevance_score": 1.0}]}
    with pytest.raises(ValueError, match="duplicate"):
        rerank._ranked(answer, 2)


def test_a_redirect_is_refused_and_not_followed():
    hits: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        # 302: the redirect urllib follows for a POST by default (as a GET to the new address); it does not follow a
        # 307 for a POST at all, so a 307 would not tell whether the client refuses redirects.
        def do_POST(self):
            hits.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path == "/v1/rerank":
                self.send_response(302)
                self.send_header("Location", "/elsewhere")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._answer()

        def do_GET(self):
            hits.append(self.path)
            self._answer()

        def _answer(self):
            body = json.dumps({"results": [{"index": 0, "relevance_score": 1.0},
                                           {"index": 1, "relevance_score": 0.5}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/v1/rerank"
        rescorer = rerank.Rescorer(rerank.settings_from({"rerank": "on", "rerank_url": url}))
        assert rescorer.usable()
        result = rescorer.scores("when does the backup run?", ["first fact", "second fact"])
    finally:
        server.shutdown()
        server.server_close()
    assert result is None
    assert rescorer.last_status.startswith("failed")
    assert hits == ["/v1/rerank"], "the client followed the redirect"


# -- tools --------------------------------------------------------------------------------------------------------

def _load_tool(name, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))  # the benchmark puts the checkout on sys.path: undone after
    spec = importlib.util.spec_from_file_location(f"_tool_{name}", _ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mcnemar_p_value_is_the_two_sided_exact_binomial(monkeypatch):
    gate = _load_tool("mcnemar_gate", monkeypatch)
    assert gate.mcnemar(0, 0) == 1.0
    assert gate.mcnemar(3, 3) == 1.0
    assert gate.mcnemar(0, 5) == pytest.approx(2 / 32)  # 5 gained, 0 lost: not significant at 0.05
    assert gate.mcnemar(0, 6) == pytest.approx(2 / 64)  # 6 / 0 is the smallest clean gain that passes
    assert gate.mcnemar(1, 9) == pytest.approx(2 * (1 + 10) / 1024)  # C(10, 0) + C(10, 1)
    assert gate.mcnemar(4, 0) == gate.mcnemar(0, 4)


def test_the_benchmark_stand_in_endpoint_is_answered(monkeypatch):
    bench = _load_tool("benchmark_rerank", monkeypatch)
    monkeypatch.setattr(rerank, "_OPENER", bench.MockRerank())
    rescorer = rerank.Rescorer(rerank.settings_from({"rerank": "on", "rerank_url": "http://127.0.0.1:9/v1/rerank"}))
    ranked = rescorer.scores("when does the backup run?", ["first fact", "second fact", "third fact"])
    assert rescorer.last_status == "answered", rescorer.last_status
    assert [i for i, _ in ranked] == [0, 1, 2]
    assert rescorer.failures == 0

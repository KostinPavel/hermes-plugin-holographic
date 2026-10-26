"""The socket handle the network stages keep (issue #26): a request's connection writes its socket into the
request's holder as soon as ``connect()`` returns, and ``cut()`` ends the read the caller has stopped waiting for.
No provider here — the opener, a holder and one local server are the whole subject; the provider-level behaviour is
the drip tests in the rerank and embeddings files. The exceptions a cut leaves are the ones the maintainer listed
on #26: ConnectionAbortedError (Windows), RemoteDisconnected / BadStatusLine where a shutdown reads as EOF.
"""

from __future__ import annotations

import http.client
import socket
import threading
import time
import urllib.request

from hermes_plugin_holographic import net


class _Server:
    """One connection per request: reads it, then answers as the mode dictates. ``stall`` never answers, ``drip``
    sends the response one byte every 0.2 s — each read succeeds on its own (0.2 s < the 2 s socket timeout), so the
    reader never times out, and at the caller's 0.5 s deadline only a couple of status-line bytes have arrived, so
    the shutdown reads as an unfinished line (BadStatusLine / RemoteDisconnected), not as a clean EOF. ``answer``
    replies at once."""

    def __init__(self, mode: str):
        self.mode = mode
        self.stop = threading.Event()
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        self.srv.settimeout(0.2)
        while not self.stop.is_set():
            try:
                conn, _ = self.srv.accept()
            except socket.timeout:
                continue
            try:
                conn.settimeout(10)
                data = b""
                while b"\r\n\r\n" not in data:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                length = 0
                for line in data.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        length = int(line.split(b":")[1])
                _, _, sent = data.partition(b"\r\n\r\n")
                while len(sent) < length:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    sent += chunk
                if self.mode == "stall":
                    self.stop.wait(60)
                elif self.mode == "drip":
                    answer = (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                              b"Content-Length: 13\r\nX-Slow: " + b"a" * 200 + b"\r\n\r\n" + b'{"results":[]}')
                    for byte in answer:
                        conn.sendall(bytes([byte]))
                        if self.stop.wait(0.2):
                            break
                elif self.mode == "answer":
                    conn.sendall(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n'
                                 b'Content-Length: 13\r\n\r\n{"results":[]}')
            except OSError:
                pass  # the shutdown ends the connection mid-write: expected
            finally:
                conn.close()
        self.srv.close()

    def close(self):
        self.stop.set()
        self.thread.join(5)


def _request(port: int):
    return urllib.request.Request(f"http://127.0.0.1:{port}/v1", data=b'{"a":1}',
                                  headers={"Content-Type": "application/json"}, method="POST")


def _read(opener, request, box):
    """What the worker thread does; whatever it raises is what the worker would record."""
    try:
        with opener.open(request, timeout=2.0) as response:
            response.read()
        box["ok"] = True
    except BaseException as exc:  # noqa: BLE001 - the worker records every failure, we assert on it
        box["error"] = exc


_ACCEPTED = (ConnectionAbortedError, http.client.RemoteDisconnected, http.client.BadStatusLine)


def test_the_holder_gets_the_socket_while_the_worker_waits_for_headers_and_cut_ends_it():
    """The socket must reach the holder before the first header byte — that is the phase nothing else bounds
    (issue #26) — and the caller's cut must end the read that is standing in it."""
    server = _Server("stall")
    try:
        request = _request(server.port)
        held = net.wire(request)
        box = {}
        t = threading.Thread(target=_read, args=(net.opener(), request, box), daemon=True)
        t.start()
        limit = time.monotonic() + 2.0
        while "sock" not in held and time.monotonic() < limit:
            time.sleep(0.001)
        assert "sock" in held, "connect() must hand the socket over before any header byte (issue #26)"
        assert t.is_alive(), "the read is waiting for the status line"
        start = time.perf_counter()
        net.cut(held)
        t.join(2)
        assert not t.is_alive(), "the shutdown must end the read the caller stopped waiting for"
        assert time.perf_counter() - start < 1.0
        assert isinstance(box["error"], _ACCEPTED), f"a cut ends the read with one of the accepted errors, got {box.get('error')!r}"
    finally:
        server.close()


def test_a_dripped_header_read_is_cut_off_at_the_caller_deadline():
    """The maintainer's drip-headers case: each byte arrives inside the socket timeout, so only the shutdown stops
    the read — and it must do so with an accepted error, not by hanging or by a silent close."""
    server = _Server("drip")
    try:
        request = _request(server.port)
        held = net.wire(request)
        box = {}
        t = threading.Thread(target=_read, args=(net.opener(), request, box), daemon=True)
        t.start()
        t.join(0.5)
        assert t.is_alive(), "at 0.5 s the status line is still coming in a byte at a time"
        start = time.perf_counter()
        net.cut(held)
        t.join(2)
        assert not t.is_alive(), "the shutdown must end the header read"
        assert time.perf_counter() - start < 1.0
        assert isinstance(box["error"], _ACCEPTED), f"got {box.get('error')!r}"
    finally:
        server.close()


def test_cut_with_no_socket_or_after_the_worker_ended_is_harmless():
    """Nothing to cut (still connecting, or the name lookup — no socket yet) and a socket the worker already
    closed must both be a no-op: the callers call cut() on every deadline, whatever the request was doing."""
    net.cut(None)
    net.cut({})
    server = _Server("answer")
    try:
        request = _request(server.port)
        held = net.wire(request)
        box = {}
        t = threading.Thread(target=_read, args=(net.opener(), request, box), daemon=True)
        t.start()
        t.join(2)
        assert box.get("ok") is True, f"the fast server must answer, got {box.get('error')!r}"
        net.cut(held)  # its socket is closed already: must not raise
    finally:
        server.close()

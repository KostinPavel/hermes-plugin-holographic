"""The socket handle the network stages keep (issue #26): a request's connection writes its socket into the
request's holder as soon as ``connect()`` returns, and ``cut()`` ends the read the caller has stopped waiting for.
No provider here — the opener, a holder and one local server are the whole subject; the provider-level behaviour is
the drip tests in the rerank and embeddings files. What a cut leaves is asserted by CATEGORY, not by type: the
maintainer asked (#26) to accept both ConnectionAbortedError (Windows) and RemoteDisconnected / BadStatusLine where
a shutdown reads as EOF, and the Windows CI added a fourth — a cut that lands while the request is still going out
is `URLError(BrokenPipeError(10058))`. Where the shutdown lands and which OS it runs on decides the type; that the
read ENDS with the caller's deadline is the property. Both tests therefore wait for the server to have the whole
request (production cuts at its deadline, long after the send) and cut from the read phase only.
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
        self.received = threading.Event()  # the whole request body is read: the worker is in its read phase now
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
                self.received.set()  # the send has landed — from here the worker only waits for the answer
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


# The category of what a cut leaves: a broken socket (OSError — ConnectionAbortedError, the Windows
# URLError(BrokenPipeError(10058)) of a cut during the send, ConnectionResetError) or a broken answer line
# (HTTPException — RemoteDisconnected, BadStatusLine). The type itself is a detail of timing and OS, and a test
# pinned to types fails where the fix works (the Windows CI run of this file).
_BROKEN = (OSError, http.client.HTTPException)


def test_the_holder_gets_the_socket_while_the_worker_waits_for_headers_and_cut_ends_it():
    """The socket must reach the holder while the worker waits for an answer that never comes — the phase nothing
    else bounds (issue #26) — and the read must then end, woken by the cut or by its own socket timeout.

    WHICH of the two wakes it is a platform detail, and the test does not pin it: Linux's shutdown() unblocks the
    blocked reader at once; Windows' does not wake a thread sitting in recv while no data arrives — measured here,
    it ended at the 2 s socket timeout of _read, 1.5 s after the cut — and the reader sees the shutdown at its next
    byte instead. That is exactly #33's picture: the stall row read 1.0x with and without the shutdown (a socket
    timeout bounds a stall either way), while the drip row needed the shutdown (28.1x -> 1.2x). The drip test below
    is where the cut is the ONLY thing that can end the read."""
    server = _Server("stall")
    try:
        request = _request(server.port)
        held = net.wire(request)
        box = {}
        t = threading.Thread(target=_read, args=(net.opener(), request, box), daemon=True)
        t.start()
        # connect() ran before the send, so by the time the server has the request the socket is in the holder:
        # no polling, and no cut landing mid-send (on Windows that is URLError(BrokenPipeError(10058)) — and
        # production cuts at its deadline, long after the send).
        assert server.received.wait(2.0), "the server has the whole request"
        assert "sock" in held, "connect() must hand the socket over before the first header byte (issue #26)"
        t.join(0.5)  # the caller's deadline
        assert t.is_alive(), "the read is waiting for a status line that never comes"
        net.cut(held)
        t.join(3)  # dead by the cut or by _read's 2 s socket timeout — never past both
        assert not t.is_alive(), "the read must end: woken by the cut or bounded by its socket timeout"
        assert isinstance(box["error"], _BROKEN), f"a cut must break the read, got {box.get('error')!r}"
    finally:
        server.close()


def test_a_dripped_header_read_is_cut_off_at_the_caller_deadline():
    """The maintainer's drip-headers case: each byte arrives inside the socket timeout, so only the shutdown stops
    the read — the type it raises is a detail of the OS, that it ENDS at the deadline is the point."""
    server = _Server("drip")
    try:
        request = _request(server.port)
        held = net.wire(request)
        box = {}
        t = threading.Thread(target=_read, args=(net.opener(), request, box), daemon=True)
        t.start()
        assert server.received.wait(2.0), "the server has the whole request"
        assert "sock" in held, "the socket is in the holder before the first status-line byte"
        t.join(0.5)  # the caller's deadline: a couple of the 0.2 s drip's bytes have arrived
        assert t.is_alive(), "at the deadline the status line is still coming in a byte at a time"
        start = time.perf_counter()
        net.cut(held)
        t.join(2)
        assert not t.is_alive(), "the shutdown must end the header read"
        assert time.perf_counter() - start < 1.0
        assert isinstance(box["error"], _BROKEN), f"a cut must break the read, got {box.get('error')!r}"
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

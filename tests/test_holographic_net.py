"""The socket handle the network stages keep (issue #26): a request's connection writes its socket into the
request's holder as soon as ``connect()`` returns, and ``cut()`` ends the read the caller has stopped waiting for.
No provider here — the opener, a holder and one local server are the whole subject; the provider-level behaviour is
the drip tests in the rerank and embeddings files. What a cut leaves is asserted by CATEGORY, not by type: the
maintainer asked (#26) to accept both ConnectionAbortedError (Windows) and RemoteDisconnected / BadStatusLine where
a shutdown reads as EOF, and the Windows CI added a fourth — a cut that lands while the request is still going out
is `URLError(BrokenPipeError(10058))`. Where the shutdown lands and which OS it runs on decides the type; that the
read ENDS with the caller's deadline is the property. Both tests therefore wait for the server to have the whole
request and cut from the read phase only — not because production always cuts after the send (the maintainer's
note: the deadline can arrive during a slow send or a handshake; the category above covers that case too), but
because a cut mid-send measures a different signal than the header read these tests are about. The last three
tests are the two gaps of the review: cancellation that arrives before the socket, the CONNECT read inside
super().connect(), and the fd moving into the SSLSocket at the wrap.
"""

from __future__ import annotations

import http.client
import socket
import ssl
import threading
import time
import urllib.request
from pathlib import Path

from hermes_plugin_holographic import net

_CERT = Path(__file__).parent / "fixtures" / "tls_cert.pem"
_KEY = Path(__file__).parent / "fixtures" / "tls_key.pem"


class _Server:
    """One connection per request: reads it, then answers as the mode dictates. ``stall`` never answers, ``drip``
    sends the response one byte every *drip* seconds — each read succeeds on its own (drip < the socket timeout),
    so the reader never times out and only a cut ends it; at 0.2 s the status line is still arriving at the 0.5 s
    deadline, so the shutdown reads as an unfinished line (BadStatusLine / RemoteDisconnected), not as a clean
    EOF. ``answer`` replies at once."""

    def __init__(self, mode: str, drip: float = 0.2):
        self.mode = mode
        self.drip = drip
        self.note: OSError | None = None  # what a cut left on this side: recorded, never swallowed
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
                        if self.stop.wait(self.drip):
                            break
                elif self.mode == "answer":
                    conn.sendall(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n'
                                 b'Content-Length: 13\r\n\r\n{"results":[]}')
            except OSError as exc:
                self.note = exc  # the shutdown ends the connection mid-write: expected, and kept, not swallowed
            finally:
                conn.close()
        self.srv.close()

    def close(self):
        self.stop.set()
        self.thread.join(5)


class _TLSServer:
    """A local https endpoint for the phases between the TCP connection and the first header byte: it accepts,
    sits on the handshake for *handshake_delay* seconds (the client is inside ``wrap_socket`` then), reads the
    request and answers like ``_Server`` — ``drip`` one byte every *drip* seconds."""

    def __init__(self, mode: str = "drip", drip: float = 0.03, handshake_delay: float = 0.0):
        self.mode = mode
        self.drip = drip
        self.handshake_delay = handshake_delay
        self.note: OSError | None = None
        self.sent = 0  # diagnostics for the assertions below: how far the answer got, and when it started
        self.drip_start: float | None = None
        self.answer_len = 0
        self.stop = threading.Event()
        self.received = threading.Event()
        self.ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.ctx.load_cert_chain(_CERT, _KEY)
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
                if self.handshake_delay:
                    self.stop.wait(self.handshake_delay)
                tls = self.ctx.wrap_socket(conn, server_side=True)
            except OSError as exc:
                self.note = exc  # a cut during the handshake ends it here: expected
                conn.close()
                continue
            with tls:
                try:
                    tls.settimeout(10)
                    data = b""
                    while b"\r\n\r\n" not in data:
                        chunk = tls.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                    self.received.set()
                    answer = (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                              b"Content-Length: 13\r\nX-Slow: " + b"a" * 200 + b"\r\n\r\n" + b'{"results":[]}')
                    self.answer_len = len(answer)
                    if self.mode == "drip":
                        self.drip_start = time.perf_counter()
                        for byte in answer:
                            tls.sendall(bytes([byte]))
                            self.sent += 1
                            if self.stop.wait(self.drip):
                                break
                    else:
                        tls.sendall(answer)
                except OSError as exc:
                    self.note = exc  # the shutdown ends the connection mid-write: expected, and kept
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


def _https_request(port: int):
    return urllib.request.Request(f"https://127.0.0.1:{port}/v1", data=b'{"a":1}',
                                  headers={"Content-Type": "application/json"}, method="POST")


def _https_opener():
    """``net.opener`` with one difference: a context that trusts the fixture's self-signed certificate —
    production trusts the system CAs, a localhost test must trust its own. Everything else under test (the
    handlers, ``_TLSConn``, the holder, ``cut``) is production's; without the trust these tests would pass by
    failing the handshake instead of by the cut."""
    ctx = ssl.create_default_context(cafile=_CERT)
    opener = urllib.request.OpenerDirector()
    opener.add_handler(net._HTTP())
    opener.add_handler(net._HTTPS(context=ctx))
    return opener


def test_a_cut_that_precedes_the_socket_still_stops_the_request(monkeypatch):
    """The cut can arrive BEFORE the socket exists — the maintainer's reproducer: a connection slower than the
    caller's wait (0.5 s here against a 0.2 s deadline). Cancellation must be retained, not lost with the empty
    holder: a socket published after the caller's deadline has to be refused, so the request stops instead of
    starting. Without that the worker goes on into the header phase and lives its whole drip — a second after
    the cut it is still alive."""
    server = _Server("drip", drip=0.03)
    real_create = socket.create_connection  # HTTPConnection.__init__ binds this per instance, at connect time

    def slow_create(*args, **kwargs):
        time.sleep(0.5)
        return real_create(*args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", slow_create)
    try:
        request = _request(server.port)
        held = net.wire(request)
        box = {}
        t = threading.Thread(target=_read, args=(net.opener(), request, box), daemon=True)
        t.start()
        time.sleep(0.2)  # the caller's deadline: the connection is still being made
        assert "sock" not in held, "no socket exists yet — that is the point of the test"
        net.cut(held)
        t.join(1.5)  # the socket arrives at ~0.5 s: it must be refused there, not read from
        assert not t.is_alive(), "a socket published after the caller's deadline must not be read from (issue #26)"
        assert isinstance(box.get("error"), _BROKEN), f"the request must stop, got {box.get('error')!r}"
    finally:
        server.close()


def test_the_connect_response_is_inside_the_cuts_reach():
    """Through an https proxy the CONNECT response is read INSIDE super().connect() — before connect() returns and
    before any socket reaches the holder, so a cut during that read used to be lost with it. The socket is
    published at creation instead, where the TCP connection already exists and the tunnel read has not started."""
    server = _Server("drip", drip=0.03)
    try:
        request = urllib.request.Request("https://example.com/v1", data=b'{"a":1}',
                                         headers={"Content-Type": "application/json"}, method="POST")
        held = net.wire(request)
        conn = net._TLSConn("127.0.0.1", server.port, timeout=0.5, holder=held)
        conn.set_tunnel("example.com", 443)
        box = {}

        def tunnel():
            try:
                conn.connect()
                box["connected"] = True
            except BaseException as exc:  # noqa: BLE001 - a cut ends the connect: we assert on it
                box["error"] = exc
            finally:
                conn.close()

        t = threading.Thread(target=tunnel, daemon=True)
        t.start()
        assert server.received.wait(2.0), "the proxy has the CONNECT request"
        t.join(0.5)  # the caller's deadline: the response is still coming a byte at a time
        assert t.is_alive(), "the tunnel read is still waiting for the rest of the response"
        assert "sock" in held, "the TCP socket must be in the holder while the CONNECT response is read"
        net.cut(held)
        t.join(1.0)
        assert not t.is_alive(), "the cut must end the CONNECT read (issue #26)"
        assert isinstance(box.get("error"), _BROKEN), f"got {box.get('error')!r}"
    finally:
        server.close()


def test_a_cut_during_the_tls_handshake_stops_the_request():
    """Between the TCP connection and the finished handshake lies a window in which connect() has not returned and
    nothing is published yet; a cut there was lost. Publishing at creation makes the socket reachable while the
    handshake is still running on it."""
    server = _TLSServer("drip", drip=0.03, handshake_delay=0.5)
    try:
        request = _https_request(server.port)
        held = net.wire(request)
        box = {}
        t = threading.Thread(target=_read, args=(_https_opener(), request, box), daemon=True)
        t.start()
        time.sleep(0.2)  # the deadline: TCP is up (localhost), the handshake waits on the server
        assert "sock" in held, "the socket must be reachable while the handshake runs (issue #26)"
        net.cut(held)
        t.join(1.0)
        assert not t.is_alive(), "the cut must end the handshake"
        assert isinstance(box.get("error"), _BROKEN), f"got {box.get('error')!r}"
    finally:
        server.close()


def test_after_the_handshake_the_cut_reaches_the_wrapped_socket():
    """The fd moves into the SSLSocket at wrap_socket (ssl.SSLSocket._create detaches it from the TCP socket), so
    the holder must be refreshed to the wrapped socket: keeping the pre-handshake object would mean fd=-1 there
    and cut() would silently find nothing — in the very phase #26 is about, the header read after the handshake.

    What the cut leaves is asserted as a bounded END, never as a parser outcome (the maintainer, #42): the outcome
    is a detail of how many bytes arrived before the shutdown. An EOF after a COMPLETE status line is not an error
    to http.client at all — _read_headers ends on b'' as happily as on the blank line, HTTPResponse even accepts
    b'HTTP/1.1 200' without CRLF, and a bodyless success records box['ok'] (ubuntu CI saw exactly that); an EOF
    mid-line raises RemoteDisconnected / BadStatusLine, a shutdown the reader observes raises OSError (BrokenPipe
    locally). Any of the three passes the outcome check below; only a record outside them would be a surprise.
    The drip is sized so the cut comes at 0.5 s while the status line (17 bytes, ending at 0.85 s with this drip)
    is still coming — the read can only end because the cut ended it."""
    server = _TLSServer("drip", drip=0.05)
    try:
        t0 = time.perf_counter()
        request = _https_request(server.port)
        held = net.wire(request)
        box = {}
        t = threading.Thread(target=_read, args=(_https_opener(), request, box), daemon=True)
        t.start()
        assert server.received.wait(2.0), "the server has the whole request"
        t_rcv = time.perf_counter()
        assert isinstance(held.get("sock"), ssl.SSLSocket), \
            f"after the wrap the holder must hold the socket the request reads from, got {held.get('sock')!r}"
        t.join(0.5)  # the deadline: the header phase is dripping
        assert t.is_alive(), "at the deadline the headers are still coming a byte at a time"
        net.cut(held)
        t_cut = time.perf_counter()
        t.join(1.0)
        t_end = time.perf_counter()
        state = (f"rcv=+{t_rcv - t0:.2f}s cut=+{t_cut - t0:.2f}s end=+{t_end - t0:.2f}s | server: "
                 f"{server.sent}/{server.answer_len} bytes, drip_start="
                 f"{f'+{server.drip_start - t0:.2f}s' if server.drip_start else 'never'}, "
                 f"stop={server.stop.is_set()}, note={server.note!r}")
        assert not t.is_alive(), f"the cut must reach the wrapped socket; {state}"
        # The END is the property (the maintainer's #42 note): which of the three records comes out is a detail of
        # timing and OS — an EOF can be a valid bodyless success here, and requiring an exception made this flaky.
        assert box.get("ok") or isinstance(box.get("error"), _BROKEN), f"unexpected outcome: box={box!r}; {state}"
    finally:
        server.close()

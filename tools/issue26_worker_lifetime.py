#!/usr/bin/env python3
"""Issue #26: how long does the rerank worker really live past ``rerank_timeout``? Run by hand.

    python tools/issue26_worker_lifetime.py [--shutdown]

The caller's side is already measured by ``test_slow_body_returns_at_rerank_timeout_not_at_the_body``
(it returns at ~rerank_timeout). What is NOT measured is the worker's own end, which is what #26 is about.
Three local servers, ``rerank_timeout`` 0.5 (the Settings floor), one real ``Rescorer`` per run:

    stall         accepts and never answers      -> the socket timeout should bound it
    drip_headers  1 byte of the status line / 0.2 s -> no overall deadline inside http.client
    drip_body     headers at once, 8 body bytes / 0.2 s -> the worker's own deadline loop bounds it

Each read succeeds on its own (0.2 s < 0.5 s timeout), so only a check on the whole operation can stop
the worker. ``--shutdown`` shows the candidate fix from #26: the caller does ``sock.shutdown(SHUT_RDWR)``
at its deadline — the missing cancellation primitive, in sync code, without an HTTP layer of our own.
"""
from __future__ import annotations

import http.client
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rerank  # the module under test, after the path above

TIMEOUT = 0.5  # rerank_timeout floor (Settings)
DRIP = 0.2  # seconds between dripped bytes: under the socket timeout, over nothing else
RUNS = 2


def _server(mode: str, stop: threading.Event) -> int:
    """A one-shot-connection server; returns the port. Writes the response as the mode dictates."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def run() -> None:
        while not stop.is_set():
            srv.settimeout(0.2)
            try:
                conn, _ = srv.accept()
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
                head, _, sent = data.partition(b"\r\n\r\n")
                while len(sent) < length:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    sent += chunk
                if mode == "stall":
                    time.sleep(60)  # never answers
                elif mode == "drip_headers":
                    answer = b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 14\r\n\r\n{"results":[]}'
                    for byte in answer:
                        conn.sendall(bytes([byte]))
                        time.sleep(DRIP)
                elif mode == "drip_body":
                    conn.sendall(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 4096\r\n\r\n')
                    payload = b'{"results":[]}' + b"x" * 4082
                    for i in range(0, len(payload), 8):
                        conn.sendall(payload[i:i + 8])
                        time.sleep(DRIP)
            except OSError:
                pass  # the worker's deadline closes the connection: expected
            finally:
                conn.close()
        srv.close()

    threading.Thread(target=run, daemon=True).start()
    return port


def _recording_opener(sock_box: dict) -> urllib.request.OpenerDirector:
    """The real opener + one hook: remember the socket as soon as it exists (after connect).

    Enough to hand the caller a handle for shutdown(); without it the socket only appears
    once the response object exists — far too late for the header phase, where it matters most.
    """
    class _Recorder(urllib.request.HTTPHandler):
        def http_open(self, req):
            class Conn(http.client.HTTPConnection):
                def connect(self):
                    super().connect()
                    sock_box["sock"] = self.sock

            return self.do_open(Conn, req)  # AbstractHTTPHandler.do_open passes req.timeout along

    return urllib.request.build_opener(_Recorder(), rerank._NoRedirect())


def measure(mode: str, shutdown: bool) -> None:
    stop = threading.Event()
    port = _server(mode, stop)
    settings = rerank.Settings({"rerank": "on", "rerank_url": f"http://127.0.0.1:{port}/v1/rerank",
                                "rerank_timeout": "0.5"})
    rescorer = rerank.Rescorer(settings)
    sock_box: dict = {}
    original = rerank._OPENER
    if shutdown:
        rerank._OPENER = _recording_opener(sock_box)
    try:
        start = time.monotonic()
        rescorer.scores("when does the backup run?", ["a first fact", "a second fact"])
        caller = time.monotonic() - start
        worker = rescorer._worker
        alive = worker is not None and worker.is_alive()
        at = None
        if shutdown and alive:
            limit = time.monotonic() + 1.0
            while "sock" not in sock_box and time.monotonic() < limit:  # connect first, then the wait
                time.sleep(0.001)
            if "sock" in sock_box:
                sock_box["sock"].shutdown(socket.SHUT_RDWR)
                at = f"{(time.monotonic() - start) * 1000:6.1f} ms"
            else:
                at = "  no socket"
        if worker is not None:
            worker.join(60)  # how long it REALLY lives (cap 60 s)
        total = time.monotonic() - start
        print(f"  {mode:12s} shutdown={str(shutdown):5s}  caller {caller * 1000:6.1f} ms | "
              f"alive_after_call={str(alive):5s} | worker end {total * 1000:7.1f} ms ({total / TIMEOUT:4.1f}x)"
              + (f" | killed at {at}" if at is not None else ""))
    finally:
        rerank._OPENER = original
        stop.set()


def main() -> None:
    shutdown = "--shutdown" in sys.argv
    print(f"rerank_timeout = {TIMEOUT} s (Settings floor); worker end = x of that deadline")
    for mode in ("stall", "drip_headers", "drip_body"):
        for _run in range(RUNS):
            measure(mode, shutdown)


if __name__ == "__main__":
    main()
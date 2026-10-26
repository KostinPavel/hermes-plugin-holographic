#!/usr/bin/env python3
"""Issue #26: how long does the rerank worker live past ``rerank_timeout``? Run by hand.

    python tools/issue26_worker_lifetime.py [--no-cut]

The caller's side is measured by ``test_slow_body_returns_at_rerank_timeout_not_at_the_body`` (it returns at
~rerank_timeout). The worker's own end is what #26 is about, and it now has an answer in production (net.py): when
the caller's wait ends, the request's socket is shut down from the caller's side, so the worker dies with the
caller. Three local servers, ``rerank_timeout`` 0.5 (the Settings floor), one real ``Rescorer`` per run:

    stall         accepts and never answers        -> the socket timeout bounds it
    drip_headers  1 byte of the status line / 0.2 s -> bounded ONLY by the shutdown (104x without it here)
    drip_body     headers at once, 8 body bytes / 0.2 s -> the worker's own deadline loop bounds it

``--no-cut`` disables the shutdown (the pre-PR behaviour) for the comparison row: the drip-headers number is the
measured window of #26. The read a shutdown ends raises RemoteDisconnected / BadStatusLine where it reads as EOF,
ConnectionAbortedError on Windows — all recorded in the worker and none escaping the turn.

Windows only: the run raises the system timer to 1 ms (``timeBeginPeriod``) and releases it after, so a wait does
not overshoot by a whole 15.6 ms tick (measured 510.9 -> 500.7 ms mean on a 0.5 s timeout; Linux needs no such
help — 500.8 ms mean in WSL).
"""
from __future__ import annotations

import importlib.util
import os
import socket
import sys
import threading
import time
from pathlib import Path

# The package tree loads as hermes_plugin_holographic from the checkout (the CI's directory name rule: a valid
# Python identifier). hermes-agent's checkout goes on sys.path first (its `agent`, `tools`, `hermes_cli`).
_REPO = Path(__file__).resolve().parent.parent
_AGENT = Path(os.environ.get("HERMES_AGENT_REPO") or Path.home() / ".hermes" / "hermes-agent").expanduser()
for _p in (str(_AGENT), str(_REPO.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    "hermes_plugin_holographic", _REPO / "__init__.py", submodule_search_locations=[str(_REPO)])
assert _spec and _spec.loader
if "hermes_plugin_holographic" not in sys.modules:
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["hermes_plugin_holographic"] = _mod
    _spec.loader.exec_module(_mod)

from hermes_plugin_holographic import net, rerank  # noqa: E402

TIMEOUT = 0.5  # rerank_timeout floor (Settings)
DRIP = 0.2  # seconds between dripped bytes: under the socket timeout, over nothing else
RUNS = 2


def _server(mode: str, stop: threading.Event) -> tuple[int, threading.Thread]:
    """A one-shot-connection server; returns the port and its thread. Writes the response as the mode dictates;
    every wait is stop.wait(), so setting *stop* ends it within one drip and the caller can join it."""
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
                    stop.wait(60)  # never answers (until the run ends)
                elif mode == "drip_headers":
                    answer = b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nX-Slow: ' + b"a" * 200 + \
                             b'\r\n\r\n{"results":[]}'
                    for byte in answer:
                        conn.sendall(bytes([byte]))
                        if stop.wait(DRIP):
                            break
                elif mode == "drip_body":
                    conn.sendall(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 4096\r\n\r\n')
                    payload = b'{"results":[]}' + b"x" * 4082
                    for i in range(0, len(payload), 8):
                        conn.sendall(payload[i:i + 8])
                        if stop.wait(DRIP):
                            break
            except OSError:
                pass  # the shutdown (or the worker's own deadline) closes the connection: expected
            finally:
                conn.close()
        srv.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return port, thread


def measure(mode: str, cut: bool) -> None:
    stop = threading.Event()
    real_cut = net.cut
    if not cut:
        net.cut = lambda _holder: None  # the pre-PR behaviour, for the comparison row (--no-cut)
    server = None
    try:
        port, server = _server(mode, stop)
        settings = rerank.Settings({"rerank": "on", "rerank_url": f"http://127.0.0.1:{port}/v1/rerank",
                                    "rerank_timeout": "0.5"})
        rescorer = rerank.Rescorer(settings)
        start = time.monotonic()
        rescorer.scores("when does the backup run?", ["a first fact", "a second fact"])
        caller = time.monotonic() - start
        worker = rescorer._worker
        alive = worker is not None and worker.is_alive()
        if worker is not None:
            worker.join(60)  # how long it REALLY lives (cap 60 s)
        total = time.monotonic() - start
        lingering = worker is not None and worker.is_alive()
        error = rescorer._last_error  # why the endpoint was paused ("no answer (timeout)" etc.)
        print(f"  {mode:12s} cut={str(cut):5s}  caller {caller * 1000:6.1f} ms | "
              f"alive_after_call={str(alive):5s} | "
              + (f"worker STILL ALIVE after {total * 1000:7.1f} ms" if lingering
                 else f"worker end {total * 1000:7.1f} ms ({total / TIMEOUT:4.1f}x)")
              + f" | paused: {error}")
    finally:
        net.cut = real_cut
        stop.set()
        if server is not None:
            server.join(5)  # every wait in it is stop.wait(): it ends within one drip


def _high_resolution_timer():
    """Windows: raise the system timer to 1 ms for the run, so a wait does not overshoot its deadline by a whole
    15.6 ms tick (measured: 510.9 -> 500.7 ms mean on a 0.5 s timeout). Returns the handle to release in the
    caller's finally, or None where no such API exists — Linux needs no help here (hrtimers: 500.8 ms mean on the
    same deadline, measured in WSL), so the run is reported as "native" and nothing is raised or restored."""
    if sys.platform != "win32":
        return None
    import ctypes
    winmm = ctypes.WinDLL("winmm")
    if winmm.timeBeginPeriod(1) != 0:  # TIMERR_NOERROR; refused: report the native timer, release nothing
        return None
    return winmm


def main() -> None:
    cut = "--no-cut" not in sys.argv
    winmm = _high_resolution_timer()
    try:
        print(f"rerank_timeout = {TIMEOUT} s (Settings floor); worker end = x of that deadline | "
              f"cut={'on (net.py, issue #26)' if cut else 'OFF (pre-PR, --no-cut)'} | "
              f"timer: {'winmm period=1ms' if winmm is not None else 'native (no raising needed)'}")
        for mode in ("stall", "drip_headers", "drip_body"):
            for _run in range(RUNS):
                measure(mode, cut)
    finally:
        if winmm is not None:
            winmm.timeEndPeriod(1)  # never leave the machine on a 1 ms tick


if __name__ == "__main__":
    main()

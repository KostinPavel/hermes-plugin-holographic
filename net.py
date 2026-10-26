"""The socket of an in-flight request, kept where the caller can reach it: cancellation instead of a join (#26).

``urllib`` offers no way to stop a read from outside — a timeout only decides *when* to stop, and a caller-side
``response.close()`` waits for the worker in ``read()`` (measured: 2.0 s for a 21-byte body at ``rerank_timeout``
0.5, removed in PR #10). ``asyncio`` cancels by construction but has no HTTP client in the standard library, and
writing one over ``open_connection`` is a lot of new code on the path every turn takes — the maintainer's answer
on #26, which also refused ``httpx``: one stdlib client, no dependency on whatever the environment happens to
provide.

So the connection hands its socket to the request's holder as soon as ``connect()`` returns, and when the
caller's wait ends the caller shuts it down from its own thread. The worker then ends with the caller's deadline
*because the read under it ends*, not because several defences hold: in the header phase — where ``http.client``
has no overall deadline and every byte resets the socket timeout — a server dripping one byte per 0.2 s held a
worker 28x past a 0.5 s ``rerank_timeout``, and 1.2-1.3x with this shutdown (the numbers of #26; the run is
``tools/issue26_worker_lifetime.py --no-cut`` against ``--shutdown`` off and on, the old way and this one).

Not covered, by construction: DNS. The name is resolved before a socket exists, so nothing here can reach it and
no timeout of ours applies — the OS resolver bounds it (measured: 22.9 ms for an address that does not exist).

Mechanics: ``AbstractHTTPHandler.do_open`` builds the connection itself and passes any extra keyword straight to
its constructor, so the handler hands it the holder taken off the request. One holder per request, no global
state, and nothing left behind when a worker gets its socket after its caller has already given up on it.
"""

from __future__ import annotations

import http.client
import socket
import urllib.request


class _Keeper:
    """A connection that remembers its socket in the request's holder (the module's whole point)."""

    def __init__(self, *args, holder: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.holder = holder

    def connect(self):
        super().connect()
        if self.holder is not None:
            # As soon as the socket exists — before the first header byte, which is the phase nothing else
            # bounds. For https this is the wrapped socket: a handshake that hangs is still bounded by the
            # socket timeout (measured ~1.1x), a header phase is not (28x without the shutdown).
            self.holder["sock"] = self.sock


class _Conn(_Keeper, http.client.HTTPConnection):
    """Plain http."""


class _TLSConn(_Keeper, http.client.HTTPSConnection):
    """https."""


class _HTTP(urllib.request.HTTPHandler):
    """``http_open`` with our connection class instead of http.client's own."""

    def http_open(self, req):
        return self.do_open(_Conn, req, holder=getattr(req, "holder", None))


class _HTTPS(urllib.request.HTTPSHandler):
    """``https_open``, the same, keeping the context the default handler was built with."""

    def https_open(self, req):
        return self.do_open(_TLSConn, req, context=self._context, holder=getattr(req, "holder", None))


def opener(*handlers):
    """The one opener the network stages use: their handlers plus the two above. Everything else stays urllib's —
    the proxy environment (``HTTPS_PROXY`` / ``NO_PROXY``), cookies, the error pages — because ``build_opener``
    only replaces the defaults a passed handler is an instance of."""
    return urllib.request.build_opener(_HTTP(), _HTTPS(), *handlers)


def wire(request: urllib.request.Request) -> dict:
    """Give *request* the holder its connection fills with the socket; returned for :func:`cut`."""
    holder: dict = {}
    request.holder = holder
    return holder


def cut(holder: dict | None) -> None:
    """The caller's wait is over: end the read the worker is standing in.

    Nothing to do when the holder has no socket yet — the request is still connecting, or still resolving the
    name (a socket does not exist then; see the module). A socket the worker has already closed raises, and is
    ignored: it ended on its own between the caller's ``is_alive()`` and this call.

    Platform note, measured (Windows CI of #42): this shutdown does NOT wake a thread blocked in ``recv`` while
    nothing arrives — that read ends at the socket timeout the stage already sets (``rerank_timeout`` /
    ``embedding_timeout``), and the shutdown is observed at the next byte that does arrive. On Linux it unblocks
    the blocked reader at once. So a *stall* is bounded by the socket timeout either way (the 1.0x stall rows of
    #33) — this cut is what bounds a server that keeps dripping, where no single read ever times out (28x).
    """
    sock = (holder or {}).pop("sock", None)
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
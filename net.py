"""The socket of an in-flight request, kept where the caller can reach it: cancellation instead of a join (#26).

``urllib`` offers no way to stop a read from outside — a timeout only decides *when* to stop, and a caller-side
``response.close()`` waits for the worker in ``read()`` (measured: 2.0 s for a 21-byte body at ``rerank_timeout``
0.5, removed in PR #10). ``asyncio`` cancels by construction but has no HTTP client in the standard library, and
writing one over ``open_connection`` is a lot of new code on the path every turn takes — the maintainer's answer
on #26, which also refused ``httpx``: one stdlib client, no dependency on whatever the environment happens to
provide.

So the connection publishes its socket to the request's holder the moment the socket exists — inside
``_create_connection``, i.e. before the proxy's CONNECT read and the TLS handshake, both of which happen before
``connect()`` returns — and when the caller's wait ends, the caller shuts it down from its own thread. The
publication and :func:`cut` share the holder's lock, so a cut landing before the socket exists is RETAINED: a
socket that appears afterwards is closed and refused instead of read from, and no request starts after its
caller gave up. At the TLS wrap the fd moves into the ``SSLSocket`` (``wrap_socket`` detaches it from the TCP
socket), so ``connect()`` re-publishes and the holder always holds what the request reads from.

One interval is *retained but not interrupted* (the maintainer's note on #42): at the wrap the raw socket is
already detached while the SSLSocket is not yet published, so a cut landing exactly there reaches neither object.
The handshake then continues until its own socket timeout — bounded, not unbounded — and the retained
cancellation refuses the connection the moment the wrapped socket is published. Every phase is therefore bounded
by the cut, by the socket timeout, or by both; none of them promises an immediate interruption.

The worker then ends with the caller's deadline *because the read under it ends*, not because several defences
hold: in the header phase — where ``http.client`` has no overall deadline and every byte resets the socket
timeout — a server dripping one byte per 0.2 s held a worker 28x past a 0.5 s ``rerank_timeout``, and 1.2-1.3x
with this shutdown (the numbers of #26; the run is ``tools/issue26_worker_lifetime.py``).

Not covered, by construction: DNS. The name is resolved before a socket exists, so nothing here can reach it and
no timeout of ours applies — the OS resolver bounds it (measured: 22.9 ms for an address that does not exist).
The exception ends at the lookup: a cut that arrives during it still applies to the socket the lookup produces.

Mechanics: ``AbstractHTTPHandler.do_open`` builds the connection itself and passes any extra keyword straight to
its constructor, so the handler hands it the holder taken off the request. One holder per request, no global
state, and nothing is swallowed silently — :func:`cut` and the publication log what they find (module logger,
debug level: every branch of theirs is an expected race).
"""

from __future__ import annotations

import http.client
import logging
import socket
import threading
import time
import urllib.request

logger = logging.getLogger(__name__)


def _publish(holder: dict | None, sock: socket.socket | None) -> None:
    """Hand the request's socket over — the one place where a socket becomes reachable by :func:`cut`.

    Called twice per connection: at creation (the socket exists; the CONNECT read and the TLS handshake have not
    run yet) and after ``connect()`` returns (the fd then belongs to the wrapped socket). Both calls take the
    holder's lock — the lock :func:`cut` takes — so no order loses the cancellation: a socket published after
    the caller's deadline is closed and refused, and the connection stops instead of sending anything.
    """
    if holder is None or sock is None:
        return
    lock = holder.get("lock")
    if lock is None:  # not a holder wire() made: nothing can cancel it, publish as plain state
        holder["sock"] = sock
        return
    with lock:
        if not holder.get("cancelled"):
            holder["sock"] = sock
            return
    logger.debug("net: the socket arrived after the caller's deadline — closing it and stopping the request")
    try:
        sock.close()
    except OSError as exc:
        logger.debug("net: closing the late socket failed: %r", exc)
    raise ConnectionAbortedError("the caller's deadline passed before the connection was ready")


class _Keeper:
    """A connection that keeps its socket in the request's holder (the module's whole point)."""

    sock: socket.socket | None  # HTTPConnection's; annotated because this mixin does not inherit from it

    def __init__(self, *args, holder: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.holder = holder
        # HTTPConnection.__init__ just set self._create_connection = socket.create_connection ("an instance
        # variable to allow unit tests to replace it"): wrap it here, so the socket is published the moment it
        # is connected — before TCP_NODELAY, before a proxy's CONNECT response is read, before the handshake.
        create = self._create_connection

        def _create(address, timeout, source_address):  # HTTPConnection.connect passes all three, positionally
            sock = create(address, timeout, source_address)
            _publish(self.holder, sock)
            return sock

        self._create_connection = _create

    def connect(self):
        super().connect()
        # Re-publish: at the TLS wrap the fd moved into the SSLSocket (the TCP socket got detached), so the
        # holder must follow it — cut() on the old object would find fd=-1 and do nothing, in the very phase
        # #26 is about. For plain http this publishes the same object again.
        _publish(self.holder, self.sock)
        if self.holder is not None:
            # Connected (TCP, a proxy's CONNECT, TLS): the request goes out next, so from here on the time is the
            # server's — what rerank measures its speed by; a timeout before this point says nothing of the speed.
            self.holder["connected"] = time.perf_counter()


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
    holder: dict = {"lock": threading.Lock(), "cancelled": False}
    request.holder = holder
    return holder


def cut(holder: dict | None) -> None:
    """The caller's wait is over: end the read the worker is standing in.

    A holder with no socket yet is not a no-op: the cancellation is retained (``cancelled``, under the holder's
    lock), so a socket that appears afterwards — a connection still in progress, a name still being resolved —
    is refused by :func:`_publish` instead of read from. Only DNS itself stays out of reach: no socket exists
    there to shut down.

    The one interval this call cannot end is the TLS wrap (see the module docstring): the raw socket is already
    detached while the SSLSocket is not yet published, so there is no object here to shut down — the handshake
    runs to its socket timeout, and the retained cancellation closes the connection when the wrapped socket
    appears.

    A socket the worker has already closed raises, and is logged rather than silently ignored: it ended on its
    own between the caller's ``is_alive()`` and this call.

    Platform note, measured (Windows CI of #42): this shutdown does NOT wake a thread blocked in ``recv`` while
    nothing arrives — that read ends at the socket timeout the stage already sets (``rerank_timeout`` /
    ``embedding_timeout``), and the shutdown is observed at the next byte that does arrive. On Linux it unblocks
    the blocked reader at once. So a *stall* is bounded by the socket timeout either way (the 1.0x stall rows of
    #33) — this cut is what bounds a server that keeps dripping, where no single read ever times out (28x).
    """
    if not holder:
        return
    lock = holder.get("lock")
    if lock is None:  # not a holder wire() made: nothing to retain, only a socket to end
        sock = holder.pop("sock", None)
    else:
        with lock:
            holder["cancelled"] = True
            sock = holder.get("sock")
    if sock is None:
        logger.debug("net: cut with no socket yet — the cancellation is retained for a late one")
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError as exc:
        logger.debug("net: cut found the socket already gone: %r", exc)
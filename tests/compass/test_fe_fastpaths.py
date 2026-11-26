# SPDX-License-Identifier: MIT
"""anyio's memory streams and uvicorn's h11 body send through `fe_fastpaths`,
against the libraries' own code.

A Starlette streaming response behind a ``BaseHTTPMiddleware``, which hands
each body message on through an anyio memory object stream, is served by
uvicorn's H11 protocol to several connections at once on one loop, over
transports that log what is written to them. The app logs each event as it
yields it into the same log, and the callbacks the loop runs are counted: a
fast path that added or dropped an await reorders the log or changes the
count, and one that framed a chunk otherwise changes its bytes. One connection
is lost part way through its stream.
"""

import asyncio

import pytest
import uvicorn
from anyio.streams import memory
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse
from starlette.routing import Route
from uvicorn.protocols.http import h11_impl
from uvicorn.server import ServerState

from atom.compass.run import fe_fastpaths
from atom.utils.compass_loop import CompassEventLoop
from tests.compass.test_compass_event_loop import _idle_traffic, run  # noqa: F401

CONNS, EVENTS, LOST = 6, 5, 2  # connection LOST is lost after its 3rd write
REPLACED = [
    (memory.MemoryObjectReceiveStream, "receive"),
    (memory.MemoryObjectSendStream, "send"),
    (h11_impl.RequestResponseCycle, "send"),
]


@pytest.fixture(autouse=True)
def own_code(monkeypatch):
    """Each method `install` replaces is the library's own, whatever an
    earlier `CompassEventLoop` in this process installed, and is put back as
    it was after the test. Yields two counts: the body messages with more to
    come that reach uvicorn's own send on a connection not lost, and the
    receivers anyio's own receive builds to wait on a memory stream."""
    calls = {"chunks": 0, "receivers": 0}
    for owner, name in REPLACED:
        current = getattr(owner, name)
        if getattr(current, fe_fastpaths._INSTALLED, False):
            current = current.__wrapped__
        monkeypatch.setattr(owner, name, current)
    send = h11_impl.RequestResponseCycle.send

    async def counted_send(cycle, message):
        if message.get("more_body", False) and not cycle.disconnected:
            calls["chunks"] += 1
        await send(cycle, message)

    monkeypatch.setattr(h11_impl.RequestResponseCycle, "send", counted_send)
    receiver = memory._MemoryObjectItemReceiver
    init = receiver.__init__

    def counted_init(self):
        calls["receivers"] += 1
        init(self)

    monkeypatch.setattr(receiver, "__init__", counted_init)
    yield calls


class _Wire(asyncio.Transport):
    """Connection `conn`'s transport: logs each write as ``(conn, bytes)``."""

    def __init__(self, conn: int, log: list, lose=None) -> None:
        super().__init__()
        self.conn, self.log, self.lose = conn, log, lose
        self.written, self.closing = 0, False

    def get_extra_info(self, name, default=None):
        addr = {"sockname": 8000, "peername": 9000 + self.conn}.get(name)
        return default if addr is None else ("127.0.0.1", addr)

    def write(self, data) -> None:
        self.log.append((self.conn, bytes(data)))
        self.written += 1
        if self.lose is not None and self.written == 3:
            self.lose()

    def is_closing(self) -> bool:
        return self.closing

    def close(self) -> None:
        self.closing = True

    def pause_reading(self) -> None:
        pass

    def resume_reading(self) -> None:
        pass


def _app(log: list) -> Starlette:
    async def events(request):
        conn = int(request.path_params["conn"])

        async def frames():
            for k in range(EVENTS):
                log.append((conn, k))
                yield f'data: {{"conn": {conn}, "k": {k}}}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(frames(), media_type="text/event-stream")

    async def passed_on(request, call_next):
        return await call_next(request)

    return Starlette(
        routes=[Route("/events/{conn}", events)],
        middleware=[Middleware(BaseHTTPMiddleware, dispatch=passed_on)],
    )


def _served(loop) -> tuple[list, int]:
    """`CONNS` streams served on `loop`: the log of every write and event,
    and how many callbacks the loop ran, task steps included."""
    log, ran = [], 0
    config = uvicorn.Config(_app(log), lifespan="off", log_level="warning")
    state = ServerState()
    for conn in range(CONNS):
        protocol = h11_impl.H11Protocol(config, state, {}, _loop=loop)
        lose = None
        if conn == LOST:
            lose = lambda p=protocol: loop.call_soon(p.connection_lost, None)
        protocol.connection_made(_Wire(conn, log, lose))
        protocol.data_received(b"GET /events/%d HTTP/1.1\r\nHost: t\r\n\r\n" % conn)
    run_handle = asyncio.Handle._run

    def counted_run(handle):
        nonlocal ran
        ran += 1
        run_handle(handle)

    asyncio.Handle._run = counted_run
    try:
        loop.run_until_complete(asyncio.wait(list(state.tasks)))
    finally:
        asyncio.Handle._run = run_handle
    return log, ran


def _own_code_served(calls) -> tuple[list, int]:
    loop = asyncio.new_event_loop()
    served = _served(loop)
    loop.close()
    assert calls["chunks"] > 0 and calls["receivers"] > 0
    calls.update(chunks=0, receivers=0)
    return served


def test_the_fast_paths_write_the_same_bytes_after_the_same_awaits(own_code):
    """Installed, the fast paths serve the streams with the writes and events
    of the libraries' own code, in the same order and in as many loop
    callbacks; no streamed chunk reaches uvicorn's own send, and anyio's own
    receive builds no receiver."""
    own = _own_code_served(own_code)
    fe_fastpaths.install()
    loop = asyncio.new_event_loop()
    assert _served(loop) == own
    loop.close()
    assert own_code == {"chunks": 0, "receivers": 0}


def test_a_compass_event_loop_serves_through_the_fast_paths(
    own_code, run  # noqa: F811
):
    """A `CompassEventLoop` installs the fast paths as it is built, and serves
    the streams as asyncio's own loop does with the libraries' own code."""
    own = _own_code_served(own_code)
    _idle_traffic(run)
    loop = CompassEventLoop()
    assert _served(loop) == own
    loop.run_forever()  # the run ends at the +inf grant
    loop.close()
    assert own_code == {"chunks": 0, "receivers": 0}


@pytest.mark.parametrize("other", ["anyio", "h11", "uvicorn"])
def test_a_library_at_another_version_keeps_its_own_code(other, monkeypatch):
    """With `other` at a version the fast paths were not written against,
    `install` leaves the methods that library's fast path replaces as they
    were, and replaces the rest once, however often it runs."""
    monkeypatch.setitem(fe_fastpaths.CHECKED, other, "0.0.0")
    before = {(o, n): getattr(o, n) for o, n in REPLACED}
    fe_fastpaths.install()
    fe_fastpaths.install()
    kept = {"anyio": REPLACED[:2], "h11": REPLACED[2:], "uvicorn": REPLACED[2:]}
    for (owner, name), was in before.items():
        now = getattr(owner, name)
        if (owner, name) in kept[other]:
            assert now is was
        else:
            assert now.__wrapped__ is was

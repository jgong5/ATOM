# SPDX-License-Identifier: MIT
"""`CompassEventLoop` against the co-hosted clock authority.

Three LPs as in a one-engine run. The frontend is the loop under test; the
engine waits in NER(+inf), and so does the traffic LP unless a test drives it.
The daemon timers are the real ones: the API server's metrics refresh loop,
and uvicorn's server tick and keep-alive timeout.
"""

import asyncio
import http.client
import importlib.util
import math
import queue
import threading
import time
import uuid
from types import SimpleNamespace

import pytest
import uvicorn

from atom.compass import clock_transport
from atom.compass.clock import NER, ClockAuthority, LpId, single_engine_table
from atom.entrypoints.openai import api_server
from atom.utils import clock
from atom.utils.clock import LPRuntime
from atom.utils.compass_loop import CompassEventLoop, HttpChannel

TABLE = single_engine_table(admission_path="serving", ipc_s=0.001, stream_s=0.002)
TRAFFIC, FRONTEND, ENGINE = LpId("traffic"), LpId("frontend"), LpId("engine")
HTTP = "traffic->frontend:http"
INF = math.inf


class _Watched:
    """The frontend's connection: each request, each grant, and each NER sent
    while a released request was unread."""

    def __init__(self, conn) -> None:
        self.conn, self.sent, self.grants, self.unread = conn, [], [], []
        self.rt = None

    def send(self, msg) -> None:
        if self.rt.inline_pending():
            self.unread.append(msg)
        self.sent.append(msg)
        if len(self.sent) > 1000:
            raise RuntimeError("1000 time requests: a timer keeps the run alive")
        self.conn.send(msg)

    def recv(self):
        g, released = self.conn.recv()
        self.grants.append(g)
        return g, released


@pytest.fixture
def run(monkeypatch):
    endpoint = f"inproc:test-compass-loop-{uuid.uuid4().hex}"
    server = clock_transport.serve(ClockAuthority(TABLE), endpoint)
    clock_transport.connect(ENGINE, endpoint).send((NER, INF, [], INF))
    conn = _Watched(clock_transport.connect(FRONTEND, endpoint))
    conn.rt = rt = LPRuntime(FRONTEND, TABLE, conn)
    rt.start_run()
    clock.install(rt)
    # Every loop the test builds, uvicorn's included, is stopped after 20 wall
    # seconds, and the test fails: a loop that never stops cannot hang it.
    loops, fired, init = [], [], CompassEventLoop.__init__

    def tracked(self):
        init(self)
        loops.append(self)

    def stop_all():
        fired.append(True)
        for loop in loops:
            if not loop.is_closed():
                loop.call_soon_threadsafe(loop.stop)

    monkeypatch.setattr(CompassEventLoop, "__init__", tracked)
    guard = threading.Timer(20, stop_all)
    guard.start()
    try:
        yield SimpleNamespace(rt=rt, conn=conn, endpoint=endpoint)
    finally:
        guard.cancel()
        clock.install(None)
        server.close()
    assert not fired, "a loop ran 20 wall seconds without stopping"


def _idle_traffic(run) -> None:
    clock_transport.connect(TRAFFIC, run.endpoint).send((NER, INF, [], INF))


def _until(cond) -> None:
    deadline = time.monotonic() + 10
    while not cond():
        assert time.monotonic() < deadline, "not reached in 10 wall seconds"
        time.sleep(0.001)


def _traffic(run, stamps: queue.Queue, before_ner) -> threading.Thread:
    """The traffic LP: stamps one request at 0, then `before_ner()`, then idles."""

    def body():
        rt = LPRuntime(TRAFFIC, TABLE, clock_transport.connect(TRAFFIC, run.endpoint))
        rt.start_run()
        stamps.put(rt.stamp_send(HTTP))
        before_ner()
        rt.next_event(INF)
        rt.close()

    thread = threading.Thread(target=body, name="traffic", daemon=True)
    thread.start()
    return thread


def _stamp(scope):
    """Stands in for the request carrier: ``x-test-stamp: <arrival> <seq>``."""
    for name, value in scope["headers"]:
        if name == b"x-test-stamp":
            arrival, seq = value.split()
            return float(arrival), int(seq)
    return None


def test_a_60_second_sleep_ends_at_lp_60_with_no_wall_wait(run):
    _idle_traffic(run)
    loop = CompassEventLoop()
    wall = time.monotonic()
    loop.run_until_complete(asyncio.sleep(60))
    wall, lp = time.monotonic() - wall, loop.time()
    loop.run_forever()  # nothing is left: the finish grants +inf and the loop stops
    loop.close()
    print(f"\n  asyncio.sleep(60): {lp} LP seconds in {wall:.4f} wall seconds")
    assert lp == 60.0 and wall < 5.0
    assert run.conn.sent == [(NER, 60.0, [], INF), (NER, INF, [], INF)]
    assert run.rt.now == INF


def test_the_metrics_refresh_is_a_daemon_and_the_finish_stops_the_loop(
    run, monkeypatch
):
    _idle_traffic(run)
    refreshed, late = [], []

    async def refresh():
        loop = asyncio.get_running_loop()
        refreshed.append(loop.time())
        # A plain callback, a daemon because the refresh loop schedules it.
        loop.call_later(4, lambda: late.append(loop.time()))

    monkeypatch.setattr(api_server, "_refresh_metrics_once", refresh)

    async def serve():
        asyncio.create_task(api_server._metrics_refresh_loop())
        await asyncio.sleep(12)  # the last essential timer
        await asyncio.Event().wait()  # serves until stopped, as uvicorn does

    # asyncio.run, as uvicorn calls it: a stop ends the main task early.
    with pytest.raises(RuntimeError, match="Event loop stopped before Future"):
        asyncio.run(serve(), loop_factory=CompassEventLoop)
    # The timers due at 14 and 15 never run, not even at +inf.
    assert refreshed == [5.0, 10.0] and late == [9.0]
    assert run.conn.sent == [
        (NER, 12.0, [], 5.0),
        (NER, 12.0, [], 9.0),
        (NER, 12.0, [], 10.0),
        (NER, 12.0, [], 14.0),
        (NER, INF, [], 14.0),
    ]
    assert run.conn.grants == [5.0, 9.0, 10.0, 12.0, INF]


def test_no_time_is_asked_while_a_station_job_is_open(run):
    _idle_traffic(run)
    loop = CompassEventLoop()

    async def go():
        later = asyncio.ensure_future(asyncio.sleep(1))  # a timer to jump to

        def do_preprocess():
            time.sleep(0.2)  # open across the loop's next select

        await loop.run_in_executor(None, do_preprocess)
        done = loop.time()
        await later
        return done

    assert loop.run_until_complete(go()) == 0.0
    assert len(loop.executor.station.jobs) == 1  # the default executor's station
    loop.run_forever()
    loop.close()


def test_closing_the_loop_before_the_finish_is_refused(run):
    _idle_traffic(run)
    loop = CompassEventLoop()
    loop.run_until_complete(asyncio.sleep(1))
    with pytest.raises(RuntimeError, match=r"before the \+inf grant"):
        loop.close()


def test_a_request_released_before_it_is_read_is_handed_over_at_its_arrival(run):
    stamps, handled, got = queue.Queue(), [], {}

    async def app(scope, receive, send):
        handled.append(asyncio.get_running_loop().time())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    server = uvicorn.Server(
        uvicorn.Config(
            HttpChannel(app, _stamp),
            host="127.0.0.1",
            port=0,
            loop="atom.utils.compass_loop:CompassEventLoop",
            lifespan="off",
            log_level="warning",
        )
    )
    traffic = _traffic(run, stamps, before_ner=lambda: None)
    arrival, seq = stamps.get(timeout=10)

    def released():
        with run.rt.lock:
            return run.rt.is_released(HTTP, seq)

    def client():
        _until(lambda: server.started and released())
        port = server.servers[0].sockets[0].getsockname()[1]
        got["conn"] = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        got["conn"].request("GET", "/", headers={"x-test-stamp": f"{arrival!r} {seq}"})
        # The connection stays open, so its keep-alive timer stays set.
        got["body"] = got["conn"].getresponse().read()

    thread = threading.Thread(target=client, name="client", daemon=True)
    thread.start()
    try:
        with pytest.raises(RuntimeError, match="Event loop stopped before Future"):
            server.run()
    finally:
        thread.join(10)
        traffic.join(10)
        if "conn" in got:
            got["conn"].close()
    assert got["body"] == b"ok" and handled == [arrival]
    assert run.conn.unread == []
    # The server tick and the keep-alive timeout never move the clock.
    assert max(g for g in run.conn.grants if g < INF) == arrival
    assert run.conn.grants[-1] == INF


def test_a_request_read_before_its_release_is_held_until_it(run):
    stamps, handled = queue.Queue(), []
    traffic = _traffic(
        run, stamps, before_ner=lambda: _until(lambda: run.rt.arrived[HTTP])
    )
    arrival, seq = stamps.get(timeout=10)

    async def app(scope, receive, send):
        handled.append(asyncio.get_running_loop().time())

    scope = {
        "type": "http",
        "headers": [(b"x-test-stamp", f"{arrival!r} {seq}".encode())],
    }
    loop = CompassEventLoop()
    try:
        loop.run_until_complete(HttpChannel(app, _stamp)(scope, None, None))
        loop.run_forever()
    finally:
        traffic.join(10)
    loop.close()
    assert handled == [arrival]
    assert run.conn.grants == [arrival, INF] and run.conn.unread == []


def test_compass_off_keeps_uvloop_and_on_names_a_loop_uvicorn_builds(run):
    clock.install(None)
    off = "uvloop" if importlib.util.find_spec("uvloop") else "auto"
    assert api_server._loop_impl() == off
    clock.install(run.rt)
    config = uvicorn.Config(lambda *_: None, loop=api_server._loop_impl())
    assert config.get_loop_factory() is CompassEventLoop

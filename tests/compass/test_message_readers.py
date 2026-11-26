# SPDX-License-Identifier: MIT
"""Readers of an LP runtime's messages answer from the state kept as messages
come and go, so a call costs the same however many messages the run brought.

`LPRuntime.holds_unstamped`; the waits of lmcache's and the Mooncake Store's
loops and shells for a released reply still on the wire (`StationLP._settle`,
`LMCacheMPShell._take`, `StoreShell._take`); the parity traffic LP's take
(`Traffic._take`). Each runs on a real `LPRuntime` under a scripted authority.
"""

import math
import pickle
import threading
import time
from concurrent.futures import Future

import msgspec
import pytest
import zmq

from atom.compass.clock import LpId, single_engine_table
from atom.compass.clock.channels import ReceiveMode, add_lmcache, add_mooncake_store
from atom.compass.kv.lmcache_mp import LMCacheMPShell
from atom.compass.kv.lmcache_server import StationLP
from atom.compass.kv.mooncake_store import StoreShell
from atom.compass.parity import traffic
from atom.utils.clock import LPRuntime

ENGINE, LMCACHE, TRAFFIC = LpId("engine"), LpId("lmcache"), LpId("traffic")
REQ = "frontend->engine:request#dp0"  # a thread channel
CTL = "frontend->engine:control#dp0"  # a thread channel
COPY = "lmcache->engine:transfer#dp0"  # inline, the shell's
TO_LMCACHE = "engine->lmcache:transfer#dp0"
STORE_LOOK = "engine->mooncake_store:lookup#dp0"
STORE_BACK = "mooncake_store->engine:lookup#dp0"
STREAM = "frontend->traffic:stream"
IPC = 0.001


class _Authority:
    """Answers each clock call with the next grant, ``(G, released)``."""

    def __init__(self, *grants):
        self.grants = list(grants)

    def send(self, msg):
        pass

    def recv(self):
        return self.grants.pop(0)


class _Wire:
    """A socket whose frames can be read only once it has been polled: a
    released reply still on the wire until its reader waits for it."""

    def __init__(self, *frames):
        self.frames, self.readable, self.polled = list(frames), [], False

    def recv_multipart(self, flags=0):
        if not self.readable:
            raise zmq.Again
        return self.readable.pop(0)

    def poll(self, *args):
        assert self.frames, "polled with nothing on the wire"
        self.readable, self.frames, self.polled = self.frames, [], True
        return zmq.POLLIN


def _runtime(lp, *grants, owner=None):
    table = single_engine_table(admission_s=IPC, ipc_s=IPC, stream_s=IPC)
    add_lmcache(table, "engine", IPC)
    add_mooncake_store(table, ["engine"], IPC)
    rt = LPRuntime(lp, table, _Authority(*grants), owner=owner)
    rt.start_run()
    return rt


def _after(lp, ch, n, batch=100):
    """`lp`'s runtime after `n` messages on `ch`, each read, released and
    handled, a grant releasing `batch` of them. Each is read and handled
    ahead of its grant: the sets end as a run's do, with no receiving thread
    for a thread channel's release to wait on."""
    batches = [range(k, min(k + batch, n)) for k in range(0, n, batch)]
    grants = [(float(b.stop), {ch: [(s, s + 0.5) for s in b]}) for b in batches]
    rt = _runtime(lp, *grants)
    for b, (g, _) in zip(batches, grants):
        for s in b:
            rt.check_arrival(ch, s + 0.5, s)
        with rt.lock:
            for s in b:
                rt.count_done_locked(ch, s)
        rt.advance_to(g)
    return rt


def _holds(rt, wire):
    def holds():
        with rt.lock:
            return rt.holds_unstamped()

    return holds


def _settle(rt, wire):
    return StationLP(rt, wire, {})._settle


def _copies(rt, wire):
    shell = LMCacheMPShell.__new__(LMCacheMPShell)
    shell._rt, shell._socks, shell._read = rt, {COPY: wire}, {}
    shell._futures = {(COPY, 0): (Future(), "store", "r0")}
    return shell._take


def _lookups(rt, wire):
    shell = StoreShell.__new__(StoreShell)
    shell._rt, shell._offered, shell._read = rt, [], {}
    shell._answer = lambda reply: None
    shell._socks, shell._look, shell._back = {STORE_LOOK: wire}, STORE_LOOK, STORE_BACK
    return shell._take


def _traffic(rt, wire):
    lp = traffic.Traffic.__new__(traffic.Traffic)
    lp.rt, lp.stream, lp.port, lp.cv = rt, STREAM, None, threading.Condition()
    lp.streams = [STREAM]
    lp.frames, lp.errors, lp.done = {}, [], 0
    return lp._take


# name: (its LP, the channel it reads, its call on a runtime and a socket)
READERS = {
    "LPRuntime.holds_unstamped": (ENGINE, REQ, _holds),
    "StationLP._settle": (LMCACHE, TO_LMCACHE, _settle),
    "LMCacheMPShell._take": (ENGINE, COPY, _copies),
    "StoreShell._take": (ENGINE, STORE_BACK, _lookups),
    "Traffic._take": (TRAFFIC, STREAM, _traffic),
}


def _us_a_call(fn):
    """`fn()`'s cost: the best of three passes of 200 calls, in microseconds a call."""
    best = math.inf
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(200):
            fn()
        best = min(best, time.perf_counter() - start)
    return best / 200 * 1e6


@pytest.mark.parametrize("reader", READERS)
def test_a_reader_costs_the_same_however_many_messages_came_before(reader):
    lp, ch, call = READERS[reader]
    early, late = (call(_after(lp, ch, n), _Wire()) for n in (10, 50_000))
    assert not early() and not late()
    a, b = _us_a_call(early), _us_a_call(late)
    assert b < 5 * a + 5, f"{a:.1f} us after 10 messages, {b:.1f} us after 50,000"


# reader: the frames before and after the header its socket reads
WAITERS = {
    "StationLP._settle": ([b"engine"], [b"request"]),
    "LMCacheMPShell._take": (
        [],
        [msgspec.msgpack.encode(0), b"", msgspec.msgpack.encode([b"", True])],
    ),
    "StoreShell._take": ([], [pickle.dumps("reply")]),
}


@pytest.mark.parametrize("reader", WAITERS)
def test_a_reader_waits_for_a_released_reply_still_on_the_wire(reader):
    lp, ch, call = READERS[reader]
    before, after = WAITERS[reader]
    rt = _runtime(lp, (1.0, {ch: [(0, 0.5)]}))
    rt.advance_to(1.0)
    wire = _Wire([*before, pickle.dumps((ch, 0.5, 0)), *after])
    call(rt, wire)()
    assert wire.polled and rt.arrived[ch] == {0}
    if reader != "StationLP._settle":  # a shell takes it: handled
        assert rt.handled[ch] == {0}


def _released(rt, ch, seq):
    for _ in range(10_000):
        with rt.lock:
            if rt.is_released(ch, seq):
                return
        time.sleep(0.001)
    raise AssertionError(f"{ch} seq {seq} is never released")


def test_holding_an_unstamped_frame_answers_what_walking_every_frame_answered():
    """A frame a thread channel brought holds one sent outside the run until
    it is handled, read ahead of its release or after it; a released frame not
    read yet holds none, nor does an inline channel's. This thread receives;
    the owner steps through three grants, waiting at each thread frame."""
    grants = [
        (1.0, {COPY: [(0, 0.4)], REQ: [(0, 0.5)]}),
        (2.0, {CTL: [(0, 1.5)]}),
        (3.0, {REQ: [(1, 2.5)]}),
    ]
    owner = threading.Thread(
        target=lambda: [rt.advance_to(g) for g, _ in grants], daemon=True
    )
    rt = _runtime(ENGINE, *grants, owner=owner)

    def holds(expected):
        with rt.lock:
            walked = any(
                rt.arrived[ch] - rt.handled[ch]
                for ch in rt.arrived
                if rt.table.recv_mode(ch) is ReceiveMode.THREAD
            )
            assert rt.holds_unstamped() == walked == expected

    def handle(ch, seq):
        rt.handed_over(ch, seq)
        rt.back_at_wait_point()

    rt.check_arrival(COPY, 0.4, 0)
    holds(False)  # an inline frame, read ahead
    owner.start()
    _released(rt, REQ, 0)
    holds(False)  # released, not read; the inline one released and read
    rt.check_arrival(REQ, 0.5, 0)
    holds(True)
    handle(REQ, 0)
    _released(rt, CTL, 0)
    holds(False)
    rt.check_arrival(REQ, 2.5, 1)
    holds(True)  # read ahead of its release
    rt.check_arrival(CTL, 1.5, 0)
    handle(CTL, 0)
    _released(rt, REQ, 1)
    holds(True)
    handle(REQ, 1)
    owner.join(5)
    assert not owner.is_alive()
    holds(False)


def test_the_traffic_lp_takes_each_released_event_read_before_or_after_its_release(
    monkeypatch,
):
    """Events 0 and 3 are read ahead of their release, 1 and 2 after it, while
    the take waits for them; each take takes what is released and not taken."""
    rt = _runtime(
        TRAFFIC,
        (1.0, {STREAM: [(0, 0.2), (1, 0.4), (2, 0.6)]}),
        (2.0, {STREAM: [(3, 1.5)]}),
    )
    take = _traffic(rt, None)
    lp = take.__self__
    monkeypatch.setattr(traffic, "post", lambda port, body, headers: iter(body))

    def read(*events):  # one response's stream: (arrival, seq, data line)
        lines = [line for a, s, d in events for line in (f": compass a={a!r} s={s}", d)]
        lp._read(lines, "")

    taken, count = [], rt.count_done_locked
    rt.count_done_locked = lambda ch, seq: (taken.append(seq), count(ch, seq))
    read((0.2, 0, "data: a"), (1.5, 3, traffic.DONE))
    rt.advance_to(1.0)
    later = threading.Timer(0.05, read, ((0.4, 1, "data: b"), (0.6, 2, "data: c")))
    later.start()
    take()
    later.join()
    assert taken == [0, 1, 2] and list(lp.frames) == [(STREAM, 3)] and lp.done == 0
    rt.advance_to(2.0)
    take()
    assert taken == [0, 1, 2, 3] and not lp.frames and lp.done == 1
    assert not lp.errors and rt.handled[STREAM] == rt.released[STREAM]

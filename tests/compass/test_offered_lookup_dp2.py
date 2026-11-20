# SPDX-License-Identifier: MIT
"""#3767 (b) on a DP2 engine: an ADD that arrives inside the group's forward
has its tier lookup stamped at its arrival.

Two members of one engine LP ask the forward's round, ``TAR(1.0)``. The
frontend sends an ADD to rank 1 at 0.5 on its request channel, which a lookup
driver makes offering. The authority grants rank 1 that arrival first; rank 1's
input thread takes the ADD there and offers its lookup through ATOM's
`EngineCore._offer_tier_lookups`, the connector waking the shell's
`_OwnedDriver`, whose pass rank 1's clock owner makes at 0.5. Rank 1 asks the
round again; rank 0 never leaves it. Everything runs on CPU, in one process,
on the in-process carrier.
"""

import math
import pickle
import threading
import uuid
from types import SimpleNamespace

import zmq

from atom.compass import clock_transport
from atom.compass import run as compass_run
from atom.compass.clock import (
    ARRIVAL,
    NER,
    TAR,
    ClockAuthority,
    LpId,
    single_engine_table,
)
from atom.compass.clock.channels import add_lmcache
from atom.compass.kv.lmcache_mp import _OwnedDriver
from atom.model_engine.engine_core import EngineCore
from atom.utils.clock import LPRuntime, WrappedSocket

INF = math.inf
IPC = 2.0**-10
ENGINE, FRONTEND = LpId("engine"), LpId("frontend")
TRAFFIC, LMCACHE = LpId("traffic"), LpId("lmcache")
REQ1, LOOK1 = "frontend->engine:request#dp1", "engine->lmcache:lookup#dp1"
FORWARD_END, ADD_AT = 1.0, 0.5


class _Timeline:
    """The authority's replies: ``(lp, from, to, kind, released)``."""

    def __init__(self):
        self.rows = []

    def record(self, lp, time_from, time_to, kind, recovered, released, sent):
        self.rows.append((lp.name, time_from, time_to, kind, released))


class _LookupClient:
    """The lookup client `_OwnedDriver` drives: each pass sends every lookup
    offered since the last on rank 1's lookup channel."""

    def __init__(self):
        self.rt, self.offered, self.sent = None, [], []

    def _drive(self, window):
        while self.offered:
            self.offered.pop(0)
            self.sent.append(self.rt.stamp_send(LOOK1))
        return False


def _runtime(lp, endpoint, table, target, member=None):
    """`lp`'s runtime, or `member`'s, owned by a thread that runs `target(rt)`."""
    box = []
    thread = threading.Thread(target=lambda: target(box[0]), daemon=True)
    conn = clock_transport.connect(lp, endpoint, member)
    box.append(LPRuntime(lp, table, conn, owner=thread))
    return box[0], thread


def _idle(rt):
    while rt.next_event(INF) != INF:
        pass


def test_an_in_step_adds_lookup_is_stamped_at_its_arrival():
    table = single_engine_table(admission_s=IPC, ipc_s=IPC, stream_s=IPC, dp=2)
    add_lmcache(table, "engine", IPC, 2, driven=True)
    timeline, endpoint = _Timeline(), f"inproc:3767b-{uuid.uuid4().hex}"
    members = compass_run._members(table, 2)
    server = clock_transport.serve(
        ClockAuthority(table, timeline=timeline, members=members), endpoint
    )
    clock_transport.connect(TRAFFIC, endpoint).send((NER, INF, [], INF))
    ctx, address = zmq.Context(), f"inproc://dp2-add-{uuid.uuid4().hex}"
    push, pull = ctx.socket(zmq.PUSH), ctx.socket(zmq.PULL)
    push.bind(address)
    pull.connect(address)
    seen, after = [], {}

    def frontend(rt):
        rt.start_run()
        rt.advance_to(ADD_AT - IPC)
        WrappedSocket(rt, push, REQ1).send(pickle.dumps(("ADD", ["r0"])))
        _idle(rt)

    def member(rt):
        rt.start_run()
        rt.advance_group([FORWARD_END] * 2, FORWARD_END)  # the forward's step
        after[rt.member] = (rt.now, rt.calls)
        _idle(rt)

    def lmcache(rt):
        rt.start_run()
        _idle(rt)

    runtimes = {
        "frontend": _runtime(FRONTEND, endpoint, table, frontend),
        "lmcache": _runtime(LMCACHE, endpoint, table, lmcache),
        "dp0": _runtime(ENGINE, endpoint, table, member, "dp0"),
        "dp1": _runtime(ENGINE, endpoint, table, member, "dp1"),
    }
    dp1 = runtimes["dp1"][0]
    client = _LookupClient()
    client.rt = dp1
    driver = _OwnedDriver(dp1, client, 4)

    def offer_lookups(reqs):  # the lookup client's `offer`: queue, wake the driver
        client.offered += reqs
        driver.notify()

    connector = SimpleNamespace(offer_lookups=offer_lookups)
    engine = SimpleNamespace(
        label="dp1", scheduler=SimpleNamespace(kv_connector=connector)
    )

    def receive():  # rank 1's input thread
        sock = WrappedSocket(dp1, pull, REQ1)
        kind, reqs = pickle.loads(sock.recv())
        seen.append((kind, dp1.read_clock()))
        EngineCore._offer_tier_lookups(engine, reqs)
        dp1.back_at_wait_point()

    threads = [thread for _, thread in runtimes.values()]
    threads.append(threading.Thread(target=receive, daemon=True))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    server.close()
    ctx.destroy(linger=0)

    assert not any(t.is_alive() for t in threads)
    assert seen == [("ADD", ADD_AT)]  # the handler ran at the ADD's arrival
    assert client.sent == [(ADD_AT + IPC, 0)]  # its lookup, stamped there
    released = [m for r in timeline.rows if r[0] == "lmcache" for m in r[4]]
    assert (LOOK1, 0, ADD_AT + IPC) in released
    engine_rows = [r[1:4] for r in timeline.rows if r[0] == "engine"]
    assert engine_rows[:2] == [(0.0, ADD_AT, ARRIVAL), (ADD_AT, FORWARD_END, TAR)]
    # Both leave the forward at its end; rank 1 asked it once more, rank 0 not.
    assert after == {"dp0": (FORWARD_END, 1), "dp1": (FORWARD_END, 2)}
    assert not any(rt.refusals for rt, _ in runtimes.values())

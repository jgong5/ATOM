# SPDX-License-Identifier: MIT
"""#3767 (b) on a DP2 engine: an ADD that arrives inside the group's forward
has its tier lookup stamped at its arrival.

Two members of one engine LP ask the forward's round, ``TAR(1.0)``. The
frontend sends an ADD to rank 1 at 0.5 on its request channel, which is
offering: a lookup driver makes it so for lmcache (`add_lmcache`), the Store
always (`add_mooncake_store`). The authority grants rank 1 that arrival first;
rank 1's input thread takes the ADD there and offers its lookup through ATOM's
`EngineCore._offer_tier_lookups`, and rank 1's clock owner makes the send at
0.5: for lmcache the pass of the shell's `_OwnedDriver` the connector wakes,
for the Store the call `StoreShell._send_lookup` hands to `LPRuntime.offer`.
Rank 1 asks the round again; rank 0 never leaves it. Everything runs on CPU, in
one process, on the in-process carrier.
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
from atom.compass.clock.channels import (
    add_lmcache,
    add_mooncake_store,
    prefill_decode_table,
)
from atom.compass.kv import mooncake_store as store_shell
from atom.compass.kv.lmcache_mp import _OwnedDriver
from atom.model_engine.engine_core import EngineCore
from atom.utils.clock import LPRuntime, WrappedSocket

INF = math.inf
IPC = 2.0**-10
ENGINE, FRONTEND = LpId("engine"), LpId("frontend")
TRAFFIC, LMCACHE, STORE = LpId("traffic"), LpId("lmcache"), LpId("mooncake_store")
REQ1, LOOK1 = "frontend->engine:request#dp1", "engine->lmcache:lookup#dp1"
STORE_LOOK1 = "engine->mooncake_store:lookup#dp1"
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


def _add_inside_the_forward(table, server, offer_lookups_on):
    """Run the scenario: `server` the LP the lookups go to, idle throughout,
    and `offer_lookups_on(rt)` rank 1's connector's ``offer_lookups`` given
    rank 1's runtime. Returns the authority's rows, what rank 1's input
    thread saw, each member's clock and calls after the forward, and the
    runtimes."""
    timeline, endpoint = _Timeline(), f"inproc:3767b-{uuid.uuid4().hex}"
    members = compass_run._members(table, 2)
    server_sock = clock_transport.serve(
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

    def idle(rt):
        rt.start_run()
        _idle(rt)

    runtimes = {
        "frontend": _runtime(FRONTEND, endpoint, table, frontend),
        server.name: _runtime(server, endpoint, table, idle),
        "dp0": _runtime(ENGINE, endpoint, table, member, "dp0"),
        "dp1": _runtime(ENGINE, endpoint, table, member, "dp1"),
    }
    dp1 = runtimes["dp1"][0]
    connector = SimpleNamespace(offer_lookups=offer_lookups_on(dp1))
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
    server_sock.close()
    ctx.destroy(linger=0)
    assert not any(t.is_alive() for t in threads)
    return timeline.rows, seen, after, runtimes


def _granted_the_arrival_first(rows, seen, after, runtimes):
    """What both cases share: the handler ran at the ADD's arrival, rank 1
    was granted it first, and both members leave the forward at its end,
    rank 1 having asked it once more."""
    assert seen == [("ADD", ADD_AT)]
    engine_rows = [r[1:4] for r in rows if r[0] == "engine"]
    assert engine_rows[:2] == [(0.0, ADD_AT, ARRIVAL), (ADD_AT, FORWARD_END, TAR)]
    assert after == {"dp0": (FORWARD_END, 1), "dp1": (FORWARD_END, 2)}
    assert not any(rt.refusals for rt, _ in runtimes.values())


def test_an_in_step_adds_lookup_is_stamped_at_its_arrival():
    table = single_engine_table(admission_s=IPC, ipc_s=IPC, stream_s=IPC, dp=2)
    add_lmcache(table, "engine", IPC, 2, driven=True)
    client = _LookupClient()

    def offer_lookups_on(dp1):
        client.rt = dp1
        driver = _OwnedDriver(dp1, client, 4)

        def offer_lookups(reqs):  # the lookup client's `offer`: queue, wake
            client.offered.extend(reqs)
            driver.notify()

        return offer_lookups

    rows, seen, after, runtimes = _add_inside_the_forward(
        table, LMCACHE, offer_lookups_on
    )
    _granted_the_arrival_first(rows, seen, after, runtimes)
    assert client.sent == [(ADD_AT + IPC, 0)]  # its lookup, stamped there
    released = [m for r in rows if r[0] == "lmcache" for m in r[4]]
    assert (LOOK1, 0, ADD_AT + IPC) in released


def test_an_in_step_adds_store_lookup_is_stamped_at_its_arrival():
    """The Store path: no lookup driver, the shell's `_send_lookup` hands the
    input thread's query to `LPRuntime.offer`, and the request channel is
    offering because the Store serves the engine."""
    table = single_engine_table(admission_s=IPC, ipc_s=IPC, stream_s=IPC, dp=2)
    add_mooncake_store(table, ["engine"], IPC, 2)
    shell = store_shell.StoreShell.__new__(store_shell.StoreShell)
    sent = []

    def offer_lookups_on(dp1):
        # The shell's state its offer path reads; its send stamps and records
        # instead of writing to LP mooncake_store's socket.
        shell.__dict__.update(
            _rt=dp1,
            _offer=dp1.offer,
            _offered=[],
            _lock=threading.Lock(),
            counts=dict.fromkeys(store_shell.COUNTS, 0),
            offered_late_s=0.0,
            _look=STORE_LOOK1,
            _send=lambda ch, payload: sent.append((ch, dp1.stamp_send(ch), payload)),
        )

        def offer_lookups(reqs):  # ATOM's Store scheduler submits a query
            for req in reqs:
                shell._send_lookup(("query", req, [f"key-{req}"]))

        return offer_lookups

    rows, seen, after, runtimes = _add_inside_the_forward(
        table, STORE, offer_lookups_on
    )
    _granted_the_arrival_first(rows, seen, after, runtimes)
    assert sent == [(STORE_LOOK1, (ADD_AT + IPC, 0), ("query", "r0", ["key-r0"]))]
    assert shell.counts["offers"] == 1 and shell.counts["offers_late"] == 0
    assert not shell._offered  # made by the owner, not queued by the shell
    released = [m for r in rows if r[0] == "mooncake_store" for m in r[4]]
    assert (STORE_LOOK1, 0, ADD_AT + IPC) in released


def test_the_store_marks_each_served_engine_s_request_channels_offering():
    """Every DP rank's request channel into every engine the Store serves,
    and nothing else, whatever the instance count."""
    table = prefill_decode_table(
        admission_s=IPC,
        ipc_s=IPC,
        stream_s=IPC,
        router_s=2 * IPC,
        router_in_s=IPC,
        kv_write_req_s=IPC,
        dp=(2, 4),
        prefill=2,
    )
    add_mooncake_store(table, ["engine-P0", "engine-P1"], IPC, 2)
    marked = {
        c.name
        for lp in table.registry.ids()
        for c in table.channels_into(lp)
        if table.offering(c.name)
    }
    assert marked == {
        f"frontend-P{k}->engine-P{k}:request#dp{r}" for k in (0, 1) for r in (0, 1)
    }

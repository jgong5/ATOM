# SPDX-License-Identifier: MIT
"""A tier lookup the engine's input thread offers is sent by the clock owner.

Under `OFFLOAD_MP_LOOKUP_THREAD`, through ATOM's `EngineCore._offer_tier_lookups`,
`_MPLookupClient` and `_LookupDriver`, over a stand-in lmcache client stamping
each lookup as `clock.RelayQueue` does: the step loop sends it at the ADD's
arrival when it idles, at the forward's end when a forward's grant releases the
ADD, each plus the lookup channel's lookahead, and the authority takes both.
"""

import math
import threading
import uuid
from types import SimpleNamespace

import zmq

from atom.compass import clock_transport
from atom.compass.clock import NER, ClockAuthority, LpId, single_engine_table
from atom.compass.clock.channels import add_lmcache
from atom.kv_transfer.offload.mp import lookup as mp_lookup
from atom.model_engine.engine_core import EngineCore
from atom.utils import clock
from atom.utils.clock import LPRuntime, WrappedSocket
from tests.test_lmcache_mp_async_lookup import _Adapter, _Future

TABLE = single_engine_table(admission_s=0.001, ipc_s=0.001, stream_s=0.002)
add_lmcache(TABLE, "engine", 0.001)
ENGINE, FRONTEND, LMCACHE = LpId("engine"), LpId("frontend"), LpId("lmcache")
REQUEST, LOOKUP = "frontend->engine:request#dp0", "engine->lmcache:lookup#dp0"
ADDS = (0.5, 0.7)  # the first while the engine idles, the second in its forward


class _StampedClient:
    """lmcache's client: each lookup stamped now, never acked."""

    def __init__(self):
        self.sent = []

    def lookup(self, key, tp_size):
        self.sent.append(clock.installed().stamp_send(LOOKUP)[0])
        return _Future(None, done=False)


def _driven(fake):
    config = SimpleNamespace(
        kv_transfer_config={"kv_connector_extra_config": {}},
        parallel_config=SimpleNamespace(data_parallel_rank=0),
    )
    return mp_lookup._MPLookupClient(
        _Adapter(fake), config=config, timeout=10.0, poll_interval=0.01, driver_window=4
    )


def _thread(target):
    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t


def test_an_offered_lookup_is_sent_by_the_owner_at_its_grant_s_clock(monkeypatch):
    endpoint = f"inproc:test-offered-lookup-{uuid.uuid4().hex}"
    server = clock_transport.serve(ClockAuthority(TABLE), endpoint)
    clock_transport.connect(LpId("traffic"), endpoint).send(
        (NER, math.inf, [], math.inf)
    )
    address = f"inproc://offered-{uuid.uuid4().hex}"
    push, pull = (zmq.Context.instance().socket(k) for k in (zmq.PUSH, zmq.PULL))
    push.bind(address)
    pull.connect(address)
    released = []  # LP lmcache's grants and the lookups released by each

    def frontend():
        rt = LPRuntime(FRONTEND, TABLE, clock_transport.connect(FRONTEND, endpoint))
        rt.start_run()
        sock = WrappedSocket(rt, push, REQUEST)
        for a in ADDS:
            rt.advance_to(a - TABLE.lookahead(REQUEST))
            sock.send(b"ADD")
        rt.next_event(math.inf)

    def lmcache():
        rt = LPRuntime(LMCACHE, TABLE, clock_transport.connect(LMCACHE, endpoint))
        rt.start_run()
        while (G := rt.next_event(math.inf)) < math.inf:
            released.append((G, len(rt.released[LOOKUP])))

    rt = LPRuntime(ENGINE, TABLE, clock_transport.connect(ENGINE, endpoint))
    rt.start_run()
    monkeypatch.setattr(clock, "_installed", rt)
    fake = _StampedClient()
    client = _driven(fake)
    offer = SimpleNamespace(offer_lookups=lambda reqs: client.offer(*reqs))
    core = SimpleNamespace(scheduler=SimpleNamespace(kv_connector=offer), label="e")
    sock = WrappedSocket(rt, pull, REQUEST)

    def input_thread():  # ATOM's: takes each ADD, offers its lookup
        for k, _ in enumerate(ADDS):
            sock.recv()
            EngineCore._offer_tier_lookups(core, (f"r{k}", lambda: list(range(8))))
        rt.back_at_wait_point()

    threads = [_thread(input_thread), _thread(frontend), _thread(lmcache)]
    assert rt.next_event(math.inf) == ADDS[0]
    rt.advance_to(1.0)
    rt.next_event(math.inf)
    for t in threads:
        t.join(timeout=10)
    server.close()

    lookahead = TABLE.lookahead(LOOKUP)
    assert fake.sent == [ADDS[0] + lookahead, 1.0 + lookahead]
    assert released == [(fake.sent[0], 1), (fake.sent[1], 2)]
    assert rt.refusals == [] and not client._driver._thread.is_alive()


class _Conn:
    """A clock connection granting each request its own target."""

    def send(self, request):
        self.t = request[1]

    def recv(self):
        return self.t, {}


def test_a_pass_waits_while_the_owner_holds_the_client_s_lock(monkeypatch):
    rt = LPRuntime(ENGINE, TABLE, _Conn())
    rt.start_run()
    monkeypatch.setattr(clock, "_installed", rt)
    fake = _StampedClient()
    client = _driven(fake)
    client.offer("r", lambda: list(range(8)))
    with client._lock:
        rt.advance_to(0.25)
    assert fake.sent == []
    rt.advance_to(0.5)  # sent at the clock of the first call after the release
    assert fake.sent == [0.25 + TABLE.lookahead(LOOKUP)]

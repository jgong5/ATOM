# SPDX-License-Identifier: MIT
"""`mooncake_store` on a simulated run: ATOM's real Store scheduler half in
`StoreShell`, its calls to LP mooncake_store on the engine's LP clock, and each
rank's copies on its lanes through ATOM's own worker paths.

Most tests play LP mooncake_store and the clock authority by hand: a ROUTER at
the endpoint the shell reads, and a clock connection that grants each request
its target, or a release the test scripts. The DP test runs the real
authority, the real LP loop over a `Master` on a thread, and two members of
one engine LP, recording the design's stale-hit renewal. No driver, no
Mooncake and no AITER: the Store's namespace, which asks AITER, is fixed.
"""

import array
import copy
import json
import math
import pickle
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import zmq
from test_memory_readings import DOCUMENT

from atom.compass import clock_transport
from atom.compass import run as compass_run
from atom.compass.clock import NER, ClockAuthority, LpId, single_engine_table
from atom.compass.clock.channels import add_mooncake_store
from atom.compass.kv import mooncake_store as shell_mod
from atom.compass.kv import mooncake_store_server as server
from atom.compass.kv.lmcache_server import reply_channel
from atom.compass.runner.overrides import NonAllocatingRunner
from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.types import LoadOperationId, SaveOperationId
from atom.kv_transfer.offload.metadata import LMCacheReqMeta, LoadSpec, SaveSpec
from atom.kv_transfer.offload.mooncake_store import keys as store_keys
from atom.kv_transfer.offload.mooncake_store import scheduler as scheduler_mod
from atom.kv_transfer.offload.mooncake_store.worker import (
    MooncakeStoreOffloadConnector,
)
from atom.model_engine.engine_core import EngineCore
from atom.model_engine.sequence import SequenceStatus
from atom.utils import clock
from atom.utils.clock import LPRuntime
from tests.compass import sync_scan

BLOCK, CHUNK = 4, 8  # tokens per KV block, per Store chunk
HOP, PER_KEY, TTL = 2.0**-12, 2.0**-14, 10.0
NAMESPACE = "atomkv1-test"
ENTRY = {
    "lookup_s": [2 * HOP, PER_KEY],
    "store_s": [0.25, 0.125],
    "load_s": [0.125, 0.0625],
    "protocol": "tcp",
    "chunk_tokens": CHUNK,
    "chunk_bytes": 64,
    "world": 1,
    "save_slots": 64,
    "load_slots": 64,
    "save_workers": 1,
    "load_workers": 1,
    "master": ["--rpc_thread_num=4", "--default_kv_lease_ttl=10000"],
    "capacity_bytes": 1 << 20,
}
STORE = {
    "kv_connector": "mooncake_store",
    "kv_role": "offload",
    "kv_connector_extra_config": {
        "mooncake_store.master": "10.0.0.1:26051",
        "mooncake_store.metadata": "http://10.0.0.1:26080/metadata",
        "mooncake_store.local_hostname": "10.0.0.2",
        "mooncake_store.protocol": "tcp",
        "mooncake_store.chunk_tokens": CHUNK,
    },
}
ENGINE, STORE_LP = LpId("engine"), LpId("mooncake_store")
TABLE = single_engine_table(admission_s=0.001, ipc_s=0.001, stream_s=0.002)
add_mooncake_store(TABLE, ["engine"], HOP)
ASK, PUT = "engine->mooncake_store:lookup#dp0", "engine->mooncake_store:transfer#dp0"
BACK = reply_channel(ASK)


def _config(dp_rank=0):
    return SimpleNamespace(
        kv_transfer_config=copy.deepcopy(STORE),
        kv_cache_block_size=BLOCK,
        decode_context_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        hf_config=SimpleNamespace(num_hidden_layers=4, model_type="glm_moe_dsa"),
        model="test-model",
        kv_cache_dtype="fp8",
        speculative_config=None,
        offload_min_load_tokens=0,
        parallel_config=SimpleNamespace(data_parallel_rank=dp_rank),
    )


def _seq(req_id: int, num_prompt_tokens: int):
    return SimpleNamespace(
        id=req_id,
        token_ids=array.array("i", range(num_prompt_tokens)),
        num_prompt_tokens=num_prompt_tokens,
        num_cached_tokens=0,
        block_table=list(range(-(-num_prompt_tokens // BLOCK))),
        status=SequenceStatus.WAITING,
        offload_loaded=False,
        offload_load_failed=False,
    )


def _keys(seq, first=0, last=None):
    hashes = scheduler_mod.prompt_chunk_hashes(seq, CHUNK)
    last = len(hashes) // store_keys.DIGEST_BYTES if last is None else last
    return store_keys.chunk_keys(NAMESPACE, 0, 1, hashes, first, last)


@pytest.fixture
def run_file(tmp_path, monkeypatch):
    """The run file the shell reads, and the Store's namespace fixed."""
    monkeypatch.setattr(scheduler_mod, "store_namespace", lambda c, n: NAMESPACE)
    run = {
        "bound_s": 60.0,
        "clock_endpoint": "inproc:unused",
        "out_dir": str(tmp_path),
        "machine": copy.deepcopy(DOCUMENT),
        server.ENTRY: copy.deepcopy(ENTRY),
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    return run


def _bind_store(run: dict):
    """A ROUTER at the endpoint the shell reads, as LP mooncake_store's."""
    router = zmq.Context.instance().socket(zmq.ROUTER)
    address = f"inproc://store-{uuid.uuid4().hex}"
    router.bind(address)
    (Path(run["out_dir"]) / server.ENDPOINT_FILE).write_text(address)
    return router


class _Conn:
    """The authority, scripted: a request is granted its own target, or the
    grant the test set in `next`."""

    def __init__(self):
        self.sent, self.next = [], None

    def send(self, request):
        self.sent.append(request)

    def recv(self):
        if self.next is not None:
            grant, self.next = self.next, None
            return grant
        return self.sent[-1][1], {}


@pytest.fixture
def shell(run_file, monkeypatch):
    lp = _bind_store(run_file)
    conn = _Conn()
    rt = LPRuntime(ENGINE, TABLE, conn)
    rt.start_run()
    monkeypatch.setattr(clock, "_installed", rt)
    built = shell_mod.StoreShell(_config())
    yield SimpleNamespace(shell=built, rt=rt, conn=conn, lp=lp)
    built.close()
    lp.close(linger=0)


def _received(router):
    assert router.poll(5000), "LP mooncake_store received nothing"
    identity, header, payload = router.recv_multipart()
    return identity, pickle.loads(header), pickle.loads(payload)


def test_a_query_is_sent_at_submit_and_its_reply_taken_at_the_next_poll(shell):
    s, rt, lp = shell.shell, shell.rt, shell.lp
    seq, lookups = _seq(1, 41), shell.shell._lookups
    keys = _keys(seq)  # five chunks

    assert lookups.submit(seq, "1")
    identity, header, (kind, rid, asked) = _received(lp)
    assert header == (ASK, HOP, 0) and kind == "query"
    assert asked == [store_keys.STORE_TAINT_KEY, *keys]
    end = HOP + PER_KEY * len(asked)
    present = dict.fromkeys(keys[:2], "engine-P1")
    reply = (rid, HOP, end + TTL, present)
    lp.send_multipart(
        [identity, pickle.dumps((BACK, end + HOP, 0)), pickle.dumps(reply)]
    )
    assert lookups.is_pending("1")  # read, not yet released

    shell.conn.next = (end + HOP, {BACK: [(0, end + HOP)]})
    rt.next_event(math.inf)
    assert lookups.poll("1") and lookups.lookup(seq, "1") == 2 * CHUNK
    # When the real client sent it: the reply's start less a hop.
    assert lookups.query_started("1") == 0.0
    assert s._replies["1"] == (present, end + TTL)
    assert s.offload_statistics()["store"]["lookups"] == 1
    assert rt.refusals == []


def test_an_offer_from_the_input_thread_is_sent_by_the_clock_owner(shell):
    s, rt, lp = shell.shell, shell.rt, shell.lp
    seq = _seq(2, 41)
    core = SimpleNamespace(scheduler=SimpleNamespace(kv_connector=s), label="e")
    offer = threading.Thread(target=EngineCore._offer_tier_lookups, args=(core, [seq]))
    offer.start()
    offer.join()
    assert not lp.poll(100) and len(rt.queued) == 1

    rt.advance_to(0.5)  # the owner makes the offer before its next clock call
    _, header, (kind, _, asked) = _received(lp)
    assert header == (ASK, HOP, 0) and kind == "query" and len(asked) == 6
    assert "2" in s._lookups.pending and rt.refusals == []


def test_a_save_puts_as_it_is_queued_and_a_load_gets_all_or_nothing(shell):
    s, rt, lp = shell.shell, shell.rt, shell.lp
    hashes = store_keys.chunk_hash_chain(list(range(32)), CHUNK)  # four chunks
    save = LMCacheReqMeta(
        req_id=7,
        token_ids=[],
        block_ids=list(range(8)),
        save_spec=SaveSpec(skip_leading_tokens=CHUNK),
        save_operation=SaveOperationId(7, 0),
        chunk_hashes=hashes,
    )
    s._save(save)
    _, header, payload = _received(lp)
    store_s = 0.25 + 0.125 * 3
    assert header == (PUT, HOP, 0)
    assert payload == (
        "put",
        store_keys.chunk_keys(NAMESPACE, 0, 1, hashes, 1, 4),
        store_keys.chunk_group_ids(NAMESPACE, hashes, 1, 4),
        0.0,
        store_s,
    )
    rank = s._ranks[0]
    rt.advance_to(store_s - 1e-3)
    s._lanes.run_due()
    assert rank.get_finished().finished_saving == set()
    rt.advance_to(store_s)
    s._lanes.run_due()
    assert rank.get_finished().finished_saving == {SaveOperationId(7, 0)}

    every = store_keys.chunk_keys(NAMESPACE, 0, 1, hashes, 0, 4)
    loads = {}
    for rid, present in ((8, dict.fromkeys(every, "engine-P1")), (9, {})):
        s._replies[str(rid)] = (present | {k: "engine" for k in every[:3]}, TTL)
        loads[rid] = LMCacheReqMeta(
            req_id=rid,
            token_ids=[],
            block_ids=list(range(8)),
            load_spec=LoadSpec(
                hbm_cached_tokens=0, lmcache_cached_tokens=32, can_load=True
            ),
            load_operation=LoadOperationId(rid, 0),
            chunk_hashes=hashes,
        )
        s._load(loads[rid])
    rt.advance_to(2.0)
    s._lanes.run_due()
    done = rank.get_finished()
    # Request 9's last reply lacked chunk 3: its get fails whole.
    assert done.finished_loading == {LoadOperationId(8, 0)}
    assert done.failed_loading == {LoadOperationId(9, 0)}
    assert s.offload_statistics()["store"] == {
        **dict.fromkeys(shell_mod.COUNTS, 0),
        "saves": 1,
        "loads": 2,
        "foreign_loads": 1,
    }


@pytest.mark.parametrize("on_refusal", ["mark", "abort"])
def test_a_load_past_its_lease_is_refused_by_name_and_counted(shell, on_refusal):
    s, rt = shell.shell, shell.rt
    s._abort = on_refusal == "abort"
    hashes = store_keys.chunk_hash_chain(list(range(16)), CHUNK)
    load = LMCacheReqMeta(
        req_id=5,
        token_ids=[],
        block_ids=list(range(4)),
        load_spec=LoadSpec(
            hbm_cached_tokens=0, lmcache_cached_tokens=16, can_load=True
        ),
        load_operation=LoadOperationId(5, 0),
        chunk_hashes=hashes,
    )
    s._replies["5"] = (dict.fromkeys(_keys(_seq(5, 16)), "engine"), 0.5)
    rt.advance_to(1.0)
    why = (
        "store:a load of 5 starts at 1.0 s, past the lease its last Store reply "
        "gave it, to 0.5 s"
    )
    if on_refusal == "abort":
        with pytest.raises(shell_mod.StoreRefusal, match="past the lease"):
            s._load(load)
    else:
        s._load(load)
    assert rt.refusals == [why]
    assert s.offload_statistics()["store"]["past_lease_loads"] == 1


def test_a_stale_hit_renewed_in_a_dp_group(run_file, monkeypatch):
    """The design's stale-hit renewal: dp0 renews a hit, waiting at an idle
    point bounded by `RENEW_TIMEOUT_S` while dp1 parks before its forward;
    the reply releases dp0, which parks, and both leave at dp0's clock."""
    table = single_engine_table(admission_s=0.001, ipc_s=0.001, stream_s=0.002, dp=2)
    add_mooncake_store(table, ["engine"], HOP, dp=2)
    endpoint = f"inproc:test-store-dp-{uuid.uuid4().hex}"
    served = clock_transport.serve(
        ClockAuthority(table, members=compass_run._members(table, 2)), endpoint
    )
    for idle in ("traffic", "frontend"):
        clock_transport.connect(LpId(idle), endpoint).send(
            (NER, math.inf, [], math.inf)
        )
    seq = _seq(3, 41)
    renewed = _keys(seq, 1, 5)  # tokens [8, 40)
    master = server.Master(copy.deepcopy(ENTRY))
    groups = store_keys.chunk_group_ids(
        NAMESPACE, scheduler_mod.prompt_chunk_hashes(seq, CHUNK), 1, 5
    )
    master.publish(master.put(renewed, groups, "engine", 0.0), 0.0)
    router = _bind_store(run_file)
    lp = []

    def store():
        # In the serving process, whose connection a poller can wait on.
        here = clock_transport.connect_here(STORE_LP, endpoint)
        srt = LPRuntime(STORE_LP, table, here)
        lp.append(
            server.MooncakeStoreLP(
                srt, router, master, lookup_s=ENTRY["lookup_s"], check_plans=None
            )
        )
        srt.start_run()
        lp[0].serve()

    asked = {}

    def member(name):
        conn = clock_transport.connect(ENGINE, endpoint, name)
        sent, send = asked.setdefault(name, []), conn.send
        conn.send = lambda request: (sent.append(request[:4]), send(request))[1]
        rt = LPRuntime(ENGINE, table, conn)
        rt.start_run()
        return rt

    rt0, out = member("dp0"), {}

    def dp1():
        rt = member("dp1")
        rt.park()  # before its forward
        out["dp1"] = rt.now
        rt.next_event(math.inf)

    threads = [threading.Thread(target=f, daemon=True) for f in (store, dp1)]
    for t in threads:
        t.start()
    monkeypatch.setattr(clock, "_installed", rt0)
    s = shell_mod.StoreShell(_config(0))
    try:
        held = s._lookups.renew(seq, CHUNK, 5 * CHUNK)
        returned = rt0.now
        rt0.park()
        out["dp0"] = rt0.now
        rt0.next_event(math.inf)
        for t in threads:
            t.join(timeout=10)
    finally:
        s.close()
        router.close(linger=0)
        served.close()

    end = HOP + PER_KEY * (1 + len(renewed))  # the station's job: the taint key too
    print("dp0 asked:", asked["dp0"], "dp1 asked:", asked["dp1"])
    print("renewal returned", held, "at", returned, "both left at", out)
    assert held == len(renewed) and returned == end + HOP
    assert asked["dp0"][0] == ("NER", math.inf, [(ASK, 0, HOP)], 1.0)
    assert asked["dp1"][0][0] == "PARK" and asked["dp0"][1][0] == "PARK"
    assert out == {"dp0": end + HOP, "dp1": end + HOP}
    assert s._replies["3"] == (dict.fromkeys(renewed, "engine"), end + TTL)
    assert s.offload_statistics()["store"]["renewals"] == 1
    assert lp[0].counts["renewals"] == 1 and rt0.refusals == []


def test_the_runner_reserves_the_store_s_buffers_before_it_substitutes(
    monkeypatch,
):
    """`get_num_blocks` reserves the Store worker's pools and staging from the
    KV budget, read off the registry before `allocate_kv_cache` substitutes the
    null worker and the shell, which it then does."""
    monkeypatch.setattr(
        KVConnectorFactory, "_registry", dict(KVConnectorFactory._registry)
    )
    pd = {"kv_connector": "compass", "kv_role": "kv_producer"}
    config = SimpleNamespace(
        kv_transfer_config={"kv_connector": "multi", "connectors": [pd, STORE]}
    )
    store = SimpleNamespace(kv_transfer_config=STORE)
    reserve = MooncakeStoreOffloadConnector.kv_budget_reserve_bytes(store)
    assert reserve > 0
    assert KVConnectorFactory.kv_budget_reserve_bytes(config) == reserve
    set_cache = []
    monkeypatch.setattr(
        "atom.utils.forward_context.set_kv_cache_data",
        lambda *a, **k: set_cache.append(k),
    )
    runner = SimpleNamespace(config=config)
    assert NonAllocatingRunner.allocate_kv_cache(runner, 64)
    entry = KVConnectorFactory._registry["mooncake_store"]
    assert (entry["worker_class"], entry["scheduler_class"]) == (
        "SimulatedKVConnector",
        "StoreShell",
    )
    assert set_cache == [{"num_blocks": 64}]


#: Each Store site the shell takes over, by the design's interception table.
SHELL_SITES = {
    "AsyncStoreLookup.submit::self.executor.submit(self._query, seq, sid)": "K4",
    "AsyncStoreLookup.submit_if::self.submit(seq, sid)": "K4",
    "AsyncStoreLookup.lookup::self.submit(seq, sid)": "K4",
    "AsyncStoreLookup.pump::f.result()": "K5",
    "AsyncStoreLookup.lookup::f.result()": "K5",
    "AsyncStoreLookup.renew::self.renew_executor.submit(self._renew, seq, start, end)": "K4",
    "AsyncStoreLookup.renew::future.result(timeout=RENEW_TIMEOUT_S)": "K3",
}


def test_the_inventory_classifies_the_store_s_sites_as_the_shell_runs_them():
    rows = {
        row["id"].split("::", 1)[1].rpartition("::")[0]: row["mechanism"]
        for row in sync_scan.load_inventory()["sites"]
        if row["file"] == "atom/kv_transfer/offload/mooncake_store/async_lookup.py"
    }
    assert {site: rows[site] for site in SHELL_SITES} == SHELL_SITES
    others = [
        r for r in sync_scan.load_inventory()["sites"] if "mooncake_store/" in r["file"]
    ]
    assert all(
        "refuses kv_connector 'mooncake_store'" not in r["mechanism_why"]
        for r in others
    )

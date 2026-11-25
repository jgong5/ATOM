# SPDX-License-Identifier: MIT
"""ATOM's `mooncake_store` offload on a simulated run, against LP `mooncake_store`.

The engine tests build the connector's scheduler half the way an engine
process does, inside `compass_run.engine`, and drive ATOM's own
`MooncakeStoreOffloadScheduler` through it on the step loop, against an
authority served here and LP `mooncake_store` started as the first prefill
frontend starts it, in a process of its own: a save put on its rank's lane, a
query sent at its submit and taken at a poll, the load it arms, a stale hit's
renewal inside a DP group whose peer parks, a renewal the master answers past
its bound, a load past its lease, and an offer made off the step loop. Times
are derived by hand from the entry's laws. No driver, no Mooncake.
"""

from __future__ import annotations

import array
import copy
import json
import logging
import math
import signal
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import atom_config_double
from test_dp_member_step_end import collectives  # noqa: F401
from test_memory_readings import DOCUMENT
from transformers import PretrainedConfig

from atom.compass import run as compass_run
from atom.compass.clock import ClockAuthority, LpId
from atom.compass.clock_transport import connect, serve
from atom.compass.kv import lmcache_mp
from atom.compass.kv import mooncake_store as shell_mod
from atom.compass.kv import mooncake_store_server as store
from atom.compass.kv.mooncake_store import (
    PAST_LEASE,
    SimulatedStoreWorker,
    StandInClient,
    StoreShell,
)
from atom.compass.runner.overrides import NonAllocatingRunner, RunnerRefusal
from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.types import (
    KVConnectorOutput,
)
from atom.kv_transfer.offload.mooncake_store import async_lookup
from atom.kv_transfer.offload.mooncake_store.scheduler import (
    MooncakeStoreOffloadScheduler,
)
from atom.kv_transfer.offload.mooncake_store.worker import (
    MooncakeStoreOffloadConnector,
)
from atom.model_engine.async_proc import AsyncIOProcManager
from atom.model_engine.engine_core import DPEngineCoreProc
from atom.model_engine.sequence import SequenceStatus
from atom.utils import clock, forward_context
from atom.utils.clock import LPRuntime
from atom.utils.distributed import utils as dist_utils

BLOCK, CHUNK = 4, 8  # tokens per KV block, per Store chunk
CHUNKS = 4  # whole chunks of a prompt
TOKENS = CHUNKS * CHUNK + 1
BLOCK_BYTES = 2**19
CHUNK_BYTES = BLOCK_BYTES * CHUNK // BLOCK
HOP, PER_KEY = 2.0**-12, 2.0**-14  # store_rpc_s, the master's per-key seconds
STORE_S, LOAD_S = [2.0**-8, 2.0**-9], [2.0**-8, 2.0**-7]
STEP = 0.01  # the step loop's clock calls, one per step
TTL = 10.0
#: The recipe's master, as pd_server_atom.sh starts it.
MASTER = [
    "--rpc_port=26051",
    "--rpc_thread_num=32",
    "--default_kv_lease_ttl=10000",
    "--eviction_high_watermark_ratio=0.90",
    "--eviction_ratio=0.05",
]
#: The default 256 MiB save and 1 GiB load pools, in 1 MiB slots.
ENTRY = {
    "lookup_s": [2 * HOP, PER_KEY],
    "store_s": STORE_S,
    "load_s": LOAD_S,
    "protocol": "tcp",
    "chunk_tokens": CHUNK,
    "chunk_bytes": CHUNK_BYTES,
    "world": 1,
    "save_slots": 256,
    "load_slots": 1024,
    "save_workers": 1,
    "load_workers": 1,
    "master": MASTER,
    "capacity_bytes": 2**34,
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
#: MiniMax-M3, a layout the Store keeps (its complete PAGE regions).
M3 = {
    "architectures": ["MiniMaxM3SparseForCausalLM"],
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "hidden_size": 256,
    "head_dim": 32,
}


def _entry(**changed):
    return {**copy.deepcopy(ENTRY), **changed}


def _config(kv=STORE, dp=1, rank=0, **overrides):
    config = atom_config_double(
        kv_transfer_config=kv,
        hf_config=PretrainedConfig.from_dict(M3),
        tensor_parallel_size=1,
        kv_cache_block_size=BLOCK,
        kv_cache_dtype="fp8",
        model="minimax-m3",
        offload_min_load_tokens=CHUNK,
        **overrides,
    )
    config.parallel_config.data_parallel_size = dp
    config.parallel_config.data_parallel_rank = rank
    return config


def _seq(req_id: int, tokens: int = TOKENS):
    return SimpleNamespace(
        id=req_id,
        token_ids=array.array("i", range(tokens)),
        num_prompt_tokens=tokens,
        num_tokens=tokens,
        num_cached_tokens=0,
        block_table=list(range(-(-tokens // BLOCK))),
        offload_loaded=False,
        offload_load_failed=False,
        cache_seed=None,
    )


def _at_step(t: float) -> float:
    """The first step at or after `t`."""
    return math.ceil(t / STEP - 1e-9) * STEP


@pytest.fixture
def factory(monkeypatch):
    """The connector factory's entries and the engine's DP hooks, as they were,
    after the test."""
    registry = KVConnectorFactory._registry
    for name in ("mooncake_store", "multi"):
        monkeypatch.setitem(registry, name, registry[name])
    for cls, name in (
        (AsyncIOProcManager, "call_func"),
        (DPEngineCoreProc, "_sync_dp_state"),
    ):
        monkeypatch.setattr(cls, name, getattr(cls, name))
    monkeypatch.setattr(clock, "_installed", None)
    monkeypatch.setattr(compass_run, "_member", None)
    monkeypatch.setattr(dist_utils, "LP_OF_RANK", None, raising=False)


class Engine:
    """An engine process's Store scheduler half on the step loop, against LP
    mooncake_store, started as the first prefill frontend starts it, and an
    authority served here; every other LP idles. At `dp` 2 this is member
    dp0, and `peer` is dp1's clock, for a thread of the test to drive."""

    def __init__(self, tmp_path, monkeypatch, dp=1, **run) -> None:
        self.run = run = {
            "clock_endpoint": "tcp://127.0.0.1:0",
            "bound_s": 60.0,
            "ipc_s": 2.0**-10,
            "admission_s": 0.54e-3,
            "stream_s": 2.0**-12,
            "machine": copy.deepcopy(DOCUMENT),
            "out_dir": str(tmp_path),
            store.ENTRY: _entry(),
            **({"data_parallel_size": dp} if dp > 1 else {}),
            **run,
        }
        table = compass_run.channel_table(run)
        members = compass_run._members(table, dp) if dp > 1 else None
        self.served = serve(
            ClockAuthority(table, bound_s=run["bound_s"], members=members),
            run["clock_endpoint"],
        )
        run["clock_endpoint"] = self.served.endpoint
        (tmp_path / "run.json").write_text(json.dumps(run))
        monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
        # Each member's plan, as its runner publishes it.
        names = [f"engine.dp{r}" for r in range(dp)] if dp > 1 else ["engine"]
        for name in names:
            now = tmp_path / compass_run.NOW_FILE.format(name)
            compass_run.publish_transfer(compass_run._block_file(now), BLOCK_BYTES, 1)
        started, real = [], subprocess.Popen
        monkeypatch.setattr(
            compass_run.subprocess,
            "Popen",
            lambda *a, **k: started.append(real(*a, **k)) or started[-1],
        )
        self.config = _config(dp=dp)
        compass_run._launch_mooncake_store(self.config, run, None)
        [self.proc] = started
        self.idle = [
            connect(lp, self.served.endpoint)
            for lp in table.registry.ids()
            if lp.name not in ("engine", store.ENTRY)
        ]
        for conn in self.idle:
            conn.send(("NER", math.inf, [], math.inf))
        self.peer = None
        if dp > 1:
            self.peer_done = threading.Event()
            self.peer_thread = threading.Thread(target=self._peer, daemon=True)
            self.peer = LPRuntime(
                compass_run.ENGINE,
                table,
                connect(compass_run.ENGINE, self.served.endpoint, "dp1"),
                owner=self.peer_thread,
            )
        with compass_run.engine(self.config):
            self.shell = KVConnectorFactory.create_connector(self.config, "scheduler")
        self.rt = clock.installed()
        self.reported = []  # (clock, what the shell handed ATOM's scheduler half)
        # What the step loop sent to LP mooncake_store, with its stamps.
        self.sent = []
        send = self.shell._send

        def recording(ch, payload):
            send(ch, payload)
            self.sent.append((self.rt.now, ch, payload))

        self.shell._send = recording
        # Each idle point the step loop asked for: (clock, t, daemon deadline).
        self.asked = []
        ask = self.rt.next_event

        def asking(t, t_daemon=math.inf):
            self.asked.append((self.rt.now, t, t_daemon))
            return ask(t, t_daemon)

        self.rt.next_event = asking
        if dp > 1:
            self.peer_thread.start()

    def _peer(self) -> None:
        """DP rank dp1: its forward dispatched at once, it parks, then follows
        the group's rounds to the finish."""
        peer = self.peer
        peer.start_run()
        self.peer_parked = peer.now
        peer.park()
        self.peer_left = peer.now
        while peer.next_event(math.inf) != math.inf:
            pass
        self.peer_done.set()

    def step(self) -> KVConnectorOutput:
        """One step: the clock to the next, then the step's completions."""
        self.rt.next_event(self.rt.now + STEP)
        out = self.shell.process_completions(KVConnectorOutput())
        done = [*out.finished_saving, *out.finished_loading, *out.failed_loading]
        if done:
            self.reported.append((self.rt.now, sorted(map(str, done))))
        return out

    def finish(self) -> dict:
        """Idle to the run's end; LP mooncake_store's hand-in."""
        try:
            while self.rt.next_event(math.inf) != math.inf:
                pass
            if self.peer is not None:
                assert self.peer_done.wait(60)
            assert self.proc.wait(120) == 0
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
            self.served.close()
        handed = Path(self.run["out_dir"]) / "lp-mooncake_store.json"
        return json.loads(handed.read_text())


def _save(shell, seq):
    """`seq` prefilled whole, its save built now."""
    shell.update_state_after_alloc(seq)
    seq.num_cached_tokens = seq.num_prompt_tokens
    [save] = shell.build_connector_meta().requests
    return save


def _answered(engine, client, sid: str) -> None:
    """Idle until `sid`'s query is answered, as the step loop polls it."""
    while not client.answered(sid):
        engine.rt.next_event(math.inf)


def test_the_factory_builds_atom_s_store_scheduler_over_stand_ins(
    tmp_path, monkeypatch, factory
):
    engine = Engine(tmp_path, monkeypatch)
    shell = engine.shell
    assert isinstance(shell, StoreShell)
    assert type(shell._inner) is MooncakeStoreOffloadScheduler
    client = shell._inner._lookup_client
    assert isinstance(client, async_lookup.AsyncStoreLookup)
    # Its workers are the shell's sends: no thread was started for them.
    assert type(client.executor).__name__ == type(client.renew_executor).__name__
    assert type(client.executor).__name__ == "_Executor"
    [pool] = client.backend._pools
    assert isinstance(pool.client, StandInClient)
    assert [type(r) for r in shell._ranks] == [MooncakeStoreOffloadConnector]
    assert shell._ranks[0]._namespace == shell._inner._namespace
    # The namespace names the run's device, as `simulated_arch` answers it.
    assert shell._look == "engine->mooncake_store:lookup#dp0"
    assert engine.finish()["refusals"] == []


def test_a_saved_prefix_is_put_on_its_lane_and_a_later_query_loads_it(
    tmp_path, monkeypatch, factory
):
    engine = Engine(tmp_path, monkeypatch)
    shell, rt = engine.shell, engine.rt
    client = shell._inner._lookup_client

    save = _save(shell, _seq(1))  # at 0
    stored = STORE_S[0] + CHUNKS * STORE_S[1]
    [(at, ch, put)] = engine.sent
    assert (at, ch) == (0.0, "engine->mooncake_store:transfer#dp0")
    kind, keys, groups, start, end = put
    assert (kind, start, end) == ("put", 0.0, stored)
    assert len(keys) == len(set(groups)) == CHUNKS
    assert all(k.startswith(f"{shell._namespace}/w0of1/") for k in keys)
    while rt.now < stored + 3 * STEP:
        engine.step()
    assert engine.reported == [
        (pytest.approx(_at_step(stored) + STEP), [str(save.req_id)])
    ]

    # A second request on the same prompt: its query is sent at its submit
    # and taken at the poll after its reply's arrival.
    b = _seq(2)
    asked = rt.now
    assert client.submit(b, "2")
    assert engine.sent[-1][1:] == (
        "engine->mooncake_store:lookup#dp0",
        ("query", 0, keys),
    )
    assert client.is_pending("2")
    _answered(engine, client, "2")
    replied = asked + HOP + CHUNKS * PER_KEY + HOP
    assert rt.now == pytest.approx(replied)
    # asked_at: the reply's start at the master less a hop.
    assert client.query_started("2") == pytest.approx(asked)
    assert shell.get_num_new_matched_tokens(b) == (CHUNKS * CHUNK, True)
    shell.update_state_after_alloc(b)
    sent = rt.now
    [load] = shell.build_connector_meta().requests
    assert (load.load_spec.hbm_cached_tokens, load.load_spec.lmcache_cached_tokens) == (
        0,
        CHUNKS * CHUNK,
    )
    loaded = sent + LOAD_S[0] + CHUNKS * LOAD_S[1]
    while rt.now < loaded + 3 * STEP:
        engine.step()
    print("\nreported", engine.reported)
    assert engine.reported[-1] == (
        pytest.approx(sent + _at_step(loaded - sent) + STEP),
        ["2"],
    )
    said = shell.offload_statistics()
    assert said["store"] | {"offers_late_max_s": 0} == {
        "lookups": 1,
        "renewals": 0,
        "loads": 1,
        "foreign_loads": 0,
        "saves": 1,
        "past_lease_loads": 0,
        "offers": 0,
        "offers_late": 0,
        "offers_late_max_s": 0,
    }
    stats = said["connector"]
    assert stats["saved_tokens"] == stats["loaded_tokens"] == CHUNKS * CHUNK
    handed = engine.finish()
    assert handed == {
        "refusals": [],
        "store": {
            "lookups": 1,
            "hit_keys": CHUNKS,
            "foreign_hit_keys": 0,
            "renewals": 0,
            "evicted": 0,
        },
    }


def test_the_lp_outlives_its_server_group_s_sigterm_and_hands_in_at_the_finish(
    tmp_path, monkeypatch, factory
):
    """`compass run` SIGTERMs its server's process group, LP mooncake_store
    among it, as the frontend's stop finishes the run: the LP keeps serving
    to its +inf grant and hands in."""
    engine = Engine(tmp_path, monkeypatch)
    client = engine.shell._inner._lookup_client
    assert client.submit(_seq(1), "1")
    engine.proc.send_signal(signal.SIGTERM)
    _answered(engine, client, "1")
    assert engine.finish()["store"]["lookups"] == 1


def test_the_query_asks_the_keys_atom_s_store_lookup_asks(
    tmp_path, monkeypatch, factory
):
    engine = Engine(tmp_path, monkeypatch)
    shell = engine.shell
    asked = []
    answer = StandInClient("m", CHUNK_BYTES)
    answer.exists = lambda keys: asked.append(keys) or [0] * len(keys)
    [pool] = shell._lookup.backend._pools
    pool.client = answer
    seq = _seq(3)
    assert shell._lookup.backend.lookup(seq, lookup_id="3") == 0
    [(taint, *keys)] = asked
    assert taint == "atomkv/store-tainted"
    assert keys == shell._keys(seq, 0, None)
    pool.client = shell._client
    engine.finish()


def test_a_stale_hit_renewed_in_a_dp_group_waits_alone_while_its_peer_parks(
    tmp_path, monkeypatch, factory, collectives  # noqa: F811
):
    """The design's stale-hit renewal: dp0 renews a hit 1.5 s old as it
    dispatches the load, a send on its lookup channel, then idles alone,
    bounded by RENEW_TIMEOUT_S, while dp1, its forward dispatched, is parked;
    the reply is released at its arrival, the renewal returns, dp0 parks, and
    both leave at dp0's clock."""
    engine = Engine(tmp_path, monkeypatch, dp=2)
    shell, rt = engine.shell, engine.rt
    client = shell._inner._lookup_client
    # dp1 parks first and stands in every round from here on.
    _save(shell, _seq(1))
    stored = STORE_S[0] + CHUNKS * STORE_S[1]
    while rt.now < stored + 3 * STEP:
        engine.step()
    b = _seq(2)
    assert client.submit(b, "2")
    _answered(engine, client, "2")
    assert shell.get_num_new_matched_tokens(b) == (CHUNKS * CHUNK, True)
    shell.update_state_after_alloc(b)
    # The hit grows stale: 1.5 s on dp0's clock alone.
    rt.next_event(rt.now + 1.5)

    renewed = rt.now
    asked = len(engine.asked)
    [load] = shell.build_connector_meta().requests
    keys = shell._keys(b, 0, CHUNKS)
    send = (renewed, "engine->mooncake_store:lookup#dp0", ("renew", 1, keys))
    assert engine.sent[-1] == send
    # The wait: an idle point, its bound a daemon deadline 1 s on.
    assert engine.asked[asked] == (renewed, math.inf, renewed + 1.0)
    reply = renewed + HOP + CHUNKS * PER_KEY + HOP
    assert rt.now == pytest.approx(reply)
    assert shell.counts["renewals"] == 1 and not engine.peer_done.is_set()
    # The renewal's lease is the one the load reads.
    assert shell._leases["2"][0] == pytest.approx(reply - HOP + TTL)
    rt.park()
    released = rt.now
    while rt.now < released + LOAD_S[0] + CHUNKS * LOAD_S[1] + 3 * STEP:
        engine.step()
    handed = engine.finish()

    timeline = [
        ("send", renewed),
        ("idle bound", renewed + 1.0),
        ("reply", reply),
        ("park dp1", engine.peer_parked),
        ("release", released),
        ("dp1 left", engine.peer_left),
    ]
    print("\n" + "\n".join(f"  {what:<10} {t!r}" for what, t in timeline))
    assert released == pytest.approx(reply) and engine.peer_left == released
    assert engine.reported[-1][1] == [str(load.req_id)]
    assert handed["store"]["renewals"] == 1 and handed["refusals"] == []


def test_a_renewal_the_master_answers_late_returns_at_its_bound(
    tmp_path, monkeypatch, factory, caplog
):
    """The master's station takes 0.3 s a key, so a query of four keys answers
    1.2 s on: the hit is stale at once, and its renewal's reply comes after
    RENEW_TIMEOUT_S. The wait returns at its bound, ATOM dispatches the load
    without it, and the late reply is taken at a later poll and dropped."""
    caplog.set_level(logging.WARNING, logger="atom")
    entry = _entry(lookup_s=[2 * HOP, 0.3])
    engine = Engine(tmp_path, monkeypatch, **{store.ENTRY: entry})
    shell, rt = engine.shell, engine.rt
    client = shell._inner._lookup_client
    _save(shell, _seq(1))
    while rt.now < 0.1:
        engine.step()
    b = _seq(2)
    assert client.submit(b, "2")
    _answered(engine, client, "2")
    assert shell.get_num_new_matched_tokens(b) == (CHUNKS * CHUNK, True)
    shell.update_state_after_alloc(b)
    renewed = rt.now
    [load] = shell.build_connector_meta().requests
    assert engine.sent[-1][2][0] == "renew"
    assert rt.now == pytest.approx(renewed + 1.0)
    assert "did not finish in 1.0s" in caplog.text
    assert load.load_spec.lmcache_cached_tokens == CHUNKS * CHUNK
    late = renewed + 2 * HOP + CHUNKS * 0.3
    while rt.now < late + STEP:
        engine.step()
        client.pump()
    assert shell._calls == {}  # taken, its future cancelled by ATOM
    assert shell.counts["past_lease_loads"] == 0
    engine.finish()


def test_a_load_whose_lane_starts_past_its_lease_is_refused_by_name(
    tmp_path, monkeypatch, factory
):
    """A lease of 1 ms, shorter than a hop: the reply's lease has run out
    before the step loop takes it, so the load it arms starts past it."""
    hop = 2.0**-9
    entry = _entry(
        lookup_s=[2 * hop, PER_KEY], master=[*MASTER, "--default_kv_lease_ttl=1"]
    )
    engine = Engine(tmp_path, monkeypatch, **{store.ENTRY: entry})
    shell, rt = engine.shell, engine.rt
    client = shell._inner._lookup_client
    _save(shell, _seq(1))
    while rt.now < 0.1:
        engine.step()
    b = _seq(2)
    assert client.submit(b, "2")
    _answered(engine, client, "2")
    assert shell.get_num_new_matched_tokens(b) == (CHUNKS * CHUNK, True)
    shell.update_state_after_alloc(b)
    [load] = shell.build_connector_meta().requests
    while not engine.reported or engine.reported[-1][1] != [str(load.req_id)]:
        engine.step()
    assert shell.counts["past_lease_loads"] == 1
    assert rt.refusals == [PAST_LEASE]
    engine.finish()


def test_an_abort_run_stops_at_a_load_past_its_lease(tmp_path, monkeypatch, factory):
    hop = 2.0**-9
    entry = _entry(
        lookup_s=[2 * hop, PER_KEY], master=[*MASTER, "--default_kv_lease_ttl=1"]
    )
    engine = Engine(tmp_path, monkeypatch, on_refusal="abort", **{store.ENTRY: entry})
    shell, rt = engine.shell, engine.rt
    client = shell._inner._lookup_client
    _save(shell, _seq(1))
    while rt.now < 0.1:
        engine.step()
    b = _seq(2)
    assert client.submit(b, "2")
    _answered(engine, client, "2")
    shell.get_num_new_matched_tokens(b)
    shell.update_state_after_alloc(b)
    shell.build_connector_meta()
    with pytest.raises(store.MooncakeStoreRefusal, match=PAST_LEASE):
        for _ in range(100):
            engine.step()
    engine.finish()


def test_an_offer_off_the_step_loop_is_sent_by_the_owner_at_its_next_call(
    tmp_path, monkeypatch, factory
):
    """The engine's input thread offers a lookup at an arrival: nothing is
    sent from it, and nothing refused; the step loop sends it at its next
    clock call, stamped at its clock then."""
    engine = Engine(tmp_path, monkeypatch)
    shell, rt = engine.shell, engine.rt
    client = shell._inner._lookup_client
    engine.step()
    b = _seq(2)
    b.status = SequenceStatus.WAITING
    offered = threading.Thread(target=shell._inner.offer_lookups, args=([b],))
    offered.start()
    offered.join()
    assert not engine.sent and shell.counts["offers"] == 1 and not rt.refusals
    at = rt.now
    engine.step()  # its clock call stamps the offer first
    [(when, ch, (kind, _, keys))] = engine.sent
    assert (when, ch, kind) == (at, "engine->mooncake_store:lookup#dp0", "query")
    assert keys == shell._keys(b, 0, None)
    _answered(engine, client, str(b.id))
    assert client.query_started(str(b.id)) == pytest.approx(at)
    assert shell.counts["offers_late"] == 0
    engine.finish()


def test_an_offer_goes_to_the_runtime_s_owner_stamped_send_where_it_has_one(
    tmp_path, monkeypatch, factory
):
    """#3767's `LPRuntime.offer`: the input thread's offer is handed to it as a
    callable, which the owner makes at the arrival it was offered at."""
    engine = Engine(tmp_path, monkeypatch)
    shell, rt = engine.shell, engine.rt
    offered = []
    shell._offer = offered.append  # as `LPRuntime.offer` queues it
    engine.step()
    b = _seq(2)
    b.status = SequenceStatus.WAITING
    thread = threading.Thread(target=shell._inner.offer_lookups, args=([b],))
    thread.start()
    thread.join()
    assert len(offered) == 1 and not engine.sent and not shell._offered
    at = rt.now
    offered.pop()()  # the owner makes it, its clock still at the arrival
    [(when, ch, (kind, _, _))] = engine.sent
    assert (when, ch, kind) == (at, "engine->mooncake_store:lookup#dp0", "query")
    assert shell.counts["offers"] == 1 and shell.counts["offers_late"] == 0
    _answered(engine, shell._inner._lookup_client, str(b.id))
    engine.finish()


def test_the_store_s_hbm_reserve_is_the_real_worker_s_before_and_after_it_substitutes(
    monkeypatch,
):
    registry = KVConnectorFactory._registry
    for name in ("mooncake_store", "multi"):
        monkeypatch.setitem(registry, name, registry[name])
    config = _config()
    real = MooncakeStoreOffloadConnector.kv_budget_reserve_bytes(config)
    assert real > 256 << 20  # the save and load pools and the staging bounds
    assert KVConnectorFactory.kv_budget_reserve_bytes(config) == real
    built = []
    monkeypatch.setattr(
        forward_context,
        "set_kv_cache_data",
        lambda _, config, **k: built.append(
            KVConnectorFactory.create_connector(config, "worker")
        ),
    )
    runner = object.__new__(NonAllocatingRunner)
    runner.config = config
    assert runner.allocate_kv_cache(100) is True
    assert [type(w) for w in built] == [SimulatedStoreWorker]
    assert KVConnectorFactory.kv_budget_reserve_bytes(config) == real
    runner.config = _config({"kv_connector": "mooncake"})
    with pytest.raises(RunnerRefusal, match="'lmcache_offload' and 'mooncake_store'"):
        runner.allocate_kv_cache(100)


def test_a_multi_beside_the_prefill_connector_builds_multishell(monkeypatch):
    registry = KVConnectorFactory._registry
    for name in ("mooncake_store", "multi"):
        monkeypatch.setitem(registry, name, registry[name])
    shell_mod.substitute()
    assert registry["multi"]["scheduler_class"] == lmcache_mp.MultiShell.__name__
    assert registry["mooncake_store"]["scheduler_class"] == StoreShell.__name__


@pytest.mark.parametrize("kv, parks", [({}, False), (STORE, True)])
def test_a_dp_member_that_offloads_to_the_store_parks(
    tmp_path, monkeypatch, factory, collectives, kv, parks  # noqa: F811
):
    run = {
        "clock_endpoint": "inproc:unused",
        "bound_s": 60.0,
        "ipc_s": 2.0**-10,
        "admission_s": 0.54e-3,
        "stream_s": 2.0**-12,
        "machine": DOCUMENT,
        "out_dir": str(tmp_path),
        "data_parallel_size": 2,
        store.ENTRY: _entry(),
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    monkeypatch.setattr(compass_run, "connect", lambda *_: None)
    call = AsyncIOProcManager.call_func
    with compass_run.engine(_config(kv, dp=2, rank=1)):
        pass
    assert (AsyncIOProcManager.call_func is not call) == parks


def test_an_engine_that_offloads_to_the_store_needs_the_entry(tmp_path):
    with pytest.raises(ValueError, match="has no mooncake_store entry"):
        compass_run._store_shell(_config(), {})
    compass_run._store_shell(_config({}), {})  # no Store, nothing to build


def test_an_engine_that_renews_store_leases_asynchronously_is_refused(monkeypatch):
    monkeypatch.setenv("ATOM_STORE_ASYNC_RENEW", "1")
    with pytest.raises(ValueError, match="ATOM_STORE_ASYNC_RENEW"):
        compass_run._store_shell(_config(), {store.ENTRY: _entry()})


def test_the_channel_table_names_the_shell_s_channels(tmp_path):
    run = {
        "bound_s": 60.0,
        "ipc_s": 2.0**-10,
        "admission_s": 0.54e-3,
        "stream_s": 2.0**-12,
        "machine": DOCUMENT,
        "out_dir": str(tmp_path),
        store.ENTRY: _entry(),
    }
    table = compass_run.channel_table(run)
    ms = LpId(store.ENTRY)
    assert {c.name for c in table.channels_into(ms)} == {
        "engine->mooncake_store:lookup#dp0",
        "engine->mooncake_store:transfer#dp0",
    }

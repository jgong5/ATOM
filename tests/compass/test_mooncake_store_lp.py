# SPDX-License-Identifier: MIT
"""LP `mooncake_store`: a shared Mooncake Store's master, its channels, its
launch, and its loop on the clock.

The master's index is driven directly. Its loop is driven through
`StationLP._at`, the test releasing each frame at its arrival and moving the
clock to the loop's next deadline, as the authority would. The end-to-end test
starts the LP the way the first prefill frontend does, in a process of its
own, against an authority served here over TCP, and plays its engine on real
ZMQ sockets. The run file `deploy.sh` builds for the recipe's 3P4D names its
channels.
"""

import copy
import json
import math
import pickle
import signal
import subprocess
import time
from collections import Counter, defaultdict
from types import SimpleNamespace

import pytest
import zmq
from conftest import atom_config_double
from test_deploy_sh import (  # noqa: F401  (planted is a fixture)
    PREFILL,
    ROUTER_LP_ARGS,
    SCRIPT,
    _tree_commit,
    _until,
    planted,
    router_hop,
)
from test_memory_readings import DOCUMENT

from atom.compass import run as compass_run
from atom.compass.clock import ClockAuthority, LpId
from atom.compass.clock_transport import connect, serve
from atom.compass.kv import mooncake_store_server as store
from atom.compass.kv.lmcache_server import reply_channel
from atom.compass.kv.mooncake_store_server import (
    Master,
    MooncakeStoreLP,
    MooncakeStoreRefusal,
)
from atom.utils import clock
from atom.utils.clock import LPRuntime

HOP = 2.0**-12  # store_rpc_s
PER_KEY = 2.0**-14
TTL = 10.0
#: The recipe's master, as pd_server_atom.sh starts it.
MASTER = [
    "--rpc_port=26051",
    "--rpc_thread_num=32",
    "--enable_http_metadata_server=true",
    "--http_metadata_server_host=10.0.0.1",
    "--http_metadata_server_port=26080",
    "--metrics_port=26090",
    "--default_kv_lease_ttl=10000",
    "--eviction_high_watermark_ratio=0.90",
    "--eviction_ratio=0.05",
    "--allocation_strategy=random",
    "--memory_allocator=offset",
    "--client_ttl=10",
]
#: One-byte objects in 4 KiB slots: the default 256 MiB save and 1 GiB load
#: pools hold this many.
SAVE_SLOTS, LOAD_SLOTS = (256 << 20) // 4096, (1 << 30) // 4096
ENTRY = {
    "lookup_s": [2 * HOP, PER_KEY],
    "store_s": [1e-3, 1e-4],
    "load_s": [1e-3, 1e-4],
    "protocol": "tcp",
    "chunk_tokens": 256,
    "chunk_bytes": 1,
    "world": 1,
    "save_slots": SAVE_SLOTS,
    "load_slots": LOAD_SLOTS,
    "save_workers": 1,
    "load_workers": 1,
    "master": MASTER,
    "capacity_bytes": 200,
}
LOOK = "engine-P0->mooncake_store:lookup#dp0"
OTHER = "engine-P1->mooncake_store:lookup#dp0"
XFER = "engine-P1->mooncake_store:transfer#dp0"


def _entry(**changed):
    return {**copy.deepcopy(ENTRY), **changed}


def _keys(chunk, world=2):
    """Each rank's key of chunk `chunk`, and its group, as ATOM keys them."""
    keys = [f"ns/w{r}of{world}/{chunk:032x}" for r in range(world)]
    return keys, [f"ns/group/{chunk:032x}"] * world


def _publish(master, chunk, at, world=2, engine="engine-P0"):
    keys, groups = _keys(chunk, world)
    master.publish(master.put(keys, groups, engine, at), at)
    return keys


def _group(chunk, world=2):
    return _keys(chunk, world)[1][0]


# --- the master's index ---------------------------------------------------------


def test_the_design_s_eviction_round_evicts_whole_groups_past_their_lease():
    """Mark 0.90, eviction ratio 0.05, a 10 s lease: a publish takes the Store
    to 0.91, and the round at the next 10 ms boundary evicts its target's
    objects in whole groups, oldest first, passing one a lookup leased."""
    m = Master(_entry())
    for chunk in range(90):  # chunk 0 is group A, chunk 1 group B
        _publish(m, chunk, 8.0 + chunk / 10)
    assert sorted(m.exists(_keys(1)[0], 32.0)) == _keys(1)[0]  # B to t1 + 4 s
    assert m.next_tick(37.0) == math.inf  # 180 of 200 bytes: at the mark
    _publish(m, 90, 37.995)
    t1 = m.next_tick(37.995)
    assert t1 == 38.0  # A's lease ran out 30 s before
    r = m.evict(t1)
    print(f"round at {t1} s: {r}")
    assert r.target == max(0.05, 182 / 200 - 0.90 + 0.05)
    assert r.goal == math.ceil(182 * r.target) == 11
    assert r.evicted == [_group(0), *map(_group, range(2, 7))] and r.objects == 12
    assert r.skipped == [(_group(1), "moved")]
    assert m.used == 170 and m.next_tick(t1) == math.inf  # 0.85
    assert m.exists(_keys(0)[0], 40.0) == {}
    both = _keys(7)[0] + _keys(1)[0]
    assert m.exists(both, 40.0).keys() == set(both)


def test_a_lease_keeps_a_group_through_a_round():
    """A put of a key the Store holds leases its group as a lookup does and
    takes no bytes; the group's later member files it at that deadline, and a
    round in that bucket passes it while the lease runs."""
    flags = ["--eviction_high_watermark_ratio=0.5", "--eviction_ratio=1"]
    m = Master(_entry(capacity_bytes=10, master=MASTER + flags))
    (k0, k1), groups = _keys(1)
    m.publish(m.put([k0], groups[:1], "engine-P0", 0.5), 0.5)
    assert m.put([k0], groups[:1], "engine-P1", 0.55) == [] and m.used == 1
    m.publish(m.put([k1], groups[1:], "engine-P1", 0.6), 0.6)
    _publish(m, 0, 1.0)
    _publish(m, 2, 2.0)
    r = m.evict(10.52)  # B leased to 10.55, filed in the bucket of 10.5
    assert r.evicted == [_group(0), _group(2)] and r.objects == 4
    assert r.skipped == [(_group(1), "lease")]
    assert m.exists([k0, k1], 10.52) == {k0: "engine-P0", k1: "engine-P1"}


def test_a_group_leaves_whole_however_far_it_overshoots_the_goal():
    m = Master(_entry(capacity_bytes=40))
    for chunk in range(5):
        _publish(m, chunk, 1.0 + chunk, world=8)
    r = m.evict(m.next_tick(5.0))  # 40 of 40
    assert r.goal == math.ceil(40 * r.target) < 8
    assert r.objects == 8 and r.evicted == [_group(0, 8)]
    assert m.used == m.filed == 32


def test_no_round_below_the_mark_nor_one_that_would_evict_nothing():
    m = Master(_entry(capacity_bytes=6))
    _publish(m, 0, 1.0)
    _publish(m, 1, 1.0)
    assert m.next_tick(1.0) == math.inf  # 4 of 6
    _publish(m, 2, 1.0)
    assert m.next_tick(1.0) == 1.0
    m.exists([k for c in range(3) for k in _keys(c)[0]], 1.003)
    # Every lease runs: the next round is at the first boundary past them all.
    assert m.next_tick(1.003) == m.next_tick(11.0) == 11.01
    assert m.evict(11.01).objects == 2
    assert m.next_tick(11.01) == math.inf and m.used == 4


def test_a_put_the_capacity_left_cannot_hold_is_refused():
    m = Master(_entry(capacity_bytes=3))
    _publish(m, 0, 1.0)
    with pytest.raises(MooncakeStoreRefusal, match="put of 2 objects.*fails it at"):
        m.put(*_keys(1), "engine-P0", 2.0)


# --- the loop -------------------------------------------------------------------


class _Clock:
    """LP mooncake_store's runtime as its loop reads it, the test its authority."""

    def __init__(self):
        self.now = 0.0
        self.released = self.arrived = defaultdict(set)
        self.seqs = Counter()

    def stamp_send(self, ch):
        self.seqs[ch] += 1
        return self.now + HOP, self.seqs[ch] - 1


class _Sock:
    def __init__(self):
        self.sent = []

    def send_multipart(self, frames):
        self.sent.append(frames)


def _loop(entry=None):
    rt, sock = _Clock(), _Sock()
    entry = entry or _entry()
    lp = MooncakeStoreLP(
        rt, sock, Master(entry), lookup_s=entry["lookup_s"], check_plans=lambda: None
    )
    return lp, rt


def _serve(lp, rt, frames, until=math.inf, asked=None):
    """Release each of `frames`, ``(arrival, channel, payload)``, at its
    arrival, moving the clock to each of the loop's deadlines below `until`,
    each appended to `asked`; return the replies as ``(channel, arrival,
    payload)``."""
    pending, seqs = sorted(frames, key=lambda f: f[0]), Counter()
    lp._at(rt.now)
    while True:
        t, daemon = lp._deadlines()
        if asked is not None:
            asked.append((t, daemon))
        at = min(t, daemon, pending[0][0] if pending else math.inf)
        if at >= until:
            break
        rt.now = at
        while pending and pending[0][0] == rt.now:
            _, ch, payload = pending.pop(0)
            lp.inbox[ch, seqs[ch]] = (rt.now, [ch.encode(), pickle.dumps(payload)])
            rt.released[ch].add(seqs[ch])
            seqs[ch] += 1
        lp._at(rt.now)
    replies = []
    for identity, header, payload in lp.sock.sent:
        ch, arrival, _ = pickle.loads(header)
        assert identity.decode() == reply_channel(ch)
        replies.append((ch, arrival, pickle.loads(payload)))
    return replies


def test_two_queries_on_one_channel_serialize_and_a_renewal_passes_a_queued_one():
    """Each channel's calls reach the master one at a time, twice the hop after
    the previous reply left; a renewal goes ahead of the queued query, behind
    the one in flight. Another channel's query does not wait for them."""
    lp, rt = _loop()
    four, two = _keys(0)[0] + _keys(1)[0], _keys(2)[0]
    got = _serve(
        lp,
        rt,
        [
            (1.0, LOOK, ("query", "q1", four)),
            (1.0, LOOK, ("query", "q2", two)),
            (1.0, OTHER, ("query", "o1", two)),
            (1.0 + HOP / 2, LOOK, ("renew", "r1", two)),
        ],
    )
    e1 = 1.0 + 4 * PER_KEY
    r_start = e1 + 2 * HOP
    e_r = r_start + 2 * PER_KEY
    q2_start = e_r + 2 * HOP
    e_q2 = q2_start + 2 * PER_KEY
    e_o = 1.0 + 2 * PER_KEY
    to_p0, to_p1 = reply_channel(LOOK), reply_channel(OTHER)
    assert [(ch, p[0]) for ch, _, p in got] == [
        (to_p1, "o1"),
        (to_p0, "q1"),
        (to_p0, "r1"),
        (to_p0, "q2"),
    ]
    # Each reply's arrival, its start and its lease deadline.
    assert [(a, p[1], p[2]) for _, a, p in got] == pytest.approx(
        [
            (e_o + HOP, 1.0, e_o + TTL),
            (e1 + HOP, 1.0, e1 + TTL),
            (e_r + HOP, r_start, e_r + TTL),
            (e_q2 + HOP, q2_start, e_q2 + TTL),
        ]
    )
    assert lp.counts == {**dict.fromkeys(store.COUNTS, 0), "lookups": 3, "renewals": 1}


def test_a_put_publishes_at_its_lane_s_end_and_a_hit_names_who_saved_it():
    lp, rt = _loop()
    keys, groups = _keys(0)
    put = ("put", keys, groups, 0.5, 2.0)  # its lane started before it arrived
    assert _serve(lp, rt, [(1.0, XFER, put)], until=1.25) == []
    assert lp.master.used == 2  # taken at its arrival
    got = _serve(
        lp,
        rt,
        [
            (1.5, LOOK, ("query", "early", keys)),
            (2.5, LOOK, ("query", "late", keys)),
            (2.5, OTHER, ("query", "own", keys)),
        ],
    )
    assert [p[0] for _, _, p in got] == ["early", "late", "own"]
    assert [p[3] for _, _, p in got] == [{}, *[dict.fromkeys(keys, "engine-P1")] * 2]
    assert lp.counts == {
        "lookups": 3,
        "hit_keys": 4,
        "foreign_hit_keys": 2,
        "renewals": 0,
        "evicted": 0,
    }


def test_a_put_of_a_key_in_flight_takes_no_bytes_grants_no_lease_publishes_nothing():
    """Engine P0 puts the keys engine P1's put holds in flight: P0's put ends
    first and publishes nothing, the keys appear at P1's end as P1's, and
    their lease starts at that end, not at P0's put."""
    lp, rt = _loop()
    keys, groups = _keys(0)
    first = (1.0, XFER, ("put", keys, groups, 1.0, 3.0))
    second = (1.5, XFER.replace("P1", "P0"), ("put", keys, groups, 1.5, 2.0))
    mid = (2.5, LOOK, ("query", "mid", keys))
    got = _serve(lp, rt, [first, second, mid], until=2.75)
    assert [p[3] for _, _, p in got] == [{}]
    assert lp.master.used == 2
    _serve(lp, rt, [], until=3.25)
    assert lp.master.groups[_group(0)].deadline == 3.0
    got = _serve(lp, rt, [(3.5, LOOK, ("query", "after", keys))])
    assert got[-1][2][3] == dict.fromkeys(keys, "engine-P1")


def test_the_loop_runs_a_round_at_its_boundary_and_none_that_would_evict_nothing():
    lp, rt = _loop(_entry(capacity_bytes=6))
    puts = [(1.0, XFER, ("put", *_keys(c), 1.0, 1.003)) for c in range(3)]
    every = [k for c in range(3) for k in _keys(c)[0]]
    asked = []
    _serve(lp, rt, [*puts, (1.004, LOOK, ("query", "q", every))], 11.0, asked)
    # The publish takes the Store over the mark, and the lookup leases every
    # group before the round it asked for runs.
    assert (math.inf, 1.01) in asked and asked[-1] == (math.inf, 11.01)
    assert lp.counts["evicted"] == 0
    _serve(lp, rt, [], 12.0, asked)
    assert lp.counts["evicted"] == 2 and asked[-1] == (math.inf, math.inf)


# --- the refusals ---------------------------------------------------------------


@pytest.mark.parametrize(
    "flags, refusal",
    [
        (["--rpc_thread_num=32", "--enable_offload=true"], "enable_offload"),
        (["--rpc_thread_num=32", "--handler_worker_thread_num=8"], "handler_worker"),
        (["--rpc_thread_num=32", "--default_kv_soft_pin_ttl=1s"], "soft_pin"),
        (["--rpc_thread_num=0"], "no positive --rpc_thread_num"),
        (["--eviction_ratio=0.05"], "no positive --rpc_thread_num"),
        (["--rpc_thread_num=32", "--eviction_ratio=2"], "eviction_ratio"),
    ],
    ids=["offload", "handler-threads", "soft-pin", "zero", "none", "ratio"],
)
def test_a_master_flag_it_neither_models_nor_accepts_is_refused_by_name(flags, refusal):
    with pytest.raises(MooncakeStoreRefusal, match=refusal):
        store.check_entry(_entry(master=flags))


def test_the_master_s_flags_read_as_mooncake_reads_them():
    flags = store.parse_master_flags(
        ["--rpc_thread_num=8", "--default_kv_lease_ttl", " 3S "]
    )
    assert (flags.default_kv_lease_ttl, flags.rpc_thread_num) == (3.0, 8)
    defaults = store.parse_master_flags(["--rpc_thread_num=1"])
    assert (defaults.default_kv_lease_ttl, defaults.eviction_ratio) == (10.0, 0.05)
    assert defaults.eviction_high_watermark_ratio == 0.9


@pytest.mark.parametrize(
    "changed, refusal",
    [
        ({"lookup_s": [0.0, PER_KEY]}, "fixed term is 0.0"),
        ({"capacity_bytes": 0}, "capacity_bytes is 0"),
        ({"protocol": "nvlink"}, "protocol is 'nvlink'"),
        ({"writers": 1}, "unknown writers"),
        ({"master": None}, "lacks master"),
    ],
    ids=["no-hop", "no-capacity", "protocol", "unknown", "no-master"],
)
def test_an_entry_the_lp_cannot_read_is_refused_by_name(changed, refusal):
    entry = _entry(**changed)
    if entry["master"] is None:
        del entry["master"]
    with pytest.raises(MooncakeStoreRefusal, match=refusal):
        store.check_entry(entry)


def test_a_published_plan_that_contradicts_the_entry_is_refused_by_name():
    entry, plans = _entry(chunk_tokens=512), {
        "engine-P0.dp0": {"block_bytes": 1, "tp": 1}
    }
    with pytest.raises(MooncakeStoreRefusal, match="engine-P0.dp0 plans a 2 B chunk"):
        store.check_plans(entry, plans, block_tokens=256)
    store.check_plans(entry, plans, block_tokens=512)
    plans["engine-P1.dp1"] = {"block_bytes": 1, "tp": 2}
    with pytest.raises(MooncakeStoreRefusal, match="engine-P1.dp1 .* at TP 2"):
        store.check_plans(entry, plans, block_tokens=512)


# --- the launch -----------------------------------------------------------------

STORE = {
    "kv_connector": "mooncake_store",
    "kv_role": "offload",
    "kv_connector_extra_config": {
        "mooncake_store.master": "10.0.0.1:26051",
        "mooncake_store.metadata": "http://10.0.0.1:26080/metadata",
        "mooncake_store.protocol": "tcp",
        "mooncake_store.local_hostname": "10.0.0.2",
    },
}
POOLS = {
    "rdma0": {"master": "10.0.0.1:1", "metadata": "http://10.0.0.1:2/metadata"},
    "rdma1": {"master": "10.0.0.3:1", "metadata": "http://10.0.0.3:2/metadata"},
}


def _store(**extra):
    return {
        **STORE,
        "kv_connector_extra_config": {**STORE["kv_connector_extra_config"], **extra},
    }


def _prefill(instance):
    """A prefill deployment of the recipe's form: the simulated P/D connector
    and the Store's in a ``multi``."""
    pd = {"kv_connector": "compass", "kv_role": "kv_producer"}
    if instance is not None:
        pd["compass_instance"] = instance
    return {"kv_connector": "multi", "connectors": [pd, STORE]}


def _config(kv, **overrides):
    return atom_config_double(kv_transfer_config=kv, **overrides)


def _run(tmp_path, **more):
    run = {
        "clock_endpoint": "tcp://127.0.0.1:0",
        "bound_s": 60.0,
        "ipc_s": 2.0**-10,
        "machine": copy.deepcopy(DOCUMENT),
        "out_dir": str(tmp_path),
        store.ENTRY: _entry(),
        **more,
    }
    return json.loads(json.dumps(run))  # as the run file holds it


@pytest.fixture
def started(monkeypatch):
    launched = []
    monkeypatch.setattr(
        compass_run.subprocess, "Popen", lambda *a, **k: launched.append(a[0])
    )
    return launched


@pytest.mark.parametrize("instance", [None, 0, 1, 2])
def test_the_first_prefill_frontend_starts_the_lp_and_every_other_only_checks(
    tmp_path, started, instance
):
    run = _run(tmp_path, prefill_instances=3)
    config = _config(_prefill(instance), kv_cache_block_size=64)
    compass_run._launch_mooncake_store(config, run, "P")
    if instance in (None, 0):
        [argv] = started
        assert argv[-2:] == [compass_run.MOONCAKE_STORE_MAIN, "64"]
    else:
        assert started == []


def test_a_decode_side_that_does_not_offload_starts_nothing(tmp_path, started):
    compass_run._launch_mooncake_store(_config({}), _run(tmp_path), "D")
    assert started == []


@pytest.mark.parametrize(
    "kv, more, side, overrides, refusal",
    [
        (_prefill(0), {store.ENTRY: None}, "P", {}, "has no mooncake_store entry"),
        ({}, {}, None, {}, "offloads no KV to a Mooncake Store and the run file has a"),
        (STORE, {}, "D", {}, "decode side offloads"),
        (STORE, {}, None, {"pipeline_parallel_size": 2}, r"world 2 \(the entry's 1\)"),
        (STORE, {}, None, {"tensor_parallel_size": 2}, r"world 2 \(the entry's 1\)"),
        (
            _store(**{"mooncake_store.chunk_tokens": 512}),
            {},
            None,
            {},
            r"chunk_tokens 512 \(the entry's 256\)",
        ),
        (
            _store(**{"mooncake_store.save_pool_mib": 1}),
            {},
            None,
            {},
            r"save_slots 256 \(the entry's 65536\)",
        ),
        (
            {
                **STORE,
                "kv_connector_extra_config": {
                    "mooncake_store.pools": POOLS,
                    "mooncake_store.local_hostname": "10.0.0.2",
                },
            },
            {},
            None,
            {},
            "addresses 2 masters",
        ),
    ],
    ids=[
        "no-entry",
        "no-deployment",
        "decode",
        "pp-world",
        "world",
        "chunk",
        "slots",
        "masters",
    ],
)
def test_each_launch_refusal_is_by_name_and_starts_nothing(
    tmp_path, started, kv, more, side, overrides, refusal
):
    run = _run(tmp_path, **more)
    if run[store.ENTRY] is None:
        del run[store.ENTRY]
    with pytest.raises(ValueError, match=refusal):
        compass_run._launch_mooncake_store(_config(kv, **overrides), run, side)
    assert started == []


# --- the channels ---------------------------------------------------------------


def test_a_pipelined_prefill_s_head_stage_alone_talks_to_the_store(tmp_path):
    """The head stage schedules, so its lookups and its ranks' puts, every
    stage's, ride its channels; a later stage has none."""
    from test_dp_slice import DP_MACHINE  # a machine stating its stage sends

    run = _run(tmp_path, router_s=2.0**-12, kv_write_req_s=2.0**-10)
    run[compass_run.PREFILL_PP] = 2
    run["machine"] = copy.deepcopy(DP_MACHINE)
    table = compass_run.channel_table(run)
    ms = LpId("mooncake_store")
    assert {c.source.name for c in table.channels_into(ms)} == {"engine-P-pp0"}
    assert {c.target.name for c in table.channels_from(ms)} == {"engine-P-pp0"}


def test_deploy_sh_s_run_file_for_the_recipe_s_3p4d_has_17_lps_and_18_store_channels(
    tmp_path, planted  # noqa: F811
):
    """Three prefill instances at DP2 and four decode at DP4 behind a DP-aware
    router, `compass run --mooncake-store` naming the entry."""
    env, flags = planted
    env["PLANTED_BUILD"] = f"{_tree_commit()} clean"
    (tmp_path / "store.json").write_text(json.dumps(ENTRY))
    flags += router_hop(tmp_path) + ["--mooncake-store", str(tmp_path / "store.json")]
    model = ["--model", str(tmp_path)]
    prefill = ["--prefill", "--", *model, *PREFILL]
    decode = ["--decode", "--", *model, "-tp", "4", "--enable-dp-attention"]
    words = ["bash", str(SCRIPT), str(tmp_path / "d"), "run", *flags]
    words += prefill * 3 + decode * 4 + ["--router", "--", *ROUTER_LP_ARGS]
    script = subprocess.Popen(
        words, env=env, cwd=tmp_path, stdout=subprocess.PIPE, text=True
    )
    try:
        _until(tmp_path / "d" / "server.json", script)
        run = json.loads((tmp_path / "d" / "run_file.json").read_text())
    finally:
        script.send_signal(signal.SIGTERM)
        out, _ = script.communicate(timeout=60)
    assert script.returncode == 0, out
    assert run[store.ENTRY] == ENTRY
    table = compass_run.channel_table(run)
    lps = sorted(lp.name for lp in table.registry.ids())
    ms = LpId("mooncake_store")
    rows = table.channels_into(ms) + table.channels_from(ms)
    print(f"{len(lps)} LPs: {', '.join(lps)}")
    print(f"{len(rows)} Store channels:")
    for c in rows:
        print(f"  {c.name}  lookahead {c.lookahead_s} s, {c.receive}")
    assert len(lps) == 17 and len(rows) == 18
    assert {c.lookahead_s for c in rows} == {HOP}
    assert {c.source.name for c in table.channels_into(ms)} == {
        f"engine-P{k}" for k in range(3)
    }
    members = compass_run._members(table, (2, 4))[LpId("engine-P2")]["dp1"]
    assert {
        "engine-P2->mooncake_store:lookup#dp1",
        "mooncake_store->engine-P2:lookup#dp1",
        "engine-P2->mooncake_store:transfer#dp1",
    } <= set(members)
    assert "mooncake_store" in compass_run._hand_ins(run)


# --- end to end -----------------------------------------------------------------


def test_the_launched_lp_answers_an_engine_on_the_run_s_clock(tmp_path, monkeypatch):
    """One engine at DP1: a put whose lane ends at 0.5 s and a query before it
    lands; at 0.75 s a query and a renewal, which goes first; the hand-in."""
    run = _run(tmp_path)
    authority = ClockAuthority(compass_run.channel_table(run), bound_s=run["bound_s"])
    served = serve(authority, run["clock_endpoint"])
    run["clock_endpoint"] = served.endpoint
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    monkeypatch.setattr(clock, "_installed", None)
    procs, real = [], subprocess.Popen
    monkeypatch.setattr(
        compass_run.subprocess, "Popen", lambda *a, **k: procs.append(real(*a, **k))
    )
    # The engine's plan, as its runner publishes it: a one-byte block.
    now = tmp_path / compass_run.NOW_FILE.format("engine")
    compass_run.publish_transfer(compass_run._block_file(now), 1, 1)
    config = _config(STORE, kv_cache_block_size=256)
    compass_run._launch_mooncake_store(config, run, None)
    [proc] = procs
    idle = [connect(LpId(lp), served.endpoint) for lp in ("traffic", "frontend")]
    for conn in idle:
        conn.send(("NER", math.inf, [], math.inf))
    table = compass_run.channel_table(run)
    rt = LPRuntime(LpId("engine"), table, connect(LpId("engine"), served.endpoint))
    named = tmp_path / store.ENDPOINT_FILE
    deadline = time.monotonic() + 120
    while not named.exists():
        assert proc.poll() is None and time.monotonic() < deadline
        time.sleep(0.05)
    ctx = zmq.Context()
    look, xfer = ctx.socket(zmq.DEALER), ctx.socket(zmq.DEALER)
    look.connect(named.read_text())
    xfer.connect(named.read_text())
    ask = "engine->mooncake_store:lookup#dp0"
    keys, groups = _keys(0)
    got = []

    def send(sock, ch, payload):
        header = pickle.dumps((ch, *rt.stamp_send(ch)))
        sock.send_multipart([header, pickle.dumps(payload)])

    def take():
        back = reply_channel(ask)
        while rt.released[back] - rt.arrived[back]:
            header, payload = look.recv_multipart()
            rt.check_arrival(*pickle.loads(header))
            got.append((pickle.loads(header)[1], pickle.loads(payload)))

    try:
        send(
            xfer,
            "engine->mooncake_store:transfer#dp0",
            ("put", keys, groups, 0.25, 0.5),
        )
        send(look, ask, ("query", "q1", keys))
        rt.next_event(math.inf)
        take()
        rt.next_event(0.75)
        send(look, ask, ("query", "q2", keys))
        send(look, ask, ("renew", "r1", keys))
        while len(got) < 3:
            rt.next_event(math.inf)
            take()
        assert rt.next_event(math.inf) == math.inf
        assert proc.wait(120) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        look.close(linger=0)
        xfer.close(linger=0)
        ctx.term()
        served.close()

    e1 = HOP + 2 * PER_KEY
    e_r = 0.75 + HOP + 2 * PER_KEY
    e_q2 = e_r + 2 * HOP + 2 * PER_KEY
    assert [p[0] for _, p in got] == ["q1", "r1", "q2"]
    # Each reply's arrival, its start and its lease deadline.
    assert [(a, *p[1:3]) for a, p in got] == pytest.approx(
        [
            (e1 + HOP, HOP, e1 + TTL),
            (e_r + HOP, 0.75 + HOP, e_r + TTL),
            (e_q2 + HOP, e_r + 2 * HOP, e_q2 + TTL),
        ]
    )
    saved = dict.fromkeys(keys, "engine")
    assert [p[3] for _, p in got] == [{}, saved, saved]
    handed = json.loads((tmp_path / "lp-mooncake_store.json").read_text())
    assert handed == {
        "refusals": [],
        "store": {
            "lookups": 2,
            "hit_keys": 2,
            "foreign_hit_keys": 0,
            "renewals": 1,
            "evicted": 0,
        },
    }


# --- a Store emptied between client runs ----------------------------------------


def test_an_entry_may_empty_the_store_between_client_runs_and_size_each():
    entry = _entry(clear_between_client_runs=True, capacity_by_client_run=[200, 50])
    store.check_entry(entry)
    assert [store.client_run_capacity(entry, k) for k in range(4)] == [200, 50, 50, 50]
    assert store.client_run_capacity(_entry(), 3) == 200


@pytest.mark.parametrize(
    "changed, refusal",
    [
        ({"clear_between_client_runs": 1}, "clear_between_client_runs is 1"),
        (
            {"clear_between_client_runs": True, "capacity_by_client_run": []},
            "not a list of positive ints",
        ),
        (
            {"clear_between_client_runs": True, "capacity_by_client_run": [100]},
            "starts at 100, and capacity_bytes, client run 0's, is 200",
        ),
        ({"capacity_by_client_run": [200, 50]}, "needs clear_between_client_runs"),
    ],
    ids=["not-bool", "empty", "first-differs", "never-emptied"],
)
def test_a_client_run_field_the_lp_cannot_read_is_refused_by_name(changed, refusal):
    with pytest.raises(MooncakeStoreRefusal, match=refusal):
        store.check_entry(_entry(**changed))


def test_a_cleared_store_holds_nothing_at_its_new_size_and_drops_pending_puts():
    """Published objects, a put still to publish and the eviction deadlines
    all go; a query after it finds nothing, and the new size bounds puts."""
    lp, rt = _loop(_entry(capacity_bytes=6))
    keys, groups = _keys(0)
    _serve(lp, rt, [(1.0, XFER, ("put", keys, groups, 1.0, 1.5))], until=2.0)
    assert lp.master.exists(keys, 2.0) == dict.fromkeys(keys, "engine-P1")
    late, late_groups = _keys(1)
    _serve(lp, rt, [(2.0, XFER, ("put", late, late_groups, 2.0, 9.0))], until=2.5)
    assert lp.events and lp.master.used == 4

    lp.clear(2)

    assert (lp.master.capacity, lp.master.used, lp.master.filed) == (2, 0, 0)
    assert not lp.master.objects and not lp.master.groups and not lp.events
    assert lp.master.next_tick(3.0) == math.inf
    got = _serve(lp, rt, [(3.0, LOOK, ("query", "after", keys + late))], until=10.0)
    assert [p[3] for _, _, p in got] == [{}]  # nor did the pending put publish
    lp.master.put(*_keys(2), "engine-P0", 10.0)  # two one-byte objects fit
    with pytest.raises(MooncakeStoreRefusal, match="of the Store's 2 B in use"):
        lp.master.put(*_keys(3), "engine-P0", 10.0)


@pytest.mark.parametrize("clear", [False, True])
def test_the_lp_hands_in_each_client_run_then_empties_the_store_where_told(
    tmp_path, monkeypatch, clear
):
    entry = _entry(capacity_by_client_run=[200, 50, 25]) if clear else _entry()
    if clear:
        entry["clear_between_client_runs"] = True
    run = {store.ENTRY: entry}
    handed, cleared = [], []
    monkeypatch.setattr(
        compass_run, "_hand_in", lambda run, lp, reasons, k, **s: handed.append(k)
    )
    lp = SimpleNamespace(counts={}, clear=cleared.append)
    ended = compass_run._store_client_run_ended(run, SimpleNamespace(refusals=[]), lp)
    for k in range(3):
        ended(k)
    assert handed == [0, 1, 2]
    assert cleared == ([50, 25, 25] if clear else [])

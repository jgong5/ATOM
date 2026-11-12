# SPDX-License-Identifier: MIT
"""ATOM's `lmcache_offload` connector on a simulated run.

The engine tests build the connector's scheduler half the way an engine
process does, inside `compass_run.engine`, against an authority served here,
and drive ATOM's own MiniMax-M3 offload scheduler through it on the step loop:
a save, then a lookup on every rank and the load it arms, each reported one
step after the step that runs it, at times derived by hand from the entry's
terms. The tier tests drive one rank's tier under ATOM's SLRU policy, and the
fit test fits an entry on log lines written here.
"""

import json
import math
from types import SimpleNamespace

import pytest
from conftest import atom_config_double
from test_dp_step_max import runner
from test_memory_readings import DOCUMENT
from test_run_refusal_summary import STAGE1
from test_runner_control_refusals import Worker
from transformers import PretrainedConfig

from atom.compass import run as compass_run
from atom.compass.backends import StepStubBackend
from atom.compass.backends.step_linear_fit import fit_offload
from atom.compass.clock import ClockAuthority
from atom.compass.clock_transport import connect, serve
from atom.compass.kv import SimulatedKVConnector
from atom.compass.kv.lmcache_offload import (
    ENTRY,
    Lookup,
    OffloadShell,
    Tier,
    chunk_keys,
)
from atom.compass.runner.overrides import NonAllocatingRunner, RunnerRefusal
from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.types import (
    KVConnectorOutput,
    LoadOperationId,
)
from atom.kv_transfer.offload.hybrid.m3.connector import (
    M3OffloadConnector,
    M3OffloadScheduler,
)
from atom.model_engine.engine_stats import EngineStats
from atom.utils import clock, forward_context
from atom.utils.clock import LPRuntime
from atom.utils.distributed import utils as dist_utils

CHUNK, BLOCK, TP = 8, 4, 2
STEP = 0.01  # the step loop's clock calls, one per step
LOOKUP_S, STORE_S, LOAD_S = [2.0**-10, 2.0**-12], [2.0**-8, 2.0**-9], [2.0**-8, 2.0**-7]
CHUNK_BYTES = 2**27
SLOTS = 6  # a 0.75 GiB tier of 128 MiB chunks
OFFLOAD = {"kv_connector": "lmcache_offload", "kv_role": "offload"}
LMCACHE = {
    "LMCACHE_CHUNK_SIZE": str(CHUNK),
    "LMCACHE_LOCAL_CPU": "True",
    "LMCACHE_MAX_LOCAL_CPU_SIZE": "0.75",
    "LMCACHE_CACHE_POLICY": "ATOM_SLRU",
    "OFFLOAD_MIN_LOAD_TOKENS": str(CHUNK),
}
#: MiniMax-M3 as ATOM's offload layout and LMCache's key namespace read it.
M3 = {
    "architectures": ["MiniMaxM3SparseForCausalLM"],
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "hidden_size": 256,
    "head_dim": 32,
}


def _entry(**fields):
    return {
        "chunk_size": CHUNK,
        "tensor_parallel_size": TP,
        "chunk_bytes": CHUNK_BYTES,
        "lookup_s": LOOKUP_S,
        "store_s": STORE_S,
        "load_s": LOAD_S,
        **fields,
    }


def _config(kv=OFFLOAD, **overrides):
    return atom_config_double(
        kv_transfer_config=kv,
        hf_config=PretrainedConfig.from_dict(M3),
        tensor_parallel_size=TP,
        kv_cache_block_size=BLOCK,
        kv_cache_dtype="fp8",
        **overrides,
    )


def _seq(name, tokens):
    return SimpleNamespace(
        id=name,
        token_ids=list(tokens),
        num_prompt_tokens=len(tokens),
        num_tokens=len(tokens),
        num_cached_tokens=0,
        block_table=list(range(-(-len(tokens) // BLOCK))),
    )


def _at_step(t: float) -> float:
    """The first step at or after `t`."""
    return math.ceil(t / STEP - 1e-9) * STEP


class Engine:
    """An engine process's connector on the step loop, against an authority
    served here; every other LP idles."""

    def __init__(self, tmp_path, monkeypatch, entry=None) -> None:
        for name, value in LMCACHE.items():
            monkeypatch.setenv(name, value)
        registry = KVConnectorFactory._registry
        for name in ("lmcache_offload", "multi"):
            monkeypatch.setitem(registry, name, registry[name])
        monkeypatch.setattr(clock, "_installed", None)
        monkeypatch.setattr(dist_utils, "LP_OF_RANK", None)
        run = {
            "clock_endpoint": "tcp://127.0.0.1:0",
            "bound_s": 60.0,
            "admission_s": 0.54e-3,
            "ipc_s": 2.0**-10,
            "stream_s": 2.0**-12,
            "machine": DOCUMENT,
            "out_dir": str(tmp_path),
            ENTRY: entry or _entry(),
        }
        table = compass_run.channel_table(run)
        self.served = serve(
            ClockAuthority(table, bound_s=run["bound_s"]), run["clock_endpoint"]
        )
        run["clock_endpoint"] = self.served.endpoint
        (tmp_path / "run.json").write_text(json.dumps(run))
        monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
        self.idle = [
            connect(lp, self.served.endpoint)
            for lp in table.registry.ids()
            if lp.name != "engine"
        ]
        for conn in self.idle:
            conn.send(("NER", math.inf, [], math.inf))
        self.config = _config()
        with compass_run.engine(self.config):
            self.shell = KVConnectorFactory.create_connector(self.config, "scheduler")
        self.rt = clock.installed()
        self.reported = []  # (step, what ATOM's scheduler half reported)

    def step(self) -> None:
        self.rt.advance_to(self.rt.now + STEP)
        out = self.shell.process_completions(KVConnectorOutput())
        done = [*out.finished_saving, *out.finished_loading, *out.failed_loading]
        if done:
            self.reported.append((self.rt.now, sorted(map(str, done))))

    def finish(self) -> None:
        try:
            while self.rt.next_event(math.inf) != math.inf:
                pass
        finally:
            self.served.close()


def test_the_factory_builds_atom_s_scheduler_half_over_a_tier_per_rank(
    tmp_path, monkeypatch
):
    engine = Engine(tmp_path, monkeypatch)
    shell = engine.shell
    assert isinstance(shell, OffloadShell)
    impl = shell._inner._impl
    assert type(impl) is M3OffloadScheduler
    assert impl.chunk_size == CHUNK
    # LMCache's default lookup scope is every worker of the replica.
    assert isinstance(impl._lookup_client, Lookup)
    assert list(impl._lookup_client.ranks) == [0, 1]
    assert [type(r) for r in shell._ranks] == [M3OffloadConnector] * TP
    assert [t.free for t in shell._tiers] == [SLOTS] * TP
    assert {type(t.policy).__name__ for t in shell._tiers} == {"SLRUCachePolicy"}
    engine.finish()


def test_a_saved_prefix_is_looked_up_on_every_rank_and_loaded_at_its_priced_end(
    tmp_path, monkeypatch
):
    engine = Engine(tmp_path, monkeypatch)
    shell, rt = engine.shell, engine.rt
    prefix = list(range(1, 4 * CHUNK + 1))  # four whole chunks

    a = _seq("a", prefix)
    shell.update_state_after_alloc(a)
    a.num_cached_tokens = len(prefix)  # its prefill has run
    [save] = shell.build_connector_meta().requests
    assert save.save_spec.skip_leading_tokens == 0
    stored = STORE_S[0] + 4 * STORE_S[1]
    while rt.now < stored + 3 * STEP:
        engine.step()
    assert [len(t.chunks) for t in shell._tiers] == [4, 4]

    # The suffix is a fifth whole chunk, asked for and not held.
    b = _seq("b", prefix + list(range(100, 100 + CHUNK)))
    asked = rt.now
    assert shell.get_num_new_matched_tokens(b) == (4 * CHUNK, True)
    assert rt.now == pytest.approx(asked + LOOKUP_S[0] + 5 * LOOKUP_S[1])
    assert [len(t.pinned) for t in shell._tiers] == [4, 4]
    shell.update_state_after_alloc(b)
    sent = rt.now
    [load] = shell.build_connector_meta().requests
    assert (load.load_spec.hbm_cached_tokens, load.load_spec.lmcache_cached_tokens) == (
        0,
        4 * CHUNK,
    )
    loaded = sent + LOAD_S[0] + 4 * LOAD_S[1]
    while rt.now < loaded + 3 * STEP:
        engine.step()

    print("\nreported", engine.reported)
    # Each copy runs at the first step at or after its end, and is reported at
    # the step after that one, as the engine's aggregation answers.
    assert engine.reported == [
        (pytest.approx(_at_step(stored) + STEP), ["a"]),
        (pytest.approx(sent + _at_step(loaded - sent) + STEP), ["b"]),
    ]
    stats = shell.get_statistics()
    assert stats["saved_tokens"] == stats["loaded_tokens"] == 4 * CHUNK
    assert (stats["save_requests"], stats["load_requests"]) == (1, 1)
    # The load released the lookup's pins on every rank.
    assert [len(t.pinned) for t in shell._tiers] == [0, 0]
    said = shell.offload_statistics()
    assert said["tier"] == {
        "stored": 8,
        "lookups": 2,
        "lookup_chunks": 10,
        "hit_chunks": 8,
        "held_chunks": 4,
    }
    assert said["copies"]["load"] == {
        "copies": 1,
        "chunks": 4,
        "seconds": pytest.approx(loaded - sent),
        "waited": 0.0,
    }
    assert LoadOperationId("b", 0) not in shell._inner._impl._load_inflight_tokens
    engine.finish()


def test_a_deployment_and_its_run_file_agree_on_the_tier(tmp_path, monkeypatch):
    run = {ENTRY: _entry()}
    with pytest.raises(ValueError, match="offloads no KV through lmcache_offload"):
        compass_run._offload_tier(_config({}), run)
    with pytest.raises(ValueError, match=f"has no {ENTRY} entry"):
        compass_run._offload_tier(_config(), {})
    dp2 = _config()
    dp2.parallel_config.data_parallel_size = 2
    with pytest.raises(ValueError, match="this deployment runs data_parallel_size 2"):
        compass_run._offload_tier(dp2, run)
    with pytest.raises(ValueError, match="lmcache_offload.load_s is"):
        compass_run._offload_tier(_config(), {ENTRY: _entry(load_s=[1.0])})


def test_an_entry_fitted_at_another_chunk_size_is_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="fitted at chunk_size 16, and this"):
        Engine(tmp_path, monkeypatch, _entry(chunk_size=16))


def test_the_compass_runner_builds_the_worker_half_that_moves_nothing(monkeypatch):
    registry = KVConnectorFactory._registry
    for name in ("lmcache_offload", "multi"):
        monkeypatch.setitem(registry, name, registry[name])
    built = []
    monkeypatch.setattr(
        forward_context,
        "set_kv_cache_data",
        lambda _, config, **k: built.append(
            KVConnectorFactory.create_connector(config, "worker")
        ),
    )
    runner_ = object.__new__(NonAllocatingRunner)
    runner_.config = _config()
    assert runner_.allocate_kv_cache(100) is True
    assert [type(w) for w in built] == [SimulatedKVConnector]
    runner_.config = _config({"kv_connector": "mooncake"})
    with pytest.raises(RunnerRefusal, match="'lmcache_mp' and 'lmcache_offload'"):
        runner_.allocate_kv_cache(100)


def _slru(slots):
    from lmcache.v1.storage_backend.cache_policy import get_cache_policy

    from atom.kv_transfer.offload.cache_policy import register_slru_policy

    register_slru_policy()
    return Tier(slots, get_cache_policy("ATOM_SLRU"), 300.0, 30.0)


def test_a_full_tier_evicts_what_slru_leaves_unprotected_and_no_pinned_chunk():
    tier = _slru(4)
    a, b, c, d = (chunk_keys(range(k, k + 3 * CHUNK), CHUNK) for k in (0, 50, 90, 140))
    assert tier.store(a[:2], 0.0) == 2
    # A hit promotes both chunks; protection holds half the tier's, so the
    # first one touched last, a[0], stays protected and a[1] goes back.
    assert tier.lookup(a[:2], "x", 1.0) == 2
    tier.unpin_lookup("x")
    assert tier.store(b[:2], 2.0) == 2
    assert tier.store(c[:2], 3.0) == 2
    # Probation goes first: a[1], then b[0]. LRU would have taken a[0] too.
    assert set(tier.chunks) == {a[0], b[1], *c[:2]}
    assert tier.lookup(c[:2], "y", 4.0) == 2
    # Two slots free up, then only pinned chunks are left: the store is cut.
    assert tier.store(d, 5.0) == 2
    assert set(tier.chunks) == {*c[:2], *d[:2]}
    assert tier.counts["evicted"] == 4
    assert tier.counts["store_cut"] == 1
    # The pin monitor, at its first check past the timeout, unpins them.
    assert tier.lookup([], "z", 300.0) == 0
    assert len(tier.pinned) == 2
    assert tier.lookup([], "z", 330.0) == 0
    assert (len(tier.pinned), tier.counts["forced_unpins"]) == (0, 2)


def test_a_chunk_stored_twice_takes_one_slot():
    tier = _slru(4)
    keys = chunk_keys(range(2 * CHUNK + 3), CHUNK)
    assert len(keys) == 2  # the tail is no whole chunk
    assert tier.store(keys, 0.0) == tier.store(keys, 1.0) == 2
    assert (tier.free, tier.counts["duplicates"]) == (2, 2)


def test_the_entry_is_fitted_on_the_copies_and_the_lookups_the_logs_state(tmp_path):
    lines = [
        f"[atom] LMCache M3 PAGE offload worker rank={r}: page_regions=301 "
        f"bytes_per_block=5056512 chunk=256 gpu_staging_chunk_bytes=10113024 "
        "gpu_staging_buffer_chunks=4"
        for r in (0, 1)
    ]
    for n in range(1, 9):
        lines.append(
            f"[OFFLOAD-SAVE-PROF] rank=0 req=r toks={256 * n} skip=0 chunks={n} "
            f"groups=1 total_bytes=1 total_ms={1.0 + 2.0 * n:.2f}"
        )
        lines.append(
            f"[OFFLOAD-LOAD-PROF] rank=0 req=r hbm=0 lmc={256 * n} retrieved=1 "
            f"status=ok chunks={n} groups=1 total_ms={0.5 + 1.5 * n:.2f}"
        )
    counts = {"lookup_n": 0, "lookup_prompt_tokens": 0, "lookup_us": 0}
    for n in range(1, 7):
        counts = {
            "lookup_n": counts["lookup_n"] + n,
            "lookup_prompt_tokens": counts["lookup_prompt_tokens"] + 256 * n * n,
            "lookup_us": counts["lookup_us"] + 300 * n + 4 * n * n,
        }
        lines.append(f"[atom] [OFFLOAD-PERF] {json.dumps(counts)} | {{}}")
    (tmp_path / "server.log").write_text("\n".join(lines) + "\n")

    entry, _ = fit_offload([tmp_path / "server.log"])

    print(json.dumps(entry))
    assert (entry["chunk_size"], entry["tensor_parallel_size"]) == (256, 2)
    assert entry["chunk_bytes"] == 10113024
    assert entry["store_s"] == pytest.approx([1e-3, 2e-3], rel=1e-6)
    assert entry["load_s"] == pytest.approx([0.5e-3, 1.5e-3], rel=1e-6)
    assert entry["lookup_s"] == pytest.approx([300e-6, 4e-6], rel=1e-6)
    assert entry["fit"]["logs"] == [str(tmp_path / "server.log")]
    other = tmp_path / "other.log"
    other.write_text(lines[0].replace("chunk=256", "chunk=128") + "\n")
    with pytest.raises(ValueError, match="an entry is fitted on one"):
        fit_offload([tmp_path / "server.log", other])


def test_a_client_run_counts_its_own_hits_and_copies(monkeypatch, tmp_path):
    """Client run 1's offload is the engine's at its end less that at client
    run 0's, a level as it stands, with the hit rates of what it counted."""
    run = {
        "admission_s": 0.54e-3,
        "ipc_s": 0.001,
        "stream_s": 0.002,
        "bound_s": 10.0,
        "clock_endpoint": "inproc:unused",
        "out_dir": str(tmp_path),
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    a = compass_run._RecordingAuthority(run)
    stats = EngineStats(enable_prefix_caching=True)
    said = {"loaded_tokens": 0, "loads_pending": 0}
    kv = SimpleNamespace(
        offload_statistics=lambda: {"connector": dict(said), "tier": {"evicted": 0}}
    )
    core = SimpleNamespace(
        runner_mgr=Worker(runner(StepStubBackend(STAGE1))),
        scheduler=SimpleNamespace(kv_connector=kv, engine_stats=stats),
    )
    monkeypatch.setattr(
        clock,
        "_installed",
        LPRuntime(compass_run.ENGINE, compass_run.channel_table(run), None),
    )
    stats.update_cache(100, 1000, 100, 100, 1000)
    said.update(loaded_tokens=400, loads_pending=3)
    compass_run._engine_hand_in(run, core, {}, client_run=0)
    stats.update_cache(300, 2000, 300, 300, 2000)
    said.update(loaded_tokens=1400, loads_pending=1)
    compass_run._engine_hand_in(run, core, {}, client_run=1)

    def counted(k):
        return compass_run._read_hand_ins(
            run, tmp_path / compass_run.CLIENT_DIR.format(k), False
        )

    out = compass_run._summary(run, a, a, 0.0, counted(1), counted(0))["offload"]
    [(lp, offload)] = out.items()
    assert lp == "engine"
    assert offload["connector"] == {"loaded_tokens": 1000, "loads_pending": 1}
    assert offload["cache"]["full_tokens"] == 2000
    assert offload["gpu_hit_rate"] == pytest.approx(300 / 2000)
    assert offload["cpu_hit_rate"] == pytest.approx(1000 / 2000)

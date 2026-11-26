# SPDX-License-Identifier: MIT
"""The Mooncake Store's scheduler half reads the LP clock, and the sync scanner
reads the Store.

ATOM's real `MooncakeStoreOffloadScheduler` runs each clock test twice: with no
runtime installed, on its modules' `time.monotonic`, and under an installed LP
runtime, on that runtime's clock while the host clock stands still. Both clocks
are moved by hand. A stand-in lookup client answers its lookups at once. No
driver, no Mooncake and no AITER.
"""

from __future__ import annotations

import array
import time
from types import SimpleNamespace

import pytest

from atom.kv_transfer.offload import _offload_common, chunked_scheduler
from atom.kv_transfer.offload.lookup_client import LookupClient
from atom.kv_transfer.offload.metadata import LoadSpec
from atom.kv_transfer.offload.mooncake_store import scheduler as scheduler_mod
from atom.kv_transfer.offload.mooncake_store.scheduler import (
    MooncakeStoreOffloadScheduler,
)
from atom.utils import clock
from tests.compass import sync_scan

BLOCK = 4  # tokens per KV block
CHUNK = 8  # tokens per Store chunk
STORE = "atom/kv_transfer/offload/mooncake_store/"
# Where the host clock stands, unmoving, while an LP runtime is installed.
HOST = 50.0


class _LP:
    """An installed runtime as `clock.now` sees it: the clock the test moves."""

    def __init__(self, state) -> None:
        self.state = state
        self.in_run = False

    def read_clock(self) -> float:
        return self.state.t


@pytest.fixture(params=["host", "lp"])
def clk(request, monkeypatch):
    """The clock the scheduler half runs on; `clk.t` is its reading.

    "host": no runtime, so each module's `time.monotonic`, here `clk.t`.
    "lp": an installed runtime's clock, `clk.t`; the host clock stays at HOST.
    """
    lp = request.param == "lp"
    state = SimpleNamespace(t=1e6 if lp else 1e3)
    host = SimpleNamespace(
        monotonic=(lambda: HOST) if lp else (lambda: state.t),
        perf_counter=time.perf_counter,
    )
    for module in (scheduler_mod, chunked_scheduler, _offload_common):
        monkeypatch.setattr(module, "time", host)
    try:  # the Store's common scheduler, where a tree split it out (kv_store)
        from atom.kv_transfer.offload.kv_store import scheduler as kv_store_scheduler
    except ImportError:
        kv_store_scheduler = None
    if kv_store_scheduler is not None:
        # Its `time.monotonic()` reads the LP clock in a simulated process as
        # compass's Store shell installs it (`_ClockedTime`), the host's here.
        from atom.compass.kv.mooncake_store import _ClockedTime

        monkeypatch.setattr(kv_store_scheduler, "time", _ClockedTime() if lp else host)
    if lp:
        clock.install(_LP(state))
    yield state
    clock.install(None)


class _Lookups(LookupClient):
    """A lookup client that answers at once: every query holds `hit` tokens and
    was sent when `sent_at` says; each renewal is recorded."""

    def __init__(self, hit: int) -> None:
        self.hit = hit
        self.sent_at: dict[str, float] = {}
        self.renewals: list[tuple[int, int]] = []

    def lookup(self, seq, lookup_id=None):
        return self.hit

    def query_started(self, sid):
        return self.sent_at.get(sid)

    def renew(self, seq, start, end):
        self.renewals.append((start, end))
        return (end - start) // CHUNK


@pytest.fixture
def store(monkeypatch):
    # The namespace names the device's KV formats, which ask AITER.
    monkeypatch.setattr(scheduler_mod, "store_namespace", lambda c, n: "atomkv1-test")
    config = SimpleNamespace(
        kv_transfer_config={
            "kv_connector": "mooncake_store",
            "kv_role": "offload",
            "kv_connector_extra_config": {
                "mooncake_store.master": "10.0.0.1:26051",
                "mooncake_store.metadata": "http://10.0.0.1:26080/metadata",
                "mooncake_store.local_hostname": "10.0.0.2",
                "mooncake_store.protocol": "tcp",
                "mooncake_store.chunk_tokens": CHUNK,
            },
        },
        kv_cache_block_size=BLOCK,
        decode_context_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        hf_config=SimpleNamespace(num_hidden_layers=4, model_type="glm_moe_dsa"),
        model="test-model",
        kv_cache_dtype="fp8",
        speculative_config=None,
        offload_min_load_tokens=0,
    )
    scheduler = MooncakeStoreOffloadScheduler(config)
    scheduler.close()
    scheduler._lookup_client = _Lookups(hit=4 * CHUNK)
    return scheduler


def _seq(req_id: int, num_prompt_tokens: int):
    return SimpleNamespace(
        id=req_id,
        token_ids=array.array("i", range(num_prompt_tokens)),
        num_prompt_tokens=num_prompt_tokens,
        num_cached_tokens=0,
        block_table=list(range(-(-num_prompt_tokens // BLOCK))),
        offload_loaded=False,
        offload_load_failed=False,
    )


def test_the_scanner_reads_the_store_and_the_inventory_classifies_its_sites():
    tree = sync_scan.repo_root_from_here()
    files = {str(p.relative_to(tree)) for p in (tree / STORE).glob("*.py")}
    assert files <= set(sync_scan.iter_scanned_files(tree))
    scanned = {site.id for site in sync_scan.scan(tree) if site.file.startswith(STORE)}
    listed = {
        row["id"]
        for row in sync_scan.load_inventory()["sites"]
        if row["file"].startswith(STORE)
    }
    assert scanned and scanned == listed


def test_a_lookup_is_dated_on_the_clock_or_when_its_query_was_sent(clk, store):
    seq = _seq(1, 41)
    assert store._fresh_tier_lookup(seq, "1") == 4 * CHUNK
    assert store._looked_up_at["1"] == clk.t
    # An answer the asynchronous lookup took earlier is as old as its query.
    store._lookup_client.sent_at["1"] = clk.t - 3.0
    store._fresh_tier_lookup(seq, "1")
    assert store._looked_up_at["1"] == clk.t - 3.0


def test_a_hit_past_the_reuse_window_is_renewed_before_its_load(clk, store):
    seq = _seq(2, 41)
    store._fresh_tier_lookup(seq, "2")
    spec = LoadSpec(hbm_cached_tokens=8, lmcache_cached_tokens=32, can_load=True)
    renewals = store._lookup_client.renewals

    clk.t += 1.0
    assert store._ensure_lookup_pin(seq, "2", spec)
    assert renewals == []
    clk.t += 0.5
    assert store._ensure_lookup_pin(seq, "2", spec)
    assert renewals == [(8, 32)]


def test_a_failed_save_pauses_saves_on_the_clock(clk, store):
    seq = _seq(3, 20)
    store.update_state_after_alloc(seq)
    seq.num_cached_tokens = 20
    [request] = store.build_connector_meta().requests

    assert store._retry_or_retire_failed_save(request.save_operation)
    assert not store._may_emit_save()
    clk.t += 0.75
    assert not store._may_emit_save()
    clk.t += 0.25
    assert store._may_emit_save()


def test_a_save_lease_ages_on_the_clock_and_a_dispatch_restarts_it(clk, store):
    seq = _seq(4, 20)
    store.activate_block_leases(seq, frozenset({3, 4}))

    clk.t += 29.5
    assert store.reclaim_stale_leases(30.0) == []
    store._refresh_save_reclaim_clock(seq)  # a save of it dispatched now
    clk.t += 29.5
    assert store.reclaim_stale_leases(30.0) == []
    clk.t += 0.5
    assert store.reclaim_stale_leases(30.0) == [frozenset({3, 4})]

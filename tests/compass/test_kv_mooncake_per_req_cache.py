# SPDX-License-Identifier: MIT
"""Mooncake's consumer takes the full transfer for a request with per-request
cache state, even when the producer's `hash_block_size` matches its own.

A block-only delta does not carry per-request state such as the SWA ring slot,
so `has_per_req_cache` alone turns the incremental offset off. Compass's
simulated decode side must find the same remote page geometry, and price the
blocks Mooncake's worker slices.
"""

from __future__ import annotations

import importlib
import itertools
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from atom.compass.kv.connector import SimulatedKVConnectorScheduler


@pytest.mark.parametrize("has_per_req_cache, expected", [(False, 4), (True, 0)])
def test_per_request_cache_forces_the_full_transfer(
    has_per_req_cache, expected, monkeypatch
):
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", MagicMock())
    mc = importlib.import_module(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    own = 16
    scheduler = object.__new__(mc.MooncakeConnectorScheduler)
    scheduler.is_producer = False
    scheduler.block_size, scheduler.dcp_size, scheduler.hash_block_size = own, 1, own
    scheduler._reqs_need_recv = {}
    scheduler.transfer_id_to_request_id = {}
    scheduler.request_id_to_transfer_id = {}
    seq = SimpleNamespace(
        id=0,
        kv_transfer_params={"do_remote_prefill": True, "hash_block_size": own},
        block_table=list(range(8)),
        has_per_req_cache=has_per_req_cache,
        num_cached_tokens=4 * own,
    )
    scheduler.update_state_after_alloc(seq)
    assert seq.kv_transfer_params["num_computed_blocks"] == expected


@pytest.mark.parametrize("consumer_dcp", [1, 2])
def test_simulated_pages_line_up_matches_mooncake(consumer_dcp, monkeypatch):
    """The simulated decode side skips its cached prefix exactly when
    Mooncake's consumer finds a remote page geometry; both then apply the same
    per-request-cache and hash-block-size guards."""
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", MagicMock())
    mc = importlib.import_module(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    own = 16
    ref = object.__new__(mc.MooncakeConnectorScheduler)
    ref.block_size, ref.dcp_size = own, consumer_dcp
    ref.hash_block_size = own * consumer_dcp
    sim = object.__new__(SimulatedKVConnectorScheduler)
    sim._block_size, sim._dcp_size = own, consumer_dcp
    sim._hash_block_size = own * consumer_dcp
    sizes = [None, own // 2, own, 2 * own, 4 * own]
    for block_size, dcp_size, hash_block_size in itertools.product(
        sizes, [None, 1, 2, 4], sizes
    ):
        blob = {
            k: v
            for k, v in (
                ("block_size", block_size),
                ("dcp_size", dcp_size),
                ("hash_block_size", hash_block_size),
            )
            if v is not None
        }
        assert sim._pages_line_up(blob) == (
            ref._remote_page_geometry(blob) is not None
        ), blob


def test_an_unsharded_producer_into_dcp2_is_priced_on_mooncakes_slices(monkeypatch):
    """With a cached prefix, a DCP-1 producer feeding a DCP-2 consumer skips
    `src_block_skip_factor` source blocks per cached consumer block. The
    simulated decode side prices as many consumer blocks as Mooncake's worker
    slices, and those carry the tokens of the source blocks it slices."""
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", MagicMock())
    mc = importlib.import_module(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    own, dcp, cached = 16, 2, 5
    blob = {
        "do_remote_prefill": True,
        "block_size": own,
        "dcp_size": 1,
        "hash_block_size": own,
        "transfer_id": 0,
        "remote_block_ids": list(range(100, 116)),
        "remote_host": "producer",
        "remote_handshake_port": 1,
    }

    def seq():
        return SimpleNamespace(
            id=0,
            kv_transfer_params=dict(blob),
            block_table=list(range(8)),
            has_per_req_cache=False,
            num_cached_tokens=cached * own * dcp,
        )

    ref = object.__new__(mc.MooncakeConnectorScheduler)
    ref.is_producer = False
    ref.block_size, ref.dcp_size, ref.hash_block_size = own, dcp, own * dcp
    ref._reqs_need_recv, ref._reqs_need_save = {}, {}
    ref.transfer_id_to_request_id, ref.request_id_to_transfer_id = {}, {}
    ref.update_state_after_alloc(seq())
    sent = []
    worker = MagicMock(is_producer=False, tp_size=1, tp_rank=0, _mla_landing=None)
    worker._has_slot_regions = False
    worker._send_on_socket = lambda addr, frames: sent.append(frames[1])
    monkeypatch.setattr(mc, "msgpack", SimpleNamespace(dumps=lambda body: body))
    mc.MooncakeConnector.start_load_kv(worker, ref.build_connector_meta())
    (body,) = sent

    sim = object.__new__(SimulatedKVConnectorScheduler)
    sim.is_producer, sim._transfer = False, (None,)
    sim._block_size, sim._dcp_size, sim._hash_block_size = own, dcp, own * dcp
    sim._rt, sim._latency_s, sim._wide = SimpleNamespace(now=0.0), 0.0, False
    sim._peers, sim._due = {"P": "engine-P"}, {}
    writes = []
    sim._socks = {("engine-P", 0): SimpleNamespace(send=writes.append)}
    sim._written = lambda pair, a, blocks: 0.0
    sim.update_state_after_alloc(seq())
    (write,) = writes
    _, priced, _, _ = pickle.loads(write)

    assert body["num_computed_blocks"] == cached
    assert priced == len(body["dst_block_ids"]) == 3
    assert len(body["src_block_ids"]) * blob["block_size"] == priced * own * dcp

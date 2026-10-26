# SPDX-License-Identifier: MIT
"""Mooncake's consumer takes the full transfer for a request with per-request
cache state, even when the producer's `hash_block_size` matches its own.

A block-only delta does not carry per-request state such as the SWA ring slot,
so `has_per_req_cache` alone turns the incremental offset off. Compass's
simulated decode side must find the same remote page geometry.
"""

from __future__ import annotations

import importlib
import itertools
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

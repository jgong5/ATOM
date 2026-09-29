# SPDX-License-Identifier: MIT
"""Mooncake's consumer takes the full transfer for a request with per-request
cache state, even when the producer's `hash_block_size` matches its own.

A block-only delta does not carry per-request state such as the SWA ring slot,
so `has_per_req_cache` alone turns the incremental offset off. The module cannot
be imported with no driver, so `update_state_after_alloc` is compiled out of the
connector's own AST and run on stand-ins for the scheduler and the sequence.
"""

from __future__ import annotations

import ast
import logging
from types import CodeType, FunctionType, SimpleNamespace

import pytest
from test_kv_blob_site import CONNECTORS


@pytest.mark.parametrize("has_per_req_cache, expected", [(False, 4), (True, 0)])
def test_per_request_cache_forces_the_full_transfer(has_per_req_cache, expected):
    path = CONNECTORS["mooncake"]
    (method,) = [
        fn
        for cls in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(cls, ast.ClassDef) and cls.name == "MooncakeConnectorScheduler"
        for fn in cls.body
        if isinstance(fn, ast.FunctionDef) and fn.name == "update_state_after_alloc"
    ]
    code = compile(ast.Module([method], []), str(path), "exec")
    (body,) = [c for c in code.co_consts if isinstance(c, CodeType)]
    own = 16
    scheduler = SimpleNamespace(
        is_producer=False,
        hash_block_size=own,
        _reqs_need_recv={},
        transfer_id_to_request_id={},
        request_id_to_transfer_id={},
    )
    seq = SimpleNamespace(
        id=0,
        kv_transfer_params={"do_remote_prefill": True, "hash_block_size": own},
        block_table=list(range(8)),
        has_per_req_cache=has_per_req_cache,
        num_cached_tokens=4 * own,
    )
    FunctionType(body, {"logger": logging.getLogger(__name__)})(scheduler, seq)
    assert seq.kv_transfer_params["num_computed_blocks"] == expected

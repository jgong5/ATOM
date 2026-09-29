# SPDX-License-Identifier: MIT
"""Mooncake's consumer takes the full transfer unless the producer's
`hash_block_size` matches its own.

The module cannot be imported with no driver -- its `aiter` import asks
`rocminfo` for the GPU arch -- so `update_state_after_alloc` is compiled out of
the file's own AST and run on stand-ins for the scheduler and the sequence. It
is the source on disk that runs, not a copy.
"""

from __future__ import annotations

import ast
import logging
from types import CodeType, FunctionType, SimpleNamespace

from test_kv_blob_site import CONNECTORS, _blob_site


def _emitted(backend: str) -> dict:
    """The backend's blob, keys only, read from its one assignment site."""
    return dict.fromkeys(k.value for k in _blob_site(CONNECTORS[backend])[0].value.keys)


def _consumer_offset(blob: dict, own: int) -> int:
    """The `num_computed_blocks` mooncake's scheduler sets on receiving *blob*."""
    path = CONNECTORS["mooncake"]
    module = ast.parse(path.read_text(encoding="utf-8"))
    (method,) = [
        fn
        for cls in module.body
        if isinstance(cls, ast.ClassDef) and cls.name == "MooncakeConnectorScheduler"
        for fn in cls.body
        if isinstance(fn, ast.FunctionDef) and fn.name == "update_state_after_alloc"
    ]
    code = compile(ast.Module([method], []), str(path), "exec")
    (body,) = [c for c in code.co_consts if isinstance(c, CodeType)]
    run = FunctionType(body, {"logger": logging.getLogger(__name__)})
    scheduler = SimpleNamespace(
        is_producer=False,
        hash_block_size=own,
        _reqs_need_recv={},
        transfer_id_to_request_id={},
        request_id_to_transfer_id={},
    )
    seq = SimpleNamespace(
        id=0,
        kv_transfer_params={**blob, "do_remote_prefill": True},
        block_table=list(range(8)),
        has_per_req_cache=False,
        num_cached_tokens=4 * own,
    )
    run(scheduler, seq)
    return seq.kv_transfer_params["num_computed_blocks"]


def test_the_consumer_takes_the_full_transfer_unless_hash_block_size_matches():
    """A moriio blob, which carries no `hash_block_size`, and a mooncake blob
    with a different one both give `num_computed_blocks = 0`. The matching size
    is the control that shows the incremental path is reachable at all."""
    own = 16
    mooncake = _emitted("mooncake")
    assert _consumer_offset(_emitted("moriio"), own) == 0
    assert _consumer_offset({**mooncake, "hash_block_size": 2 * own}, own) == 0
    assert _consumer_offset({**mooncake, "hash_block_size": own}, own) == 4

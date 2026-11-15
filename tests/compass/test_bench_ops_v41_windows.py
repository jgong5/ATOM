# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's DSpark window read over its FP8 pool under `compass measure
ops`: it takes the ring rows' addresses and pointers ATOM builds for its
drafter's stage layers at the step's state slots, and reads a ring row an
address.
"""

from types import SimpleNamespace

import torch
from test_bench_ops import _graph, _leaf, _Runner, _t
from test_capture_packed_window import _own_process

from atom.compass.bench import ops
from atom.compass.capture import driverless
from atom.utils import forward_context as fctx


def test_the_drafter_s_window_read_takes_its_stage_layers_rows_at_the_step_s_slots(
    monkeypatch, request, tmp_path
):
    # V4.1's FP8 pool of 2 pages and 4 state slots, a ring of 5 rows of 33
    # bytes a layer; the drafter's stages read layers 2 and 1, a decode's 2
    # rows in slots 3 and 1: 20 ring rows, each by its byte address tagged FP8
    if _own_process(request):
        return
    driverless.install(str(tmp_path), "gfx1250")
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import (
        V41PoolGeometry,
    )
    from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec

    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    geometry = V41PoolGeometry(
        3, ((0, 1),), 32, 4, 32, 32, kv_dtype="fp8", speculative_tokens=1
    )
    cache, slots = PagedAttentionCache(geometry, 2, 4, "cpu"), torch.tensor([3, 1])
    runner = _Runner()
    runner.attn_metadata_builder = SimpleNamespace(geometry=geometry)
    runner.model.append(torch.nn.Module())
    runner.model[-1].spec = LayerAttentionSpec(0, 0, AttentionMode.FULL)
    stages = SimpleNamespace(_context_layer_ids=lambda d: torch.tensor([2, 1]))
    runner.drafter, prepare = SimpleNamespace(model=stages), runner.prepare_model

    def prepare_model(batch):
        out = prepare(batch)
        md = fctx._forward_context.attn_metadata
        vars(md).update(cache=cache, state_slot_out=slots.int())
        return out

    i32, i64 = "torch.int32", "torch.int64"
    pool = _t([cache.backing.numel()], "kv", dtype="torch.uint8")
    args = ("addresses", _t([20], "activation", dtype=i64)), ("first", 0)
    args += ("ptr", _t([2], "activation", dtype=i32)), ("end", 1)
    leaf = _leaf(ops._WINDOWS, ("pool", pool), *args)
    runner.prepare_model, stub = prepare_model, ops.Stub(runner, 10_000)
    stub.leaves = ops.leaves(_graph(leaf, layers=1)["region"])
    stub._prepare((0, 0))

    [(sig, (op, instances))] = stub.leaves.items()
    [(_, bound)] = stub.placed
    ring = {layer: geometry.window_bytes(layer, 2).ring_start for layer in (2, 1)}
    rows = [
        ring[layer] + slot * geometry.state_bytes + 33 * i
        for layer in (2, 1)
        for slot in (3, 1)
        for i in range(5)
    ]
    assert bound[sig, "addresses"][0] == "read_windows.addresses"
    assert bound[sig, "addresses"][1].tolist() == [2 * row + 1 for row in rows]
    assert bound[sig, "ptr"][1].tolist() == [0, 20]
    # each a row of its slot's ring, 33 bytes
    assert stub._plan(op, instances, {}, 2, 2, None, sig, True)[3] == [0, 20 * 33]

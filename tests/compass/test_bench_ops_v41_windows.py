# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's DSpark window read over its FP8 pool under `compass measure
ops`: it takes the ring rows' addresses and pointers ATOM builds for its
drafter's stage layers at the step's state slots, and reads a ring row an
address.
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import torch
from test_bench_ops import _graph, _leaf, _Runner, _t

from atom.compass.bench import ops
from atom.compass.capture import driverless
from atom.utils import forward_context as fctx

#: The window read.
READ = "atom.model_ops.attentions.deepseek_v41.packed_rows:gather_prefix_rows"


def _read(runner, pool):
    """`runner` over V4.1's FP8 pool, a ring of 5 rows of 33 bytes a layer,
    and the leaves of a window read of 20 rows from a pool of `pool` bytes."""
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import (
        V41PoolGeometry,
    )
    from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec

    geometry = V41PoolGeometry(
        3, ((0, 1),), 32, 4, 32, 32, kv_dtype="fp8", speculative_tokens=1
    )
    runner.attn_metadata_builder = SimpleNamespace(geometry=geometry)
    runner.model.append(torch.nn.Module())
    runner.model[-1].spec = LayerAttentionSpec(0, 0, AttentionMode.FULL)
    args = ("addresses", _t([20], "activation", dtype="torch.int64")), ("first", 0)
    args += ("ptr", _t([2], "activation", dtype="torch.int32")), ("end", 1)
    leaf = _leaf(READ, ("pool", _t([pool], "kv", dtype="torch.uint8")), *args)
    return geometry, ops.leaves(_graph(leaf, layers=1)["region"])


def test_the_drafter_s_window_read_takes_its_stage_layers_rows_at_the_step_s_slots(
    monkeypatch, request, tmp_path
):
    # the drafter's stages read layers 2 and 1, a decode's 2 rows in state
    # slots 3 and 1 of 4: 20 ring rows, each by its byte address tagged FP8
    if "COMPASS_ARCH_DECLARED" not in os.environ:
        # V4.1's cache imports aiter, which needs an arch declared: once per
        # process, so in a process of its own.
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
            + [f"{__file__}::{request.node.name}"],
            env={**os.environ, "COMPASS_ARCH_DECLARED": "1"},
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        assert done.returncode == 0, done.stdout[-4000:]
        return
    driverless.install(str(tmp_path), "gfx1250")
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache

    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    runner, slots = _Runner(), torch.tensor([3, 1], dtype=torch.int32)
    geometry, found = _read(runner, 1)
    cache, prepare = PagedAttentionCache(geometry, 2, 4, "cpu"), runner.prepare_model
    runner.drafter = SimpleNamespace(model=torch.nn.Module())
    runner.drafter.model._context_layer_ids = lambda device: torch.tensor([2, 1])

    def prepare_model(batch):
        out = prepare(batch)
        md = fctx._forward_context.attn_metadata
        vars(md).update(cache=cache, state_slot_out=slots)
        return out

    runner.prepare_model, stub = prepare_model, ops.Stub(runner, 10_000)
    stub.leaves = found
    stub._prepare((0, 0))

    [sig] = found
    [(_, bound)] = stub.placed
    assert {p: name for (s, p), (name, _) in bound.items() if s == sig} == {
        "addresses": "read_windows.addresses",
        "ptr": "read_windows.ptr",
    }
    ring = {layer: geometry.window_bytes(layer, 2).ring_start for layer in (2, 1)}
    rows = [
        ring[layer] + slot * geometry.state_bytes + 33 * i
        for layer in (2, 1)
        for slot in (3, 1)
        for i in range(5)
    ]
    assert bound[sig, "addresses"][1].tolist() == [2 * row + 1 for row in rows]
    assert bound[sig, "ptr"][1].tolist() == [0, 20]


def test_the_drafter_s_window_read_reads_a_ring_row_an_address_at_its_slots():
    # 20 rows of 33 bytes, each in the ring of a state slot the point's rows
    # hold, from a pool of 999 bytes on 2 blocks
    runner = _Runner()
    _, found = _read(runner, 999)
    [(sig, (op, instances))] = found.items()
    plan = ops.Stub(runner, 10_000)._plan(op, instances, {}, 2, 2, None, sig, True)
    assert plan[3] == [0, 20 * 33]

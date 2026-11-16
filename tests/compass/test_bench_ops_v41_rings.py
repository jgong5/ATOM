# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's window writes and window index builds under `compass
measure ops`: each takes its layer's window ring of the point's pool, where
ATOM's forward places it, not where the capture's pool placed it.
"""

import dataclasses
from types import SimpleNamespace

import pytest
from test_bench_ops import _graph, _leaf, _Runner

from atom.compass.bench import ops
from atom.compass.ir.nodes import Reduced
from atom.model_ops.attentions.pool_layout.v4_pool_geometry import WindowParams
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.utils import forward_context as fctx

#: A window write, which takes its layer's ring.
WRITE = "atom.model_ops.attentions.deepseek_v41.packed_rows:_launch_write_window"
#: The pages of the capture's pool, and of the point's.
CAPTURED, POINT = 2, 7
#: A window ring's class, as the capture names it.
RING = f"{WindowParams.__module__}:{WindowParams.__qualname__}"


def _geometry(kv_dtype):
    """V4.1's pool of 3 layers, its ring 5 rows."""
    return V41PoolGeometry(
        3, ((0, 1),), 32, 4, 128, 32, kv_dtype=kv_dtype, speculative_tokens=1
    )


def _taken(windows, cache):
    """Copy 0's `window` of a window write given each of `windows` as the
    capture states it, on a point whose forward context holds `cache`."""
    leaves = [
        _leaf(WRITE, ("window", Reduced(RING, tuple(dataclasses.asdict(w).items()))))
        for w in windows
    ]
    runner = _Runner()
    prepare = runner.prepare_model

    def prepare_model(batch):
        out = prepare(batch)
        vars(fctx._forward_context.attn_metadata).update(cache=cache)
        return out

    runner.prepare_model, stub = prepare_model, ops.Stub(runner, 10_000)
    stub.leaves = ops.leaves(_graph(*leaves, layers=1)["region"])
    stub._prepare((0, 0))
    return [
        stub._copy(op, sig, instances, 0)[0]["window"]
        for sig, (op, instances) in stub.leaves.items()
    ]


@pytest.mark.parametrize(
    "kv_dtype, form",
    [("fp8", "window_bytes"), ("fp8", "window"), ("bf16", "window")],
)
def test_a_v41_window_ring_is_its_layer_s_of_the_point_s_pool(kv_dtype, form):
    # each layer's ring as a 2-page pool placed it, taken on a 7-page pool
    geometry = _geometry(kv_dtype)
    ring = getattr(geometry, form)
    cache = SimpleNamespace(geometry=geometry, num_pages=POINT)

    got = _taken([ring(layer, CAPTURED) for layer in range(3)], cache)

    assert got == [ring(layer, POINT) for layer in range(3)]


def test_a_ring_where_no_layer_s_starts_refuses_its_leaf():
    geometry = _geometry("fp8")
    cache = SimpleNamespace(geometry=geometry, num_pages=POINT)
    ring = geometry.window_bytes(1, CAPTURED)
    off = dataclasses.replace(ring, ring_start=ring.ring_start + 1)

    with pytest.raises(ValueError, match="no layers' window rings start where"):
        _taken([off], cache)


def test_a_ring_beside_no_v41_cache_is_taken_as_captured():
    # DeepSeek-V4's rings start inside a state slot, wherever its pool ends
    captured = _geometry("fp8").window_bytes(2, CAPTURED)

    assert _taken([captured], None) == [captured]

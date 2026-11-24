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


def _taken(windows, cache, kept=None):
    """Copy 0's `window` of a window write given each of `windows` as the
    capture states it, on a point whose forward context holds `cache`: of each
    leaf of their graph, or of those at `kept` alone, the graph's leaves left
    to measure."""
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
    found = list(ops.leaves(_graph(*leaves, layers=1)["region"]).items())
    stub.leaves = dict(found[i] for i in kept or range(len(found)))
    stub._prepare((0, 0))
    return [
        stub._copy(op, sig, instances, 0)[0]["window"]
        for sig, (op, instances) in stub.leaves.items()
    ]


def test_a_v41_window_ring_is_its_layer_s_of_the_point_s_pool():
    # each layer's ring as a 2-page pool placed it, taken on a 7-page pool
    geometry = _geometry("bf16")
    ring = geometry.window
    cache = SimpleNamespace(geometry=geometry, num_pages=POINT)

    got = _taken([ring(layer, CAPTURED) for layer in range(3)], cache)

    assert got == [ring(layer, POINT) for layer in range(3)]


@pytest.mark.parametrize("kept", [[0], [0, 1], [1, 2]], ids=["0", "0-1", "1-2"])
def test_a_point_holding_some_of_a_graph_s_rings_takes_the_graph_s_pool(kept):
    # the graph's other leaves measured already: those kept take their layers'
    # rings on the point's pool
    geometry = _geometry("bf16")
    ring = geometry.window
    cache = SimpleNamespace(geometry=geometry, num_pages=POINT)

    got = _taken([ring(layer, CAPTURED) for layer in range(3)], cache, kept)

    assert got == [ring(layer, POINT) for layer in kept]


def test_a_ring_no_layer_s_refuses_its_leaf():
    geometry = _geometry("bf16")
    ring = geometry.window(1, CAPTURED)
    off = dataclasses.replace(ring, ring_start=ring.ring_start + 1)
    cache = SimpleNamespace(geometry=geometry, num_pages=POINT)

    with pytest.raises(ValueError, match="no layers' window rings start where"):
        _taken([off], cache)


def test_an_fp8_window_s_ring_on_its_own_planes_is_taken_as_captured():
    # its ring starts at its planes' first row, whatever the pool's pages and
    # whichever its layer
    geometry = _geometry("fp8")
    captured = [geometry.window(2, CAPTURED)]
    cache = SimpleNamespace(geometry=geometry, num_pages=POINT)

    assert _taken(captured, cache) == captured


def test_a_ring_beside_no_v41_cache_is_taken_as_captured():
    # DeepSeek-V4's rings start inside a state slot, wherever its pool ends
    captured = _geometry("fp8").window(2, CAPTURED)

    assert _taken([captured], None) == [captured]

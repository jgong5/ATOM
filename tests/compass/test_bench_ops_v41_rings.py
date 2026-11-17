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


def _coinciding():
    """V4.1's FP8 pool of 6 layers and no drafter, 4 layers' rings a page:
    layer L's ring on P pages starts where layer L + 4's does on P - 1."""
    return V41PoolGeometry(6, ((0, 1),), 16, 16, 128, 32, kv_dtype="fp8")


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


@pytest.mark.parametrize("form", ["window_bytes", "window"])
def test_rings_that_coincide_take_the_one_pool_that_holds_all_the_point_s(form):
    # layer 0's ring on 2 pages starts where layer 4's does on 1, and layer 5's
    # on 2 where layer 1's does on 3: only 2 pages hold each of the six
    geometry = _coinciding()
    ring = getattr(geometry, form)
    cache = SimpleNamespace(geometry=geometry, num_pages=POINT)
    assert ring(0, CAPTURED) == ring(4, CAPTURED - 1)

    got = _taken([ring(layer, CAPTURED) for layer in range(6)], cache)

    assert got == [ring(layer, POINT) for layer in range(6)]


@pytest.mark.parametrize("form", ["window_bytes", "window"])
@pytest.mark.parametrize("kept", [[0], [0, 1], [4, 5]], ids=["0", "0-1", "4-5"])
def test_a_point_holding_some_of_a_graph_s_rings_takes_the_graph_s_pool(form, kept):
    # the graph's other leaves measured already: alone, layers 0 and 1's rings
    # on 2 pages are 4 and 5's on 1, and 4 and 5's are 0 and 1's on 3
    geometry = _coinciding()
    ring = getattr(geometry, form)
    cache = SimpleNamespace(geometry=geometry, num_pages=POINT)

    got = _taken([ring(layer, CAPTURED) for layer in range(6)], cache, kept)

    assert got == [ring(layer, POINT) for layer in kept]


def _off(geometry):
    ring = geometry.window_bytes(1, CAPTURED)
    return dataclasses.replace(ring, ring_start=ring.ring_start + 1)


@pytest.mark.parametrize(
    "geometry, ring, why",
    [
        (_geometry("fp8"), _off, "no layers'"),
        # alone, layer 0's ring on 2 pages is layer 4's on 1
        (_coinciding(), lambda g: g.window_bytes(0, CAPTURED), "2 layers'"),
    ],
    ids=["no-layer", "two-layers"],
)
def test_a_ring_not_one_layer_s_refuses_its_leaf(geometry, ring, why):
    cache = SimpleNamespace(geometry=geometry, num_pages=POINT)

    with pytest.raises(ValueError, match=f"{why} window rings start where"):
        _taken([ring(geometry)], cache)


def test_a_ring_beside_no_v41_cache_is_taken_as_captured():
    # DeepSeek-V4's rings start inside a state slot, wherever its pool ends
    captured = _geometry("fp8").window_bytes(2, CAPTURED)

    assert _taken([captured], None) == [captured]

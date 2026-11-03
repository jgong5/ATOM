# SPDX-License-Identifier: MIT
"""The step bench's MoE routing under expert parallelism: each layer of the
forward context keeps its own routing function, and the top-k ids it returns
are replaced by a uniform draw, the same on every pass and on every rank.

A layer is the attributes `route_uniformly` reads of ATOM's `FusedMoE`, its
own routing function a stand-in called as `FusedMoE.select_experts` calls one.
"""

from types import SimpleNamespace

import pytest
import torch

from atom.compass.bench import routing

EXPERTS, K = 16, 2


class _Route:
    """A layer's own routing: every real row to experts 0 to k - 1, rows from
    `valid` on to no expert (-1), and past top-k `shared` columns of id 99, as
    ATOM's buffer holds fused shared experts; each call counted."""

    def __init__(self, valid=None, shared=0):
        self.valid, self.shared, self.calls = valid, shared, 0

    def __call__(self, hidden_states, gating_output, topk, renormalize):
        self.calls += 1
        rows = gating_output.shape[0]
        ids = torch.arange(topk, dtype=torch.int32).repeat(rows, 1)
        ids[rows if self.valid is None else self.valid :] = -1
        shared = torch.full((rows, self.shared), 99, dtype=torch.int32)
        weights = torch.rand(rows, topk)
        self.weights = weights
        return weights, torch.cat([ids, shared], 1)


def _layer(route=None, experts=EXPERTS, k=K):
    return SimpleNamespace(
        expert_layout=SimpleNamespace(num_routed=experts),
        top_k=k,
        custom_routing_function=route if route is not None else _Route(),
    )


def _routed(layer, rows, k=K):
    """The weights and ids `layer` routes `rows` tokens to."""
    return layer.custom_routing_function(
        hidden_states=torch.zeros(rows, 4),
        gating_output=torch.zeros(rows, EXPERTS),
        topk=k,
        renormalize=True,
    )


def _context():
    """A forward context of two MoE layers and an attention layer."""
    return {
        "model.layers.0.mlp.experts": _layer(),
        "model.layers.0.self_attn.attn": SimpleNamespace(kv_cache=None),
        "model.layers.1.mlp.experts": _layer(),
    }


def test_each_layer_routes_each_token_to_distinct_drawn_experts_on_every_pass():
    context = _context()
    own = {n: getattr(m, "custom_routing_function", None) for n, m in context.items()}
    assert routing.route_uniformly(context, 64, "cpu") == 2

    drawn = {}
    for name in ("model.layers.0.mlp.experts", "model.layers.1.mlp.experts"):
        layer = context[name]
        weights, ids = _routed(layer, 8)
        assert torch.equal(weights, own[name].weights)  # the layer's own weights
        assert ids.dtype == torch.int32 and ids.shape == (8, K)
        assert all(len(set(row)) == K for row in ids.tolist())
        assert 0 <= ids.min() and ids.max() < EXPERTS
        # The same ids on every pass, a shorter pass the leading rows.
        assert torch.equal(_routed(layer, 8)[1], ids)
        assert torch.equal(_routed(layer, 3)[1], ids[:3])
        assert own[name].calls == 3
        drawn[name] = ids
    assert not torch.equal(*drawn.values())  # each layer draws its own
    assert context["model.layers.0.self_attn.attn"].kv_cache is None


def test_every_rank_draws_the_same_ids():
    ranks = [_context(), _context()]
    for context in ranks:
        routing.route_uniformly(context, 64, "cpu")

    for name in ("model.layers.0.mlp.experts", "model.layers.1.mlp.experts"):
        a, b = (_routed(context[name], 64)[1] for context in ranks)
        assert torch.equal(a, b)


def test_a_row_routed_to_no_expert_stays_so_and_shared_columns_are_kept():
    layer = _layer(_Route(valid=5, shared=1))
    routing.route_uniformly({"experts": layer}, 64, "cpu")

    _, ids = _routed(layer, 8)

    assert (ids[:5, :K] >= 0).all() and (ids[5:, :K] == -1).all()
    assert (ids[:, K] == 99).all()


def test_the_draw_is_independent_and_uniform_over_the_routed_experts():
    """V4.1-Flash's 256 experts, top-6. Counts over the experts follow
    chi-squared at 255 degrees of freedom, mean 255 and spread 22.6: a draw
    that favours experts lands above the band, one that deals them out in
    turn, as a balanced router does, below it."""
    experts, k, rows = 256, 6, 4096
    ids = routing.uniform_ids(rows, experts, k, 0, "cpu")

    assert all(len(set(row)) == k for row in ids.tolist())
    counts = torch.bincount(ids.flatten(), minlength=experts).double()
    mean = rows * k / experts
    chi2 = float(((counts - mean) ** 2 / mean).sum())
    assert 255 - 6 * 22.6 < chi2 < 255 + 6 * 22.6
    assert torch.equal(routing.uniform_ids(rows, experts, k, 0, "cpu"), ids)
    assert not torch.equal(routing.uniform_ids(rows, experts, k, 1, "cpu"), ids)


def test_a_layer_with_no_routing_function_of_its_own_refuses_naming_it():
    layer = _layer()
    layer.custom_routing_function = None

    with pytest.raises(ValueError, match="model.layers.3.mlp.experts routes with no"):
        routing.route_uniformly({"model.layers.3.mlp.experts": layer}, 64, "cpu")


def test_a_pass_of_more_rows_than_the_draw_holds_refuses():
    layer = _layer()
    routing.route_uniformly({"experts": layer}, 8, "cpu")

    with pytest.raises(ValueError, match="a pass routes 9 rows, and the bench drew 8"):
        _routed(layer, 9)

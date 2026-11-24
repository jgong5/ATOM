# SPDX-License-Identifier: MIT
"""Uniform MoE routing: the draw both benches take, and the step bench's routing
under expert parallelism.

`uniform_ids` draws each token's top-k expert ids independently and uniformly,
distinct, from a layer's routed experts, seeded: the same ids for the same
arguments on one kind of device. The op bench draws a router's ids with it
(`ops.Stub._route`).

Under expert parallelism a rank's MoE time follows the rows routed to its own
experts, and a replayed batch's filler tokens route neither as the workload's
tokens do nor uniformly. So the step bench routes every MoE layer of the
forward context, the target's and the drafter's, by a draw (`route_uniformly`),
seeded by the layer's name: every rank draws the same ids, and a pass takes the
same ones on every repeat and graph replay. The layer's own routing function
still runs, its kernel and weights included; the draw replaces the top-k ids it
returns, except on a row it routes to no expert, past the pass's real rows,
which stays so. The draw holds the most rows a pass routes, given; a pass of
more refuses.

A layer with no routing function of its own refuses, naming it: ATOM picks
some MoE paths by whether a layer has one (`model_ops/moe.py::Mxfp4MoEMethod.apply`,
`models/qwen4_exp.py::Qwen4ExpSparseMoeBlock`), so setting one could measure a
path the target does not run.
"""

import zlib

import torch


def uniform_ids(rows: int, experts: int, k: int, seed: int, device) -> torch.Tensor:
    """`rows` tokens' top-`k` expert ids, each token's drawn independently and
    uniformly, distinct, from `experts`, by `seed`."""
    generator = torch.Generator(device).manual_seed(seed)
    # double, not the default dtype ATOM's runner sets to the model's: a
    # bfloat16 or float32 draw ties, and the device's top-k breaks a tie
    # differently from call to call
    draw = torch.rand(
        rows, experts, generator=generator, device=device, dtype=torch.double
    )
    return draw.topk(k).indices


class UniformRouting:
    """A MoE layer's routing function `route`, the top-k ids it returns replaced
    by the leading rows of `ids`, where it routes a row to an expert."""

    def __init__(self, route, ids: torch.Tensor) -> None:
        self.route, self.ids = route, ids

    def __call__(self, hidden_states, gating_output, topk, renormalize):
        weights, ids = self.route(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
        )
        rows = ids.shape[0]
        if rows > len(self.ids):
            raise ValueError(
                f"a pass routes {rows} rows, and the bench drew {len(self.ids)}"
            )
        routed = ids[:, :topk]
        routed.copy_(torch.where(routed < 0, routed, self.ids[:rows]))
        return weights, ids


def route_uniformly(layers: dict, rows: int, device) -> int:
    """Each MoE layer of `layers`, ATOM's forward context by name, routed by a
    draw of `rows` tokens over its routed experts, seeded by its name, and how
    many there are; or a refusal naming a layer that has no routing function
    of its own."""
    routed = 0
    for name, layer in layers.items():
        if not hasattr(layer, "custom_routing_function"):
            continue
        route = layer.custom_routing_function
        if route is None:
            raise ValueError(
                f"the MoE layer {name} routes with no routing function of its "
                "own, and ATOM picks some MoE paths by whether a layer has one: "
                "the bench routes uniformly only through a layer's own"
            )
        experts, k = layer.expert_layout.num_routed, layer.top_k
        ids = uniform_ids(rows, experts, k, zlib.crc32(name.encode()), device)
        layer.custom_routing_function = UniformRouting(route, ids.to(torch.int32))
        routed += 1
    return routed

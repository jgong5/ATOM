# SPDX-License-Identifier: MIT
"""A backend prices each step of a run at one graph as a backend new to that
step does, while from the graph's second step on it prices again only what a
step moved: a leaf whose price list answer reads nothing a step moves is not
asked again, attention is at each step that moves its rows' context, and a
data-parallel step's routed experts, whose work reads the group's tokens, only
at a step that moves them. A step whose attention leaves the list's measured
range names its roofline term, and the next step back inside it the list's.
"""

import collections

import pytest
from test_fit_ops import SPEC, _at
from test_op_graph_listed_steps import ATTENTION, GEMM, _attention, _gemm
from test_op_work import I32, leaf, t

from atom.compass.backends import BatchView, RequestShape, op_graph
from atom.compass.backends.op_graph import OpGraphBackend
from atom.compass.fit import ops

DP = 8
MOE = "aiter.fused_moe_.default"
#: Each step's one row's context and its DP group's real tokens: the context
#: moves at every step, past the measured range at 5000, the tokens at the
#: third and the sixth; the last step is the first again.
STEPS = ((1000, 5), (1001, 5), (1002, 6), (2000, 6), (5000, 6), (2001, 7), (1000, 5))


def _experts():
    """The routed experts of the group's rows: 64 of hidden 4096 and
    intermediate 1024 in bf16, top 2, each too large for the kernel floor."""
    return leaf(
        MOE,
        [[DP, 4096]],
        hidden_states=t([DP, 4096]),
        w1=t([64, 2048, 4096], role="weight"),
        w2=t([64, 4096, 1024], role="weight"),
        topk_weight=t([DP, 2], "torch.float32"),
        topk_ids=t([DP, 2], I32),
        quant_type=0,
    )


def _region():
    """Two layers of a GEMM, attention, and the routed experts beside a GEMM,
    each leaf its own object."""

    def layer():
        experts = {"par": [_experts(), _gemm()], "join": "resource_bound"}
        return {"seq": [_gemm(), _attention(), experts]}

    return {"seq": [layer(), layer()]}


def _backend(region, listed: bool) -> OpGraphBackend:
    """A backend of `region`, with a price list of the GEMM flat over context
    and attention moving with it, both measured at 1000 and 3000."""
    price_list = None
    if listed:
        points = [_at(_gemm(), "decode B=1", 5e3, c) for c in (1000, 3000)]
        points += [_at(_attention(), "decode B=1", c * 10.0, c) for c in (1000, 3000)]
        params = ops.fit(points)
        ops.held_out(params)
        price_list = ops.PriceList(params)
    return OpGraphBackend(
        SPEC, lambda *_, **__: region, kv="bf16", dp=DP, price_list=price_list
    )


def _step(context: int, tokens: int) -> BatchView:
    return BatchView((RequestShape(1, context, True),), moe_rows=DP, moe_tokens=tokens)


@pytest.mark.parametrize("listed", [True, False], ids=["price_list", "rooflines"])
def test_a_run_s_steps_price_as_a_backend_new_to_each_prices_it(listed):
    region = _region()
    new = [_backend(region, listed).price(_step(*s)).terms for s in STEPS]
    run = _backend(region, listed)

    assert [run.price(_step(*s)).terms for s in STEPS] == new
    # the steps move the experts' and attention's prices, and the list's names
    by_step = [{t.name: t.seconds for t in terms} for terms in new]
    assert len({got["par.resource_bound"] for got in by_step}) == 3
    if listed:
        assert [f"op.{ATTENTION}" in got for got in by_step] == [
            context > 3000 for context, _ in STEPS
        ]


def test_a_later_step_prices_only_what_it_moved_again(monkeypatch):
    works, work = collections.Counter(), op_graph.work

    def counted(leaf, *args):
        works[leaf["op"]] += 1
        return work(leaf, *args)

    monkeypatch.setattr(op_graph, "work", counted)
    run = _backend(_region(), True)
    listed, asked = run.resolver.sources[0], collections.Counter()
    price = listed.price
    monkeypatch.setattr(
        listed, "price", lambda r: asked.update([r[0]["op"]]) or price(r)
    )
    for s in STEPS:
        run.price(_step(*s))

    # the GEMM at the graph's first two steps, the second planning it;
    # attention at each step but the last, a view priced before
    assert asked == {GEMM: 2, ATTENTION: len(STEPS) - 1}
    # the experts there too, and at each later step that moves the tokens
    assert works[MOE] == 4

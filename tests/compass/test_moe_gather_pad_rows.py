# SPDX-License-Identifier: MIT
"""A padded data-parallel gather's routed experts in both analytical backends.

The gather pads every rank to the rung, and a pad row routes to no expert:
only the gather's real rows (`moe_tokens`) route, and the group's dummy
forwards, each one row of token 0 at position 0, reach one token's experts.
A fused shared expert still takes every gathered row, and the exchanges
still move every row. Each is checked against a count by hand.
"""

from math import comb

import pytest
from test_backend_step_roofline_v4 import BOUND, EP2, EP_OFF, LINK, _module, _price
from test_op_work_v4 import close, experts, moe, triton

from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.op_work import work
from atom.compass.backends.step_roofline import Module
from atom.compass.backends.work import Work, activated_experts, max_rank_load


def gathered(rows, real, dummies=0):
    """This rank's three decode rows in a graph of 4, gathered over two ranks
    into `rows` rows, `real` of them real, `dummies` the group's dummies'."""
    return BatchView(
        (RequestShape(1, 9, True),) * 3,
        capture_rung=4,
        moe_rows=rows,
        moe_tokens=real,
        group_dummies=dummies,
    )


def _exchanges(rows):
    """The gathered hidden states and router logits, and the scatter back."""
    return 2e-6 + rows * (32 + 8) * 2 / 2 / LINK + 2e-6 + rows * 32 * 2 / 2 / LINK


@BOUND
def test_only_the_real_rows_route_and_the_dummies_reach_one_tokens_experts(spec):
    # 7 real rows of 16, three of them dummies': 7 rows' routes through the
    # experts that 5 tokens reach, at 7 rows' share of the rows' tensors.
    moe_ep_off = Module("experts", "moe", EP_OFF, "fp4", ranks=2)
    hit = activated_experts(8, 2, 5)
    want = Work({"bf16": 2 * 14 * (32 + 16) * 32}, hit * 816 + 7 * 144)

    got = _price([moe_ep_off], gathered(16, 7, dummies=3), spec)["roofline.moe"]
    assert got == pytest.approx(_module(want, spec) + _exchanges(16))


@BOUND
def test_under_expert_parallelism_the_busiest_rank_takes_the_real_rows_share(spec):
    # EP2: the busiest rank's share of 5 real rows' routes at the full
    # intermediate; MoRI sends this rank's 4 graph rows as before.
    ep2 = Module("experts", "moe", EP2, "fp4", ranks=2, act="fp8", wire=("fp4", "fp8"))
    load, hit = max_rank_load(5, 2, 8, 2), activated_experts(8, 2, 5)
    row = 32 * (0.5 + 2) + 16
    want = Work({"fp8": 2 * load * (64 + 32) * 32}, hit / 2 * 1632 + load / 2 * row)
    sent = 4 * 32 * 2 * (1 - comb(4, 2) / comb(8, 2))
    exchanges = 2e-6 + sent * 0.5 / 2 / LINK + 2e-6 + sent / 2 / LINK

    got = _price([ep2], gathered(8, 5), spec)["roofline.moe"]
    assert got == pytest.approx(_module(want, spec) + exchanges)


@BOUND
def test_a_fused_shared_expert_still_takes_every_gathered_row(spec):
    # 5 real rows of 8 route to 2 experts each; all 8 take the shared one,
    # whose ids and weights make three a row.
    shared = Module("experts", "moe", EP_OFF + (1,), "fp4", ranks=2)
    hit = activated_experts(8, 2, 5) + 1
    want = Work({"bf16": 2 * (10 + 8) * (32 + 16) * 32}, hit * 816 + 5 * (128 + 24))

    got = _price([shared], gathered(8, 5), spec)["roofline.moe"]
    assert got == pytest.approx(_module(want, spec) + _exchanges(8))


def test_the_op_level_experts_route_only_the_gathers_real_rows():
    # A leaf of 8 gathered rows, 5 real, two of them dummies': 5 rows' routes
    # through the experts 4 tokens reach.
    hit = activated_experts(16, 2, 4)

    close(work(moe(8, 16), gathered(8, 5, dummies=2), 1, "bf16"), experts(8, 10, hit))


def test_the_op_level_triton_experts_route_only_the_gathers_real_rows():
    # EP4, four of sixteen experts a rank: the busiest rank's share of 5 real
    # rows' routes, of the experts 5 tokens reach.
    load, hit = max_rank_load(5, 2, 16, 4), activated_experts(16, 2, 5) / 4
    share = load / 16
    weights = (32 * 64 + 64 * 2, 16 * 64 + 64 * 1)
    io = (8 * 64 + 8 * 2 + 16 * 4 + 16 * 32, 16 * 32 + 8 * 2 + 16 * 8 + 8 * 64 * 2)
    view = gathered(8, 5)

    up, down = (work(triton("a8", 8, 4, g), view, 1, "bf16", 4) for g in (0, 1))

    for got, flops, held, moved in zip((up, down), (64 * 64, 32 * 64), weights, io):
        close(got, Work({"fp8": 2 * load * flops}, held * hit + moved * share))

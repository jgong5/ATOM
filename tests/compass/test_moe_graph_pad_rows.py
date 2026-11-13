# SPDX-License-Identifier: MIT
"""One rank's padded decode graph in both analytical backends' MoE.

On one rank ATOM runs the experts on the step's own tokens alone: a graph's
pad rows route to no expert and take no fused shared expert, while the
output's all-reduce still moves every row. Each is checked against a count by
hand.
"""

import pytest
from test_backend_step_roofline_v4 import BOUND, EP2, _module, _price
from test_op_work_v4 import close, experts, moe, triton

from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.op_work import work
from atom.compass.backends.step_roofline import Module
from atom.compass.backends.work import (
    Work,
    activated_experts,
    allreduce,
    max_rank_load,
)

#: Three decode tokens replayed in a graph of 8 rows, on one rank.
PADDED = BatchView((RequestShape(1, 9, True),) * 3, capture_rung=8)
#: The 8 rows' output, all-reduced over the module's two ranks.
REDUCE = allreduce(2, 8 * 32 * 2)


@BOUND
def test_under_expert_parallelism_the_busiest_rank_takes_the_tokens_share(spec):
    # EP2 on one data-parallel rank, with a fused shared expert: the busiest
    # rank's share of the 3 tokens' routes, and the shared expert on the 2 of
    # them aiter deals it; the 5 pad rows take neither.
    dims = EP2[:6] + (1,) + EP2[7:] + (1,)
    ep2 = Module("experts", "moe", dims, "fp4", ranks=2, act="fp8", wire=("fp4", "fp8"))
    load, hit = max_rank_load(3, 2, 8, 2), activated_experts(8, 2, 3) / 2 + 1
    row = 32 * (0.5 + 2) + 24
    want = Work({"fp8": 2 * (load + 2) * (64 + 32) * 32}, hit * 1632 + load / 2 * row)

    assert _price([ep2], PADDED, spec)["roofline.moe"] == pytest.approx(
        _module(want + REDUCE, spec)
    )


def test_the_op_level_experts_route_only_the_step_s_own_tokens():
    # A leaf of 8 graph rows, 3 of them the step's tokens: 6 routes through
    # the experts 3 tokens reach, at 3 rows' share of the rows' tensors.
    hit = activated_experts(16, 2, 3)

    close(work(moe(8, 16), PADDED, 1, "bf16"), experts(8, 6, hit))


def test_the_op_level_experts_under_expert_parallelism_take_the_tokens_share():
    # EP4, four of sixteen experts a rank: the busiest rank's share of the 3
    # tokens' routes.
    load, hit = max_rank_load(3, 2, 16, 4), activated_experts(16, 2, 3) / 4

    close(work(moe(8, 4), PADDED, 1, "bf16", 4), experts(8, load, hit))


def test_the_op_level_triton_experts_route_only_the_step_s_own_tokens():
    # EP4: the busiest rank's share of the 3 tokens' routes, of the experts
    # 3 tokens reach.
    load, hit = max_rank_load(3, 2, 16, 4), activated_experts(16, 2, 3) / 4
    share = load / 16
    weights = (32 * 64 + 64 * 2, 16 * 64 + 64 * 1)
    io = (8 * 64 + 8 * 2 + 16 * 4 + 16 * 32, 16 * 32 + 8 * 2 + 16 * 8 + 8 * 64 * 2)

    up, down = (work(triton("a8", 8, 4, g), PADDED, 1, "bf16", 4) for g in (0, 1))

    for got, flops, held, moved in zip((up, down), (64 * 64, 32 * 64), weights, io):
        close(got, Work({"fp8": 2 * load * flops}, held * hit + moved * share))

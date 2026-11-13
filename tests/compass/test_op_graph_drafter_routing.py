# SPDX-License-Identifier: MIT
"""A drafter's experts on one rank, run at each pass's own rows.

ATOM's experts run on the rows the pass hands them (`num_local_tokens`):
`EagleProposer.propose`'s first pass the step's tokens, unpadded, and each
later pass its sequences, one row each, however wide it runs; a block
drafter's block, `DSparkProposer.propose`, its sequences' `drafts` rows each,
however wide it pads. The op-level backend states each pass by its place in
the drafter's graph, not by a leaf's rows. Each pass's leaves are checked
against their work at a view of the rows that pass runs.
"""

import pytest
from test_op_graph import by_name, g
from test_op_work import F32, I32, leaf, t
from test_op_work_v4 import FP4, SPEC

from atom.compass.backends import BatchView, CostRefused, RequestShape
from atom.compass.backends.cost import fold_seconds
from atom.compass.backends.op_graph import KV, OpGraphBackend
from atom.compass.backends.op_work import work
from atom.compass.backends.work import price

U8 = "torch.uint8"
MOE = "draft op.aiter.fused_moe_.default"
#: Three rows, each verifying two drafts, replayed at rung 4: the drafter's
#: later pass runs at 4 rows.
MTP2 = BatchView((RequestShape(3, 90, True),) * 3, capture_rung=4, draft_bs=4)
#: Three rows, each verifying three drafts, replayed at rung 4: the block runs
#: at the rung's 12 rows.
BLOCK3 = BatchView((RequestShape(4, 90, True),) * 3, capture_rung=4)


def moe(rows):
    """V4-Pro's routed experts at TP8 over `rows` rows, as a trace captures
    them: top 6 of 384, hidden 7168, fp4 weights. Their weights outweigh a
    kernel's floor, so the experts its routes reach set its price."""
    return leaf(
        "aiter.fused_moe_.default",
        [[rows, 7168]],
        hidden_states=t([rows, 7168]),
        w1=t([384, 768, 3584], FP4, role="weight"),
        w2=t([384, 7168, 192], FP4, role="weight"),
        w1_scale=t([294912, 224], U8, role="weight"),
        w2_scale=t([2752512, 16], U8, role="weight"),
        topk_weight=t([rows, 6], F32),
        topk_ids=t([rows, 6], I32),
        quant_type=3,
    )


def backend(drafter, **kwargs):
    """The op-level backend over a target of one GEMM, a block drafter's
    draft-KV pass of another, and `drafter`, the drafter's graph."""
    regions = {False: g(12, 64, 64), True: drafter, KV: g(12, 32, 64)}
    return OpGraphBackend(
        SPEC, lambda *_, draft=False, **__: regions[draft], kv="bf16", **kwargs
    )


def ran(*leaves):
    """Each ``(leaf, view)``'s work priced as one leaf, folded in run order."""
    return fold_seconds(price(work(op, v, 1, "bf16"), SPEC) for op, v in leaves)


def tokens(n, q):
    """A view of `n` decode rows of `q` tokens each, as a pass runs them."""
    return BatchView((RequestShape(q, 90, True),) * n)


def test_an_mtp_drafter_s_later_pass_routes_only_the_step_s_sequences():
    # The first pass runs the 9 tokens and routes them all; the later pass
    # runs 4 rows, of which ATOM hands the experts the 3 sequences'.
    cost = backend({"seq": [moe(9), g(9, 64, 64), moe(4)]}, drafts=2)

    first = by_name(cost.price(MTP2))[MOE]
    assert first == pytest.approx(ran((moe(9), MTP2), (moe(4), tokens(3, 1))))
    # A step of a graph priced before reads each pass's rows again.
    assert by_name(cost.price(MTP2))[MOE] == first


def test_a_later_pass_as_wide_as_the_first_is_told_apart_by_its_place():
    # One row verifying two drafts, its later passes at 3 rows: both passes'
    # leaves hold 3 rows, the first routing 3 tokens and the later 1.
    view = BatchView((RequestShape(3, 90, True),), draft_bs=3)
    cost = backend({"seq": [moe(3), g(3, 64, 64), moe(3)]}, drafts=2)

    assert by_name(cost.price(view))[MOE] == pytest.approx(
        ran((moe(3), view), (moe(3), tokens(1, 1)))
    )


def test_a_block_drafter_s_block_routes_its_sequences_drafts_each():
    # Each of the block's three layers runs the rung's 4 rows of 3 drafts, of
    # which ATOM hands the experts the 3 sequences' 9.
    cost = backend({"seq": [moe(12), moe(12), moe(12)]}, drafts=3, block=True)

    assert by_name(cost.price(BLOCK3))[MOE] == pytest.approx(
        ran(*[(moe(12), tokens(3, 3))] * 3)
    )


#: Two passes of one region, a repeat: its leaf runs in both.
SHARED = {"repeat": moe(9), "count": 2, "index": "l0", "start": 0, "step": 1}


@pytest.mark.parametrize(
    "drafter, count",
    [({"seq": [moe(9), moe(4), moe(4)]}, 3), (SHARED, 2)],
    ids=["uneven", "shared"],
)
def test_an_mtp_drafter_whose_passes_its_graph_does_not_tell_apart_is_refused(
    drafter, count
):
    with pytest.raises(CostRefused, match=f"{count} routed leaves do not run as"):
        backend(drafter, drafts=2).price(MTP2)


@pytest.mark.parametrize(
    "view",
    [
        BatchView((RequestShape(9, 9, False),), produces_output=False),
        BatchView(MTP2.requests, moe_rows=72, moe_tokens=72, dp_rank=0),
    ],
    ids=["middle_chunk", "data_parallel"],
)
def test_a_drafter_that_samples_nothing_or_gathers_routes_as_the_step_does(view):
    # A middle chunk's drafter writes its draft KV over the chunk's 9 tokens;
    # a gather's real rows are its own. Neither is told apart by pass.
    cost = backend({"seq": [moe(9)] * 3}, drafts=2)

    assert by_name(cost.price(view))[MOE] == pytest.approx(ran(*[(moe(9), view)] * 3))

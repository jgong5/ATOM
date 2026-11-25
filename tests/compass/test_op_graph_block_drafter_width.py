# SPDX-License-Identifier: MIT
"""A block drafter's propose on a decode padded to a rung, asked at the rows
each of its leaves ran.

`DSparkProposer.propose` runs its block at the step's `running_bs`, the rung
its target replayed, `drafts` tokens a row, and stages its anchors from the
rows the target verified. DeepSeek-V4.1-Flash's DSpark5 at a decode of 9 rows
replayed at rung 16 runs its block's GEMMs over 80 rows, gates its MoE beside
a one-element count of its valid rows, and stages its 9 anchors beside the
target's 96 positions. `measure ops --plan` measures the block's leaves at the
rung and the anchors at the step's own shape, which `fit ops` reads at its
rows.
"""

import pytest
from test_op_graph import by_name, g, listed, pt
from test_op_graph_target_width import ARGMAX, Asked, argmax
from test_op_work import F32, I32, leaf, t
from test_work import SPEC

from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.op_graph import OpGraphBackend
from atom.compass.backends.provenance import Method

I64 = "torch.int64"
ANCHORS = "atom.spec_decode.dspark_proposer:DSparkProposer._stage_anchors"
COMMIT = "atom.model_ops.attentions.deepseek_v41.speculative:_launch_commit_cursors"
GATING = "aiter.topk_gating_fwd.default"
GEMM = "aiter.gemm_a16w16.default"
#: DSpark5: a decode row verifies 5 drafts and one.
DRAFTS = 5


def anchors(rows, width):
    """`rows` anchors staged into the block, from the target's positions at
    `width`."""
    return leaf(
        ANCHORS,
        next_token_ids=t([rows], I32),
        positions=t([width * (DRAFTS + 1)], I64, role="metadata"),
        rows=t([rows], I32),
    )


def commit(rows):
    """The cursor commit of `rows` requests."""
    return leaf(
        COMMIT,
        cursor=t([128, 4], I64, role="kv"),
        cursors=t([rows, DRAFTS + 1, 4], I64, role="kv"),
        anchors=t([rows], I32),
        starts=t([rows], I32, role="metadata"),
        slots=t([rows], I32, role="metadata"),
        count=rows,
    )


def gating(tokens):
    """The block's MoE gating over `tokens`, beside its valid rows' count."""
    return leaf(
        GATING,
        topk_weights=t([tokens, 3], F32),
        topk_indices=t([tokens, 3], I32),
        gating_output=t([tokens, 128]),
        correction_bias=t([128], F32, role="weight"),
        num_valid_rows=t([1], I32, role="metadata"),
    )


def drafter(rows, width, dp=1):
    """The drafter's graph at `decode T=rows B=width`, its block's tokens
    gathered over `dp` ranks in its MoE."""
    block = width * DRAFTS
    return {
        "seq": [
            anchors(rows, width),
            g(block, 64, 64),
            g(dp * block, 32, 64),
            gating(block),
            argmax(width),
            commit(rows),
        ]
    }


def step(rows, rung, **kwargs):
    """A decode of `rows` rows of 6 tokens over 500 of context, replayed at
    `rung`."""
    decode = (RequestShape(DRAFTS + 1, 500, True),) * rows
    return BatchView(decode, capture_rung=rung, produces_output=True, **kwargs)


def priced(graphs, price_list, dp=1):
    """A DSpark5 run's op-level backend over `graphs`, the drafter's under
    True, its target's and its draft-KV pass's empty."""
    return OpGraphBackend(
        SPEC,
        lambda *_, draft=False, **__: graphs.get(draft, {"seq": []}),
        kv="bf16",
        dp=dp,
        drafts=DRAFTS,
        block=True,
        price_list=price_list,
    )


@pytest.mark.parametrize(
    "rows,rung,dp,wide",
    [
        (9, 16, 1, 16),
        # A one-element count of valid rows is not its one row.
        (1, 2, 1, 2),
        # Its MoE gathers both ranks' blocks.
        (9, 16, 2, 16),
        # A decode that replayed no graph runs its block at its rows.
        (9, None, 1, 9),
    ],
    ids=["nine-rows-at-rung-16", "one-row-at-rung-2", "dp2", "eager"],
)
def test_a_padded_decode_asks_its_block_drafter_at_its_block_s_width(
    rows, rung, dp, wide
):
    listing = Asked()
    graphs = {True: drafter(rows, rung or rows, dp)}
    gathered = dp * (rung or rows) * (DRAFTS + 1) if dp > 1 else None
    view = step(rows, rung, moe_rows=gathered)

    priced(graphs, listing, dp).price(view)

    block, at_rows, at_width = (rung or rows) * DRAFTS, (rows, rows), (wide, wide)
    assert listing.asked == [
        (ANCHORS, rows, *at_rows),
        (GEMM, block, *at_width),
        (GEMM, dp * block, *at_width),
        (GATING, block, *at_width),
        (ARGMAX, rung or rows, *at_width),
        (COMMIT, 128, *at_rows),
    ]


def test_a_padded_decode_prices_its_block_drafter_at_the_points_plan_measured():
    # The plan measures the block's leaves on the drafter's graph at rung 16
    # and the anchors and commit at its own shape, each at two contexts it
    # does not move with.
    block, own = drafter(16, 16)["seq"][1:5], [anchors(9, 16), commit(9)]
    points = [
        pt(op, shape, ns, context)
        for ops, shape in ((block, "decode B=16"), (own, "decode T=9 B=16"))
        for ns, op in enumerate(ops, start=1)
        for context in (100, 1000)
    ]
    graphs = {True: drafter(9, 16)}

    cost = priced(graphs, listed(*points)).price(step(9, 16))

    # Each leaf at a measured point of its own: its block's at the rung's,
    # its anchors' and commit's at its rows'.
    assert {term.provenance.method for term in cost.draft_terms} == {Method.MEASURED}
    assert by_name(cost) == {
        f"draft op.{ANCHORS}.price_list": pytest.approx(1e-9),
        f"draft op.{GEMM}.price_list": pytest.approx(3e-9),
        f"draft op.{GATING}.price_list": pytest.approx(3e-9),
        f"draft op.{ARGMAX}.price_list": pytest.approx(4e-9),
        f"draft op.{COMMIT}.price_list": pytest.approx(2e-9),
    }

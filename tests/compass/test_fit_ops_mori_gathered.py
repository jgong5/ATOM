# SPDX-License-Identifier: MIT
"""MORI's dispatch and combine at a DP-attention shape stating G: a skewed
exchange takes about what its largest sender sends, not G, so its points are a
family of their own over EP times that sender's rows, which `fit ops` fits,
`OpGraphBackend` prices a step stating G at by the most tokens a rank of its
group scheduled, and `measure ops --plan` does not densify."""

import json

import pytest
from test_op_graph import by_name
from test_op_work import leaf, t
from test_work import SPEC

from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.op_graph import OpGraphBackend
from atom.compass.bench import ops as bench
from atom.compass.capture.peers import MORI
from atom.compass.fit import ops as fit

DP = EP = 8
HIDDEN, TOP_K = 7168, 6
#: A bf16 row's bytes, what a dispatch's output moves a row.
ROW = HIDDEN * 2
KERNEL = "EpDispatchIntraNodeKernel_bf16"
DISPATCH = f"{MORI}.dispatch"


def _dispatch(rows, gathered=None):
    """An EP8 dispatch of this rank's `rows` as the capture states it in a DP
    group of 8 gathering `gathered` rows, by default every rank's `rows`
    (`peers.install_mori`): its output each row a rank receives."""
    n = gathered or DP * rows
    return leaf(
        DISPATCH,
        [[n, HIDDEN], [n, TOP_K], [n, TOP_K], [1]],
        input=t([rows, HIDDEN]),
        weights=t([rows, TOP_K], "torch.float32"),
        scales=None,
        indices=t([rows, TOP_K], "torch.int32"),
        block_num=128,
        warp_per_block=16,
        num_ep_ranks=EP,
    )


def _point(op, shape, ns):
    """`measure ops`' point of `op` at `shape`, its `exchange` as it stores it."""
    point = {"shape": shape, "context": None, "op": op["op"], "cold_ns": ns}
    point |= {"leaf": bench.signature(op), "kernels": [KERNEL]}
    return point | {"exchange": bench.exchange(op)}


#: Rank 0 prefills 4608 tokens and each peer one: its G point, beside the
#: shapes stating no G at 512 and 4608 tokens a rank.
POINTS = [
    _point(_dispatch(512), "prefill T=512 B=1", 100e3),
    _point(_dispatch(4608), "prefill T=4608 B=1", 800e3),
    _point(_dispatch(4608, 4615), "prefill T=4608 B=1 G=4615", 890e3),
]


def _listed(region):
    params = fit.fit(POINTS)
    fit.held_out(params)
    listed = fit.PriceList(params)
    return params, OpGraphBackend(
        SPEC, lambda *_, **__: region, kv="bf16", dp=DP, ep=EP, price_list=listed
    )


def test_a_g_point_sits_at_ep_times_its_largest_rank_s_rows_and_a_g_step_there():
    params, backend = _listed(_dispatch(1, 4615))
    # rank 0 runs one token; a peer the 4608 rank 0 ran at the G point
    step = BatchView(
        (RequestShape(1, 1, False),),
        moe_rows=4615,
        group_max_tokens=4608,
        produces_output=True,
        dp_rank=0,
    )

    apart = [
        f["points"]
        for name, f in params["families"].items()
        if name.startswith('["exchange G"')
    ]

    assert apart == [[[[EP * 4608 * ROW], 890e3, [KERNEL]]]]
    assert by_name(backend.price(step)) == {
        f"op.{DISPATCH}.price_list": pytest.approx(890e-6)
    }


def test_each_micro_batch_s_g_exchange_is_priced_at_its_group_s_largest_rank():
    _, backend = _listed(_dispatch(1, 4615))
    # rank 0 runs a token in each micro-batch; a peer 4608 in each, another
    # peer in each
    split = ((1, 4608, 1, 1, 1, 1, 1, 1), (1, 1, 4608, 1, 1, 1, 1, 1))
    step = BatchView(
        (RequestShape(2, 2, False),),
        moe_rows=9230,
        group_max_tokens=4609,
        produces_output=False,
        tbo_tokens=split,
        dp_rank=0,
    )

    # both on the comm stream, one after the other
    assert by_name(backend.price(step)) == {
        "op.tbo.price_list": pytest.approx(2 * 890e-6)
    }


def test_the_plan_densifies_no_mori_family_whose_shapes_state_g():
    # the line stating no G bends at 200 tokens a rank; its G points, which
    # would bend it too, lie on no line it is halved over
    records = [
        _point(_dispatch(t), f"prefill T={t} B=1", ns)
        for t, ns in ((100, 100e3), (200, 300e3), (300, 300e3))
    ]
    records += [
        _point(_dispatch(t, g), f"prefill T={t} B=1 G={g}", ns)
        for t, g, ns in ((150, 157, 160e3), (250, 260, 140e3), (290, 297, 290e3))
    ]
    families = bench._families(records, bench._dependent(records))

    want = bench._densify(families, set())

    assert sorted(str(shape) for shape, _ in want) == [
        "prefill T=150 B=1",
        "prefill T=250 B=1",
    ]
    assert {json.loads(f)[0] for f in set().union(*want.values())} == {"exchange"}

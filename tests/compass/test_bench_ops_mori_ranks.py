# SPDX-License-Identifier: MIT
"""`compass measure ops` on MORI's dispatch and combine over the ranks of a
DP-attention layout: where a step's DP group gathers G rows, each rank sends
its own share of them, as the capture split them, and not rank 0's rows, the
capture's, from every rank."""

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from test_bench_ops import _graph, _t
from test_bench_ops_mori import HIDDEN, TOP_K, _atom, _Device, _Handle, _Moe, _stub
from test_bench_ops_ranks import _config, _leaves, _measuring_rank

from atom.compass import ir
from atom.compass.bench import ops, ranks
from atom.compass.capture.forward import prefill
from atom.compass.capture.peers import MORI
from atom.utils import forward_context as fctx

#: Rank 0's rows at prefill T=13 on two DP ranks, and the rows the group
#: gathers where its peer holds one row.
T, G = 13, 14
LEAVES = ("dispatch", "combine")
NAMES = [f"{MORI}.{leaf}" for leaf in LEAVES]


def _exchanged(name, rows, most, ranks=2):
    """MORI's `name` leaf as the capture states rank 0's call of `rows` rows on
    `ranks` EP ranks, each receiving at most `most` (`peers.install_mori`)."""
    ids = ("indices", _t([rows, TOP_K], "activation", dtype="torch.int32"))
    launch = (("block_num", 64), ("warp_per_block", 8), ("num_ep_ranks", ranks))
    if name == "dispatch":
        sent = _t([rows, HIDDEN], "activation", dtype="torch.bfloat16")
        weights = ("weights", _t([rows, TOP_K], "activation"))
        args = (("input", sent), weights, ("scales", None), ids, *launch)
        out = ((most, HIDDEN), (most, TOP_K), (most, TOP_K), (1,))
    else:
        got = _t([most, HIDDEN], "activation", dtype="torch.bfloat16")
        args = (("input", got), ("weights", None), ids, *launch)
        out = ((16384, HIDDEN),)
    return ir.Op(f"{MORI}.{name}", "declared", args, out)


class _Exchange:
    """MORI's handle over the gloo group of two ranks: a dispatch adds every
    rank's rows to each rank's count, as a routing of each row to every rank
    does, and a combine copies the counted rows of its input, failing where
    the input holds fewer, where MORI's kernel faults reading past it."""

    def __init__(self):
        self.config = SimpleNamespace(
            world_size=2,
            hidden_dim=HIDDEN,
            num_experts_per_token=TOP_K,
            data_type=torch.bfloat16,
            scale_dim=0,
            scale_type_size=4,
        )
        self.count, self.sent = torch.zeros((), dtype=torch.int64), set()

    def dispatch(self, input, weights, scales, indices, **launch):
        rows = [None, None]
        dist.all_gather_object(rows, len(input))
        self.count += sum(rows)
        self.sent.add(len(input))
        return input, weights, scales, indices, self.count

    def combine(self, input, weights, indices, **launch):
        if len(input) < int(self.count):
            raise RuntimeError(f"{int(self.count)} rows copied from {len(input)}")
        self.count.zero_()
        return input, None


def _exchange_rank(config, todo, gathered, llc, rank):
    """`_measuring_rank` measuring MORI's dispatch and combine of rank 0's T
    rows at prefill T on two DP ranks gathering `gathered` rows (None: each
    rank's T), on `_Exchange`, at a cache of `llc` bytes; with the rows each
    of the rank's dispatches sent."""
    measure, handle = ops.measure, _Exchange()

    def on_dp_group(config, todo, _, rank):
        dp = SimpleNamespace(rank_in_group=dist.get_rank(), world_size=2)
        sys.modules["aiter.dist.parallel_state"].get_dp_group = lambda: dp
        base, _ = _atom(pytest.MonkeyPatch(), SimpleNamespace(_mori_ops=[handle]))
        ops._runner(config, rank).model.append(_Moe(base()))
        most = gathered or 2 * T
        found = ops.leaves(
            _graph(*(_exchanged(n, T, most) for n in LEAVES), layers=1)["region"]
        )
        return measure(config, [(prefill(T, 1, gathered), None, found)], llc, rank)

    ops.measure = on_dp_group
    return _measuring_rank(config, todo, None, rank), handle.sent


@pytest.mark.parametrize(
    ("gathered", "sent"),
    [(G, [{T}, {G - T}]), (None, [{T}, {T}])],
    ids=["G", "every_rank_T"],
)
def test_each_rank_s_mori_exchange_sends_its_share_of_the_rows_its_group_gathers(
    gathered, sent
):
    config = _config(2, dp_attention=True)
    ranks.layout(config)

    by_rank = ranks.launch(_exchange_rank, config, [[], []], gathered, 10_000)

    # The combine copies at most the G rows its input holds: from rank 0's T
    # rows sent from both ranks it would copy 2 T.
    assert [_leaves(records) for records, _ in by_rank] == [
        [(n, None) for n in NAMES]
    ] * 2
    assert [rows for _, rows in by_rank] == sent


def test_every_rank_times_the_warm_graph_where_any_rank_has_several_copies():
    # Under a cache of 100 bytes rank 0's dispatch of 13 rows takes one copy
    # and its peer's of one row several: a rank that alone timed its warm
    # graph would dispatch with no peer to meet.
    config = _config(2, dp_attention=True)
    ranks.layout(config)

    by_rank = ranks.launch(_exchange_rank, config, [[], []], G, 100)

    assert [_leaves(records) for records, _ in by_rank] == [
        [(n, None) for n in NAMES]
    ] * 2
    [first, peer] = [
        [r for r in records if r["op"] == NAMES[0]] for records, _ in by_rank
    ]
    assert first[0]["copies"] == 1 < peer[0]["copies"]
    assert first[0]["warm_ns"] is not None and peer[0]["warm_ns"] is not None


def test_a_mori_leaf_whose_rows_no_split_gives_refuses_by_name_before_any_call(
    monkeypatch,
):
    # Rank 0 sends 13 rows and the leaf states a rank receives at most 12 over
    # two EP ranks: no G a DP group of two gathers gives that.
    device = _Device()
    stub, calls = _stub(monkeypatch, SimpleNamespace(_mori_ops=[_Handle(device)]))
    dp = SimpleNamespace(rank_in_group=0, world_size=2)
    monkeypatch.setitem(
        sys.modules,
        "aiter.dist.parallel_state",
        SimpleNamespace(get_dp_group=lambda: dp),
    )
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    rule = SimpleNamespace(block_size=16, batch=lambda view, block, slot: (block, slot))
    found = ops.leaves(_graph(_exchanged("dispatch", T, 12), layers=1)["region"])

    [record] = stub.point(rule, prefill(T, 1, G), None, found)

    assert record["refused"] == (
        "ValueError: MORI's dispatch states 12 rows a rank receives from 2 EP "
        "ranks, 13 of them this rank's: no split of a DP group of 2 gives that"
    )
    assert calls == [] and device.ran == []

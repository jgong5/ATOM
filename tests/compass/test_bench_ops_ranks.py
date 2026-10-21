# SPDX-License-Identifier: MIT
"""`compass measure ops` over every rank of a layout, with no device.

Ranks are spawned processes that meet in a gloo group at the store `launch`
holds; collective leaves are a family over their message bytes in the plan's
coverage and in `fit ops`.
"""

import contextlib
import datetime
import json
import os
import sys
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from test_bench_ops import _graph, _leaf, _Runner, _t
from test_bench_ops_plan import CONFIG, _records, bench  # noqa: F401  the fixture

from atom.compass import ir
from atom.compass.bench import ops, ranks
from atom.compass.capture.forward import decode, prefill
from atom.compass.fit import ops as fit
from atom.utils import forward_context as fctx

GATHER = "atom.model_ops.moe:all_gather_with_padding"


def _config(tp, dp=1, dp_attention=False, pp=1):
    parallel = SimpleNamespace(
        data_parallel_size=dp,
        data_parallel_size_local=dp,
        data_parallel_rank=0,
        data_parallel_rank_local=0,
        data_parallel_master_ip="127.0.0.1",
        data_parallel_base_port=0,
        _managed_distributed_store=False,
    )
    return SimpleNamespace(
        parallel_config=parallel,
        tensor_parallel_size=tp,
        enable_dp_attention=dp_attention,
        pipeline_parallel_size=pp,
        prefill_context_parallel_size=1,
    )


def test_dp_attention_runs_each_tp_rank_as_a_dp_rank_of_one():
    dpa, tp = _config(8, dp_attention=True), _config(8)

    assert (ranks.layout(dpa), ranks.layout(tp)) == (8, 8)
    assert (dpa.tensor_parallel_size, dpa.parallel_config.data_parallel_size) == (1, 8)
    assert dpa.parallel_config.data_parallel_size_local == 8
    assert (tp.tensor_parallel_size, tp.parallel_config.data_parallel_size) == (8, 1)
    assert ranks.layout(dpa) == 8  # a rewritten layout stays as it is
    with pytest.raises(ValueError, match="pipeline and context parallelism"):
        ranks.layout(_config(1, pp=2))


def _op(name):
    return ({"op": name}, [{}])


#: Each rank's points: rank 1 names a leaf rank 0 does not, and a point of its own.
TODO = [
    [(decode(2), 20, {"a": _op("a"), "b": _op("b")})],
    [(decode(2), 20, {"b": _op("b"), "c": _op("c")}), (prefill(8, 1), None, {})],
]


def _rank(config, todo, rank):
    """One rank: join the gloo group at `launch`'s store, then its union of
    `todo`, and the least and the most of every rank's number."""
    pc = config.parallel_config
    me = pc.data_parallel_rank * config.tensor_parallel_size + rank
    store = dist.TCPStore(pc.data_parallel_master_ip, pc.data_parallel_base_port)
    dist.init_process_group("gloo", store=store, rank=me, world_size=2)
    try:
        group = dist.group.WORLD
        got = [(str(s), c, list(f)) for s, c, f in ranks.union(todo[me], group)]
        agreed = ranks.agreed(10 - me, group), ranks.agreed(10 - me, group, most=True)
        return (pc.data_parallel_rank, rank), got, agreed
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "config",
    [_config(2, dp_attention=True), _config(2)],
    ids=["dp_attention", "tp"],
)
def test_two_ranks_naming_different_points_each_measure_their_union_in_one_order(
    config,
):
    ranks.layout(config)

    by_rank = ranks.launch(_rank, config, TODO)

    union = [
        (str(decode(2)), 20, ["a", "b", "c"]),
        (str(prefill(8, 1)), None, []),
    ]
    dpa = config.enable_dp_attention
    assert [r[0] for r in by_rank] == ([(0, 0), (1, 0)] if dpa else [(0, 0), (0, 1)])
    assert [r[1:] for r in by_rank] == [(union, (9, 10))] * 2
    assert config.parallel_config._managed_distributed_store is False


def test_a_point_is_rank_0_s_with_every_rank_s_time_refused_where_any_refused(
    monkeypatch,
):
    def launch(fn, config, todo, llc):
        assert fn is ops.measure
        return [
            [{"leaf": "a", "cold_ns": 5.0}, {"leaf": "b", "cold_ns": 1.0}],
            [{"leaf": "a", "cold_ns": 7.0}, {"leaf": "b", "refused": "no room"}],
        ]

    monkeypatch.setattr(ranks, "launch", launch)

    got = ranks.measure(None, [], 1, gpus=2)

    assert got == [
        {"leaf": "a", "cold_ns": 5.0, "rank_cold_ns": [5.0, 7.0]},
        {"leaf": "b", "cold_ns": 1.0, "rank_cold_ns": [1.0, None]}
        | {"refused": "no room"},
    ]


NEG, ABS = "aten.neg.default", "aten.abs.default"


def _found(*names):
    x = ("self", _t([2, 3], "activation"))
    return ops.leaves(_graph(*(_leaf(n, x) for n in names), layers=1)["region"])


#: The gloo timeout of a rank whose peer raises building its runner.
RUNNER_WAIT = 60


def _measuring_rank(config, todo, refuse, rank):
    """One rank of `ops.measure` in a gloo group: ATOM's runner, the device
    and its kernels stand-ins, every cross-rank call on the group; rank 1
    refusing `NEG` at `refuse`: from its copy 0, from its copy 1, its first
    call or its capture; or raising as it builds its runner."""
    from atom.compass.bench import replay

    pc = config.parallel_config
    me = pc.data_parallel_rank * config.tensor_parallel_size + rank
    store = dist.TCPStore(pc.data_parallel_master_ip, pc.data_parallel_base_port)
    # a rank left waiting fails, not hangs
    wait = datetime.timedelta(seconds=RUNNER_WAIT if refuse == "runner" else 10)
    dist.init_process_group("gloo", store=store, rank=me, world_size=2, timeout=wait)
    runner, captures = _Runner(), []
    runner.exit = dist.destroy_process_group
    fctx._forward_kv_cache_context.kv_cache_data = {}

    def build(config, rank):
        if (me, refuse) == (1, "runner"):
            raise RuntimeError("rank 1 has no runner")
        return runner

    ops._runner = build
    ops._kernels = lambda fn, kwargs: (["k"], 1.0)
    ops._together = dist.barrier
    ranks.cpu_group = lambda: dist.group.WORLD
    replay.ReplayRule = lambda config: SimpleNamespace(
        block_size=16, batch=lambda view, block, slot: (block, slot)
    )
    sys.modules["aiter.dist.parallel_state"] = SimpleNamespace(
        graph_capture=contextlib.nullcontext
    )
    torch.cuda.CUDAGraph = lambda: SimpleNamespace(replay=lambda: None)
    torch.cuda.Event = lambda enable_timing: SimpleNamespace(
        record=lambda: None, synchronize=lambda: None, elapsed_time=lambda e: 1.0
    )

    def no(*args, **kwargs):
        raise ValueError("rank 1 refuses")

    def graph(*args, **kwargs):
        captures.append(1)
        return no() if (me, refuse, len(captures)) == (1, "capture", 1) else None

    copy, function = ops.Stub._copy, ops._function
    first = {"copy": 0, "later_copy": 1}.get(refuse) if me == 1 else None

    def copying(s, op, sig, inst, c, *a):
        refused = first is not None and op["op"] == NEG and c >= first
        return no() if refused else copy(s, op, sig, inst, c, *a)

    ops.Stub._copy = copying
    if (me, refuse) == (1, "call"):
        ops._function = lambda name: no if name == NEG else function(name)
    torch.cuda.graph = lambda *a, **k: contextlib.nullcontext(graph())
    return ops.measure(config, todo[me], 10_000, rank)


def _leaves(records):
    return [(r["op"], r.get("refused")) for r in records]


def test_ranks_measuring_different_points_each_measure_their_union_in_one_order():
    config = _config(2)
    ranks.layout(config)
    todo = [[(decode(2), 20, _found(ABS))], [(decode(2), 20, _found(NEG, ABS))]]

    by_rank = ranks.launch(_measuring_rank, config, todo, None)

    assert [_leaves(r) for r in by_rank] == [[(ABS, None), (NEG, None)]] * 2


@pytest.mark.parametrize("refuse", ["copy", "later_copy", "call", "capture"])
def test_a_leaf_one_rank_refuses_every_rank_refuses_and_the_next_is_measured(
    refuse, tmp_path, monkeypatch
):
    config = _config(2)
    ranks.layout(config)
    todo = [[(decode(2), 20, _found(NEG, ABS))]] * 2
    journal = tmp_path / "points.jsonl"
    monkeypatch.setenv(ops.JOURNAL, str(journal))

    by_rank = ranks.launch(_measuring_rank, config, todo, refuse)

    own = "ValueError: rank 1 refuses"
    assert [_leaves(r) for r in by_rank] == [
        [(NEG, f"ValueError: {ranks.PEER_REFUSED}"), (ABS, None)],
        [(NEG, own), (ABS, None)],
    ]
    assert [r.get("refused") for r in ranks.merge(by_rank)] == [own, None]
    # each rank journals its own records, which `_measured` reads by rank
    lines = [Path(f"{journal}.{r}").read_text().splitlines() for r in (0, 1)]
    assert [_leaves(map(json.loads, r)) for r in lines] == [_leaves(r) for r in by_rank]
    assert not journal.exists()


def test_a_rank_raising_before_its_first_cross_rank_call_ends_the_others():
    config = _config(2)
    ranks.layout(config)
    todo = [[(decode(2), 20, _found(NEG))]] * 2

    start = time.monotonic()
    with pytest.raises(RuntimeError, match="rank 1 has no runner"):
        ranks.launch(_measuring_rank, config, todo, "runner")

    assert time.monotonic() - start < RUNNER_WAIT  # not rank 0's gloo timeout


def test_a_rank_that_dies_costs_only_the_point_the_ranks_were_measuring(
    monkeypatch,
):
    todo = [(decode(2), 20, {s: _op(s) for s in "abc"})]
    launches = []

    def launch(fn, config, todo, llc):
        names = [sig for *_, found in todo for sig in found]
        launches.append(names)
        point = {"shape": str(decode(2)), "context": 20}
        by_rank = [
            [point | {"op": s, "leaf": s, "cold_ns": float(r)} for s in names]
            for r in (1, 2)
        ]
        if len(launches) > 1:
            return by_rank
        # Rank 1 dies measuring b, which rank 0 finished: the pool ends rank 0.
        for r, finished in enumerate((2, 1)):
            path = f"{os.environ[ops.JOURNAL]}.{r}"
            ops._journal(path, by_rank[r][:finished])
        raise BrokenProcessPool("rank 1 was killed")

    monkeypatch.setattr(ranks, "launch", launch)

    got = ops._measured(None, todo, 1, gpus=2)

    assert launches == [["a", "b", "c"], ["c"]]
    assert [(r["leaf"], r.get("rank_cold_ns"), r.get("refused")) for r in got] == [
        ("a", [1.0, 2.0], None),
        ("b", None, "the measuring process died"),
        ("c", [1.0, 2.0], None),
    ]


def test_a_lifted_collective_runs_on_the_group_atom_calls_it_on_or_refuses(
    monkeypatch,
):
    # moe's own signatures, which take the group the capture drops
    moe = SimpleNamespace(
        all_gatherv=lambda x, sizes, group: x, tp_gather=lambda x, group: x
    )
    dp = SimpleNamespace(get_dp_group=lambda: "dp")
    monkeypatch.setitem(sys.modules, "atom.model_ops.moe", moe)
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", dp)

    gatherv = ops._function("atom.model_ops.moe:all_gatherv")

    assert (gatherv.func, gatherv.keywords) == (moe.all_gatherv, {"group": "dp"})
    with pytest.raises(ValueError, match="tp_gather takes a group"):
        ops._function("atom.model_ops.moe:tp_gather")


def _gather(rows):
    x = ("x", _t([rows, 8], "activation"))
    return ir.Op(GATHER, "declared", (x, ("use_cag", True)), ((8 * rows, 8),))


def _gather_graph(store, config, shape, arch=None):
    rows = shape.tokens if shape.phase == "prefill" else shape.seqs
    return _graph(_gather(rows), layers=2)


def test_a_collective_is_one_family_over_its_bytes_across_decode_and_prefill(
    bench,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    # Measured at decode widths 1 and 4 (8 and 32 rows gathered, float32), one
    # kernel: a prefill of 2 tokens gathers bytes between them, one of 8 past.
    monkeypatch.setattr(ops, "graph", _gather_graph)
    store, root, _, _ = bench
    records = []
    for width in (1, 4):
        graph = _gather_graph(None, None, decode(width))
        [(sig, (op, _))] = ops.leaves(graph["region"]).items()
        point = {"shape": str(decode(width)), "context": 20, "op": GATHER}
        records.append(point | {"leaf": sig, "kernels": ["ag"], "cold_ns": 1.0})
    graded = _records(
        tmp_path / "graded",
        ([[2, 2, False]], None),
        ([[8, 8, False]], None),
        ([[1, 30, True], [1, 9000, True]], 2),
    )

    report = ops.coverage(store, root, CONFIG, [graded], records, "gfx942")

    assert ops.exchange(op)[1] == 8 * 4 * 8 * 4
    assert report["by phase"] == {
        "prefill": {"steps": 2, "covered": 1},
        "decode": {"steps": 1, "covered": 1},
    }
    assert report["left out, by leaf kind"] == {
        GATHER: {"outside the measured range": 1}
    }


def _sig(rows):
    [sig] = ops.leaves(_graph(_gather(rows), layers=1)["region"])
    return sig


def test_two_kernels_of_a_collective_densify_to_the_rows_halfway():
    fam = ops.exchange(json.loads(_sig(1)))[0]
    points = [
        (256, ("one",), {"shape": str(decode(2)), "context": 20}),
        (4096, ("two",), {"shape": str(prefill(32, 2)), "context": None}),
    ]

    want = ops._densify({fam: points}, set())

    assert want == {(prefill(17, 1), None): {fam}}


@pytest.mark.parametrize(
    "step", [decode, lambda rows: prefill(rows, 1)], ids=["decode", "prefill"]
)
@pytest.mark.parametrize(
    "middle, want",
    [
        # 26 at 2048 bytes against 20 on the line from 256 to 4096, one kernel.
        ((26.0, 26.0), {4, 12}),
        ((20.0, 20.0), set()),
        # 18 and 22: on the line at their mean, as `fit ops` merges them.
        ((18.0, 22.0), set()),
    ],
)
def test_a_collective_halves_either_side_of_a_size_off_its_ends_line(
    step, middle, want
):
    # Each size read at two contexts; the second reading at 256 bytes, 6% off
    # the first, is a repeat, not a bend.
    fam = ops.exchange(json.loads(_sig(1)))[0]
    points = [
        (256 * w, ("one",), {"shape": str(step(w)), "context": c, "cold_ns": ns})
        for w, readings in ((1, (10.0, 10.6)), (8, middle), (16, (30.0, 30.0)))
        for c, ns in zip((20, 3000), readings)
    ]

    got = ops._densify({fam: points}, set())

    assert set(got) == {(step(rows), 20) for rows in want}
    assert all(fams == {fam} for fams in got.values())


def test_fit_ops_prices_a_collective_between_two_message_sizes():
    points = []
    for rows, ns in ((1, 10.0), (3, 30.0)):
        sig = _sig(rows)
        point = {"shape": str(decode(rows)), "context": 20, "op": GATHER}
        point |= {"leaf": sig, "cold_ns": ns, "kernels": ["ag"]}
        points.append(point | {"exchange": ops.exchange(json.loads(sig))})

    [family] = fit.fit(points)["families"].values()

    assert family["dims"] == ["bytes"]
    at = {tuple(c): (ns, k) for c, ns, k in family["points"]}
    one = 8 * 8 * 4  # one row gathered from each of eight ranks, float32
    assert fit.interpolate(at, (2 * one,), ["bytes"]) == (20.0, ["ag"])

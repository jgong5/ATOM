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
import subprocess
import sys
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

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


C10D = "_c10d_functional.all_gather_into_tensor.default"


def test_a_capture_names_a_functional_collective_s_group_by_its_coordinator(
    request, tmp_path
):
    if "COMPASS_ARCH_DECLARED" not in os.environ:
        # The declared device and group are process-global: a process of its own.
        env = {**os.environ, "COMPASS_ARCH_DECLARED": "1"}
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
            + [f"{__file__}::{request.node.name}"],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        assert done.returncode == 0, done.stdout[-4000:]
        return
    from torch.distributed import _functional_collectives as functional
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    from atom.compass.capture import driverless, forward

    driverless.install(str(tmp_path), "gfx942")
    driverless.group(4, 2)
    from aiter.dist import parallel_state

    tp, dp = parallel_state.get_tp_group(), parallel_state.get_dp_group()
    recorder = forward._Recorder({}, driverless.driverless_mode(ShapeEnv()))
    with recorder.fake_mode, recorder:
        x = torch.empty(4, 8, device="cuda")
        recorder.on = True
        for group in (tp.device_group, dp.device_group, dist.group.WORLD):
            functional.all_gather_tensor(x, 0, group)

    named = [dict(op.args).get("group_name") for _, op in recorder.items]
    # the default group belongs to no coordinator: its name stays torch's
    assert [n for n in named if n] == ["tp:0", "dp:0", dist.group.WORLD.group_name]


def _layout_rank(config, todo, rank):
    """`_measuring_rank`, measuring `C10D` on aiter's coordinators `tp:0`, each
    rank's group of itself, and `dp:0`, of both ranks, their groups created in
    another order than a capture's (DP first); on `ep:0`, which no rank holds;
    and on the name torch gives the DP group here."""
    measure = ops.measure

    def on_groups(config, todo, llc, rank):
        dp = dist.new_group([0, 1])
        tp = [dist.new_group([r]) for r in (0, 1)][dist.get_rank()]
        sys.modules["aiter.dist.parallel_state"]._groups = {
            key: (lambda g=g: SimpleNamespace(device_group=g))
            for key, g in (("tp:0", tp), ("dp:0", dp))
        }
        x = ("input", _t([2, 8], "activation"))
        gathers = [
            _leaf(C10D, x, ("group_size", size), ("group_name", name))
            for name, size in (
                ("tp:0", 1),
                ("dp:0", 2),
                ("ep:0", 2),
                (dp.group_name, 2),
            )
        ]
        found = ops.leaves(_graph(*gathers, layers=1)["region"])
        return measure(config, [(decode(2), 20, found)], llc, rank)

    ops.measure = on_groups
    return _measuring_rank(config, todo, None, rank)


def test_a_c10d_leaf_runs_on_its_coordinator_s_group_whatever_torch_named_it():
    config = _config(2, dp_attention=True)
    ranks.layout(config)

    by_rank = ranks.launch(_layout_rank, config, [[], []])

    merged = {
        json.loads(r["leaf"])["args"]["group_name"]: r for r in ranks.merge(by_rank)
    }
    tp, dp, ep = (merged.pop(name) for name in ("tp:0", "dp:0", "ep:0"))
    [(torch_name, raw)] = merged.items()
    # a TP leaf of group size 1 on the DP group of two ranks would fail in torch
    assert [len(r["rank_cold_ns"]) for r in (tp, dp)] == [2, 2]
    assert "refused" not in tp and "refused" not in dp
    for record, name in ((ep, "ep:0"), (raw, torch_name)):
        assert record["refused"].startswith(
            f"ValueError: {C10D} on coordinator '{name}', which rank 0 does not hold"
        )


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


class _AtomRunner(_Runner):
    """ATOM's `ModelRunner` as `_runner` builds it, each config it was built
    on kept in `built`."""

    built: ClassVar[list] = []
    tokenID_processor = SimpleNamespace(is_deferred_out=False)

    def __init__(self, rank, config):
        super().__init__()
        self.config = config
        _AtomRunner.built.append(config)

    def get_num_blocks(self):
        return {"num_kvcache_blocks": 64}

    def allocate_kv_cache(self, blocks):
        self.kv_cache = torch.empty(blocks)

    def exit(self):
        pass


def _atom_runner(monkeypatch):
    """`_AtomRunner` in place of ATOM's, whose module imports need a driver."""
    engine = SimpleNamespace(ModelRunner=_AtomRunner, RapidServeModelRunner=None)
    monkeypatch.setitem(sys.modules, "atom.model_engine.model_runner", engine)
    monkeypatch.setattr(_AtomRunner, "built", [])


def _dpa_config(tp, dp_attention=True):
    config = _config(tp, dp_attention=dp_attention)
    config.compilation_config = SimpleNamespace(level=0, static_forward_context={})
    config.enforce_eager, config.model, config.dp_logical_size = True, "m", 0
    return config


def _layout(config):
    """The key's parallel layout, read off `config`."""
    pc = config.parallel_config
    return {
        "tp": config.tensor_parallel_size,
        "dp": pc.data_parallel_size,
        "pp": 1,
        "pcp": 1,
        "decode_context_parallel_size": 1,
        "enable_expert_parallel": False,
        "enable_dp_attention": config.enable_dp_attention,
    }


@pytest.mark.parametrize("managed", [False, True], ids=["alone", "launched"])
def test_one_dp_rank_alone_is_built_as_atom_s_simulated_dp_attention(
    managed, monkeypatch
):
    _atom_runner(monkeypatch)
    config = _dpa_config(2)
    ranks.layout(config)
    config.parallel_config._managed_distributed_store = managed

    runner = ops._runner(config, 0)

    [built] = _AtomRunner.built
    pc = built.parallel_config
    got = built.tensor_parallel_size, pc.data_parallel_size, built.dp_logical_size
    assert got == ((1, 2, 0) if managed else (1, 1, 2))
    assert pc.data_parallel_size_local == got[1]
    assert built.num_kvcache_blocks == 64
    assert list(runner.kv_held) == ["kv_cache"]  # `trace ops` reads it as KV
    assert ops.Stub(runner, 10_000).alone == got[2]
    assert config.parallel_config.data_parallel_size == 2  # the layout's own


def _dpa_main(argv, bench, tmp_path, monkeypatch, config):  # noqa: F811
    """`ops.main` over `argv` and ATOM's arguments for `config` on GPU 0, a
    gather and a negation measured in this process on `_AtomRunner`."""
    _, root, _, run = bench
    _atom_runner(monkeypatch)
    monkeypatch.setattr("atom.compass.bench.measure.atom_config", lambda a: config)
    monkeypatch.setattr("atom.compass.run.memory.key_fields", _layout)
    monkeypatch.setattr("atom.compass.drive.gpu.hip_visible", lambda g: "0")
    monkeypatch.setattr(
        "atom.compass.drive.gpu.quiet", lambda *a: contextlib.nullcontext()
    )
    monkeypatch.setattr("atom.compass.drive.identity.identity", lambda *a: run)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    placed = SimpleNamespace(
        block_size=16, batch=lambda view, block, slot: (block, slot)
    )

    def rule(c):  # the runner's own config, DP 1 with its KV pool sized
        assert c is _AtomRunner.built[-1]
        return placed

    monkeypatch.setattr("atom.compass.bench.replay.ReplayRule", rule)
    x = ("self", _t([2, 8], "activation"))
    graph = _graph(_gather(2), _leaf(NEG, x), layers=1)
    monkeypatch.setattr(ops, "graph", lambda *a: graph)
    monkeypatch.setattr(ops, "_child", lambda fn, *args: fn(*args))
    monkeypatch.setattr(ops, "_kernels", lambda fn, kwargs: (["k"], 1.0))
    monkeypatch.setattr(ops, "_replay", lambda fn, calls: [1.0] * ops.REPEATS)
    argv += ["--width", "2", "--context", "20", "--llc-mib", "1", "--gpu", "0"]
    argv += ["--store", root, "--out", str(tmp_path / "out")]
    tp = str(config.tensor_parallel_size)
    atom = ["--model", "m", "-tp", tp, "--enforce-eager", "--level", "0"]
    atom += ["--enable-dp-attention"] if config.enable_dp_attention else []
    ops.main([*argv, "--", *atom])


def test_one_gpu_at_dp_attention_measures_a_rank_s_leaves_and_refuses_collectives(
    bench, tmp_path, monkeypatch  # noqa: F811
):
    config = _dpa_config(2)

    with pytest.raises(ValueError, match="1 points refused") as refused:
        _dpa_main([], bench, tmp_path, monkeypatch, config)

    [built] = _AtomRunner.built
    widths = [
        (c.tensor_parallel_size, c.parallel_config.data_parallel_size)
        for c in (built, config)
    ]
    assert widths == [(1, 1), (1, 2)]
    lines = (tmp_path / "out" / "ops.jsonl").read_text().splitlines()
    got = {r["op"]: r for r in map(json.loads, lines)}
    why = "ValueError: a collective over the layout's 2 ranks, measured across them"
    assert got[GATHER]["refused"].startswith(why)
    assert GATHER in str(refused.value)
    assert "refused" not in got[NEG] and got[NEG]["cold_ns"] == 1.0
    # keyed by the deployed layout, two DP ranks of TP 1, not the runner's
    store, _, _, run = bench
    point = decode(2), 20, got[NEG]["leaf"]
    assert store.directory_for(ops.keys(config, run)(*point)).exists()
    assert not store.directory_for(ops.keys(built, run)(*point)).exists()


def test_one_gpu_refuses_a_tp_layout_wider_than_one_rank(
    bench, tmp_path, monkeypatch  # noqa: F811
):
    with pytest.raises(ValueError, match="names 1 GPUs and ATOM's arguments resolve 2"):
        _dpa_main([], bench, tmp_path, monkeypatch, _dpa_config(2, False))

    assert _AtomRunner.built == []

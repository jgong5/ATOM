# SPDX-License-Identifier: MIT
"""`compass measure ops`, the parts that run with no device.

Graphs are built by `trace ops`'s own folding (`capture.graph.build`); the
stub runs beside a stand-in runner whose input preparation sets ATOM's forward
context, as ATOM's runner does, and whose forward raises.
"""

import contextlib
import dataclasses
import functools
import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from test_bench_replay import _line, rule  # noqa: F401  the rule fixture
from test_filler_tokens import model_dir  # noqa: F401  the model_dir fixture
from test_parity_record import _unpriced  # noqa: F401  an autouse fixture

from atom.compass import cli, ir
from atom.compass.artifacts import ArtifactStore, Kind, machine_spec_digest
from atom.compass.bench import ops
from atom.compass.capture import driverless
from atom.compass.capture.forward import decode, prefill
from atom.compass.capture.graph import Traced, build, to_json
from atom.compass.spec import MachineSpec
from atom.compass.spec.fields import PINNED
from atom.utils import block_convert
from atom.utils import forward_context as fctx

F32 = "torch.float32"
KEY = "model.layers.{L}.attn"
TREE = pathlib.Path(__file__).resolve().parents[2]


def _t(shape, role, stride=None, dtype=F32):
    stride = stride or [s for s in torch.empty(shape).stride()]
    return ir.Tensor(tuple(shape), dtype, tuple(stride), role)


def _leaf(name, *args, key=None):
    ref = key and ir.ContextRef(key)
    return ir.Op(name, "opaque_leaf", args, (), context_ref=ref)


def _graph(*layer_ops, layers=4):
    """`layer_ops` in each of `layers` layers, as JSON."""
    items = [(i, op) for i in range(layers) for op in layer_ops]
    return json.loads(to_json(build(items, Traced("x tp=1 decode eager", "B=2"))))


def test_each_leaf_holds_every_instance_s_context_key_and_module_path():
    attn = _leaf(
        "aiter.attention",
        ("q", _t([2, 8], "activation")),
        ("layer", ir.ContextRef(KEY)),
        ("rope", ir.ModuleRef("model.rope")),
        key=KEY,
    )
    gemm = _leaf("aiter.gemm", ("x", _t([2, 8], "activation")))

    found = ops.leaves(_graph(attn, gemm)["region"])

    by_op = {op["op"]: instances for op, instances in found.values()}
    assert by_op == {
        "aiter.attention": [
            {"layer": f"model.layers.{i}.attn", "rope": "model.rope"} for i in range(4)
        ],
        "aiter.gemm": [{}],
    }
    assert [op["occurrences"] for op, _ in found.values()] == [4, 4]


def test_stated_values_part_signatures_and_stating_nothing_keeps_the_key():
    plain = _leaf("aiter.decode", ("x", _t([2, 8], "activation")))
    csa, hca = (
        dataclasses.replace(plain, stated=(("compress_ratio", r), ("index_topk", k)))
        for r, k in ((4, 1024), (128, None))
    )

    found = ops.leaves(_graph(plain, csa, hca, layers=1)["region"])

    assert [op.get("stated") for op, _ in found.values()] == [
        None,
        {"compress_ratio": 4, "index_topk": 1024},
        {"compress_ratio": 128, "index_topk": None},
    ]
    # The key a leaf stating nothing had before any leaf stated anything.
    op, _ = next(iter(found.values()))
    leaf = {"op": "aiter.decode", "args": op["args"], "out": []}
    assert ops.signature(op) == json.dumps(leaf, sort_keys=True, separators=(",", ":"))


LIVE = {
    "kv_indptr": torch.arange(9, dtype=torch.int32),
    # Its own shape and stride, with more room than `kv_indptr`.
    "cu_seqlens_q": torch.arange(12, dtype=torch.int32)[:3],
    "slots": torch.arange(64, dtype=torch.int32)[8:],
}


@pytest.mark.parametrize(
    "slot, shape, picked",
    [
        ("kv_indptr_convert", [3], "kv_indptr"),  # a name the slot holds
        ("self", [3], "cu_seqlens_q"),  # then its own shape and stride
        ("self", [5], "kv_indptr"),  # then the tightest with room
        ("self", [40], "slots"),
    ],
)
def test_a_metadata_argument_is_a_view_of_live_metadata(slot, shape, picked):
    spec = {"shape": shape, "dtype": "torch.int32", "stride": [1], "role": "metadata"}

    name, view = ops._meta(spec, slot, LIVE)

    assert name == picked
    assert (
        view.untyped_storage().data_ptr() == LIVE[picked].untyped_storage().data_ptr()
    )
    assert torch.equal(view, LIVE[picked][: shape[0]])


def test_metadata_with_no_room_refuses_by_slot():
    spec = {"shape": [99], "dtype": "torch.int32", "stride": [1], "role": "metadata"}

    with pytest.raises(ValueError, match=r"no room for indices \[99\] torch.int32"):
        ops._meta(spec, "indices", LIVE)


@dataclasses.dataclass
class _Meta:
    tables: torch.Tensor
    lens: dict
    also: torch.Tensor
    owner: torch.nn.Module


def test_a_snapshot_clones_every_tensor_it_reaches_and_keeps_aliases():
    base = torch.arange(24.0)
    tables = base.view(4, 6)[:, 1:3]
    meta = _Meta(tables, {"lens": [base[2:5]]}, tables, torch.nn.Linear(2, 2))

    twin = ops._deep(meta, {})

    assert twin.also is twin.tables and twin.owner is meta.owner
    for old, new in [(tables, twin.tables), (base[2:5], twin.lens["lens"][0])]:
        assert new.untyped_storage().data_ptr() != base.untyped_storage().data_ptr()
        assert new.stride() == old.stride() and torch.equal(new, old)


def test_a_module_copy_holds_tensors_of_its_own():
    module = torch.nn.Linear(2, 2)
    module.register_buffer("cache", torch.ones(3))
    module.cos_cache = torch.full((4,), 2.0)

    twin = ops._module_copy(module)

    for name in ("weight", "bias", "cache", "cos_cache"):
        old, new = getattr(module, name), getattr(twin, name)
        assert torch.equal(old, new) and old.data_ptr() != new.data_ptr()
    assert sum(t.numel() for t in ops._own(module)) == 4 + 2 + 3 + 4


class _Runner:
    """ATOM's runner as the stub sees it: a model whose parameters are on the
    meta device, its persistent buffers, an input preparation that writes a
    batch's blocks and slots into them and sets the forward context on them,
    between the staging gate's two ends, and a forward that raises."""

    device = torch.device("cpu")

    def __init__(self, contexts=()):
        self.model = torch.nn.Sequential(
            *(torch.nn.Linear(3, 3, device="meta") for _ in range(2))
        )
        self.forward_vars = {"block_tables": torch.zeros(4, 8, dtype=torch.int32)}
        self.positions = torch.zeros(2, dtype=torch.int64)
        self.config = SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context=dict(contexts)),
            num_kvcache_blocks=64,
        )
        self.batches, self.staged = [], []

    def _gate_staging_reuse(self):
        self.staged.append("begin")

    def _mark_staging_h2d_enqueued(self):
        self.staged.append("finish")

    def prepare_model(self, batch):
        self.batches.append(batch)
        self.staged.append(batch)
        block, slot = batch
        self.forward_vars["block_tables"].fill_(block)
        self.positions.fill_(slot)
        meta = fctx.AttentionMetaData(block_tables=self.forward_vars["block_tables"])
        context = fctx.Context(positions=self.positions)
        fctx._forward_context = fctx.ForwardContext(attn_metadata=meta, context=context)
        return torch.zeros(2, dtype=torch.int32), None, None, None, True, False

    def forward(self, batch):
        raise AssertionError("the op bench runs no forward")


def test_a_point_places_its_batch_for_its_context_keys_and_refuses_by_name(
    monkeypatch,
):
    # Two layers of 4 blocks' KV, 256 bytes a block: 1056 bytes a call, so
    # 10 copies past twice a 5000-byte cache, 5 placements of 4 blocks and 2 slots.
    layers = {f"model.layers.{i}.attn": SimpleNamespace(layer_num=i) for i in (0, 1)}
    kv = {f"layer_{i}": SimpleNamespace(k_cache=torch.zeros(64, 16, 4)) for i in (0, 1)}
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", kv)
    runner = _Runner(layers)
    found = ops.leaves(
        _graph(
            _leaf(
                "aten.nowhere.default",
                ("q", _t([2, 4], "activation")),
                ("layer", ir.ContextRef(KEY)),
                key=KEY,
            ),
            _leaf("aten.add.Tensor", ("self", _t([2, 3], "kv"))),
            _leaf(
                "aten.add_.Tensor",
                ("self", _t([2, 8], "metadata", dtype="torch.int32")),
            ),
            layers=2,
        )["region"]
    )
    placed = SimpleNamespace(
        block_size=16, batch=lambda view, block, slot: (block, slot)
    )
    stub = ops.Stub(runner, 5_000)

    got = stub.point(placed, decode(2), 20, found)

    assert runner.batches == [(4 * p, 2 * p) for p in range(5)]
    assert runner.staged == [s for b in runner.batches for s in ("begin", b, "finish")]
    why = {r["op"]: r["refused"] for r in got}
    assert "'nowhere'" in why["aten.nowhere.default"]
    assert "a kv tensor argument [2, 3]" in why["aten.add.Tensor"]
    [meta] = [r for r in got if r["op"] == "aten.add_.Tensor"]
    assert meta["metadata"] == {"self": "block_tables"}
    assert {r["placements"] for r in got} == {5}


def test_each_placement_keeps_its_own_forward_context_and_metadata():
    runner = _Runner()
    stub = ops.Stub(runner, 10_000)
    meta = _t([2, 8], "metadata", dtype="torch.int32")
    stub.leaves = ops.leaves(
        _graph(_leaf("aten.add_.Tensor", ("self", meta)), layers=1)["region"]
    )

    for p in range(3):
        stub._prepare((4 * p, 2 * p))

    for p, (snap, views) in enumerate(stub.placed):
        [(name, view)] = views.values()
        assert name == "block_tables"
        assert set(snap.attn_metadata.block_tables.flatten().tolist()) == {4 * p}
        assert set(view.flatten().tolist()) == {4 * p}
        assert snap.context.positions.tolist() == [2 * p] * 2


def _measuring(monkeypatch):
    """A stand-in runner `measure` builds, a rule that refuses a batch of two
    rows, and two points of two leaves each: two rows, then four. A leaf times
    1 ns, with or without a device to time it on."""
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    monkeypatch.setattr(ops.Stub, "_time", lambda *_: {"cold_ns": 1.0})
    runner = _Runner()
    runner.exit = lambda: runner.batches.append("exit")

    def batch(view, block, slot):
        if len(view["rows"]) == 2:
            raise ValueError("no room for 2 rows")
        return len(view["rows"]), slot

    placed = SimpleNamespace(block_size=16, batch=batch)
    monkeypatch.setattr(ops, "_runner", lambda config, rank: runner)
    monkeypatch.setattr("atom.compass.bench.replay.ReplayRule", lambda c: placed)
    x = ("self", _t([2, 3], "activation"))
    found = ops.leaves(
        _graph(_leaf("aten.neg.default", x), _leaf("aten.abs.default", x))["region"]
    )
    return runner, [(decode(2), 20, found), (decode(4), 20, found)]


def test_a_shape_the_rule_refuses_refuses_its_leaves_and_the_next_is_measured(
    monkeypatch, tmp_path
):
    runner, todo = _measuring(monkeypatch)
    journal = tmp_path / "points.jsonl"
    monkeypatch.setenv(ops.JOURNAL, str(journal))

    got = ops.measure(None, todo, 10_000)

    lines = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [r.get("refused") for r in lines] == [r.get("refused") for r in got]
    assert [(r["shape"], r["op"], r.get("refused")) for r in got] == [
        (str(decode(2)), "aten.neg.default", "ValueError: no room for 2 rows"),
        (str(decode(2)), "aten.abs.default", "ValueError: no room for 2 rows"),
        (str(decode(4)), "aten.neg.default", None),
        (str(decode(4)), "aten.abs.default", None),
    ]
    assert {b[0] for b in runner.batches[:-1]} == {4}
    assert runner.batches[-1] == "exit"


def test_measure_times_a_point_atom_prepared_and_runs_no_forward(monkeypatch):
    # ATOM's runner and its model both raise on a forward; a GEMM over two
    # layers is timed on random weights, one a layer, on what ATOM prepared.
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    runner = _Runner()
    runner.exit = lambda: None
    runner.model.forward = runner.forward
    placed = SimpleNamespace(
        block_size=16, batch=lambda view, block, slot: (block, slot)
    )
    monkeypatch.setattr(ops, "_runner", lambda config, rank: runner)
    monkeypatch.setattr("atom.compass.bench.replay.ReplayRule", lambda c: placed)
    monkeypatch.setattr(ops, "_kernels", lambda fn, kwargs: (["k"], 1.0))
    timed = []
    monkeypatch.setattr(
        ops, "_replay", lambda fn, calls: timed.append(calls) or [1.0] * ops.REPEATS
    )
    mm = _leaf(
        "aten.mm.default",
        ("self", _t([2, 3], "activation")),
        ("mat2", _t([3, 3], "weight")),
    )
    found = ops.leaves(_graph(mm, layers=2)["region"])

    [record] = ops.measure(None, [(decode(2), 20, found)], 10_000)

    assert "refused" not in record and record["cold_ns"] == 1.0
    assert runner.staged == [s for b in runner.batches for s in ("begin", b, "finish")]
    assert record["placements"] == len(runner.batches) > 0
    weights = {id(kwargs["mat2"]) for kwargs, _ in timed[0]}
    assert len(weights) == 2


def test_an_error_in_atom_s_preparation_fails_measure_with_that_error(monkeypatch):
    runner, todo = _measuring(monkeypatch)

    def prepare_model(batch):
        runner.batches.append(batch)
        raise ValueError("cannot begin: owner is active")

    runner.prepare_model = prepare_model

    with pytest.raises(ValueError, match="cannot begin: owner is active") as raised:
        ops.measure(None, todo, 10_000)
    assert raised.value.__notes__ == [f"measuring {decode(4)} at context 20"]
    assert runner.batches == [(4, 0), "exit"]


class _Launches:
    """A Triton kernel that records the grid of each launch and runs nothing."""

    def __init__(self):
        self.grids = []

    def __getitem__(self, grid):
        self.grids.append(grid)
        return lambda *args, **kwargs: None


def test_a_decode_point_takes_an_integer_affine_in_the_context_at_its_own(
    monkeypatch,
    rule,  # noqa: F811
):
    kernel = _Launches()
    monkeypatch.setattr(block_convert, "kv_indices_generate_kernel", kernel)
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})

    def call(stub, op, sig, instances, n):
        kwargs, _ = stub._copy(op, sig, instances, 0)
        ops._function(op["op"])(**kwargs)
        return {}

    monkeypatch.setattr(ops.Stub, "_time", call)
    runner = _Runner()
    stub = ops.Stub(runner, 10_000)
    ints = [_t(shape, "metadata", dtype="torch.int32") for shape in ([2, 8], [16], [3])]
    leaf = _leaf(
        "atom.utils.block_convert:kv_indices_generate_triton",
        *zip(("block_tables", "kv_indices_convert", "kv_indptr_convert"), ints),
        ("ratio", 1),
        ("max_ctx_len", ir.Affine(1, 0)),
    )
    found = ops.leaves(_graph(leaf, layers=1)["region"])
    placed = SimpleNamespace(
        block_size=16, batch=lambda view, block, slot: (block, slot)
    )

    [record] = stub.point(placed, decode(2), 8192, found)

    # ATOM's own step at decode B = 2, context 8192 passes its rows' longest
    # context.
    batch = rule(max_model_len=8192).batch(_line([[1, 8192, True]] * 2))
    tables = torch.zeros(2, 8, dtype=torch.int32)
    block_convert.kv_indices_generate_triton(
        tables, tables, tables, 1, batch.context_lens.max()
    )
    assert "refused" not in record
    assert kernel.grids == [(2, 2), (2, 2)]


def test_an_embedding_reads_the_token_ids_atom_prepared():
    runner = _Runner()
    # An index buffer whose name holds the slot's, which the name rule picks.
    runner.forward_vars["spec_state_indices"] = torch.zeros(8, dtype=torch.int32)
    stub = ops.Stub(runner, 10_000)
    embedding = _leaf(
        "aten.embedding.default",
        ("weight", _t([16, 3], "weight")),
        ("indices", _t([2], "metadata", dtype="torch.int32")),
    )
    stub.leaves = ops.leaves(_graph(embedding, layers=1)["region"])

    stub._prepare((0, 0))

    [(_, views)] = stub.placed
    assert [name for name, _ in views.values()] == ["input_ids"]


def test_a_call_counts_its_module_s_tensors_and_cycles_over_every_layer():
    layers = {f"model.layers.{i}.attn": SimpleNamespace(layer_num=i) for i in (0, 1)}
    kv = {f"layer_{i}": SimpleNamespace(k_cache=torch.zeros(64, 16, 4)) for i in (0, 1)}
    stub = ops.Stub(_Runner(layers), 50)
    attn = {"op": "x", "args": {"layer": {"context": None}}}
    rope = {"op": "x", "args": {"rope": {"module": None}}}
    pairs = [{"layer": key} for key in layers]

    # 4 blocks of 256 bytes is past twice a 50-byte cache, and both layers still
    # cycle.
    assert stub._plan(attn, pairs, kv, 4, 2) == (2, 1024, 1)
    # A Linear(3, 3) holds 12 floats, 48 bytes: 3 copies pass twice 50, on one
    # placement.
    assert stub._plan(rope, [{"rope": "0"}], kv, 4, 2) == (3, 48, 1)


def test_a_context_module_with_no_kv_layer_is_sized_by_its_own_tensors(monkeypatch):
    # DeepSeek-V4's Indexer: bound by its layer name, its pool on itself, no
    # layer_num and no entry in ATOM's KV registry.
    indexer = torch.nn.Module()
    indexer.register_buffer("kv_cache", torch.zeros(4, 16, 4))
    runner = _Runner({KEY.format(L=0): indexer})
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    monkeypatch.setattr(ops.Stub, "_time", lambda *_: {"cold_ns": 1.0})
    placed = SimpleNamespace(block_size=16, batch=lambda view, b, s: (b, s))
    q = ("q", _t([2, 4], "activation"))
    leaf = _leaf("aten.nowhere.default", q, ("layer", ir.ContextRef(KEY)), key=KEY)
    found = ops.leaves(_graph(leaf, layers=1)["region"])
    stub = ops.Stub(runner, 5_000)

    [record] = stub.point(placed, decode(2), 20, found)

    # 32 bytes of q and 1024 of pool a call: 10 copies pass twice 5000, on one
    # placement.
    assert "refused" not in record
    assert (record["cold_ns"], record["footprint_bytes"]) == (1.0, 10 * 1056)
    assert record["placements"] == 1


def test_a_call_cycles_a_weight_per_occurrence_on_the_cold_copies_inputs():
    # One copy already passes twice a 1-byte cache. The leaf occurs twice in
    # the step: copy 1 is copy 0 with a second weight, so a weight count past
    # the cache-sized one allocates no activation.
    runner = _Runner()
    stub = ops.Stub(runner, 1)
    stub.placed = [(fctx.ForwardContext(), {})]
    x = {"shape": [2, 3], "dtype": F32, "stride": [3, 1], "role": "activation"}
    w = x | {"shape": [3, 3], "role": "weight"}
    gemm = {"op": "x", "args": {"x": x, "w": [w]}, "occurrences": 2}
    graphs = []

    def replay(fn, calls):
        graphs.append(calls)
        return [1.0] * ops.REPEATS

    with (
        patch.object(ops, "_function", return_value=lambda **kwargs: None),
        patch.object(ops, "_kernels", return_value=(["k"], 1.0)),
        patch.object(ops, "_replay", side_effect=replay),
    ):
        n, _, _ = stub._plan(gemm, [{}], {}, 4, 2)
        record = stub._time(gemm, "sig", [{}], n)

    (first, snap), (second, snap1) = graphs[0][:2]
    assert second["x"] is first["x"] and snap1 is snap
    weights = [first["w"][0], second["w"][0]]
    assert weights[0] is not weights[1]
    assert all(t.shape == (3, 3) and t.stride() == (3, 1) for t in weights)
    assert all(t.device == runner.device and t.std() > 0 for t in weights)
    assert (n, record["copies"]) == (1, 2)


def test_a_call_that_fills_a_graph_alone_cycles_no_second_weight():
    # The leaf occurs twice, but one call fills `GRAPH_NS`: the graph holds
    # one call, so one copy, and no warm graph.
    stub = ops.Stub(_Runner(), 1)
    stub.placed = [(fctx.ForwardContext(), {})]
    x = {"shape": [2, 3], "dtype": F32, "stride": [3, 1], "role": "activation"}
    w = x | {"shape": [3, 3], "role": "weight"}
    gemm = {"op": "x", "args": {"x": x, "w": [w]}, "occurrences": 2}
    graphs = []

    def replay(fn, calls):
        graphs.append(len(calls))
        return [1.0] * ops.REPEATS

    with (
        patch.object(ops, "_function", return_value=lambda **kwargs: None),
        patch.object(ops, "_kernels", return_value=(["k"], ops.GRAPH_NS * 2)),
        patch.object(ops, "_replay", side_effect=replay),
    ):
        record = stub._time(gemm, "sig", [{}], 1)

    assert graphs == [1]
    assert (record["copies"], record["calls"], record["warm_ns"]) == (1, 1, None)


def test_a_copy_past_the_cold_ones_takes_its_own_cold_copy_s_inputs_and_is_logged(
    monkeypatch, capsys
):
    # A [2, 3] input and a [3, 3] weight are 60 bytes a call: 2 copies pass
    # twice a 30-byte cache. The leaf occurs in four layers, so copies 2 and 3
    # are copies 0 and 1 with the third and fourth weights.
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    runner = _Runner()
    stub = ops.Stub(runner, 30)
    mm = _leaf(
        "aten.mm.default",
        ("self", _t([2, 3], "activation")),
        ("mat2", _t([3, 3], "weight")),
    )
    found = ops.leaves(_graph(mm, layers=4)["region"])
    placed = SimpleNamespace(
        block_size=16, batch=lambda view, block, slot: (block, slot)
    )
    graphs = []

    def replay(fn, calls):
        graphs.append(calls)
        return [1.0] * ops.REPEATS

    with (
        patch.object(ops, "_kernels", return_value=(["k"], 1.0)),
        patch.object(ops, "_replay", side_effect=replay),
    ):
        [record] = stub.point(placed, decode(2), 20, found)

    cold = [kwargs for kwargs, _ in graphs[0]]
    assert cold[1]["self"] is not cold[0]["self"]
    assert cold[2]["self"] is cold[0]["self"] and cold[3]["self"] is cold[1]["self"]
    assert len({id(c["mat2"]) for c in cold[:4]}) == 4
    assert all(c["mat2"] is cold[i % 4]["mat2"] for i, c in enumerate(cold))
    assert record["copies"] == 4 and "over 4 copies" in capsys.readouterr().out


def test_copy_c_rotates_weights_and_instances_and_clones_past_the_first_round():
    runner = _Runner()
    rope = torch.nn.Module()
    rope.register_buffer("cos_sin_cache", torch.ones(4))
    runner.model.append(rope)
    stub = ops.Stub(runner, 10_000)
    op = {
        "op": "aten.mm.default",
        "args": {
            "self": {
                "shape": [2, 3],
                "dtype": F32,
                "stride": [3, 1],
                "role": "metadata",
            },
            "mat2": {"shape": [3, 3], "dtype": F32, "stride": [3, 1], "role": "weight"},
            "rope": {"module": None},
            "layer": {"context": None},
            "out_dtype": "torch.bfloat16",
        },
        "occurrences": 4,
    }
    live = torch.ones(2, 3)
    stub.placed = [(f"snap{p}", {("sig", "self"): ("x", live + p)}) for p in (0, 1)]
    instances = [{"rope": "2", "layer": f"model.layers.{i}.attn"} for i in (0, 1)]

    copies = [stub._copy(op, "sig", instances, c) for c in range(6)]

    weights = [kw["mat2"] for kw, _ in copies]
    assert [kw["layer"] for kw, _ in copies] == [
        "model.layers.0.attn",
        "model.layers.1.attn",
    ] * 3
    assert [snap for _, snap in copies] == [
        "snap0",
        "snap0",
        "snap1",
        "snap1",
        "snap0",
        "snap0",
    ]
    # one random weight per occurrence of the leaf, copy c taking the c-th,
    # every one a slice of one tensor
    assert len({w.data_ptr() for w in weights[:4]}) == 4
    assert {w.untyped_storage().data_ptr() for w in weights} == {
        weights[0].untyped_storage().data_ptr()
    }
    assert weights[4] is weights[0] and weights[5] is weights[1]
    assert all(w.shape == (3, 3) and w.stride() == (3, 1) for w in weights)
    assert copies[3][0]["self"] is stub.placed[1][1][("sig", "self")][1]
    assert copies[4][0]["self"].data_ptr() != copies[0][0]["self"].data_ptr()
    assert copies[3][0]["rope"] is rope
    assert copies[4][0]["rope"] is not rope
    assert copies[0][0]["out_dtype"] is torch.bfloat16


def test_a_referenced_module_s_meta_parameters_are_filled_as_load_dummy_xavier(
    request, tmp_path
):
    # Every parameter of ATOM's model is on the meta device: a leaf reaching
    # one through a module path or its layer's context key reads it on the
    # device, filled as ATOM's `--load_dummy xavier` fills it.
    if "COMPASS_ARCH_DECLARED" not in os.environ:
        # ATOM's loader imports aiter, which needs an arch declared: once per
        # process, so in a process of its own.
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
    driverless.install(str(tmp_path), "gfx942")
    layers = {"module": torch.nn.Module(), "context": torch.nn.Module()}
    for layer in layers.values():
        layer.A_log = torch.nn.Parameter(torch.empty(4, device="meta"))
        layer.conv1d = torch.nn.Linear(3, 2, device="meta")
    runner = _Runner({KEY: layers["context"]})
    runner.model.append(layers["module"])
    stub = ops.Stub(runner, 10_000)
    stub.placed = [(fctx.ForwardContext(), {})]

    for kind, key in (("module", "2"), ("context", KEY)):
        op = {"op": "x", "args": {"layer": {kind: None}}}
        stub._copy(op, "sig", [{"layer": key}], 0)

    for layer in layers.values():
        assert {p.device.type for p in layer.parameters()} == {"cpu"}
        # 1-D and bias as xavier fills them; the declared device stubs uniform_
        assert torch.equal(layer.A_log, torch.ones(4))
        assert torch.equal(layer.conv1d.bias, torch.zeros(2))
    assert runner.model[0].weight.is_meta  # a module no leaf references


@pytest.mark.parametrize("copies", [1, 8])
def test_every_cold_and_warm_graph_holds_calls_calls_at_any_copies(copies):
    # A replay's fixed cost is spread over a graph's calls: a count that moved
    # with the copies would move each price by its copies.
    stub = ops.Stub(_Runner(), 10_000)
    stub.placed = [(fctx.ForwardContext(), {})]
    graphs = []

    def replay(fn, calls):
        graphs.append(len(calls))
        return [1.0] * ops.REPEATS

    with (
        patch.object(ops, "_function", return_value=lambda **kwargs: None),
        patch.object(ops, "_kernels", return_value=(["k"], 1.0)),
        patch.object(ops, "_replay", side_effect=replay),
    ):
        record = stub._time({"op": "x", "args": {"n": 1}}, "sig", [{}], copies)

    assert graphs == [ops.CALLS] * (2 if copies > 1 else 1)
    assert (record["copies"], record["calls"]) == (copies, ops.CALLS)


@pytest.mark.parametrize(
    "call_ns, calls, copies",
    [
        (ops.GRAPH_NS / 1000, ops.CALLS, 8),  # short: every graph holds CALLS
        (ops.GRAPH_NS / 10, 10, 8),
        (ops.GRAPH_NS / 3.5, 3, 3),  # fewer calls than copies: one copy a call
        (ops.GRAPH_NS * 50, 1, 1),  # past the cap: at least one call
    ],
)
def test_a_long_call_fills_a_graph_to_graph_ns_and_has_no_warm_graph(
    call_ns, calls, copies
):
    stub = ops.Stub(_Runner(), 10_000)
    stub.placed = [(fctx.ForwardContext(), {})]
    graphs = []

    def replay(fn, cycle):
        graphs.append(len(cycle))
        return [1.0] * ops.REPEATS

    with (
        patch.object(ops, "_function", return_value=lambda **kwargs: None),
        patch.object(ops, "_kernels", return_value=(["k"], call_ns)),
        patch.object(ops, "_replay", side_effect=replay),
    ):
        record = stub._time({"op": "x", "args": {"n": 1}}, "sig", [{}], 8)

    assert graphs == [calls] * (2 if calls == ops.CALLS else 1)
    assert (record["copies"], record["calls"]) == (copies, calls)
    assert (record["warm_ns"] is None) == (calls < ops.CALLS)


def test_a_replay_rests_the_host_a_share_of_each_replay_after_it(monkeypatch):
    # Replay i takes i + 1 ms on a stand-in clock; the host idles `REST` of that
    # after every replay, warmups included, which keeps a leaf off the lower
    # memory clock it reaches back to back.
    assert ops.REST > 0
    clock, slept = [0], []

    class Graph:
        def replay(self):
            clock[0] += len(slept) + 1

    class Event:
        def __init__(self, enable_timing):
            self.synchronize = lambda: None

        def record(self):
            self.at = clock[0]

        def elapsed_time(self, end):
            return end.at - self.at

    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(ops.time, "sleep", slept.append)

    ops._replay(lambda: None, [({}, fctx._forward_context)])

    replays = range(ops.WARMUPS + ops.REPEATS)
    assert slept == [(i + 1) * ops.REST / 1e3 for i in replays]


def test_copies_past_the_cache_sized_ones_cycle_instances_on_their_activations(
    monkeypatch,
):
    # One call's 1056 bytes pass twice a 600-byte cache in two copies: four
    # layers take four copies, and only the first two allocate activations.
    layers = {f"model.layers.{i}.attn": SimpleNamespace(layer_num=i) for i in range(4)}
    kv = {
        f"layer_{i}": SimpleNamespace(k_cache=torch.zeros(64, 16, 4)) for i in range(4)
    }
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", kv)
    runner = _Runner(layers)
    q = ("q", _t([2, 4], "activation"))
    found = ops.leaves(
        _graph(_leaf("x", q, ("layer", ir.ContextRef(KEY)), key=KEY))["region"]
    )
    placed = SimpleNamespace(
        block_size=16, batch=lambda view, block, slot: (block, slot)
    )
    graphs = []

    def replay(fn, calls):
        graphs.append(calls)
        return [1.0] * ops.REPEATS

    stub = ops.Stub(runner, 600)
    with (
        patch.object(ops, "_function", return_value=lambda **kwargs: None),
        patch.object(ops, "_kernels", return_value=(["k"], 1.0)),
        patch.object(ops, "_replay", side_effect=replay),
    ):
        [record] = stub.point(placed, decode(2), 20, found)

    assert [kw["layer"] for kw, _ in graphs[0][:4]] == list(layers)
    qs = [kw["q"] for kw, _ in graphs[0]]
    assert qs[0] is not qs[1]
    assert all(q is qs[c % 2] for c, q in enumerate(qs))
    assert record["copies"] == 4


def test_a_graph_over_placements_reads_one_clone_of_metadata_none_changed(
    monkeypatch,
):
    # The leaf prepares on the host for the last `cu_seqlens_q` it saw, by
    # identity, as linear attention does its chunk indices; a capture cannot.
    # Every placement's forward leaves it unchanged.
    layers = {f"model.layers.{i}.attn": SimpleNamespace(layer_num=i) for i in (0, 1)}
    kv = {f"layer_{i}": SimpleNamespace(k_cache=torch.zeros(64, 16, 4)) for i in (0, 1)}
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", kv)
    cu = torch.tensor([0, 1, 2], dtype=torch.int32)
    meta = functools.partial(fctx.AttentionMetaData, cu_seqlens_q=cu)
    monkeypatch.setattr(fctx, "AttentionMetaData", meta)
    runner = _Runner(layers)
    q = ("q", _t([2, 4], "activation"))
    leaf = _leaf("x", q, ("layer", ir.ContextRef(KEY)), key=KEY)
    found = ops.leaves(_graph(leaf, layers=2)["region"])
    placed = SimpleNamespace(
        block_size=16, batch=lambda view, block, slot: (block, slot)
    )
    prepared, capturing = {}, []

    def call(**kwargs):
        cu = fctx.get_forward_context().attn_metadata.cu_seqlens_q
        if prepared.get("cu") is not cu:
            assert not capturing, "operation not permitted when stream is capturing"
            prepared["cu"] = cu

    @contextlib.contextmanager
    def graph(g, capture_error_mode=None):
        capturing.append(g)
        yield
        capturing.clear()

    def event(enable_timing):
        return SimpleNamespace(
            record=lambda: None, synchronize=lambda: None, elapsed_time=lambda e: 1.0
        )

    monkeypatch.setattr(torch.cuda, "graph", graph)
    monkeypatch.setattr(
        torch.cuda, "CUDAGraph", lambda: SimpleNamespace(replay=lambda: None)
    )
    monkeypatch.setattr(torch.cuda, "Event", event)
    stub = ops.Stub(runner, 5_000)
    with (
        patch.object(ops, "_function", return_value=call),
        patch.object(ops, "_kernels", return_value=(["k"], 1.0)),
    ):
        [record] = stub.point(placed, decode(2), 20, found)

    assert "refused" not in record
    assert record["placements"] == 5


def test_a_replay_prices_each_timed_replay_per_call_each_under_its_context(
    monkeypatch,
):
    # Replay i of the graph takes i + 1 ms on a stand-in clock; warmups are not
    # timed, and a price is per call of the graph.
    clock, seen = [0, 0], []

    class Graph:
        def replay(self):
            clock[1] += 1
            clock[0] += clock[1]

    class Event:
        def __init__(self, enable_timing):
            self.synchronize = lambda: None

        def record(self):
            self.at = clock[0]

        def elapsed_time(self, end):
            return end.at - self.at

    monkeypatch.setattr(fctx, "_forward_context", fctx._forward_context)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", Event)
    calls = [({"x": i}, f"snap{i}") for i in range(4)]

    times = ops._replay(lambda x: seen.append((x, fctx._forward_context)), calls)

    assert seen == [(i, f"snap{i}") for i in range(4)]
    timed = range(ops.WARMUPS, ops.WARMUPS + ops.REPEATS)
    assert times == [(i + 1) * 1e6 / 4 for i in timed]


def test_a_copy_the_profiler_misses_at_one_context_does_not_change_the_kind(
    monkeypatch,
):
    # The same leaf at two contexts of one width, cold time flat: the profiler
    # recorded its device-to-device copy at one and missed it at the other.
    cuda, cpu = torch.autograd.DeviceType.CUDA, torch.autograd.DeviceType.CPU

    def event(name, start, device=cuda):
        span = SimpleNamespace(start=start, elapsed_us=lambda: 1.0)
        return SimpleNamespace(name=name, device_type=device, time_range=span)

    profiled = {
        2436: [
            "Memcpy DtoD (Device -> Device)",
            "attn",
            "Memcpy HtoD (Host -> Device)",
        ],
        452: ["attn"],
    }

    @contextlib.contextmanager
    def profile(activities):
        names = profiled[context]
        yield SimpleNamespace(
            events=lambda: [event("aten::copy_", -1, cpu)]
            + [event(n, -i) for i, n in enumerate(names)]
        )

    monkeypatch.setattr("torch.profiler.profile", profile)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    records = []
    for context in profiled:
        kernels, _ = ops._kernels(lambda: None, {})
        records.append(
            {"shape": "decode B=2", "context": context, "op": "aiter.attn"}
            | {"leaf": "l", "kernels": kernels, "cold_ns": 31_700.0}
        )

    assert [r["kernels"] for r in records] == [["attn"], ["attn"]]
    assert ops._dependent(records) == {"aiter.attn": False}


def test_a_batch_takes_the_blocks_and_slots_it_is_placed_at(rule):  # noqa: F811
    line = _line([[1, 40, True], [1, 17, True]])

    batch = rule("qwen3_next").batch(line, block=10, slot=3)

    assert [list(table) for table in batch.block_tables] == [[10, 11, 12], [13, 14]]
    assert batch.state_slots == [[3], [4]]
    with pytest.raises(ValueError, match="hold 15 KV blocks .* the pool has 14"):
        rule(blocks=14).batch(line, block=10)


@pytest.mark.parametrize(
    "shape, context, rows",
    [
        # Four rows of 40 tokens take 12 of the pool's 14 blocks.
        (decode(4), 40, [[1, 40, True]] * 4),
        # Four of 100 would take 28: the first keeps 7, three share the rest.
        (decode(4), 100, [[1, 100, True]] + [[1, 32, True]] * 3),
        # 51 prefill tokens over two rows, the capture's split, 30 cached before each.
        (prefill(51, 2), 30, [[26, 56, False], [25, 55, False]]),
        (prefill(51, 2), None, [[26, 26, False], [25, 25, False]]),
    ],
)
def test_point_rows_turn_ragged_past_the_pool(rule, shape, context, rows):  # noqa: F811
    got = ops.point_rows(shape, context, 14, 16)

    assert got == rows
    rule(blocks=14).batch({"rows": got, "produces_output": True})


def test_a_point_runs_and_records_the_rows_the_pool_holds(monkeypatch):
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    runner, views = _Runner(), []
    placed = SimpleNamespace(
        block_size=16,
        batch=lambda view, block, slot: views.append(view["rows"]) or (block, slot),
    )
    stub = ops.Stub(runner, 10_000)
    x = ("self", _t([2, 3], "activation"))
    found = ops.leaves(_graph(_leaf("aten.neg.default", x), layers=1)["region"])

    [record] = stub.point(placed, decode(2), 600, found)

    # 38 blocks a row, 64 in the pool: the second row takes the 26 left.
    assert views == [[[1, 600, True], [1, 416, True]]]
    assert record["rows"] == views[0]


@pytest.mark.parametrize(
    "argv, refusal",
    [
        (["--width", "4", "--context", "9", "--gpu", "0,1"], "names 2 GPUs"),
        (["--context", "9", "--gpu", "0"], "name a shape"),
        (["--width", "4", "--gpu", "0"], "a decode shape needs --context"),
    ],
)
def test_measure_ops_refuses_before_it_touches_a_gpu(tmp_path, argv, refusal):
    argv += ["--llc-mib", "256", "--store", str(tmp_path), "--out", str(tmp_path / "o")]

    with pytest.raises(SystemExit, match=f"^compass measure ops: .*{refusal}"):
        cli.main(["measure", "ops", *argv])
    assert not (tmp_path / "o").exists()


def test_a_measured_point_is_keyed_by_the_spec_of_the_platform_it_ran_on(
    tmp_path, monkeypatch
):
    spec = MachineSpec.from_mapping(
        json.loads((TREE / "scripts/compass/machines/mi308x-hjbog39.json").read_text())
    )
    stack = {c: spec.value(f"device.software_pinned_to.{c}") for c in PINNED}
    name, arch = spec.value("device.name"), spec.value("device.arch")
    run = {
        "device": [{"index": 6, "name": name, "arch": arch}],
        "stack": stack,
        "ptl": spec.values.get("device.ptl", "disabled"),
        "host_key": {"model": "sha256:" + "0" * 64, "host_cpu": "cpu"},
        "source": {"revision_kind": "tree", "revision": "a" * 40},
    }
    layout = {"dp": 1, "pp": 1, "pcp": 1, "tp": 1, "decode_context_parallel_size": 1}
    layout |= {"enable_expert_parallel": False, "enable_dp_attention": False}
    monkeypatch.setattr("atom.compass.run.memory.key_fields", lambda c: layout)
    monkeypatch.setattr("atom.compass.run.step_grid.keyed", lambda c: {"level": 0})
    store = ArtifactStore(str(tmp_path / "store"))
    key = ops.keys(None, run)
    measured = {"shape": "decode B=2", "context": 9, "leaf": "a", "cold_ns": 1.0}
    refused = measured | {"leaf": "b", "refused": "no"}

    ops.publish(
        store, key, None, run, "compass measure ops", [measured, refused], tmp_path
    )

    at = key("decode B=2", 9, "a")
    assert at.kind is Kind.OP_MEASUREMENT
    assert at.values["machine_spec"] == machine_spec_digest(spec)
    assert at.values["source_root"] == "tree-" + "a" * 40
    assert store.directory_for(at).exists()
    assert not store.directory_for(key("decode B=2", 9, "b")).exists()
    lines = [json.loads(x) for x in (tmp_path / "ops.jsonl").read_text().splitlines()]
    assert [x["token"] for x in lines] == [
        at.token(),
        key("decode B=2", 9, "b").token(),
    ]


def test_the_capture_child_starts_with_every_gpu_hidden_below_hip(monkeypatch):
    # The capture sets HIP_VISIBLE_DEVICES itself, so only ROCm's own mask
    # keeps it off every GPU.
    started = {}
    monkeypatch.setattr(ops, "_child", lambda fn, argv: started.update(os.environ))
    monkeypatch.setattr("atom.compass.capture.forward.device_arch", lambda: "gfx942")
    config = SimpleNamespace(model="m", tensor_parallel_size=1)

    ops._capture(config, [decode(4)], "store")

    assert started["ROCR_VISIBLE_DEVICES"] == ""

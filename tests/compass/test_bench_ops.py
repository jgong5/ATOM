# SPDX-License-Identifier: MIT
"""`compass measure ops`, the parts that run with no device.

Graphs are built by `trace ops`'s own folding (`capture.graph.build`); the
stub runs beside a stand-in runner whose input preparation sets ATOM's forward
context, as ATOM's runner does, and whose forward raises.
"""

import collections
import contextlib
import dataclasses
import functools
import gc
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
from atom.compass.ir.nodes import Reduced
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
        self.pool_plan, self.kv_held = SimpleNamespace(entries={}), {}
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


def test_leaves_share_a_metadata_view_s_clone_and_placements_share_it_unchanged():
    runner = _Runner()
    runner.forward_vars["scratch"] = torch.zeros(6)
    stub = ops.Stub(runner, 10_000)
    tables = _t([2, 8], "metadata", dtype="torch.int32")
    scratch = ("out", _t([6], "metadata"))
    stub.leaves = ops.leaves(
        _graph(
            _leaf("aten.add_.Tensor", ("self", tables), scratch),
            _leaf("aten.mul_.Tensor", ("self", tables), scratch),
            layers=1,
        )["region"]
    )

    for p in range(2):
        stub._prepare((4 * p, 2 * p))

    def views(p, path):
        return [v for (_, at), (_, v) in stub.placed[p][1].items() if at == path]

    # both leaves read one clone at a placement; the block tables changed
    # between placements, the scratch did not
    (a0, b0), (a1, b1) = views(0, "self"), views(1, "self")
    assert a0 is b0 and a1 is b1 and a0 is not a1
    assert len({id(v) for p in (0, 1) for v in views(p, "out")}) == 1


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
    # layers is timed on random weights, one a layer, on what ATOM prepared:
    # its 60 bytes a call pass twice a 30-byte cache in 2 copies.
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

    [record] = ops.measure(None, [(decode(2), 20, found)], 30)

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


def test_a_v4_placement_s_csa_top_k_names_rows_of_its_own_blocks():
    # Decode rows at positions 1023, 511 and 15 see 256, 128 and 4 compressed
    # rows at ratio 4, 64 to a block: 4, 2 and 1 blocks. Each slice is the
    # top-k section, 8 ids or as many as the row sees, then a window of 4 the
    # builder wrote.
    class Csa(torch.nn.Module):
        # ATOM's CSA layer, whose package needs aiter's JIT to import: its fill
        # writes each token's top-k ids at the head of its slice as rows of
        # the blocks they fall in, as `csa_translate_pack` does
        indexer, compress_ratio = SimpleNamespace(index_topk=8), 4

        def _fill_csa_paged_compress(self, md, topk, positions, tokens):
            for t in range(tokens):
                start = int(md.kv_indptr_csa[t])
                k = int(md.kv_indptr_csa[t + 1]) - start - min(int(positions[t]) + 1, 4)
                ids = topk[t, :k].long()
                blocks = md.block_tables[md.batch_id_per_q_token[t], ids // 64]
                rows = blocks * md.envelope_rows + ids % 64
                md.kv_indices_csa[start : start + k] = rows

    i32 = functools.partial(torch.tensor, dtype=torch.int32)
    runner, envelope = _Runner(), 100
    runner.model.append(Csa())

    def prepare_model(batch):
        b = batch[0]
        tables = i32([[b, b + 1, b + 2, b + 3], [b + 4, b + 5, 0, 0], [b + 6, 0, 0, 0]])
        meta = fctx.AttentionMetaData(block_tables=tables, state=fctx.AttnState.DECODE)
        vars(meta).update(
            kv_indptr_csa=i32([0, 12, 24, 32]),
            kv_indices_csa=torch.full((32,), -1, dtype=torch.int32),
            batch_id_per_q_token=i32([0, 1, 2]),
            envelope_rows=envelope,
        )
        context = fctx.Context(positions=torch.tensor([1023, 511, 15]))
        fctx._forward_context = fctx.ForwardContext(attn_metadata=meta, context=context)
        return torch.zeros(3, dtype=torch.int32), None, None, None, True, False

    runner.prepare_model = prepare_model
    stub = ops.Stub(runner, 10_000)
    stub.leaves = {}

    for p in range(2):
        stub._prepare((8 * p, 0))

    sections = []
    for p, (snap, _) in enumerate(stub.placed):
        ids = snap.attn_metadata.kv_indices_csa.long()
        rows = [ids[:8], ids[12:20], ids[24:28]]
        assert (
            ids[8:12].eq(-1).all() and ids[20:24].eq(-1).all() and ids[28:].eq(-1).all()
        )
        for got, blocks in zip(rows[:2], ([0, 1, 2, 3], [4, 5])):
            assert set((got // envelope).tolist()) == {8 * p + b for b in blocks}
            assert (got % envelope < 64).all() and len(set(got.tolist())) == 8
        assert rows[2].tolist() == [(8 * p + 6) * envelope + i for i in range(4)]
        sections.append(ids)
    assert not torch.equal(*sections)


def test_a_v4_metadata_slot_is_a_view_of_the_buffer_atom_passes_or_refuses():
    # Each buffer ATOM passes is wider than one of its dtype `_meta`'s rules
    # take: one of the slot's own spec, or a tighter one its name also holds.
    # No HCA indptr is live, so the persistent decode's refuses past a decoy.
    nm, ps = "aiter.mla:mla_decode_fwd_v4_nm", "aiter.mla_decode_v4_ps_asm.default"
    csa = "aiter.ops.flydsl.kernels.fused_compress_attn:flydsl_fused_compress_attn"
    state = "atom.model_ops.v4_kernels.state_writes:update_compressor_states"
    qkn = "aiter._fused_qk_norm_rope_group_quant_kernel.default"
    hca = "aiter.ops.flydsl.kernels.fused_compress_attn_hca:flydsl_hca_compress_attn"
    bf16 = "atom.model_ops.v4_kernels.paged_decode:_sparse_attn_v4_paged_decode_triton"

    def i32(*shape):
        return torch.zeros(shape, dtype=torch.int32)

    def meta(slot, *shape):
        return slot, _t(list(shape), "metadata", dtype="torch.int32")

    runner = _Runner()
    runner.forward_vars |= {
        "context_lens": i32(3),
        "top_ks": i32(4, 4),
        "v4_compress_plan_4": i32(8, 4),
        "v4_compress_plan_128": i32(8, 4),
        "v4_write_plan_4": i32(4, 4),
        "v4_write_plan_128": i32(8, 4),
    }
    v4 = {
        "kv_indptr_swa": i32(5),
        "kv_indptr_csa": i32(9),
        "state_slot_in": i32(8),
        "state_slot_out": i32(8),
        "batch_id_per_q_token": i32(8),
    }
    prepare = runner.prepare_model

    def prepare_model(batch):
        prepared = prepare(batch)
        vars(fctx.get_forward_context().attn_metadata).update(v4)
        return prepared

    runner.prepare_model = prepare_model
    slots = meta("state_slot_mapping", 3)
    leaves = [
        dataclasses.replace(
            _leaf(name, meta("kv_indptr", 5)), stated=(("compress_ratio", r),)
        )
        for name, r in ((nm, 4), (ps, 128), (bf16, 4))
    ]
    leaves += [
        _leaf(csa, ("ratio", 4), slots, meta("plan_gpu", 4, 4)),
        _leaf(hca, ("ratio", 128), slots, meta("plan_gpu", 4, 4)),
        _leaf(state, ("ratio", 128), slots, meta("write_plan", 4, 4)),
        _leaf(qkn, meta("batch_id_per_token", 3)),
    ]
    stub = ops.Stub(runner, 10_000)
    stub.leaves = ops.leaves(_graph(*leaves, layers=1)["region"])

    stub._prepare((0, 0))

    [(_, views)] = stub.placed
    got = {(stub.leaves[s][0]["op"], p): n for (s, p), (n, _) in views.items()}
    assert got == {
        (nm, "kv_indptr"): "kv_indptr_csa",
        (ps, "kv_indptr"): "ATOM passes kv_indptr kv_indptr_hca: not live",
        (bf16, "kv_indptr"): "kv_indptr_csa",
        (csa, "state_slot_mapping"): "state_slot_in",
        (csa, "plan_gpu"): "v4_compress_plan_4",
        (hca, "state_slot_mapping"): "state_slot_in",
        (hca, "plan_gpu"): "v4_compress_plan_128",
        (state, "state_slot_mapping"): "state_slot_out",
        (state, "write_plan"): "v4_write_plan_128",
        (qkn, "batch_id_per_token"): "batch_id_per_q_token",
    }


def test_v4_index_writer_takes_atom_s_state_slots_and_dest_rows_by_ratio():
    # Bound by size, the state slots were `context_lens` and every destination
    # row `top_ks`. The dict's items are out of ratio order, so an item binds
    # by its key, not its place; no HCA buffer is live, so its item refuses.
    write = (
        "atom.model_ops.v4_kernels.paged_decode_indices:write_v4_paged_decode_indices"
    )
    rows = _t([8], "metadata", dtype="torch.int32")
    runner = _Runner()
    runner.forward_vars |= {
        "context_lens": torch.zeros(3, dtype=torch.int32),
        "top_ks": torch.zeros(8, dtype=torch.int32),
        "v4_meta_state_slot_out": torch.zeros(16, dtype=torch.int32),
        "v4_swa_dest_dense": torch.zeros(16, dtype=torch.int32),
        "v4_swa_dest_csa": torch.zeros(32, dtype=torch.int32),
    }
    leaf = _leaf(
        write,
        ("state_slot_per_seq", _t([3], "metadata", dtype="torch.int32")),
        ("dest_rows", Reduced("builtins:dict", ((128, rows), (0, rows), (4, rows)))),
    )
    stub = ops.Stub(runner, 10_000)
    stub.leaves = ops.leaves(_graph(leaf, layers=1)["region"])

    stub._prepare((0, 0))

    [(_, views)] = stub.placed
    assert {p: n for (_, p), (n, _) in views.items()} == {
        "state_slot_per_seq": "v4_meta_state_slot_out",
        "dest_rows.0.1": "ATOM passes dest_rows.0.1 v4_swa_dest_hca: not live",
        "dest_rows.1.1": "v4_swa_dest_dense",
        "dest_rows.2.1": "v4_swa_dest_csa",
    }


def test_v4_qk_norm_takes_the_dest_rows_of_its_stated_ratio():
    # Bound by size, every ratio's destination rows were `top_ks`. No HCA
    # buffer is live, so the ratio-128 leaf refuses by that buffer's name.
    qkn = "aiter._fused_qk_norm_rope_group_quant_kernel.default"
    runner = _Runner()
    runner.forward_vars |= {
        "top_ks": torch.zeros(8, dtype=torch.int32),
        "v4_swa_dest_dense": torch.zeros(16, dtype=torch.int32),
        "v4_swa_dest_csa": torch.zeros(32, dtype=torch.int32),
    }
    rows = ("swa_dest_row", _t([8], "metadata", dtype="torch.int32"))
    leaves = [
        dataclasses.replace(_leaf(qkn, rows), stated=(("compress_ratio", r),))
        for r in (0, 4, 128)
    ]
    stub = ops.Stub(runner, 10_000)
    stub.leaves = ops.leaves(_graph(*leaves, layers=1)["region"])

    stub._prepare((0, 0))

    [(_, views)] = stub.placed
    got = {
        stub.leaves[s][0]["stated"]["compress_ratio"]: n
        for (s, _), (n, _) in views.items()
    }
    assert got == {
        0: "v4_swa_dest_dense",
        4: "v4_swa_dest_csa",
        128: "ATOM passes swa_dest_row v4_swa_dest_hca: not live",
    }


def test_v4_index_leaves_write_the_indptrs_atom_passes_and_a_decode_its_split_plan(
    monkeypatch,
):
    # The leaves of a placement share a metadata view's clone, and ATOM's
    # index builder writes its three indptrs. Bound by size, all three were
    # `cu_seqlens_q`, which the CSA decode then read as its split plan.
    indices = "atom.model_ops.v4_kernels.paged_decode_indices"
    build, write = (
        f"{indices}:build_v4_paged_decode_indptr",
        f"{indices}:write_v4_paged_decode_indices",
    )
    nm = "aiter.mla:mla_decode_fwd_v4_nm"
    cu = "atom.model_ops.decode_input_ids:fill_deferred_decode_ids"
    indptr = _t([3], "metadata", dtype="torch.int32")
    v4 = {
        "cu_seqlens_q": torch.arange(3, dtype=torch.int32),
        **{
            f"kv_indptr_{k}": torch.zeros(9, dtype=torch.int32)
            for k in ("swa", "csa", "hca")
        },
        # aiter's uniform plan of 5 splits a row, a row of ATOM's table
        "split_plan_csa": (
            5,
            (torch.arange(1, 17)[:, None] * torch.arange(9)).int()[4],
        ),
    }
    runner, read = _Runner(), {}
    prepare = runner.prepare_model

    def prepare_model(batch):
        prepared = prepare(batch)
        vars(fctx.get_forward_context().attn_metadata).update(v4)
        return prepared

    def written(swa_indptr, csa_indptr, hca_indptr):
        for t, per_row in ((swa_indptr, 2), (csa_indptr, 6), (hca_indptr, 3)):
            t.copy_(torch.arange(3) * per_row)

    calls = {
        build: written,
        write: lambda csa_indptr: read.update(write=csa_indptr.tolist()),
        nm: lambda kv_indptr, split_indptr: read.update(
            nm=kv_indptr.tolist(), split=split_indptr.tolist()
        ),
        cu: lambda cu: read.update(cu=cu.tolist()),
    }

    def time(self, op, sig, instances, n):
        calls[op["op"]](**self._copy(op, sig, instances, 0)[0])
        return {"cold_ns": 1.0}

    runner.prepare_model = prepare_model
    monkeypatch.setattr(ops.Stub, "_time", time)
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    found = ops.leaves(
        _graph(
            _leaf(build, *((f"{k}_indptr", indptr) for k in ("swa", "csa", "hca"))),
            _leaf(write, ("csa_indptr", indptr)),
            dataclasses.replace(
                _leaf(nm, ("kv_indptr", indptr), ("split_indptr", indptr)),
                stated=(("compress_ratio", 4),),
            ),
            _leaf(cu, ("cu", indptr)),
            layers=1,
        )["region"]
    )
    placed = SimpleNamespace(block_size=16, batch=lambda view, b, s: (b, s))

    got = ops.Stub(runner, 5_000).point(placed, decode(2), 20, found)

    assert [r for r in got if "refused" in r] == []
    assert read == {
        "write": [0, 6, 12],
        "nm": [0, 6, 12],
        "split": [0, 5, 10],
        "cu": [0, 1, 2],
    }


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


def test_a_context_module_s_pool_view_is_sized_by_the_batch_s_blocks(monkeypatch):
    # DeepSeek-V4's Indexer as ATOM binds it: a view of the pool a block a row,
    # FP4 data and scales, beside tensors of its own, one 0-dim; no layer_num.
    def indexer():
        module = torch.nn.Module()
        module.register_buffer("kv_cache", torch.zeros(64, 16, 4))
        module.register_buffer("own", torch.zeros(3))
        module.register_buffer("scale", torch.zeros(()))
        module.kv_scale = torch.zeros(64, 16)
        return module

    runner = _Runner({KEY.format(L=i): indexer() for i in (0, 1)})
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    monkeypatch.setattr(ops.Stub, "_time", lambda *_: {"cold_ns": 1.0})
    placed = SimpleNamespace(block_size=16, batch=lambda view, b, s: (b, s))
    q = ("q", _t([2, 4], "activation"))
    leaf = _leaf("aiter.indexer", q, ("layer", ir.ContextRef(KEY)), key=KEY)
    found = ops.leaves(_graph(leaf, layers=2)["region"])

    [record] = ops.Stub(runner, 5_000).point(placed, decode(2), 20, found)

    # Two rows of 20 tokens read 4 of the pool's 64 blocks, 320 bytes each,
    # beside 32 of q and 16 of its own: 1328 a call, 8 copies past twice 5000,
    # cycling the two layers over 4 placements of the batch.
    assert record["footprint_bytes"] == 8 * 1328
    assert record["placements"] == 4
    assert runner.batches == [(4 * p, 2 * p) for p in range(4)]


def test_a_kv_argument_is_a_view_of_the_runner_s_kv_and_its_copies_cycle_placements(
    monkeypatch,
):
    # DeepSeek-V4: the pools are runner attributes, each layer's view of the
    # plane from its base row to the end, the state slots a strided view; the
    # capture's pool was sized otherwise.
    plane, state = torch.zeros(100, 4), torch.zeros(6, 2, 4)
    runner = _Runner()
    runner.pool_plan = SimpleNamespace(entries={"state": 6})
    runner.kv_held = {
        "v4_unified_kv": [plane, plane[2:]],
        "v4_state": state,
        "num_kv_heads": 1,
    }
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    kv = ("kv", _t([60, 1, 1, 4], "kv", stride=[4, 4, 4, 1]))
    found = ops.leaves(
        _graph(
            _leaf(
                "x", ("q", _t([2, 4], "activation")), kv, ("st", _t([5, 2, 4], "kv"))
            ),
            _leaf("y", ("kv", _t([500, 4], "kv"))),
            layers=1,
        )["region"]
    )
    placed = SimpleNamespace(
        block_size=16, batch=lambda view, block, slot: (block, slot)
    )
    graphs = []

    def replay(fn, calls):
        graphs.append(calls)
        return [1.0] * ops.REPEATS

    with (
        patch.object(ops, "_function", return_value=lambda **kwargs: None),
        patch.object(ops, "_kernels", return_value=(["k"], 1.0)),
        patch.object(ops, "_replay", side_effect=replay),
    ):
        got = {
            r["op"]: r
            for r in ops.Stub(runner, 1000).point(placed, decode(2), 20, found)
        }

    # 32 bytes of q, and per state slot of the two rows 16 of kv and 32 of
    # state: 128 a call, 16 copies past twice 1000 bytes, each its own
    # placement, which the pool's 64 blocks hold; 6 slots hold 3 of two rows,
    # and the placements past them take those slots again.
    record = got["x"]
    assert record["metadata"] == {"kv": "v4_unified_kv.1", "st": "v4_state"}
    assert (record["copies"], record["placements"]) == (16, 16)
    assert runner.batches == [(4 * p, 2 * p % 6) for p in range(16)]
    for c, (kwargs, snap) in enumerate(graphs[0][:16]):
        assert kwargs["kv"].untyped_storage().data_ptr() == plane.data_ptr()
        assert kwargs["kv"].storage_offset() == 8
        assert kwargs["kv"].shape == (60, 1, 1, 4)
        assert kwargs["st"].data_ptr() == state.data_ptr()
        assert set(snap.attn_metadata.block_tables.flatten().tolist()) == {4 * c}
    assert "ATOM's live KV has no room for kv [500, 4]" in got["y"]["refused"]


def test_a_compressor_state_is_a_layer_of_its_own_field_and_every_slot_stays_in_it():
    # DeepSeek-V4's compressor state: a field per compressor kind, each a
    # `[layers, slots, rows, dim]` view strided by its plane's slot. The HCA
    # state sits in the wide plane, the CSA indexer's in the narrow one after
    # it, the tighter; the capture's pool held fewer slots.
    from atom.model_ops.attentions.pool_layout.entry_arena import (
        EntryField,
        EntryMajorArena,
    )

    slots, f32 = 8, torch.float32
    pool = torch.zeros(slots * (4096 + 1024), dtype=torch.uint8)
    wide, narrow = pool[: slots * 4096], pool[slots * 4096 :]
    hca = EntryMajorArena(
        [EntryField("hca", 2, (4, 8), f32)], slots, "cpu", wide, slot_stride=4096
    )
    idx = EntryMajorArena(
        [EntryField("idx", 2, (2, 4), f32)], slots, "cpu", narrow, slot_stride=1024
    )
    runner = SimpleNamespace(
        kv_held={
            "v4_hca_main_kv_state": hca.view("hca"),
            "v4_csa_idx_kv_state": idx.view("idx"),
        }
    )
    spec = {"shape": [2, 4, 8], "stride": [1024, 8, 1], "dtype": F32, "role": "kv"}

    name, view = ops._kv(spec, "kv_state", ops._held(runner, {}))

    # the end of the last slot ATOM hands out, at the view's slot stride
    last = view.storage_offset() + (slots - 1) * view.stride(0) + 4 * 8
    assert last <= pool.numel() // f32.itemsize
    assert name == "v4_hca_main_kv_state.1"
    assert view.storage_offset() == hca.view("hca")[1].storage_offset()


def test_a_v4_compressor_s_kv_cache_is_its_layer_s_block_view_and_no_other():
    # DeepSeek-V4's main compressors write their layer's rows of every block
    # of the plane, a view strided by a block's rows of every layer that no
    # runner attribute holds: 4 a block in a CSA layer, 2 in an HCA one. A
    # layer's view of the plane from its base row to the end is the tighter.
    plane = torch.zeros(8 * 10, 4)
    blocks = plane.view(8, 10, 4)

    def layer(base, rows):
        compressor = torch.nn.Module()
        compressor.kv_cache, compressor.kv_cache_rope = (
            blocks[:, base : base + rows],
            None,
        )
        return torch.nn.ModuleDict(
            {"attn": torch.nn.ModuleDict({"compressor": compressor})}
        )

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([layer(0, 4), layer(4, 2), layer(6, 4)])
    runner = SimpleNamespace(
        model=model, kv_held={"v4_unified_kv": [plane, plane[30:]]}
    )
    held = ops._held(runner, {})

    def bound(rows):
        spec = {"shape": [6, rows, 4], "stride": [40, 4, 1], "dtype": F32, "role": "kv"}
        return ops._kv(spec, "kv_cache", held)

    (hca, hca_view), (csa, csa_view), (other, none) = bound(2), bound(4), bound(3)
    assert (hca, hca_view.storage_offset()) == ("layers.1.attn.compressor.kv_cache", 16)
    assert (csa, csa_view.storage_offset()) == ("layers.2.attn.compressor.kv_cache", 24)
    assert none is None
    assert other.startswith("ATOM's KV holds no tensor laid out as kv_cache [6, 3, 4]")


@pytest.mark.parametrize("faulted", [False, True])
def test_a_rank_whose_device_faulted_dies_and_one_that_refused_measures_on(
    monkeypatch, tmp_path, faulted
):
    # A HIP fault is sticky: a rank that refused the leaf and went on would
    # hold its peers in their next collective until that timed out. One that
    # journaled the leaf first would have the next leaf refused in its place.
    def synchronize(device):
        if faulted:
            raise RuntimeError("HIP error: an illegal memory access was encountered")

    exits, journal = [], tmp_path / "points.jsonl"
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(
        ops.os, "_exit", lambda code: exits.append((code, journal.exists()))
    )
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    monkeypatch.setattr(ops.Stub, "_time", lambda *_: 1 / 0)
    runner = _Runner()
    runner.device = torch.device("cuda", 0)
    placed = SimpleNamespace(block_size=16, batch=lambda view, b, s: (b, s))
    found = ops.leaves(
        _graph(_leaf("x", ("q", _t([2, 4], "activation"))), layers=1)["region"]
    )

    [record] = ops.Stub(runner, 5_000, journal).point(placed, decode(2), 20, found)

    assert record["refused"] == "ZeroDivisionError: division by zero"
    # the journal holds no record of the leaf when the rank ends
    assert exits == ([(1, False)] if faulted else [])


def test_a_plane_argument_is_sized_by_the_rows_of_the_batch_s_blocks(monkeypatch):
    # DeepSeek-V4's decode attention states its layer's window and compress
    # ratio and reads a plane indexed by rows, beside a model's state slots;
    # a leaf stating the ratio alone writes a row per state slot; one stating
    # an indexer's top-k reads at most that many rows a request.
    runner = _Runner()
    runner.pool_plan = SimpleNamespace(entries={"state": 7})
    runner.config.kv_cache_block_size = 16
    runner.kv_held = {"v4_unified_kv": [torch.zeros(100, 4)]}
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    monkeypatch.setattr(ops.Stub, "_time", lambda *_: {"cold_ns": 1.0})
    leaf = _leaf(
        "aiter.decode", ("q", _t([2, 4], "activation")), ("kv", _t([60, 4], "kv"))
    )
    found = ops.leaves(
        _graph(
            dataclasses.replace(
                leaf, stated=(("window_size", 8), ("compress_ratio", 4))
            ),
            dataclasses.replace(
                leaf, name="aiter.write", stated=(("compress_ratio", 4),)
            ),
            dataclasses.replace(
                leaf,
                name="aiter.sparse",
                stated=(("window_size", 8), ("compress_ratio", 4), ("index_topk", 3)),
            ),
            layers=1,
        )["region"]
    )
    placed = SimpleNamespace(block_size=16, batch=lambda view, b, s: (b, s))

    got = {
        r["op"]: r for r in ops.Stub(runner, 1000).point(placed, decode(2), 20, found)
    }

    # Two rows of 20 tokens hold 4 blocks of 16, 16 rows at ratio 4: 256 bytes
    # of kv and 32 of q a call, 7 copies past twice 1000 bytes. Per slot, 32
    # of kv: 32 copies, each its own placement, of which the pool holds 16;
    # 7 slots hold 3 of two rows, the rest take the first 6 again. Past 4 x 3
    # tokens a request reads 3 of its 5 rows: 96 bytes of kv, 16 copies.
    assert got["aiter.decode"]["footprint_bytes"] == 7 * 288
    assert got["aiter.write"]["footprint_bytes"] == 32 * 64
    assert got["aiter.sparse"]["footprint_bytes"] == 16 * 128
    assert got["aiter.decode"]["placements"] == 16
    assert runner.batches == [(4 * p, 2 * p % 6) for p in range(16)]


@pytest.mark.parametrize(
    "entries, footprint", [({"state": 7}, 21 * 96), ({}, 13 * 160)]
)
def test_a_strided_kv_argument_is_sized_by_its_entry_s_span(
    monkeypatch, entries, footprint
):
    # DeepSeek-V4's compressor views: an entry of 2 rows of 4 floats, strided
    # by 16 rows, as a state slot is by the whole slot and a block by every
    # layer's rows.
    runner = _Runner()
    runner.pool_plan = SimpleNamespace(entries=entries)
    runner.kv_held = {"v4_pool": torch.zeros(1000)}
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    monkeypatch.setattr(ops.Stub, "_time", lambda *_: {"cold_ns": 1.0})
    kv = ("kv", _t([8, 2, 4], "kv", stride=[64, 4, 1]))
    leaf = _leaf("aiter.compress", ("q", _t([2, 4], "activation")), kv)
    found = ops.leaves(_graph(leaf, layers=1)["region"])
    placed = SimpleNamespace(block_size=16, batch=lambda view, b, s: (b, s))

    [record] = ops.Stub(runner, 1000).point(placed, decode(2), 20, found)

    # 32 bytes of q and 32 of kv an entry: per state slot of the two rows 96 a
    # call, 21 copies past twice 1000 bytes; with no state, per block of the 4
    # the rows hold, 160 a call and 13 copies.
    assert record["footprint_bytes"] == footprint


@pytest.mark.parametrize(
    "shape, footprint", [(decode(2), 7 * 288), (prefill(6, 2), 3 * 800)]
)
def test_a_rope_table_adds_a_row_a_token_not_its_span(monkeypatch, shape, footprint):
    # DeepSeek-V4's compressor: cos and sin tables of 1000 positions, a row
    # of 32 bf16 at each, read one row at a token's position.
    runner = _Runner()
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    monkeypatch.setattr(ops.Stub, "_time", lambda *_: {"cold_ns": 1.0})
    table = _t([1000, 1, 1, 32], "weight", [32, 32, 32, 1], "torch.bfloat16")
    q = ("q", _t([2, 4], "activation"))
    leaf = _leaf("aiter.compress", q, ("cos_cache", table), ("sin_cache", table))
    found = ops.leaves(_graph(leaf, layers=1)["region"])
    placed = SimpleNamespace(block_size=16, batch=lambda view, b, s: (b, s))

    [record] = ops.Stub(runner, 1000).point(placed, shape, 20, found)

    # 32 bytes of q and 64 a token from each table: 2 decode tokens make 288 a
    # call, 7 copies past twice 1000 bytes; 6 prefill tokens 800 and 3 copies.
    # By span, the tables' 128000 bytes would make one copy.
    assert record["footprint_bytes"] == footprint


def test_a_slot_named_as_a_v4_hca_workspace_field_is_a_view_of_that_field(
    monkeypatch,
):
    # aiter's workspace is a NamedTuple, one per device: this runner's is at
    # index 1, another's at 0. A live buffer of o_acc's own spec and one with
    # room for cnt would take them by `_meta`'s rules. The module stands in for
    # hca_persist, whose package needs aiter's JIT to import.
    work = collections.namedtuple("Workspace", "o_acc cnt")
    other = work(torch.randn(4, 2, 3), torch.arange(5, dtype=torch.int32))
    workspace = work(torch.randn(4, 2, 3), torch.arange(7, 12, dtype=torch.int32))
    hca = SimpleNamespace(_workspaces={0: other, 1: workspace})
    monkeypatch.setitem(sys.modules, "atom.model_ops.v4_kernels.hca_persist", hca)
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    runner = _Runner()
    runner.device = torch.device("cpu", 1)
    runner.forward_vars["partials"] = torch.zeros(4, 2, 3)
    cnt = ("cnt", _t([5], "metadata", dtype="torch.int32"))
    found = ops.leaves(
        _graph(
            _leaf(ops.HCA_DECODE, ("o_acc", _t([4, 2, 3], "metadata")), cnt),
            # another operator's slot of a field's name keeps the live pick
            _leaf("aten.add_.Tensor", cnt),
            _leaf(ops.HCA_DECODE, ("cnt", _t([9], "metadata", dtype="torch.int32"))),
            layers=1,
        )["region"]
    )
    placed = SimpleNamespace(
        block_size=16, batch=lambda view, block, slot: (block, slot)
    )
    graphs = []

    def replay(fn, calls):
        graphs.append(calls)
        return [1.0] * ops.REPEATS

    with (
        patch.object(ops, "_function", return_value=lambda **kwargs: None),
        patch.object(ops, "_kernels", return_value=(["k"], 1.0)),
        patch.object(ops, "_replay", side_effect=replay),
    ):
        got = ops.Stub(runner, 1000).point(placed, decode(2), 20, found)

    asm, add, wide = got
    assert asm["metadata"] == {"o_acc": "o_acc", "cnt": "cnt"}
    assert add["metadata"] == {"cnt": "block_tables"}
    assert "ATOM's live HCA workspace has no room for cnt" in wide["refused"]
    for kwargs, _ in graphs[0]:
        assert torch.equal(kwargs["o_acc"], workspace.o_acc)
        assert torch.equal(kwargs["cnt"], workspace.cnt)


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


def test_a_packed_fp4_activation_and_weight_are_built_at_their_shape_and_stride():
    # No cast reaches torch.float4_e2m1fn_x2, so a draw cast to it refuses.
    stub = ops.Stub(_Runner(), 1)
    x = {
        "shape": [2, 3],
        "dtype": "torch.float4_e2m1fn_x2",
        "stride": [4, 1],
        "role": "activation",
    }
    built = [stub._tensor(x, None, False), stub._weight(x | {"role": "weight"}, 0, 2)]

    for t in built:
        assert t.dtype == torch.float4_e2m1fn_x2
        assert (t.shape, t.stride()) == ((2, 3), (4, 1))
        assert t.view(torch.uint8).unique().numel() > 1


def test_a_weight_captured_preshuffled_is_built_marked_and_only_that_one():
    # aiter's fused_moe_ picks its kernel by its weights' `is_shuffled`, which
    # no shape or dtype shows: a leaf built without it runs another kernel.
    moe = _leaf(
        "aiter.fused_moe_",
        ("w1", ir.Tensor((4, 3), F32, (3, 1), "weight", shuffled=True)),
        ("w2", _t([3, 4], "weight")),
    )
    [(op, _)] = ops.leaves(_graph(moe)["region"]).values()
    stub = ops.Stub(_Runner(), 1)

    copies = [stub._weight(op["args"]["w1"], c, 2) for c in (0, 1)]

    assert all(t.is_shuffled is True for t in copies)
    assert not hasattr(stub._weight(op["args"]["w2"], 0, 2), "is_shuffled")


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


def test_a_leaf_that_occurs_once_takes_a_weight_per_cache_sized_copy(monkeypatch):
    # A [2, 3] input and a [3, 3] weight are 60 bytes a call: 3 copies pass
    # twice a 60-byte cache. The leaf occurs once in the step, and each copy
    # still takes a weight of its own, as it takes an input of its own.
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    stub = ops.Stub(_Runner(), 60)
    mm = _leaf(
        "aten.mm.default",
        ("self", _t([2, 3], "activation")),
        ("mat2", _t([3, 3], "weight")),
    )
    found = ops.leaves(_graph(mm, layers=1)["region"])
    placed = SimpleNamespace(
        block_size=16, batch=lambda view, block, slot: (block, slot)
    )
    monkeypatch.setattr(ops, "_kernels", lambda fn, kwargs: (["k"], 1.0))
    graphs = []
    monkeypatch.setattr(
        ops, "_replay", lambda fn, calls: graphs.append(calls) or [1.0] * ops.REPEATS
    )

    [record] = stub.point(placed, decode(2), 20, found)

    cold = [kwargs for kwargs, _ in graphs[0]]
    assert [op["occurrences"] for op, _ in found.values()] == [1]
    assert len({id(c["mat2"]) for c in cold[:3]}) == 3
    assert all(c["mat2"] is cold[i % 3]["mat2"] for i, c in enumerate(cold))
    assert record["copies"] == 3


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


def test_a_capture_waits_for_the_driver_to_clear_a_release_and_a_capped_wait_is_recorded(
    monkeypatch,
):
    # Each capture returns 8 GiB the allocator held. The cold graph's clear
    # lands at the card's third reading after its release, and its wait ends
    # there; the warm graph's never lands, and its wait ends at the cap.
    gib = 1 << 30
    readings = [20 * gib, 20 * gib, 20 * gib, 12 * gib, 20 * gib]
    read, captured = [], []
    monkeypatch.setattr(ops, "CLEAR_S", 0.05)
    monkeypatch.setattr(ops, "_waited", ops.Counter())
    monkeypatch.setattr(
        ops, "_vram_used", lambda: read.append(1) or readings[min(len(read), 5) - 1]
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", iter([8 * gib, 0] * 2).__next__)

    @contextlib.contextmanager
    def graph(g, capture_error_mode=None):
        captured.append(len(read))
        yield

    def event(enable_timing):
        return SimpleNamespace(
            record=lambda: None, synchronize=lambda: None, elapsed_time=lambda e: 1.0
        )

    monkeypatch.setattr(torch.cuda, "graph", graph)
    monkeypatch.setattr(
        torch.cuda, "CUDAGraph", lambda: SimpleNamespace(replay=lambda: None)
    )
    monkeypatch.setattr(torch.cuda, "Event", event)
    stub = ops.Stub(_Runner(), 10_000)
    stub.placed = [(fctx.ForwardContext(), {})]
    with (
        patch.object(ops, "_function", return_value=lambda **kwargs: None),
        patch.object(ops, "_kernels", return_value=(["k"], 1.0)),
    ):
        record = stub._time({"op": "x", "args": {"n": 1}}, "sig", [{}], 8)

    assert captured[0] == 4  # the cold graph captured once the 12 GiB reading came
    assert len(read) > 6  # the warm graph's wait read until the cap
    assert record["release_capped"] == 1
    assert ops.CLEAR_S <= record["release_wait_s"] < ops.CLEAR_S + 1
    assert not ops._waited  # taken by the point


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


def test_a_capture_that_fails_refuses_its_leaf_and_the_next_draws_its_activations(
    monkeypatch,
):
    # HIP's capture end refuses a graph a call invalidated and leaves its
    # stream current and still capturing, and the generator in capture: a
    # draw then raises, and so does the next capture on that stream. A graph
    # a garbage collection frees in a capture would end the process.
    world = {"stream": "main", "capturing": set(), "generator": "free"}
    collecting = []
    randn = torch.randn

    def draw(*args, **kwargs):
        if world["generator"] != "free":
            raise RuntimeError("Offset increment outside graph capture")
        return randn(*args, **kwargs)

    class Capture:
        capture_stream = "capture"

        def __init__(self, graph, capture_error_mode):
            restore = lambda *_: world.update(stream=self.found)
            self.stream_ctx = SimpleNamespace(__exit__=restore)

        def __enter__(self):
            if "capture" in world["capturing"]:
                raise RuntimeError("Cannot register the state during capturing")
            self.found = world["stream"]
            collecting.append(gc.isenabled())
            world.update(stream="capture", generator="capturing")
            world["capturing"].add("capture")

        def __exit__(self, *exc):
            if world.pop("invalid", False):
                raise RuntimeError("operation failed due to a previous error")
            world["capturing"].discard("capture")
            world["generator"] = "free"
            self.stream_ctx.__exit__()

    @contextlib.contextmanager
    def stream(s):
        found, world["stream"] = world["stream"], s
        yield
        world["stream"] = found

    def function(name):
        def call(**kwargs):
            if name == "aten.neg.default" and world["stream"] == "capture":
                world["invalid"] = True

        return call

    generator = SimpleNamespace(
        clone_state=lambda: "fresh",
        graphsafe_set_state=lambda s: world.update(generator=s == "fresh" and "free"),
    )
    event = SimpleNamespace(record=lambda: None, synchronize=lambda: None)
    event.elapsed_time = lambda end: 1.0
    for name, value in {
        "CUDAGraph": lambda: SimpleNamespace(replay=lambda: None),
        "graph": Capture,
        "Event": lambda enable_timing: event,
        "current_stream": lambda: world["stream"],
        "stream": stream,
        "is_current_stream_capturing": lambda: world["stream"] in world["capturing"],
        "default_generators": (generator,),
        "current_device": lambda: 0,
    }.items():
        monkeypatch.setattr(torch.cuda, name, value)
    monkeypatch.setattr(torch, "randn", draw)
    monkeypatch.setattr(ops, "_end_capture", world["capturing"].discard, raising=False)
    monkeypatch.setattr(ops, "_function", function)
    monkeypatch.setattr(ops, "_kernels", lambda fn, kwargs: (["k"], 1.0))
    monkeypatch.setattr(ops.time, "sleep", lambda s: None)
    time = ops.Stub._time
    _, [_, point] = _measuring(monkeypatch)
    monkeypatch.setattr(ops.Stub, "_time", time)

    neg, abs_ = ops.measure(None, [point], 10_000)

    assert "previous error" in neg["refused"]
    assert "refused" not in abs_ and abs_["cold_ns"] > 0
    assert world == {"stream": "main", "capturing": set(), "generator": "free"}
    assert collecting == [False] * 3 and gc.isenabled()


def test_a_device_argument_is_the_rank_s_own_device():
    # The capture ran on device 0: on another rank a call there would leave
    # the rank's device, and its capture.
    stub = ops.Stub(_Runner(), 10_000)
    stub.placed = [(fctx.ForwardContext(), {})]
    args = {"size": [1, 2], "dtype": "torch.float32", "device": "cuda:0"}
    op = {"op": "aten.empty.memory_format", "args": args}

    kwargs, _ = stub._copy(op, "sig", [{}], 0)

    assert kwargs == {**args, "dtype": torch.float32, "device": _Runner.device}


@pytest.mark.parametrize("splits, room", [(1, 2 * 2 * 4 * 8), (5, 2 * 4 * 8)])
def test_v4_nm_output_at_one_kv_split_has_room_for_the_fp32_stride(splits, room):
    # At one split aiter's kernel writes its BF16 output at the FP32 split
    # stride; an exact buffer let it write past the end and kill the process.
    stub = ops.Stub(_Runner(), 10_000)
    stub.placed = [(fctx.ForwardContext(), {})]
    out = {"shape": [2, 4, 8], "dtype": "torch.bfloat16", "stride": [32, 8, 1]}
    args = {"output": out | {"role": "activation"}, "num_kv_splits": splits}
    op = {"op": "aiter.mla:mla_decode_fwd_v4_nm", "args": args}

    built = stub._copy(op, "sig", [{}], 0)[0]["output"]

    assert (built.shape, built.stride(), built.storage_offset()) == (
        (2, 4, 8),
        (32, 8, 1),
        0,
    )
    assert built.dtype == torch.bfloat16 and ops._room(built) == room


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

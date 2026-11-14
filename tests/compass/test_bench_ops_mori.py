# SPDX-License-Identifier: MIT
"""`compass measure ops` on MORI's dispatch and combine: each leaf is called at
MORI's entry point on the handle ATOM builds for the model's MoE, never twice in
a row, and priced at its own call's device time."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from test_bench_ops import _Runner

from atom.compass.bench import ops
from atom.compass.capture.peers import MORI
from atom.utils import forward_context as fctx

EXPERTS, TOP_K, HIDDEN, RANKS, ROWS = 16, 2, 8, 8, 4
#: Each kernel's time on the fake device, in ns: a combine with no rows counted
#: skips the copy of its input that a counted one makes.
COST = {"dispatch": 300.0, "combine": 200.0, "combine of no rows": 120.0, "fill": 20.0}
LAUNCH = {"block_num": 64, "warp_per_block": 8}


def _spec(shape, dtype):
    stride = list(torch.empty(shape).stride())
    return {"shape": shape, "dtype": dtype, "stride": stride, "role": "activation"}


IDS = _spec([ROWS, TOP_K], "torch.int32")
DISPATCH = {
    "op": f"{MORI}.dispatch",
    "args": {
        "input": _spec([ROWS, HIDDEN], "torch.float8_e4m3fn"),
        "weights": _spec([ROWS, TOP_K], "torch.float32"),
        "scales": _spec([ROWS, 2], "torch.float32"),
        "indices": IDS,
        **LAUNCH,
        "num_ep_ranks": RANKS,
    },
    "occurrences": 1,
}
COMBINE = {
    "op": f"{MORI}.combine",
    "args": {
        "input": _spec([RANKS * ROWS, HIDDEN], "torch.bfloat16"),
        "weights": None,
        "indices": IDS,
        **LAUNCH,
        "num_ep_ranks": RANKS,
    },
    "occurrences": 1,
}


class _Device:
    """Runs each kernel it is handed in order and keeps the time it took."""

    def __init__(self):
        self.time, self.ran = 0.0, []

    def run(self, name, launch=None):
        assert launch in (None, LAUNCH), f"{name} launched at {launch}"
        self.time += COST[name]
        self.ran.append(name)


class _Count:
    """MORI's count of the rows a rank received, as its dispatch returns it."""

    def __init__(self, device):
        self.device, self.rows = device, 0

    def fill_(self, rows):
        self.device.run("fill")
        self.rows = rows

    def __int__(self):
        return self.rows


class _Handle:
    """MORI's handle as its kernels keep state: a dispatch refuses to follow one
    no combine has followed, and adds the rows its peers sent to the count; a
    combine reduces onto the last dispatch's ids, copies the counted rows of
    its input and zeroes the count. It sends rows in `wire`, built for HIDDEN
    values a row, which a packed wire holds in fewer columns."""

    def __init__(self, device, wire=torch.float8_e4m3fn, columns=HIDDEN):
        self.config = SimpleNamespace(
            world_size=RANKS,
            hidden_dim=HIDDEN,
            num_experts_per_token=TOP_K,
            data_type=wire,
            scale_dim=2,
            scale_type_size=4,
        )
        self.device, self.count, self.columns = device, _Count(device), columns
        self.ids, self.open, self.sent, self.routed = None, False, [], []

    def dispatch(self, input, weights, scales, indices, **launch):
        assert not self.open, "a dispatch before the last one's combine"
        assert input.shape == (ROWS, self.columns)
        assert input.dtype == self.config.data_type
        assert weights.shape == (ROWS, TOP_K) and weights.dtype == torch.float32
        assert scales.shape == (ROWS, 2) and scales.element_size() == 4
        assert ((0 <= indices) & (indices < EXPERTS)).all()
        self.device.run("dispatch", launch)
        self.ids, self.open = indices, True
        self.count.rows += RANKS * ROWS  # as many as its peers sent here
        self.sent.append(input)
        self.routed.append(tuple(indices.flatten().tolist()))
        return input, weights, scales, indices, self.count

    def combine(self, input, weights, indices, **launch):
        assert indices is self.ids, "a combine onto another dispatch's ids"
        assert weights is None and len(input) >= self.count.rows
        self.device.run("combine" if self.count.rows else "combine of no rows", launch)
        self.count.rows, self.open = 0, False
        return input, None


class _Profile:
    """torch's profiler on the fake device: the kernels run while it was open."""

    device = None

    def __init__(self, activities):
        pass

    def __enter__(self):
        self.start = len(self.device.ran)
        return self

    def __exit__(self, *exc):
        self.names = self.device.ran[self.start :]

    def export_chrome_trace(self, path):
        events = [
            {"cat": "kernel", "name": n, "ts": i, "dur": COST[n] / 1e3, "args": {}}
            for i, n in enumerate(self.names)
        ]
        Path(path).write_text(json.dumps({"traceEvents": events}))


def _replayer(device):
    def replay(fn, calls, flush=None):
        """A graph of `calls`: its first call run once before its capture, then
        each replay running every call, its device time over the calls."""
        fn(**calls[0][0])
        times = []
        for _ in range(ops.WARMUPS + ops.REPEATS):
            start = device.time
            for kwargs, _ in calls:
                fn(**kwargs)
            times.append((device.time - start) / len(calls))
        return times[ops.WARMUPS :]

    return replay


def _atom(monkeypatch, made):
    """ATOM's MoE module as `Stub._mori` reads it: its quant methods' base,
    whose `_maybe_make_prepare_finalize` returns `made`, each call recorded."""
    calls = []

    class FusedMoEMethodBase:
        moe = SimpleNamespace(in_dtype=torch.bfloat16)
        routed_rows_wire = "wire"

        def get_fused_moe_quant_config(self, layer):
            return ("quant config", layer)

        @staticmethod
        def _maybe_make_prepare_finalize(moe, quant_config, wire):
            calls.append((moe, quant_config, wire))
            return made

    module = ModuleType("atom.model_ops.moe")
    module.FusedMoEMethodBase = FusedMoEMethodBase
    monkeypatch.setitem(sys.modules, "atom.model_ops.moe", module)
    return FusedMoEMethodBase, calls


class _Moe(torch.nn.Module):
    """An MoE layer as the stub reads one: its experts, top-k and method."""

    def __init__(self, method):
        super().__init__()
        self.global_num_experts, self.top_k = EXPERTS, TOP_K
        self.quant_method = method


def _stub(monkeypatch, made, moe=True):
    base, calls = _atom(monkeypatch, made)
    runner = _Runner()
    if moe:
        runner.model.append(_Moe(base()))
    stub = ops.Stub(runner, 10_000)
    stub.placed = [(fctx.ForwardContext(), {})]
    return stub, calls


def _time(stub, op, device):
    _Profile.device = device
    with (
        patch.object(ops, "_replay", side_effect=_replayer(device)),
        patch.object(torch.profiler, "profile", _Profile),
        patch.object(torch.cuda, "synchronize", lambda: None),
    ):
        return stub._time(op, "sig", [{}], 4)


@pytest.fixture
def mori(monkeypatch):
    device = _Device()
    handle = _Handle(device)
    stub, calls = _stub(monkeypatch, SimpleNamespace(_mori_ops=[handle]))
    return SimpleNamespace(device=device, handle=handle, stub=stub, calls=calls)


def test_a_dispatch_is_priced_at_its_own_call_through_rounds_with_a_combine(mori):
    record = _time(mori.stub, DISPATCH, mori.device)

    # Its rounds, less a combine alone after its count's fill, plus the fill.
    round_, alone, fill = 500.0, 220.0, 20.0
    assert record["parts_ns"] == {"round": round_, "combine": alone, "fill": fill}
    assert record["cold_ns"] == record["warm_ns"] == COST["dispatch"]
    assert record["kernels"] == ["dispatch"]
    # Each round's copy draws its own ids, which its combine reduces onto.
    assert len(record["routing"]["activated"]) == record["copies"] == 4
    assert len(set(mori.handle.routed)) == 4
    assert not mori.handle.open


def test_a_combine_is_priced_alone_on_one_dispatch_s_rows_counted_again(mori):
    record = _time(mori.stub, COMBINE, mori.device)

    assert record["parts_ns"] == {"combine": 220.0, "fill": 20.0}
    assert record["cold_ns"] == record["warm_ns"] == COST["combine"]
    assert record["kernels"] == ["combine"]
    assert "combine of no rows" not in mori.device.ran
    # One dispatch of zeros at the handle's format, on the first copy's ids,
    # which every later combine reduces onto with no dispatch between.
    [sent] = mori.handle.sent
    assert not sent.float().any()
    assert len(record["routing"]["activated"]) == 1


def test_both_leaves_take_the_handle_atom_builds_for_its_first_moe_layer_once(mori):
    _time(mori.stub, DISPATCH, mori.device)
    _time(mori.stub, COMBINE, mori.device)

    [layer] = [m for m in mori.stub.modules.values() if isinstance(m, _Moe)]
    [(moe, quant_config, wire)] = mori.calls
    method = layer.quant_method
    assert (moe, quant_config, wire) == (method.moe, ("quant config", layer), "wire")


def test_a_handle_for_other_shapes_than_the_leaf_s_refuses_naming_both(mori):
    other = {**DISPATCH, "args": DISPATCH["args"] | {"num_ep_ranks": 4}}

    with pytest.raises(
        ValueError, match=r"\(8, 8, 2\), and the leaf states \(4, 8, 2\)"
    ):
        _time(mori.stub, other, mori.device)
    assert mori.device.ran == []


def test_no_mori_handle_refuses_naming_what_atom_builds(monkeypatch):
    class RcclPrepareAndFinalize:
        pass

    stub, _ = _stub(monkeypatch, RcclPrepareAndFinalize())
    with pytest.raises(ValueError, match="RcclPrepareAndFinalize for 2, not MORI"):
        _time(stub, COMBINE, _Device())

    stub, _ = _stub(monkeypatch, None, moe=False)
    with pytest.raises(ValueError, match="no MoE layer"):
        _time(stub, DISPATCH, _Device())


@pytest.mark.parametrize("op", [DISPATCH, COMBINE], ids=["dispatch", "combine"])
def test_a_packed_fp4_wire_holds_two_values_a_column(monkeypatch, op):
    # ATOM_MORI_FP4_DISPATCH: ATOM builds the handle for the hidden size, and
    # the dispatch sends its rows as float4_e2m1fn_x2, two values a column.
    device, fp4 = _Device(), torch.float4_e2m1fn_x2
    handle = _Handle(device, fp4, HIDDEN // 2)
    stub, _ = _stub(monkeypatch, SimpleNamespace(_mori_ops=[handle]))
    packed = _spec([ROWS, HIDDEN // 2], "torch.float4_e2m1fn_x2")
    leaf = op["op"].rsplit(".", 1)[1]
    if leaf == "dispatch":
        op = {**op, "args": op["args"] | {"input": packed}}

    record = _time(stub, op, device)

    assert record["cold_ns"] == COST[leaf]
    assert {(t.shape, t.dtype) for t in handle.sent} == {((ROWS, HIDDEN // 2), fp4)}

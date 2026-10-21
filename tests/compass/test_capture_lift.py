# SPDX-License-Identifier: MIT
"""A raw Triton or FlyDSL launch lifts to the function that issues it.

From the launch, capture walks past closures and generic launchers (the kernel
is one of the frame's arguments, or an attribute of its `self`) to the first
function its `module:qualname` names. An untraced step finds these functions;
in the traced step each call of one is a leaf whose arguments are taken at
entry, what it runs inside folded in, and it is called again from its
contract to check it launches the same. Arguments that are objects are held as
pickle reduces them, or as the runner attribute they are.

The rule runs here on functions written in this file, then through
`compass trace ops` on Qwen3.8-27B and on DeepSeek-V4-Pro's Triton experts.
"""

import dataclasses
import enum
import types

import pytest
import torch
from test_capture_ir import WIDTHS, _ops, capture
from test_capture_v4_pro import capture as capture_v4
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from atom.compass import ir
from atom.compass.capture import driverless, forward
from atom.compass.ir.nodes import Reduced

_RECORDER: dict = {}


@pytest.fixture
def recorder(monkeypatch):
    """A recorder the functions below launch through, on a fake device."""
    recorder = forward._Recorder({}, driverless.driverless_mode(ShapeEnv()))
    monkeypatch.setitem(_RECORDER, "now", recorder)
    with forward._entries(recorder):
        yield recorder


def steps(recorder, untraced, traced):
    """An untraced step, then a traced one, each calling its function on a fake
    `[4, 8]` tensor: the traced step's items by name."""
    with recorder.fake_mode, recorder:
        x = torch.empty(4, 8, device="cuda")
        for on, step in ((False, untraced), (True, traced)):
            recorder.start()
            recorder.on = on
            step(x)
    return [op.name for _, op in recorder.items]


def _kernel():
    pass


def _tail_kernel():
    pass


def _run(kernel, *args):
    """A generic launcher: the kernel is its argument."""
    _RECORDER["now"].launch(f"triton.{kernel.__name__}", kernel)


class _Launcher:
    """A generic launcher holding its kernel."""

    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, *args):
        _RECORDER["now"].launch(f"triton.{self.kernel.__name__}", self.kernel)


def _scaled(x, scale):
    scale = scale * 2
    y = torch.empty_like(x)

    def launch():
        _run(_kernel, x, y, scale)

    launch()
    _Launcher(_tail_kernel)(y)
    return y.view(-1)


def _outer(x):
    _scaled(x, 0.5)
    _run(_kernel, x)


def _maybe(x, launch):
    x.add(1)
    if launch:
        _run(_kernel, x)


@torch.library.custom_op("compass_lift::inner", mutates_args=())
def _inner(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


_inner.register_fake(torch.empty_like)


def _dispatching(x, launch):
    _inner(x)
    if launch:
        _run(_kernel, x)


_TAIL: list = []


def _stateful(x):
    _run(_kernel, x)
    if _TAIL:
        _TAIL.pop()
        _run(_tail_kernel, x)


class _Mode(enum.Enum):
    FAST = 1


@dataclasses.dataclass
class _Plan:
    rows: torch.Tensor
    mode: _Mode
    table: dict


def _planned(x, plan, pool):
    # A replay passes these rebuilt: an object, an enum member and a dict, and
    # the runner's own attribute.
    assert plan.mode is _Mode.FAST and plan.table[2].shape == plan.rows.shape
    assert pool is _RECORDER["pool"]
    _run(_kernel, x, plan.rows)


def test_a_launch_lifts_past_closures_and_generic_launchers_with_its_entry_arguments(
    recorder,
):
    names = steps(
        recorder,
        lambda x: _scaled(x, 0.5),
        lambda x: (x.add(1), _scaled(x, 0.5), x.mul(2)),
    )

    assert names == ["aten.add.Tensor", f"{__name__}:_scaled", "aten.mul.Tensor"]
    leaf = recorder.items[1][1]
    assert leaf.kind is ir.NodeKind.OPAQUE_LEAF
    # As called: the function doubles `scale` before its launches.
    x = ir.Tensor((4, 8), "torch.float32", (8, 1), ir.Role.METADATA)
    assert dict(leaf.args) == {"x": x, "scale": 0.5}


def test_a_launching_function_called_inside_another_folds_into_the_outermost(
    recorder,
):
    names = steps(recorder, _outer, _outer)

    assert names == [f"{__name__}:_outer"]


def test_a_call_that_launches_nothing_gives_back_what_it_ran(recorder):
    names = steps(recorder, lambda x: _maybe(x, True), lambda x: _maybe(x, False))

    assert names == ["aten.add.Tensor"]


def test_an_operator_outside_torch_s_folds_into_the_leaf_and_keeps_it_one(recorder):
    names = steps(
        recorder,
        lambda x: _dispatching(x, True),
        lambda x: (_dispatching(x, True), _dispatching(x, False)),
    )

    # The second call launched nothing raw, but dispatched an operator that
    # hides its kernels: still a leaf.
    assert names == [f"{__name__}:_dispatching"] * 2


def test_a_launch_from_a_function_the_untraced_step_did_not_find_is_refused(
    recorder,
):
    with pytest.raises(
        ValueError,
        match=r"triton._kernel lifts to .* in _maybe, which launched nothing in "
        "the untraced step",
    ):
        steps(recorder, lambda x: _maybe(x, False), lambda x: _maybe(x, True))


def test_a_leaf_whose_replay_launches_other_kernels_is_refused(recorder):
    def traced(x):
        _TAIL.append(True)
        _stateful(x)

    with pytest.raises(
        ValueError,
        match=r"_stateful, called again from its contract, launches "
        r"\['triton._kernel'\], not \['triton._kernel', 'triton._tail_kernel'\]",
    ):
        steps(recorder, _stateful, traced)


def test_an_object_argument_is_held_as_pickle_reduces_it_or_as_the_runner_s(
    recorder, monkeypatch
):
    pool = object()
    recorder.runner = types.SimpleNamespace(pool=pool)
    monkeypatch.setitem(_RECORDER, "pool", pool)

    def call(x):
        rows = torch.empty(3, dtype=torch.int32, device="cuda")
        _planned(x, _Plan(rows, _Mode.FAST, {2: rows}), pool)

    steps(recorder, call, call)

    leaf = recorder.items[-1][1]
    assert leaf.name == f"{__name__}:_planned"
    rows = ir.Tensor((3,), "torch.int32", (1,), ir.Role.METADATA)
    plan = (
        ("rows", rows),
        ("mode", Reduced(f"{__name__}:_Mode", 1)),
        ("table", Reduced("builtins:dict", ((2, rows),))),
    )
    assert dict(leaf.args)["plan"] == Reduced(f"{__name__}:_Plan", plan)
    assert dict(leaf.args)["pool"] == ir.ModuleRef("runner.pool")


def test_an_argument_neither_a_value_nor_reducible_is_refused_by_its_type(
    recorder,
):
    with pytest.raises(ValueError, match="_maybe has no contract a graph can hold"):
        steps(
            recorder,
            lambda x: _maybe(x, True),
            lambda x: _maybe(x, lambda: None),
        )


# --- through `compass trace ops` ------------------------------------------------

#: Qwen3.8-27B's decode leaves that are functions, each by the function that
#: launches its kernels.
QWEN_LEAVES = {
    "aiter.ops.flydsl.topk.topk_per_row_argmax:topk_per_row_argmax",
    "atom.model_ops.decode_input_ids:fill_deferred_decode_ids",
    "atom.model_ops.layernorm:fused_qk_norm",
    "atom.model_ops.triton_mrope:try_mrope_qk_fused",
    "atom.utils.block_convert:kv_indices_generate_triton",
}


def test_qwen_s_decode_leaves_are_the_functions_that_launch_their_kernels(
    tmp_path_factory,
):
    graph, _ = capture(1, WIDTHS, tmp_path_factory)["decode B=2"]
    ops = _ops(graph["region"])

    # The deferred ids are first filled on the runner's second step, so only
    # two untraced steps before the traced one find their function.
    assert {op["op"] for op in ops if ":" in op["op"]} == QWEN_LEAVES
    # The greedy pick's selection around the FlyDSL argmax is the graph's.
    [argmax] = [op for op in ops if op["op"].endswith(":topk_per_row_argmax")]
    assert set(argmax["args"]) == {"scores", "row_lens", "indices"}
    assert {"aten.le.Scalar", "aten.where.self"} <= {op["op"] for op in ops}


def test_v4_pro_s_triton_experts_capture_at_the_functions_that_launch_them(
    tmp_path_factory, monkeypatch
):
    # TP8 at EP off, where ATOM's Triton experts once took a method argument.
    monkeypatch.setenv("ATOM_USE_TRITON_MOE", "1")
    graph, _ = capture_v4(tmp_path_factory, "--width 2 -- --kv_cache_dtype fp8")[
        "decode B=2"
    ]

    names = {op["op"] for op in _ops(graph["region"])}
    triton = "aiter.ops.triton.moe."
    assert {
        triton + "moe_op_gemm_a16w4:moe_gemm_a16w4",
        triton + "moe_routing.routing:sort_tokens_fused",
        triton + "moe_routing.topk:topk",
    } <= names
    assert "aiter.fused_moe_.default" not in names

# SPDX-License-Identifier: MIT
"""CPU-only cover for capturing a model whose forward issues collectives.

The question these pin is whether a fake-tensor trace of a tensor-parallel
model needs the process group substituted, or whether the collectives are
already operations the trace can record. The answer is different for the two
kinds of collective ATOM reaches, and the difference is a property of the
dispatcher rather than of the model:

* A collective exposed as a **registered custom operator with a registered
  fake implementation** is recorded, with its real shapes, and its body never
  runs -- so it needs no process group, no transport and no substitution.
* A collective that calls **`torch.distributed`'s legacy entry points**
  reaches `c10d::*`, which carry a backend kernel and nothing else. There is
  no meta kernel to run, so `FakeTensorMode` declines them.
* Torch's **functional** collectives do carry a kernel the mode can run, and
  produce the width-scaled shape, so they are what the legacy call sites can
  be routed through when one has to be.

Nothing here builds a model, opens a device or talks to a peer.
"""

from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from atom.compass.capture.fake_trace import Recorder

# The legacy entry points, by the operator each one dispatches to. Enumerated
# rather than counted: if a future torch grows a meta kernel for one of these,
# the corresponding call site stops needing to be routed anywhere, and that
# should show up as a failure here rather than as unexplained extra code.
LEGACY_C10D_OPS = (
    "c10d::allreduce_",
    "c10d::broadcast_",
    "c10d::_allgather_base_",
    "c10d::allgather_",
    "c10d::gather_",
    "c10d::_reduce_scatter_base_",
)

# The functional forms, which do carry one.
FUNCTIONAL_C10D_OPS = (
    "_c10d_functional::all_reduce",
    "_c10d_functional::all_gather_into_tensor",
    "_c10d_functional::reduce_scatter_tensor",
    "_c10d_functional::broadcast",
    "_c10d_functional::wait_tensor",
)


def _fake_mode() -> tuple[ShapeEnv, FakeTensorMode]:
    shape_env = ShapeEnv()
    return shape_env, FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True)


# --------------------------------------------------------------------------
# a collective that is a registered custom op


def _register_stub_collective() -> str:
    """A custom op shaped like the one a TP all-reduce goes through.

    The body raises. A fake implementation returns a tensor of the input's
    shape. That is the whole pattern: under a fake trace the registered fake
    answers and the body is unreachable, which is why such a collective needs
    no group to be captured.
    """
    name = "compass_capture_test::stub_all_reduce"
    if hasattr(
        torch.ops.compass_capture_test,
        "stub_all_reduce",
    ):
        return name

    @torch.library.custom_op(name, mutates_args=())
    def stub_all_reduce(x: torch.Tensor, group_name: str) -> torch.Tensor:
        raise AssertionError(
            "the body of a collective custom op ran under a fake trace; the "
            "registered fake implementation should have answered instead"
        )

    @stub_all_reduce.register_fake
    def _(x: torch.Tensor, group_name: str) -> torch.Tensor:
        return torch.empty_like(x)

    return name


def test_a_collective_custom_op_is_recorded_and_its_body_never_runs():
    """The property the whole TP capture rests on.

    The group name is one that does not exist. If anything but the registered
    fake answered, the body would raise on it.
    """
    _register_stub_collective()
    _shape_env, fake_mode = _fake_mode()
    rec = Recorder()
    with fake_mode:
        x = torch.empty(16384, 5120, dtype=torch.bfloat16)
    with fake_mode, torch._C._EnablePythonDispatcher(), rec:
        y = torch.ops.compass_capture_test.stub_all_reduce(x, "no such group")

    assert list(y.shape) == [16384, 5120]
    recorded = [o for o in rec.ops if "stub_all_reduce" in o.op]
    assert len(recorded) == 1, [o.op for o in rec.ops]
    # An all-reduce does not change shape, and the record has to say so: a
    # capture that drops the collective prices the width as free.
    assert recorded[0].in_shapes == [["16384", "5120"]]
    assert recorded[0].out_shapes == [["16384", "5120"]]


# --------------------------------------------------------------------------
# the legacy entry points, and why they are the ones that need routing


@pytest.mark.parametrize("op", LEGACY_C10D_OPS)
def test_a_legacy_c10d_collective_has_no_kernel_a_fake_trace_can_run(op: str):
    """Neither a meta kernel nor a backend-agnostic one.

    This is the measured reason a `torch.distributed.all_gather_into_tensor`
    call site cannot simply be traced: `FakeTensorMode` has nothing to run for
    it and refuses. It is not that the collective would be issued for real.
    """
    for key in ("Meta", "CompositeExplicitAutograd"):
        assert not torch._C._dispatch_has_kernel_for_dispatch_key(op, key), (
            f"{op} now has a {key} kernel; a fake trace can run it directly "
            "and the call site routing this operator no longer needs to exist"
        )


@pytest.mark.parametrize("op", FUNCTIONAL_C10D_OPS)
def test_the_functional_collectives_do_have_one(op: str):
    """The other half of the same fact, and the reason routing is possible."""
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        op, "CompositeExplicitAutograd"
    ), f"{op} lost the kernel that lets a fake trace run it"


def test_a_functional_all_gather_records_the_width_scaled_shape():
    """What a captured gather must carry, and does, with no group at all.

    The output is `width x` the input on the gathered dimension. A capture
    that got this wrong would corrupt every operator after it.
    """
    _shape_env, fake_mode = _fake_mode()
    rec = Recorder()
    with fake_mode:
        x = torch.empty(2, 124160, dtype=torch.bfloat16)
    with fake_mode, torch._C._EnablePythonDispatcher(), rec:
        y = torch.ops._c10d_functional.all_gather_into_tensor(x, 2, "no such group")
        y = torch.ops._c10d_functional.wait_tensor(y)

    assert list(y.shape) == [4, 124160]
    gathers = [o for o in rec.ops if "all_gather_into_tensor" in o.op]
    assert len(gathers) == 1, [o.op for o in rec.ops]
    assert gathers[0].in_shapes == [["2", "124160"]]
    assert gathers[0].out_shapes == [["4", "124160"]]


# --------------------------------------------------------------------------
# a group of width 2 in one process


@pytest.fixture
def two_rank_group():
    """A process group that reports two ranks, from this one process.

    A width-2 group is what makes every weight shard two ways and what stops
    the collective wrappers taking their `world_size == 1` shortcut. It needs
    no second process and no transport.
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        pytest.skip("a process group is already initialised in this process")
    from torch.testing._internal.distributed.fake_pg import FakeStore

    dist.init_process_group(backend="fake", store=FakeStore(), rank=0, world_size=2)
    try:
        yield dist
    finally:
        dist.destroy_process_group()


def test_a_width_two_group_exists_in_one_process(two_rank_group):
    assert two_rank_group.get_world_size() == 2
    assert two_rank_group.get_rank() == 0


def test_fake_tracing_declines_a_legacy_collective_rather_than_issuing_it(
    two_rank_group,
):
    """The refusal itself, on a real call through a real group.

    The important half is *which* failure it is: an unsupported operator, not
    a collective that went to a transport. Nothing is communicated either way.
    """
    from torch._subclasses.fake_tensor import UnsupportedOperatorException

    _shape_env, fake_mode = _fake_mode()
    rec = Recorder()
    with (
        pytest.raises(UnsupportedOperatorException, match="c10d.allreduce_"),
        fake_mode,
        torch._C._EnablePythonDispatcher(),
        rec,
    ):
        x = torch.empty(4, 8, dtype=torch.bfloat16)
        two_rank_group.all_reduce(x)
    assert not [
        o for o in rec.ops if "c10d" in o.op
    ], "a declined collective must leave nothing in the inventory"

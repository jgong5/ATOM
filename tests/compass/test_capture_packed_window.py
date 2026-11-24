# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's FP8 window write holds no `BatchStep` in its leaf.

`write_packed_window` takes a step, whose `request_*` fields are numpy
arrays, and passes the fields its kernel reads to a launcher, which is the
leaf. Compass off, `_write_window` launches over the step's width with
those fields and the window's scalars; under capture, a write on steps
ATOM's own producer builds at both `CONTEXTS` is one leaf of tensors and
scalars equal at both, replaying to that kernel.
"""

import os
import subprocess
import sys

import numpy
import pytest
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from atom.compass import ir
from atom.compass.capture import driverless, forward
from atom.compass.ir.nodes import Reduced

LEAF = "atom.model_ops.attentions.deepseek_v41.packed_rows:_launch_write_window"
WINDOW_PARAMS = "atom.model_ops.attentions.pool_layout.v4_pool_geometry:WindowParams"
REQUESTS, PAGES, LAYER = 4, 64, 3


def _own_process(request):
    """Run the calling test again in a process of its own, with an arch
    declared there: importing the window write imports aiter, which needs
    one, once per process. True in the parent, which then has nothing left
    to do."""
    if "COMPASS_ARCH_DECLARED" in os.environ:
        return False
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
        + [f"{__file__}::{request.node.name}"],
        env={**os.environ, "COMPASS_ARCH_DECLARED": "1"},
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-4000:]
    return True


def _geometry():
    """V4.1-Flash's pool at its served page, FP8 rows and an FP8 index plane."""
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import (
        V41PoolGeometry,
    )

    return V41PoolGeometry(
        43,
        ((2, 2), (8, 2), (14, 2), (20, 1), (24, 2), (30, 1)),
        128,
        128,
        512,
        128,
        kv_dtype="fp8",
        speculative_tokens=5,
    )


def _host_step(spans, **kwargs):
    """`spans` staged on the host as ATOM's own producer stages them."""
    from atom.model_ops.attentions.deepseek_v41.metadata import prepare_batch_step

    return prepare_batch_step(
        spans,
        "cpu",
        block_tables=[[i] for i in range(len(spans))],
        **kwargs,
    )


def _rows(geometry, width, device):
    """`(values, scales)`: FP8 values then E8M0 scales in each row of one
    pool-row buffer, as the FP8 pool's KV seam hands them to the write."""
    from atom.model_ops.blockscale_kernels.quantization import FP8_DTYPE

    dim = geometry.head_dim
    rows = torch.empty(width, geometry.row_bytes, dtype=torch.uint8, device=device)
    return rows[:, :dim].view(FP8_DTYPE), rows[:, dim:].view(torch.float8_e8m0fnu)


def _same(launched, expected):
    """A tensor by its storage address, dtype, shape and stride; anything
    else by its type and value."""
    if isinstance(expected, torch.Tensor):
        return isinstance(launched, torch.Tensor) and (
            launched.data_ptr(),
            launched.dtype,
            launched.shape,
            launched.stride(),
        ) == (expected.data_ptr(), expected.dtype, expected.shape, expected.stride())
    return type(launched) is type(expected) and launched == expected


@pytest.mark.parametrize("cached", [False, True], ids=["every-row", "cached-until"])
def test_compass_off_the_fp8_window_write_launches_with_its_steps_fields(
    request, tmp_path, monkeypatch, cached
):
    if _own_process(request):
        return
    driverless.install(str(tmp_path), "gfx950")
    import triton
    from triton.runtime.jit import JITFunction

    from atom.model_ops.attentions.deepseek_v41 import packed_rows
    from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan

    launched = []

    def run(kernel, *args, grid, warmup, **kwargs):
        launched.append((kernel, grid, args, kwargs))

    monkeypatch.setattr(JITFunction, "run", run)
    geometry = _geometry()
    dim = geometry.head_dim
    # A prefill of three requests run at a width of 12 rows over 4 requests:
    # past the 9 rows they own, padding.
    spans = (
        RequestSpan(0, 0, 0, 5, 2),
        RequestSpan(1, 130, 5, 1, 0),
        RequestSpan(2, 7, 6, 3, 1),
    )
    step = _host_step(spans, is_prefill=True, running_bs=4, running_tokens=12)
    values, scales = _rows(geometry, step.width, "cpu")
    pool = torch.empty(4096, dtype=torch.uint8)
    window = geometry.window_bytes(LAYER, PAGES)
    until = torch.tensor([0, 131, 8, 0], dtype=torch.int32) if cached else None

    packed_rows.write_packed_window(values, scales, pool, step, window, dim, until)

    # Each field read off the step and the window, in the kernel's order.
    expected = (
        values.view(torch.uint8),
        scales.view(torch.uint8),
        values.stride(-2),
        scales.stride(-2),
        pool,
        step.positions,
        step.batch_ids,
        step.cu_seqlens_q,
        step.slots,
        window.ring_start,
        step.positions if until is None else until,
        until is not None,
        window.slot_rows,
        window.ring_slots,
        dim,
        triton.next_power_of_2(dim),
    )
    [(kernel, grid, args, kwargs)] = launched
    assert kernel is packed_rows._write_window
    assert grid == (12,) and not kwargs
    assert len(args) == len(expected)
    for slot, (arg, want) in enumerate(zip(args, expected)):
        assert _same(arg, want), (slot, arg, want)


def _on_device(step):
    """`step` with the tensors the write reads as the device's, faked."""

    def device(tensor):
        return torch.empty_strided(
            tensor.shape, tensor.stride(), dtype=tensor.dtype, device="cuda"
        )

    for name in ("positions", "cu_seqlens_q", "slots", "batch_ids"):
        setattr(step, name, device(getattr(step, name)))
    return step


def _write(geometry, step, pool):
    """The FP8 pool's window write of `step`'s rows, as `PagedAttentionCache.
    write_window` makes it on a step no request of which recomputes rows."""
    from atom.model_ops.attentions.deepseek_v41.packed_rows import (
        write_packed_window,
    )

    values, scales = _rows(geometry, step.width, "cuda")
    window = geometry.window_bytes(LAYER, PAGES)
    write_packed_window(values, scales, pool, step, window, geometry.head_dim)


def _held(value):
    """Whether `value` is a tensor, a scalar or a window's five scalars, and
    so the same at both contexts by being itself."""
    if isinstance(value, Reduced):
        return value.cls == WINDOW_PARAMS and all(
            type(v) is int for _, v in value.state
        )
    return value is None or isinstance(value, (ir.Tensor, bool, int, str))


def test_v41_s_fp8_window_write_is_one_leaf_holding_no_step(request, tmp_path):
    if _own_process(request):
        return
    driverless.install(str(tmp_path), "gfx950")
    from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan

    geometry = _geometry()
    recorder = forward._Recorder({}, driverless.driverless_mode(ShapeEnv()))
    traces, steps = [], []
    with forward._patches(recorder):
        for context in forward.CONTEXTS:
            # A decode of `REQUESTS` rows, each holding `context` tokens.
            spans = tuple(RequestSpan(i, context - 1, i, 1, i) for i in range(REQUESTS))
            steps.append(_host_step(spans, max_q_len=1))
            with recorder.fake_mode, recorder:
                step = _on_device(steps[-1])
                pool = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
                for _ in range(2):  # the untraced steps find the launcher
                    _write(geometry, step, pool)
                recorder.start()
                recorder.on = True
                _write(geometry, step, pool)
            recorder.on = False
            traces.append(recorder.items)
        items = forward._affine(f"decode B={REQUESTS}", *traces)
        leaves = [op for _, op in items if op.kind is ir.NodeKind.OPAQUE_LEAF]
        with recorder.fake_mode:
            replays = {op.name: recorder._replay(op) for op in leaves}

    # The host state the write no longer takes moves with the context.
    first, second = steps
    assert isinstance(first.request_positions, numpy.ndarray)
    assert first.request_positions.tolist() != second.request_positions.tolist()
    assert [op.name for op in leaves] == [LEAF]
    [leaf] = leaves
    moving = [(slot, value) for slot, value in leaf.args if not _held(value)]
    assert not moving, moving
    assert replays == {LEAF: ["triton._write_window"]}

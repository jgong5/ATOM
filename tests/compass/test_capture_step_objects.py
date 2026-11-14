# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's index, window and compress launches hold no per-step object.

`fill_step_indptrs`, `build_window_rows`, `build_indices` and
`rope_quant_window` take a `BatchStep`, and `fused_compress_attn` a
`CompressPlan`, whose host state moves with the context: the requests and
their positions, numpy arrays, a plan's live boundary count. Each passes the
fields its kernel reads to a launcher, which is the leaf. A decode of 16
requests runs here through the five at both `CONTEXTS`, as capture traces a
decode shape, on steps and plans ATOM's own producers build at each context.
"""

import os
import subprocess
import sys

import numpy
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from atom.compass import ir
from atom.compass.capture import driverless, forward
from atom.compass.ir.nodes import Reduced

INDICES = "atom.model_ops.attentions.deepseek_v41.indices"
#: Each launcher, in the order the forward below reaches it, and the kernels
#: its replay launches.
LAUNCHERS = {
    f"{INDICES}:_launch_indptr_scan": ["triton._indptr_scan_all"],
    "atom.model_ops.deepseek_v41.rope_window:_launch_rope_window": [
        "triton._rope_window_kernel"
    ],
    "atom.model_ops.v4_kernels.fused_compress:_launch_fused_compress": [
        "triton._fused_compress_attn_kernel"
    ],
    f"{INDICES}:_launch_indices": ["triton._indices"],
    f"{INDICES}:_launch_window_rows": ["triton._window_rows"],
}
WINDOW_PARAMS = "atom.model_ops.attentions.pool_layout.v4_pool_geometry:WindowParams"
REQUESTS, COLUMNS, PAGES = 16, 8, 64
HEADS, HEAD_DIM, ROPE_DIM, RATIO = 16, 512, 64, 2


def _geometry():
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import (
        V41PoolGeometry,
    )

    return V41PoolGeometry(
        4,
        ((1, 1), (2, RATIO)),
        256,
        128,
        HEAD_DIM,
        128,
        layer_ratios=(0, 1, RATIO),
        index_topk=512,
        index_block_rows=8,
    )


def _host_step(geometry, context):
    """A decode step of `REQUESTS` rows at `context` and its compress plans,
    staged on the host into fixed-width buffers as the serving builder stages
    them."""
    from atom.model_ops.attentions.deepseek_v41.metadata import (
        RequestSpan,
        prepare_batch_step,
        visible_buffer_name,
    )
    from atom.model_ops.v4_kernels import make_compress_plans
    from atom.utils import CpuGpuBuffer

    def buffer(*shape):
        return CpuGpuBuffer(*shape, dtype=torch.int32, device="cpu", pin_memory=False)

    ratios = (1, RATIO)
    buffers = {
        "positions": buffer(REQUESTS),
        "cu_seqlens_q": buffer(REQUESTS + 1),
        "batch_id_per_q_token": buffer(REQUESTS),
        "block_tables": buffer(REQUESTS, COLUMNS),
        **{visible_buffer_name(ratio): buffer(REQUESTS) for ratio in ratios},
    }
    pages = -(-context // geometry.block_size)
    spans = tuple(RequestSpan(i, context - 1, i, 1, i) for i in range(REQUESTS))
    step = prepare_batch_step(
        spans,
        "cpu",
        block_tables=[[i * COLUMNS + j for j in range(pages)] for i in range(REQUESTS)],
        buffers=buffers,
        max_q_len=1,
        ratios=ratios,
    )
    step.plans = make_compress_plans(
        numpy.ones(REQUESTS, dtype=numpy.int32),
        numpy.full(REQUESTS, context, dtype=numpy.int32),
        geometry.compress_ratios,
        plan_buffers={
            ratio: {
                "compress": buffer(4 * REQUESTS, 4),
                "write": buffer(4 * REQUESTS, 4),
            }
            for ratio in ratios
        },
        running_bs=REQUESTS,
        max_q_len=1,
        extra_write=0,
    )
    return step


def _on_device(step):
    """`step` with its tensors and its plans' as the device's, faked."""

    def device(tensor):
        return torch.empty_strided(
            tensor.shape, tensor.stride(), dtype=tensor.dtype, device="cuda"
        )

    for name in ("positions", "cu_seqlens_q", "slots", "batch_ids", "block_tables"):
        setattr(step, name, device(getattr(step, name)))
    step.visible = {ratio: device(t) for ratio, t in step.visible.items()}
    for plan in step.plans.values():
        plan.compress_plan_gpu = device(plan.compress_plan_gpu)
        plan.write_plan_gpu = device(plan.write_plan_gpu)
    # The cache's fixed zeros (`PagedAttentionCache.begin_step`).
    step.swa_replay_start = step.swa_cached_until = torch.zeros(
        REQUESTS, dtype=torch.int32, device="cuda"
    )
    return step


def _forward(geometry, step):
    """A layer's index, window and compress work on `step`, in a decode's
    order: the indptrs, the ring write, the boundary compress, then the
    selection's indices and the window's rows."""
    from atom.model_ops.attentions.deepseek_v41.indices import (
        build_indices,
        build_window_rows,
        fill_step_indptrs,
    )
    from atom.model_ops.deepseek_v41.rope_window import rope_quant_window
    from atom.model_ops.v4_kernels.fused_compress import fused_compress_attn

    def empty(*shape, dtype=torch.int32):
        return torch.empty(*shape, dtype=dtype, device="cuda")

    bf16, f32, window = torch.bfloat16, torch.float32, geometry.window(1, PAGES)
    rope = empty(4096, ROPE_DIM // 2, dtype=bf16)
    indptrs = fill_step_indptrs(
        step,
        geometry,
        {r: (empty(REQUESTS + 1), empty(REQUESTS + 1)) for r in geometry.layer_ratios},
    )
    rope_quant_window(
        empty(REQUESTS, HEADS, HEAD_DIM, dtype=bf16),
        empty(REQUESTS, HEAD_DIM, dtype=bf16),
        rope,
        rope,
        step.positions,
        rope_dim=ROPE_DIM,
        ring=(empty(PAGES * 1024, HEAD_DIM, dtype=bf16), step, window),
    )
    plan = step.plans[RATIO]
    rows = plan.compress_plan_gpu.shape[0]
    fused_compress_attn(
        kv_in=empty(REQUESTS, HEAD_DIM, dtype=f32),
        score_in=empty(REQUESTS, HEAD_DIM, dtype=f32),
        kv_state=empty(REQUESTS, geometry.compress_ring_slots, HEAD_DIM, dtype=f32),
        score_state=empty(REQUESTS, geometry.compress_ring_slots, HEAD_DIM, dtype=f32),
        plan=plan,
        state_slot_mapping=step.slots,
        ape=empty(RATIO, HEAD_DIM, dtype=f32),
        rms_weight=empty(HEAD_DIM, dtype=f32),
        rms_eps=1e-20,
        cos_cache=rope,
        sin_cache=rope,
        kv_cache=None,
        block_tables=None,
        k_per_block=0,
        overlap=False,
        ratio=RATIO,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_DIM,
        quant_mode="none",
        latent_out=empty(rows, HEAD_DIM, dtype=bf16),
        rotated_out=empty(rows, HEAD_DIM, dtype=bf16),
    )
    selected = empty(REQUESTS, geometry.batch_topk(RATIO))
    _, _, _, eptr = build_indices(
        selected, indptrs[RATIO], step, geometry, window, 2, RATIO
    )
    build_window_rows(step, geometry, window, eptr)


def _held(value):
    """Whether `value` is a tensor, a scalar, a tuple of them or a window's
    five scalars, and so the same at both contexts by being itself."""
    if isinstance(value, tuple):
        return all(map(_held, value))
    if isinstance(value, Reduced):
        return value.cls == WINDOW_PARAMS and all(
            type(v) is int for _, v in value.state
        )
    return value is None or isinstance(value, (ir.Tensor, bool, int, float, str))


def test_v41_s_step_objects_stay_out_of_the_leaves_that_launch_its_kernels(
    request, tmp_path
):
    if "COMPASS_ARCH_DECLARED" not in os.environ:
        # The patches import aiter, which needs an arch declared: once per
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
    driverless.install(str(tmp_path), "gfx950")
    geometry = _geometry()
    recorder = forward._Recorder({}, driverless.driverless_mode(ShapeEnv()))
    traces, steps = [], []
    with forward._patches(recorder):
        for context in forward.CONTEXTS:
            steps.append(_host_step(geometry, context))
            with recorder.fake_mode, recorder:
                step = _on_device(steps[-1])
                for _ in range(2):  # the untraced steps find the launchers
                    _forward(geometry, step)
                recorder.start()
                recorder.on = True
                _forward(geometry, step)
            recorder.on = False
            traces.append(recorder.items)
        items = forward._affine("decode B=16", *traces)
        leaves = [op for _, op in items if op.kind is ir.NodeKind.OPAQUE_LEAF]
        with recorder.fake_mode:
            replays = {op.name: recorder._replay(op) for op in leaves}

    # What the launches no longer take moves: the requests' positions, and a
    # ratio-2 plan's boundary count, every row at one context and none at the
    # other.
    first, second = steps
    assert first.requests != second.requests
    assert first.plans[RATIO].num_compress != second.plans[RATIO].num_compress
    assert [op.name for op in leaves] == list(LAUNCHERS)
    for op in leaves:
        moving = [(slot, value) for slot, value in op.args if not _held(value)]
        assert not moving, (op.name, moving)
    assert replays == LAUNCHERS

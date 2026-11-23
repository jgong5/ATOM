# SPDX-License-Identifier: MIT
"""First calls of AITER's native A8 GEMM variants at warmup (#3743).

AITER runs a native A8 (MXFP8) linear's GEMM as a FlyDSL kernel picked by the
token count M through its tuned table. A FlyDSL kernel's first call in a
process traces the whole kernel in Python, 0.15 to 1 s each on MI355X, before
it finds the compiled artifact on disk; a variant first met while serving
stalled that forward and, under DP, every rank's at the next collective
(79 s of a 630 s DeepSeek-V4.1 c=128 window). `prewarm_native_a8` makes those
first calls at warmup instead. Apart from `blockscale` so it imports without
a GPU.
"""

import torch


def _gemm():
    from atom.model_ops.blockscale import native_quant_linear

    return native_quant_linear


#: The token counts a native A8 GEMM is first-called at when AITER's tuned
#: table names none for its shape: the padded-M ladder that table is tuned on.
PREWARM_M = (
    *(1, 2, 4, 8, 16),
    *range(32, 1024 + 1, 16),
    *range(1088, 2048 + 1, 64),
    *range(4096, 16384 + 1, 2048),
    32768,
)


def _tuned_ms(n: int, k: int, w_scale) -> tuple[int, ...]:
    """The M rows AITER's mxscale bpreshuffle table holds for an N x K weight
    with scales `w_scale` on this GPU, each the first M of a kernel variant;
    () when it holds none or the table cannot be read."""
    try:
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx
        from aiter.ops.gemm_op_a8w8 import _load_mxscale_bpreshuffle_tuned
        from aiter.ops.gemm_op_common import mxscale_w_scale_block

        w_scale_block = mxscale_w_scale_block(tuple(w_scale.shape), n, k)
    except (ImportError, ValueError):
        return ()
    gfx, cu_num = get_gfx(), get_cu_num()
    rows = _load_mxscale_bpreshuffle_tuned()
    return tuple(
        sorted(
            {
                key[2]
                for key in rows
                if (key[0], key[1], key[3], key[4], key[5])
                == (gfx, cu_num, n, k, w_scale_block)
            }
        )
    )


def prewarm_native_a8(model: torch.nn.Module, max_tokens: int) -> int:
    """First-call every native A8 GEMM variant `model`'s linears can meet,
    each distinct weight shape at each M of AITER's tuned table for it (or
    `PREWARM_M` when it has none) up to `max_tokens` rows, and the first M
    above it, which a forward of up to `max_tokens` pads to: the calls are
    local, no collective, so each rank warms its own process. Returns the
    GEMM calls made.

    AITER runs these GEMMs as FlyDSL kernels whose first call in a process
    traces the whole kernel in Python, 0.15 to 1 s each on MI355X, before it
    finds the compiled artifact on disk. A variant first met while serving
    stalled that forward, and under DP every rank's at the next collective: 79
    s of a 630 s DeepSeek-V4.1 c=128 window (#3743). Zero activations, so the
    outputs are discarded and nothing a forward reads changes."""
    seen, calls = set(), 0
    for module in model.modules():
        group_rows = getattr(module, "native_a8_group_rows", None)
        weight = getattr(module, "weight", None)
        if group_rows is None or weight is None or weight.device.type != "cuda":
            continue
        shuffled = getattr(weight, "is_shuffled", False)
        scale = module.weight_scale
        key = (tuple(weight.shape), weight.dtype, group_rows, shuffled, scale.shape)
        if key in seen:
            continue
        seen.add(key)
        n, stored_k = weight.shape
        k = stored_k * (2 if weight.dtype == torch.float4_e2m1fn_x2 else 1)
        ladder = (_tuned_ms(n, k, scale) if shuffled else ()) or PREWARM_M
        ms = [m for m in ladder if m <= max_tokens]
        ms += [m for m in ladder if m > max_tokens][:1]
        for m in ms:
            x = torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device)
            _gemm()(
                x,
                weight,
                scale,
                weight_group_rows=group_rows,
                weight_preshuffled=shuffled,
            )
            calls += 1
    return calls

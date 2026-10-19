# SPDX-License-Identifier: MIT
"""DeepSeek-V4-Pro's attention kernels, priced from each leaf's own arguments.

Each extractor matches a hand count on one shape: what a leaf reaches by index
(the KV pool, the rotary cache) stays out of its bytes, a prefill's products
run over the entries its index lists name, and a decode, whose lists are sized
for the longest row, counts no products.
"""

from test_op_work import I32, ONE, leaf, rows, t

from atom.compass.backends.op_work import work
from atom.compass.backends.work import Work

FP8, F32, U8 = "torch.float8_e4m3fn", "torch.float32", "torch.uint8"
I64 = "torch.int64"
V4 = "atom.model_ops.v4_kernels."
POOL = 1 << 20


def test_the_grouped_output_projection_reads_each_group_s_weights_once():
    op = leaf(
        "aiter.batched_gemm_a8w8_mxscale.default",
        [[3, 2, 8]],
        x=t([3, 2, 64], FP8),
        wo_a=t([2, 8, 64], FP8, role="weight"),
        x_scale=t([3, 2, 2], U8),
        w_scale=t([2, 1, 2], U8, role="weight"),
    )

    # Per group: x [3, 64] read, [8, 64] weights read, [3, 8] written, all fp8.
    moved = 2 * (3 * 64 + 8 * 64 + 3 * 8) + 3 * 2 * 2 + 2 * 2
    assert work(op, ONE, 1, "fp8") == Work({"fp8": 2 * 2 * 3 * 8 * 64}, moved)


def test_a_fused_qk_norm_leaves_the_pools_it_writes_by_row_out():
    op = leaf(
        "aiter._fused_qk_norm_rope_group_quant_kernel.default",
        q=t([2, 4, 16]),
        kv=t([2, 16]),
        k_weight=t([16], role="weight"),
        positions=t([2], I64, role="metadata"),
        cos_cache=t([POOL, 8], role="weight"),
        sin_cache=t([POOL, 8], role="weight"),
        q_nope_scale_buff=t([2, 4, 16], FP8),
        q_rope_buff=t([2, 4, 4]),
        k_nope_scale_buff=t([2, 1, 16], FP8),
        k_rope_buff=t([2, 1, 4]),
        swa_nope_scale_buff=t([POOL, 16], FP8, role="kv"),
        swa_rope_buff=t([POOL, 4], role="kv"),
    )

    read = 2 * 4 * 16 * 2 + 2 * 16 * 2 + 16 * 2 + 2 * 8
    written = 2 * 4 * 16 + 2 * 4 * 4 * 2 + 2 * 16 + 2 * 4 * 2
    assert work(op, ONE, 1, "fp8") == Work({}, read + written)


def test_a_fused_rms_quant_writes_its_values_and_scales_in_their_formats():
    op = leaf(
        "aiter._aiter_rms_quant.default",
        [[2, 256], [2, 2]],
        x=t([2, 256]),
        weight=t([256], role="weight"),
        eps=1e-6,
        value_dtype=FP8,
        e8m0_scale=True,
    )

    moved = 2 * 256 * 2 + 256 * 2 + 2 * 256 + 2 * 2
    assert work(op, ONE, 1, "fp8") == Work({}, moved)


def test_a_prefill_attends_each_entry_its_index_lists_name():
    op = leaf(
        "aiter.pa_sparse_prefill_fp8_opus.default",
        [[3, 2, 16]],
        q_nope=t([3, 2, 16], FP8),
        q_rope=t([3, 2, 4]),
        unified_kv_nope=t([POOL, 16], FP8, role="kv"),
        unified_kv_rope=t([POOL, 4], role="kv"),
        kv_indices_prefix=t([5], I32, role="metadata"),
        kv_indptr_prefix=t([4], I32, role="metadata"),
        kv_nope=t([3, 16], FP8),
        kv_rope=t([3, 4]),
        kv_indices_extend=t([6], I32, role="metadata"),
        kv_indptr_extend=t([4], I32, role="metadata"),
        attn_sink=t([2], F32, role="weight"),
        softmax_scale=0.25,
    )

    # 5 prefix and 6 in-chunk entries over 2 heads of width 16; the pool's rows
    # are read by index and left out.
    q = 3 * 2 * 16 + 3 * 2 * 4 * 2
    kv = 3 * 16 + 3 * 4 * 2
    lists = (5 + 4 + 6 + 4) * 4
    moved = q + kv + lists + 2 * 4 + 3 * 2 * 16 * 2
    assert work(op, rows((3, 3), decode=False), 1, "fp8") == Work(
        {"fp8": 4 * 2 * 16 * 11}, moved
    )


def test_a_decode_counts_q_the_sink_and_o_alone():
    op = leaf(
        V4 + "paged_decode:_sparse_attn_v4_paged_decode_asm",
        [[2, 4, 16]],
        unified_kv=t([POOL, 16], FP8, role="kv"),
        kv_indices=t([POOL], I32, role="metadata"),
        kv_indptr=t([3], I32, role="metadata"),
        attn_sink=t([4], F32, role="weight"),
        unified_kv_rope=t([POOL, 4], role="kv"),
        q_packed_in=t([2, 4, 16], FP8),
        q_rope_in=t([2, 4, 4]),
        num_kv_splits=16,
    )

    moved = 2 * 4 * 16 + 2 * 4 * 4 * 2 + 4 * 4 + 2 * 4 * 16 * 2
    assert work(op, rows((1, 9), (1, 300)), 1, "fp8") == Work({}, moved)

# SPDX-License-Identifier: MIT
"""DeepSeek-V4-Pro's attention kernels, priced from each leaf's own arguments.

Each extractor matches a hand count on one shape: what a leaf reaches by index
(the KV pool, the rotary cache) stays out of its bytes, a prefill's products
run over the entries its index lists name, and a decode, whose lists are sized
for the longest row, and the indexer run over the entries the layer geometry
their leaves state gives each row.
"""

import pytest
from test_op_work import I32, ONE, leaf, rows, t

from atom.compass.backends.op_work import work
from atom.compass.backends.resolver import CostRefused
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


def decode(**stated):
    """A paged decode over 2 rows of 4 heads of width 16, stating `stated`."""
    op = leaf(
        "aiter.mla:mla_decode_fwd_v4_nm",
        [],
        q=t([2, 4, 16], FP8),
        qrope=t([2, 4, 4]),
        kv_buffer=t([POOL, 1, 1, 16], FP8, role="kv"),
        kvrope=t([POOL, 1, 1, 4], role="kv"),
        output=t([2, 4, 16]),
        kv_indptr=t([3], I32, role="metadata"),
        kv_page_indices=t([POOL], I32, role="metadata"),
        sink=t([4], F32, role="weight"),
        num_kv_splits=16,
    )
    return op | ({"stated": stated} if stated else {})


@pytest.mark.parametrize(
    "ratio, top, entries",
    [
        # Rows at 9 and 300: the last 4 tokens each, and 9 // 4 = 2 and
        # 300 // 4 = 75 compressed entries, at most 16.
        (4, 16, 4 + 2 + 4 + 16),
        # No indexer: every entry, 9 // 8 = 1 and 300 // 8 = 37.
        (8, None, 4 + 1 + 4 + 37),
        # The window alone.
        (0, None, 4 + 4),
    ],
    ids=["csa", "hca", "window"],
)
def test_a_decode_reads_each_row_s_window_and_compressed_entries(ratio, top, entries):
    op = decode(window_size=4, compress_ratio=ratio, index_topk=top)

    # Q, its rotary part and the sink read, O written in bf16; each entry's
    # 16 fp8 values read once, and two products of width 16 per head and entry.
    moved = 2 * 4 * 16 + 2 * 4 * 4 * 2 + 4 * 4 + 2 * 4 * 16 * 2 + entries * 16
    assert work(op, rows((1, 9), (1, 300)), 1, "fp8") == Work(
        {"fp8": 4 * 4 * 16 * entries}, moved
    )


def indexer(tokens):
    op = leaf(
        "aiter.indexer_score_topk.default",
        [[tokens, 16]],
        q_quant=t([tokens, 4, 16], FP8),
        weights=t([tokens, 4]),
        layer_name={"context": "layers.2.attn.indexer"},
        topk=16,
    )
    return op | {"stated": {"compress_ratio": 4}}


@pytest.mark.parametrize(
    "op",
    [decode(window_size=4, compress_ratio=4, index_topk=16), indexer(2)],
    ids=["decode", "indexer"],
)
def test_a_drafter_s_later_pass_reads_one_query_token_a_row(op):
    # Q of 2 tokens over two rows each verifying two drafts: a later draft pass.
    later = work(op, rows((3, 9), (3, 300)), 1, "fp8")

    assert later == work(op, rows((1, 9), (1, 300)), 1, "fp8")


def test_a_decode_stating_no_geometry_is_refused_by_name():
    with pytest.raises(CostRefused, match="states no window_size, compress_ratio"):
        work(decode(), rows((1, 9)), 1, "fp8")


@pytest.mark.parametrize(
    "q, scale, fmt",
    [(t([4, 4, 8], U8), t([4, 1, 1], U8), "fp4"), (t([4, 4, 16], FP8), None, "fp8")],
    ids=["fp4", "fp8"],
)
def test_the_indexer_scores_each_new_query_against_every_entry_before_it(q, scale, fmt):
    op = leaf(
        "aiter.indexer_score_topk.default",
        [[4, 16]],
        q_quant=q,
        weights=t([4, 4]),
        q_scale=scale,
        layer_name={"context": "layers.2.attn.indexer"},
        topk=16,
    )
    op["stated"] = {"compress_ratio": 4}

    # 4 new tokens at positions 9 to 12 over 4 heads of width 16, packed two
    # to a byte in fp4: 2 + 2 + 2 + 3 entries scored, the row's 12 // 4 read.
    size = {"fp4": 0.5, "fp8": 1}[fmt]
    boundary = 4 * 4 * 16 * size + 4 * 4 * 2 + 4 * 16 * 1 + (4 if scale else 0)
    assert work(op, rows((4, 12), decode=False), 1, "fp8") == Work(
        {fmt: 2 * 4 * 16 * 9}, boundary + 3 * 16 * size
    )


def test_a_compressor_state_update_counts_no_kv_or_score_row():
    # A 4096-token chunk at ratio 4 with overlap: the plan names 8 rows per
    # sequence, so the kernel reads 8 of the 4096 kv and score rows.
    op = leaf(
        V4 + "state_writes:update_compressor_states",
        kv=t([4096, 512], F32),
        score=t([4096, 512], F32),
        ape=t([4, 512], F32, role="weight"),
        kv_state=t([POOL, 8, 512], F32, role="kv"),
        score_state=t([POOL, 8, 512], F32, role="kv"),
        write_plan=t([8, 4], I32, role="metadata"),
        state_slot_mapping=t([1], I32, role="metadata"),
        ratio=4,
        overlap=True,
    )

    moved = 4 * 512 * 4 + 4
    assert work(op, rows((4096, 4096), decode=False), 1, "fp8") == Work({}, moved)


def test_a_bf16_cache_s_decode_reads_the_same_entries_at_its_width():
    op = leaf(
        V4 + "paged_decode:_sparse_attn_v4_paged_decode_triton",
        [],
        q=t([2, 4, 16]),
        unified_kv=t([POOL, 16], role="kv"),
        kv_indices=t([POOL], I32, role="metadata"),
        kv_indptr=t([3], I32, role="metadata"),
        attn_sink=t([4], F32, role="weight"),
        softmax_scale=0.25,
    ) | {"stated": {"window_size": 4, "compress_ratio": 8, "index_topk": None}}

    # Q and the sink read; each of 4 + 1 + 4 + 37 entries' 16 bf16 values read
    # once, two products of width 16 per head and entry.
    entries = 4 + 1 + 4 + 37
    moved = 2 * 4 * 16 * 2 + 4 * 4 + entries * 16 * 2
    assert work(op, rows((1, 9), (1, 300)), 1, "bf16") == Work(
        {"bf16": 4 * 4 * 16 * entries}, moved
    )

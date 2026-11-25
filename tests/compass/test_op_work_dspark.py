# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's DSpark drafter kinds, priced from each leaf's own arguments:
each extractor matches a hand count on one shape."""

import pytest
from test_op_work import F32, I32, leaf, rows, t

from atom.compass.backends.op_work import work
from atom.compass.backends.work import Work

I64 = "torch.int64"
TP = 4
DSPARK = "atom.model_ops.deepseek_v41.dspark:"
MARKOV = "atom.model_ops.dspark_markov_sample:"
STEP = rows((1, 9), (1, 9))


@pytest.mark.parametrize("packed, written", [(False, 768 * 2), (True, 768 * 33 / 32)])
def test_the_draft_kv_tail_writes_its_rows_back_and_reads_norms_and_positions(
    packed, written
):
    # 4 rows of 3 stages of width 64 read once and written back, in bf16 where
    # the pass rounds them in place, in fp8 with a scale byte per 32 where it
    # packs them; the stages' norm weights and the 4 positions read once.
    op = leaf(
        DSPARK + "fused_draft_kv_tail",
        kv=t([1, 4, 3, 64]),
        norm_weight=t([3, 64], role="weight"),
        positions=t([4], I64),
        cos_cache=t([1024, 32], role="weight"),
        sin_cache=t([1024, 32], role="weight"),
        eps=1e-20,
        packed=packed,
    )

    read = 768 * 2 + 3 * 64 * 2 + 4 * 8
    assert work(op, STEP, TP, "bf16") == Work({}, read + written)


def test_the_window_gather_moves_each_layer_s_ring_of_each_slot_in_and_out():
    # 3 layers of 2 slots, each slot's ring 6 rows of width 16 in bf16, read
    # and written once; the layer and slot indices read once.
    op = leaf(
        DSPARK + "gather_window_rows",
        window=t([5, 8, 6, 16], role="KV"),
        layers=t([3], I64),
        slots=t([2], I32),
    )

    assert work(op, STEP, TP, "bf16") == Work({}, 2 * 6 * 6 * 16 * 2 + 3 * 8 + 2 * 4)


def test_the_block_indices_write_each_row_s_positions_and_its_queries_indices():
    # 2 anchors, a block of 3 over a ring of 6: per row its 3 block and 6 ring
    # positions in int64, and for each of its 3 queries an int32 index of the
    # 6 ring slots and the 3 block rows; the anchors read once.
    op = leaf(
        DSPARK + "draft_step_indices", positions=t([2], I64), ring=6, window=4, width=3
    )

    each = (3 + 6) * 8 + 3 * (6 + 3) * 4
    assert work(op, STEP, TP, "bf16") == Work({}, 2 * 8 + 2 * each)


def test_the_block_state_writes_its_copies_of_each_block_row_with_a_pre_mix():
    # 2 anchors of width 8, a block of 3, 4 copies: 24 rows of 8 bf16 written,
    # each with an fp32 pre-mix; the anchors and the noise row read once.
    op = leaf(
        DSPARK + "build_block_state",
        anchor_embed=t([2, 8]),
        noise_embed=t([8], role="weight"),
        width=3,
        hc_mult=4,
    )

    read = 2 * 8 * 2 + 8 * 2
    assert work(op, STEP, TP, "bf16") == Work({}, read + 24 * (8 * 2 + 4))


def test_the_block_attention_attends_each_query_s_listed_entries():
    # 2 rows of 3 queries, 2 heads of width 8, each query over the 5 entries
    # its row lists: two products of width 8 per head and entry. Q, KV, the
    # sink and the indices read once, O written as Q.
    op = leaf(
        "atom.model_ops.sparse_attn_v4:_sparse_attn_triton",
        q=t([2, 3, 2, 8]),
        kv=t([2, 5, 8], role="KV"),
        attn_sink=t([2], F32, role="weight"),
        topk_idxs=t([2, 3, 5], I32),
        softmax_scale=0.35,
    )

    q = 2 * 3 * 2 * 8 * 2
    moved = q + 2 * 5 * 8 * 2 + 2 * 4 + 2 * 3 * 5 * 4 + q
    assert work(op, STEP, TP, "bf16") == Work({"bf16": 4 * 2 * 8 * 2 * 3 * 5}, moved)


def test_the_markov_head_is_a_gemm_onto_its_shard_s_logits_and_its_picks():
    # 2 rows over a shard of 6 ids at rank 4: each row's W1 row against the
    # shard's 6 W2 rows, base_logits read as the GEMM's output term; the ids,
    # the packed pairs and the published rows read or written once.
    op = leaf(
        MARKOV + "_launch",
        base_logits=t([2, 6]),
        prev_ids=t([2], I32),
        markov_w1=t([10, 4], role="weight"),
        markov_w2=t([10, 4], role="weight"),
        out=t([2], I32),
        vocab_offset=0,
        packed=t([2, 2], F32),
        embed_out=t([2, 4]),
    )

    gemm = 2 * 4 * 2 + 6 * 4 * 2 + 2 * 6 * 2
    picks = 2 * 4 + 2 * 4 + 2 * 2 * 4 + 2 * 4 * 2
    assert work(op, STEP, TP, "bf16") == Work({"bf16": 2 * 2 * 6 * 4}, gemm + picks)


def test_the_whole_vocabulary_markov_head_writes_the_w1_rows_it_picks_as_its_output():
    # 2 rows over a vocabulary of 6 ids at rank 4: each row's W1 row against
    # the 6 W2 rows, base_logits read as the GEMM's output term; the previous
    # and the picked ids read or written once, and the 2 picked W1 rows of 4
    # bf16 written as the output.
    op = leaf(
        "aiter.dspark_markov_argmax.default",
        [[2, 4]],
        base_logits=t([2, 6]),
        prev_ids=t([2], I32),
        markov_w1=t([6, 4], role="weight"),
        markov_w2=t([6, 4], role="weight"),
        out=t([2], I32),
    )

    gemm = 2 * 4 * 2 + 6 * 4 * 2 + 2 * 6 * 2
    picks = 2 * 4 + 2 * 4 + 2 * 4 * 2
    assert work(op, STEP, TP, "bf16") == Work({"bf16": 2 * 2 * 6 * 4}, gemm + picks)


def test_the_argmax_gather_reads_each_rank_s_pairs_and_writes_the_ids():
    # 4 ranks' (value, id) pairs for 2 rows in fp32 read once; the 2 ids read
    # and written.
    op = leaf(MARKOV + "argmax_gather", pairs=t([4, 2, 2], F32), out=t([2], I32))

    assert work(op, STEP, TP, "bf16") == Work({}, 4 * 2 * 2 * 4 + 2 * 2 * 4)


def test_staging_the_anchors_reads_the_next_ids_and_their_rows():
    # The 2 next ids and the 2 rows they sit at, read once.
    op = leaf(
        "atom.spec_decode.dspark_proposer:DSparkProposer._stage_anchors",
        self={"module": "runner.drafter"},
        running_bs=2,
        next_token_ids=t([2], I32),
        positions=t([12], I64),
        rows=t([2], I32),
    )

    assert work(op, STEP, TP, "bf16") == Work({}, 2 * 4 + 2 * 4)


def test_the_cursor_commit_copies_each_accepted_row_into_its_slot():
    # 2 requests' accepted cursor rows of 4 int64 read from `cursors` and
    # written to their slots of `cursor`; anchors, starts and slots read once.
    op = leaf(
        "atom.model_ops.attentions.deepseek_v41.speculative:_launch_commit_cursors",
        count=2,
        cursor=t([8, 4], I64, role="KV"),
        cursors=t([2, 6, 4], I64),
        anchors=t([2], I32),
        starts=t([2], I32),
        slots=t([2], I32),
    )

    assert work(op, STEP, TP, "bf16") == Work({}, 2 * 2 * 4 * 8 + 3 * 2 * 4)

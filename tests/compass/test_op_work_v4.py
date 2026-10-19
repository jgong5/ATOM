# SPDX-License-Identifier: MIT
"""DeepSeek-V4-Pro's op kinds, priced from each leaf's own arguments.

Each extractor matches a hand count on one shape, the MoE's in expectation
under uniform routing; every leaf of the nightly DPA-TP8 captures, EP off and
EP8, has work that prices, but the attention, which is refused by name; and the
op-level backend refuses a V4-Pro step for that reason alone.
"""

from math import comb

import pytest
from test_capture_v4_pro import DPA_TP8, EP8, SHAPES, capture
from test_op_work import I32, ONE, _view, leaf, rows, t
from test_work import SPEC

from atom.compass.backends.op_graph import OpGraphBackend
from atom.compass.backends.op_work import work
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.work import (
    Collective,
    Work,
    activated_experts,
    max_rank_load,
    price,
)
from atom.compass.bench.ops import leaves

F32, FP8, FP4 = "torch.float32", "torch.float8_e4m3fn", "torch.float4_e2m1fn_x2"
E8M0 = "torch.float8_e8m0fnu"
ATTENTION = "aiter.v4_attention_with_output.default"
MORI = "mori.ops.dispatch_combine:EpDispatchCombineOp."


def moe(rows, local, mask=None):
    """Routed experts of hidden 64 and intermediate 32, top 2, fp4 weights."""
    return leaf(
        "aiter.fused_moe_.default",
        [[rows, 64]],
        hidden_states=t([rows, 64]),
        w1=t([local, 64, 32], FP4, role="weight"),
        w2=t([local, 64, 16], FP4, role="weight"),
        topk_weight=t([rows, 2], F32),
        topk_ids=t([rows, 2], I32),
        expert_mask=mask and t([mask], I32, role="weight"),
        quant_type=3,
    )


def experts(routed, groups):
    """Gate-up `[routed, 64] x [64, 64]` then down `[routed, 32] x [32, 64]`,
    bf16 rows, fp4 weights read once per activated expert, bf16 outputs."""
    flops = 2 * routed * 64 * 64 + 2 * routed * 64 * 32
    moved = routed * (64 * 2 + 64 * 2) + groups * 64 * 64 * 0.5
    moved += routed * (32 * 2 + 64 * 2) + groups * 32 * 64 * 0.5
    return Work({"bf16": flops}, moved)


def close(got, want):
    assert got.flops_by_format == pytest.approx(want.flops_by_format)
    assert (got.bytes, got.comm) == (pytest.approx(want.bytes), want.comm)


def test_the_routed_experts_run_every_route_on_one_rank():
    hit = activated_experts(16, 2, 8)

    close(work(moe(8, 16), ONE, 1, "bf16"), experts(16, hit))


@pytest.mark.parametrize("mask", [16, 17], ids=["mask", "mask_and_sentinel"])
def test_under_expert_parallelism_the_busiest_rank_s_share_runs(mask):
    # Four of sixteen experts here: four ranks, whether or not the mask
    # carries the sentinel slot for non-local routes.
    load, hit = max_rank_load(8, 2, 16, 4), activated_experts(16, 2, 8) / 4

    close(work(moe(8, 4, mask), ONE, 1, "bf16"), experts(load, hit))


def test_a_block_scaled_fp8_gemm_reads_its_scales_once():
    op = leaf(
        "aiter.gemm_a8w8_blockscale_preshuffle_impl.default",
        [[4, 128]],
        x=t([4, 256], FP8),
        weight=t([128, 256], FP8, role="weight"),
        x_scale=t([4, 2], E8M0),
        w_scale=t([1, 2], E8M0, role="weight"),
        dtype="torch.bfloat16",
        prefix="layers.0.ffn.shared_experts.gate_up_proj",
    )

    moved = 4 * 256 + 128 * 256 + 4 * 128 * 2 + 8 + 2
    assert work(op, ONE, 1, "bf16") == Work({"fp8": 2 * 4 * 128 * 256}, moved)


def test_a_block_quant_writes_its_values_and_scales_in_their_formats():
    op = leaf(
        "aiter.per_1x128_e8m0_quant.default",
        [[4, 256], [4, 2]],
        x=t([4, 256]),
        quant_dtype=FP8,
        transpose_scale=True,
    )

    assert work(op, ONE, 1, "bf16") == Work({}, 4 * 256 * 2 + 4 * 256 + 4 * 2)


def test_a_hyper_connection_mix_projects_the_widened_residual():
    op = leaf(
        "aiter.mhc_pre.default",
        [[2, 4, 1], [2, 4, 4], [2, 64]],
        residual=t([2, 4, 64]),
        fn=t([24, 256], I32, role="weight"),
        hc_scale=t([3], F32, role="weight"),
        hc_base=t([24], F32, role="weight"),
        norm_weight=t([64], role="weight"),
    )

    read = 2 * 4 * 64 * 2 + 24 * 256 * 4 + 3 * 4 + 24 * 4 + 64 * 2
    written = (2 * 4 + 2 * 4 * 4 + 2 * 64) * 2
    assert work(op, ONE, 1, "bf16") == Work({"bf16": 2 * 2 * 24 * 256}, read + written)


@pytest.mark.parametrize(
    "name, rows_in, rows_out, sizes, sent",
    [
        # Gathers send 7/8 of the gathered rows; reduce-scatters 7/8 of theirs.
        ("all_gather_with_padding", 2, 16, None, 16),
        ("all_gatherv", 2, 16, [2] * 8, 16),
        ("reduce_scatter_with_unpadding", 16, 2, None, 16),
        ("reduce_scatterv", 16, 2, [2] * 8, 16),
    ],
)
def test_a_dp_exchange_spans_the_group_its_shapes_state(
    name, rows_in, rows_out, sizes, sent
):
    args = {"x": t([rows_in, 64])} | ({"sizes": sizes} if sizes else {})
    op = leaf(f"atom.model_ops.moe:{name}", [[rows_out, 64]], **args)

    assert work(op, ONE, 1, "bf16") == Work(
        {}, (rows_in + rows_out) * 64 * 2, (Collective(8, 7 / 8 * sent * 64 * 2),)
    )


def test_mori_sends_each_token_to_the_ranks_holding_its_experts():
    weights, indices = t([2, 2], F32), t([2, 2], I32)
    dispatch = leaf(
        MORI + "dispatch",
        [[16, 64], [16, 2], [16, 2], [1]],
        input=t([2, 64]),
        weights=weights,
        scales=None,
        indices=indices,
    )
    combine = leaf(
        MORI + "combine", [[4096, 64]], input=t([16, 64]), weights=None, indices=indices
    )
    # Eight ranks, top 2: another rank holds one of a token's experts with
    # chance at least 1 - (7/8)^2, which sixteen experts over eight ranks meet.
    reach = 1 - (7 / 8) ** 2
    assert reach <= 1 - comb(14, 2) / comb(16, 2)

    for op, row in ((dispatch, 64 * 2 + 2 * 4 + 2 * 4), (combine, 64 * 2)):
        sent = 2 * 7 * reach * row
        [c] = work(op, ONE, 1, "bf16").comm
        assert (c.ranks, c.bytes) == (8, pytest.approx(sent))


def test_an_all_to_all_sends_all_but_its_own_split():
    def op(sizes):
        return leaf(
            "_c10d_functional.all_to_all_single.default",
            [[64]],
            input=t([64]),
            output_split_sizes=sizes,
            input_split_sizes=sizes,
            group_name="7",
        )

    assert work(op([8] * 8), ONE, 1, "bf16").comm == (Collective(8, 7 / 8 * 128),)
    assert work(op([22] + [6] * 7), ONE, 1, "bf16").comm == (
        Collective(8, 42 / 64 * 128),
    )
    with pytest.raises(CostRefused, match="no input_split_sizes"):
        work(op([]), ONE, 1, "bf16")


def test_a_hashed_route_does_not_count_its_whole_table():
    op = leaf(
        "atom.model_ops.triton_hash_topk:hash_topk_triton",
        ids=t([8], I32),
        gating_output=t([8, 16]),
        tid2eid=t([129280, 2], I32, role="weight"),
        out_ids=t([8, 2], I32),
        out_weights=t([8, 2], F32),
    )

    assert work(op, ONE, 1, "bf16") == Work({}, 8 * 4 + 8 * 16 * 2 + 8 * 2 * 8)


def test_an_index_build_is_one_launch():
    op = leaf(
        "atom.model_ops.v4_kernels.paged_decode_indices:build_v4_paged_decode_indptr",
        batch_id_per_q_token=t([1], I32, role="metadata"),
        swa_indptr=t([2], I32, role="metadata"),
        T_pad=1,
    )

    assert work(op, ONE, 1, "bf16") == Work()


def test_the_attention_is_refused_by_name():
    op = leaf(
        ATTENTION,
        [[1, 64]],
        x=t([1, 64]),
        positions=t([1], "torch.int64"),
        layer_name={"context": "layers.0.attn"},
    )

    with pytest.raises(CostRefused, match=f"{ATTENTION}: the leaf states the layer"):
        work(op, ONE, 1, "fp8")


@pytest.mark.parametrize("ep", ["", EP8], ids=["ep_off", "ep8"])
def test_every_v4_pro_leaf_prices_but_the_attention(ep, tmp_path_factory):
    graphs = capture(tmp_path_factory, SHAPES + DPA_TP8 + ep)
    kinds, refused = set(), set()
    for shape, (graph, _) in graphs.items():
        view = _view(shape)
        for op, _ in leaves(graph["region"]).values():
            kinds.add(op["op"])
            try:
                w = work(op, view, 1, "fp8")
            except CostRefused:
                refused.add(op["op"])
                continue
            assert w is None or price(w, SPEC) > 0

    assert refused == {ATTENTION}
    assert {"aiter.fused_moe_.default", "aiter.mhc_pre.default"} <= kinds
    exchanges = {MORI + "dispatch", MORI + "combine"}
    assert exchanges <= kinds if ep else not exchanges & kinds

    # The op-level backend refuses a V4-Pro step for the attention alone.
    graph, _ = graphs["decode B=2"]
    backend = OpGraphBackend(SPEC, lambda *shape: graph["region"], kv="fp8")
    with pytest.raises(CostRefused) as refusal:
        backend.price(rows((1, 1024), (1, 1024)))
    assert [r.reason.split(":")[0] for r in refusal.value.declined] == [
        f"the op kind {ATTENTION}"
    ]

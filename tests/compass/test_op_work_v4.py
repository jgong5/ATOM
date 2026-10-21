# SPDX-License-Identifier: MIT
"""DeepSeek-V4-Pro's op kinds, priced from each leaf's own arguments.

Each extractor matches a hand count on one shape, the MoE's in expectation
under uniform routing; the whole attention as one leaf is refused by name;
every leaf of the nightly DPA-TP8 captures, EP off and EP8, has work that
prices, the attention's own kernels among them; and the op-level backend
prices a V4-Pro decode step and a prefill step. Under `ATOM_USE_TRITON_MOE`
every leaf prices too, the experts at bf16 at EP off, fp8 at EP8, and fp4
under `ATOM_USE_TRITON_MOE_A4W4`.
"""

from dataclasses import replace
from math import comb

import pytest
import test_capture_v4_pro
from test_capture_v4_pro import DPA_TP8, EP8, SHAPES, capture
from test_op_work import BF16, I32, ONE, _view, leaf, rows, t
from test_work import machine

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
E8M0, U8 = "torch.float8_e8m0fnu", "torch.uint8"
ATTENTION = "aiter.v4_attention_with_output.default"
MORI = "mori.ops.dispatch_combine:EpDispatchCombineOp."
DECODE = "aiter.mla:mla_decode_fwd_v4_nm"
#: A machine stating an fp4 rate, as gfx950 does: V4-Pro's experts run at it.
SPEC = machine(fp4_flops=10e15)


def moe(rows, local, mask=None, quant_type=3):
    """Routed experts of hidden 64 and intermediate 32, top 2, fp4 weights
    with a one-byte scale per 32 of them."""
    return leaf(
        "aiter.fused_moe_.default",
        [[rows, 64]],
        hidden_states=t([rows, 64]),
        w1=t([local, 64, 32], FP4, role="weight"),
        w2=t([local, 64, 16], FP4, role="weight"),
        w1_scale=t([local * 64, 2], U8, role="weight"),
        w2_scale=t([local * 64, 1], U8, role="weight"),
        topk_weight=t([rows, 2], F32),
        topk_ids=t([rows, 2], I32),
        expert_mask=mask and t([mask], I32, role="weight"),
        quant_type=quant_type,
    )


def experts(rows, routed, groups, mask=0, fmt="fp4"):
    """Gate-up `[routed, 64] x [64, 64]` then down `[routed, 32] x [32, 64]`,
    at `fmt`, fp4 where the op quantizes its bf16 rows to the weights'. M is
    the boundary alone: each activated expert's weights and scales once, the
    bf16 rows in and out, their routes and the mask at the share `routed / 2`
    of the `rows`."""
    flops = 2 * routed * 64 * 64 + 2 * routed * 64 * 32
    io = (rows * (64 * 2 + 2 * 4 + 2 * 4 + 64 * 2) + mask * 4) * routed / (2 * rows)
    return Work({fmt: flops}, io + groups * ((64 * 64 + 32 * 64) * 0.5 + 64 * 3))


def close(got, want):
    assert got.flops_by_format == pytest.approx(want.flops_by_format)
    assert (got.bytes, got.comm) == (pytest.approx(want.bytes), want.comm)


@pytest.mark.parametrize("quant_type, fmt", [(3, "fp4"), (0, "bf16")])
def test_the_routed_experts_run_every_route_on_one_rank(quant_type, fmt):
    hit = activated_experts(16, 2, 8)

    got = work(moe(8, 16, quant_type=quant_type), ONE, 1, "bf16")
    close(got, experts(8, 16, hit, fmt=fmt))


@pytest.mark.parametrize("mask", [16, 17], ids=["mask", "mask_and_sentinel"])
def test_under_expert_parallelism_the_busiest_rank_s_share_runs(mask):
    # Four of sixteen experts here: four ranks, whether or not the mask
    # carries the sentinel slot for non-local routes.
    load, hit = max_rank_load(8, 2, 16, 4), activated_experts(16, 2, 8) / 4

    close(work(moe(8, 4, mask), ONE, 1, "bf16"), experts(8, load, hit, mask))


TRITON = "aiter.ops.triton.moe."
GEMM = TRITON + "moe_op_gemm_{0}w4:moe_gemm_{0}w4"
#: The activation each Triton experts GEMM kind takes, as its leaf states it:
#: dtype and packed width of a 64-wide row.
ACTS = {"a16": (BF16, 64), "a8": (FP8, 64), "a4": (U8, 32)}


def triton(kind, rows, local, gemm2=False):
    """One of the Triton experts' GEMMs over the routed experts of `moe`'s
    shapes: the gate-up, which gathers the rows and applies SwiGLU, or the
    down, which scatters its routes back into the rows."""
    dtype, packed = ACTS[kind]
    k, n = (32, 64) if gemm2 else (64, 64)
    slot = "scatter_indx" if gemm2 else "gather_indx"
    state = [["gate_scal", t([rows * 2], F32)], ["n_expts_tot", local]]
    return leaf(
        GEMM.format(kind),
        x=t([rows * 2, packed // 2] if gemm2 else [rows, packed], dtype),
        w=t([local, k // 2, n], U8, role="weight"),
        x_scales=t([rows, 2], U8),
        w_scales=t([local, n, k // 32], U8, role="weight"),
        bias=None,
        routing_data={
            "object": TRITON + "moe_routing.routing:RoutingData",
            "state": state + [["n_expts_act", 2]],
        },
        **{slot: t([rows * 2], I32)},
        gammas=t([rows * 2], F32) if gemm2 else None,
        out_dtype=BF16 if gemm2 else FP8,
        apply_swiglu=not gemm2,
        y_out=None,
    )


@pytest.mark.parametrize("ep", [1, 4], ids=["ep_off", "ep4"])
def test_the_triton_experts_gemms_are_the_routed_experts_products(ep):
    # Sixteen experts in all: every one local at EP off, four a rank at EP4.
    local = 16 // ep
    load = 8 * 2 if ep == 1 else max_rank_load(8, 2, 16, ep)
    hit = activated_experts(16, 2, 8) / ep
    share = load / 16

    up, down = (work(triton("a8", 8, local, g), ONE, 1, "bf16", ep) for g in (0, 1))

    weights = (32 * 64 + 64 * 2, 16 * 64 + 64 * 1)
    # Gate-up: the fp8 rows, their scales and gather index, and SwiGLU's half
    # written in fp8; down: the routes' fp8 rows, scales, index and weights,
    # and the tokens' bf16 rows.
    io = (8 * 64 + 8 * 2 + 16 * 4 + 16 * 32, 16 * 32 + 8 * 2 + 16 * 8 + 8 * 64 * 2)
    for got, flops, held, moved in zip((up, down), (64 * 64, 32 * 64), weights, io):
        close(got, Work({"fp8": 2 * load * flops}, held * hit + moved * share))


@pytest.mark.parametrize("kind, fmt", [("a16", "bf16"), ("a8", "fp8"), ("a4", "fp4")])
def test_the_triton_experts_run_at_the_activation_format_their_kind_names(kind, fmt):
    for gemm2 in (False, True):
        got = work(triton(kind, 8, 16, gemm2), ONE, 1, "bf16")

        assert got.flops_by_format == {fmt: 2 * 16 * 64 * (32 if gemm2 else 64)}


def test_a_triton_experts_gemm_with_no_routes_is_refused_by_name():
    op = triton("a8", 8, 16)
    op["args"]["gather_indx"] = None

    with pytest.raises(CostRefused, match="moe_gemm_a8w4: no top k in routing_data"):
        work(op, ONE, 1, "bf16")


def test_an_mx_downcast_writes_its_values_and_a_scale_byte_per_32():
    op = leaf(
        TRITON + "quant_moe:downcast_to_mxfp",
        src_tensor=t([8, 64]),
        out_quant_type=FP8,
        axis=-1,
    )

    assert work(op, ONE, 1, "bf16") == Work({}, 8 * 64 * 2 + 8 * 64 + 8 * 2)


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
        # Uneven: the group is the eight `sizes`, not the rows' ratio.
        ("all_gatherv", 3, 10, [3] + [1] * 7, 10),
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
def test_every_v4_pro_leaf_prices(ep, tmp_path_factory):
    graphs = capture(tmp_path_factory, SHAPES + DPA_TP8 + ep)
    kinds = set()
    for shape, (graph, _) in graphs.items():
        view = _view(shape)
        for op, _ in leaves(graph["region"]).values():
            kinds.add(op["op"])
            w = work(op, view, 1, "fp8")
            assert w is None or price(w, SPEC) > 0

    assert ATTENTION not in kinds
    assert {"aiter.fused_moe_.default", "aiter.mhc_pre.default"} <= kinds
    assert {"aiter.pa_sparse_prefill_fp8_opus.default", DECODE} <= kinds
    exchanges = {MORI + "dispatch", MORI + "combine"}
    assert exchanges <= kinds if ep else not exchanges & kinds

    # The op-level backend prices a V4-Pro decode step and a prefill step.
    for shape, view in (
        ("decode B=2", rows((1, 1024), (1, 1024))),
        (
            "prefill T=512 B=1",
            replace(rows((512, 512), decode=False), produces_output=True),
        ),
    ):
        graph, _ = graphs[shape]
        backend = OpGraphBackend(SPEC, lambda *s, g=graph: g["region"], kv="fp8")
        assert backend.price(view).seconds > 0


@pytest.mark.parametrize(
    "ep, a4w4, fmt",
    [("", "0", "bf16"), (EP8, "0", "fp8"), (EP8, "1", "fp4")],
    ids=["ep_off", "ep8", "ep8_a4w4"],
)
def test_v4_pro_s_triton_experts_price_at_the_format_their_kernels_run(
    ep, a4w4, fmt, tmp_path_factory, monkeypatch
):
    monkeypatch.setenv("ATOM_USE_TRITON_MOE", "1")
    monkeypatch.setenv("ATOM_USE_TRITON_MOE_A4W4", a4w4)
    # `capture` keys its cache by the arguments alone, not the environment.
    monkeypatch.setattr(test_capture_v4_pro, "_CAPTURES", {})
    graphs = capture(tmp_path_factory, "--width 1 --prefill 512 1" + DPA_TP8 + ep)
    width = 8 if ep else 1

    formats = set()
    for shape, (graph, _) in graphs.items():
        view = _view(shape)
        for op, _ in leaves(graph["region"]).values():
            w = work(op, view, 1, "fp8", width)
            assert w is None or price(w, SPEC) > 0
            gemm = op["op"].startswith(TRITON + "moe_op_gemm_")
            formats |= set(w.flops_by_format) if gemm else set()
    assert formats == {fmt}

    # The backend hands its EP width to the experts: a rank runs its share of
    # every rank's experts, not every route over its own.
    graph, _ = graphs["prefill T=512 B=1"]
    view = replace(_view("prefill T=512 B=1"), produces_output=True)
    seconds = [
        OpGraphBackend(SPEC, lambda *s: graph["region"], kv="fp8", ep=n).price(view)
        for n in (width, 1)
    ]
    assert seconds[0].seconds < seconds[1].seconds if ep else seconds[0].seconds > 0

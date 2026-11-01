# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's op kinds, priced from each leaf's own arguments and what it
states.

Each extractor matches a hand count on one shape; the sparse attention and the
indexer's scorer read their entries at the geometry their leaf states, over
the queries the leaf holds, and are refused by name where it states none, as
the mono K2 is; every leaf of the V4.1-Flash TP4 captures, EP off and EP4,
and gfx950's EP4 and mono EP-off decodes prices; and the op-level backend
prices a V4.1-Flash decode step and a prefill step on MI455 A0's spec, a
decode step over a longer context higher.
"""

import json

import pytest
from test_capture_v41_flash import ARGS, COMPILED, EP4, MONO, SHAPES, TP, TREE, capture
from test_op_work import BF16, F32, I32, leaf, rows, t

from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.op_graph import OpGraphBackend
from atom.compass.backends.op_work import work
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.work import Work, allgather, allreduce, gemm, price
from atom.compass.bench.ops import leaves
from atom.compass.capture.forward import V41_ATTENTION, V41_LAYER_POST, V41_SCORERS
from atom.compass.spec import MachineSpec

FP8, E8M0, I64 = "torch.float8_e4m3fn", "torch.float8_e8m0fnu", "torch.int64"
MI455 = MachineSpec.from_mapping(
    json.loads((TREE / "scripts/compass/machines/mi455-a0.json").read_text())
)
#: Window 4, every 2 tokens compressed, the top 3 read.
GEOMETRY = {"window_size": 4, "compress_ratio": 2, "index_topk": 3}
FP4, U8 = "torch.float4_e2m1fn_x2", "torch.uint8"
V41_MONO = "atom.models.deepseek_v41.mono."
K1 = V41_MONO + "kernels.attn_pre:launch_attn_pre"
STEP_BEGIN = "atom.mono.runtime.step_begin:_launch_step_begin"
#: The mono decode's kernels, every one in a decode it serves.
MONOS = {K1, V41_LAYER_POST, STEP_BEGIN, V41_MONO + "step_meta:_launch_step_meta"}


def build(name, **fields):
    """A mono kernel's build key, as a leaf holds it."""
    return {"object": f"{name}Build", "state": [[k, v] for k, v in fields.items()]}


def weight(shape, dtype=FP8):
    return t(shape, dtype, role="weight")


#: What a mono kernel reaches by index or zeroes apart, each far wider than
#: a step of 2 rows reads.
TABLES = {
    "scratch": t([1 << 20], U8),
    "cos": weight([4096, 4], BF16),
    "ilog": t([2, 1 << 16], F32),
}


def attn_pre(fold, tok_par=False):
    """K1 of 2 rows over 2 hc copies of width 8: 12 mixes, a 16-row QKV
    and a 32-row Q product; no indexer feed."""
    return leaf(
        K1,
        key=build("AttnPre", tokens=2, fold=fold, tok_par=tok_par),
        res_in=t([1, 2, 2, 8]),
        pend=t([1, 2, 8]) if fold else 0,
        res_out=t([1, 2, 2, 8]),
        hc_fn=weight([12, 16], F32),
        wqkv=weight([16, 8]),
        wqkv_s=weight([1, 1], E8M0),
        wqb=weight([32, 16]),
        ring=t([4096, 8], role="kv"),
        iq_w=0,
        iw_w=0,
        **TABLES,
    )


def layer_post(stated=True, **fields):
    """K2 of 2 rows at width 8 whose indexer runs over 2 heads of width 16,
    attending 2 heads of width 8: 128 output-projection and 32 router weights,
    and 4 experts whose width 2 the loader pads to 4, 2 a route."""
    op = leaf(
        V41_LAYER_POST,
        key=build(
            "LayerPost", tokens=2, experts=4, topk=2, tp=TP, index_fp4=False, **fields
        ),
        iact=1,
        iq=t([2, 2, 16], FP8),
        q=t([1, 2, 2, 8]),
        pool=t([4096, 8], role="kv"),
        woa=weight([8, 16]),
        gate_w=weight([4, 8], BF16),
        w13=weight([4, 8, 4], FP4),
        w13_s=weight([4, 2], U8),
        w2=weight([4, 8, 2], FP4),
        w2_s=weight([4, 1], U8),
        out=t([1, 2, 8]),
        **dict.fromkeys(("wob", "hc_fn", "sgu", "sw2"), 0),
        **TABLES,
    )
    held = {**GEOMETRY, "index_candidates": 100, "expert_width": 2}
    return {**op, "stated": held} if stated else op


def sparse(q_rows, stated=GEOMETRY):
    """V4.1's sparse attention over `q_rows` queries of 2 heads of width 8."""
    op = leaf(
        V41_ATTENTION,
        [[q_rows, 2, 8]],
        q=t([q_rows, 2, 8]),
        unified_kv=t([4096, 8], role="KV"),
        kv_indices_prefix=t([q_rows * 3], I32),
        kv_indptr_prefix=t([q_rows + 1], I32),
        kv_indices_extend=t([q_rows * 4], I32),
        attn_sink=t([2], F32, role="weight"),
        softmax_scale=0.35,
        out=t([q_rows, 2, 8]),
    )
    return {**op, "stated": stated} if stated else op


def scorer(ratio, candidates, stated=True):
    """V4.1's FP8 scorer of one query over 4 index heads of width 16."""
    op = leaf(
        V41_SCORERS[0],
        q_fp8=t([1, 1, 4, 16], FP8),
        kv_cache=t([1024, 8, 1, 20], "torch.uint8", role="KV"),
        weights=t([1, 4], F32),
        out_logits=t([1, 4096], F32),
        context_lens=t([1], I32),
        kv_indices=t([1, 512], I32),
    )
    held = {"compress_ratio": ratio, "index_candidates": candidates}
    return {**op, "stated": held} if stated else op


def test_the_sparse_attention_reads_the_window_and_the_top_compressed_entries():
    # Row 1 at context 300: its last 4 tokens and the top 3 of its 150
    # compressed entries. Row 2 at context 5: 4 and 2. Two products of width
    # 8 per head and entry, each entry read once in bf16, Q, sink and O once.
    got = work(sparse(2), rows((1, 300), (1, 5)), TP, "bf16")

    entries = 4 + 3 + 4 + 2
    io = 2 * 2 * 8 * 2 + 2 * 4 + 2 * 2 * 8 * 2
    assert got == Work({"bf16": 4 * 2 * 8 * entries}, entries * 8 * 2 + io)


def test_a_leaf_of_fewer_queries_than_the_step_runs_each_row_s_last_ones():
    # A replayed prefill's tail: a 512-token prompt whose decoder layer runs
    # its last 4 rows, each over the context before it.
    tail = work(sparse(4), rows((512, 512), decode=False), TP, "bf16")

    assert tail == work(sparse(4), rows((4, 512), decode=False), TP, "bf16")
    assert tail != work(sparse(4), rows((4, 4), decode=False), TP, "bf16")


def test_the_scorer_scores_every_compressed_entry_or_at_most_its_candidates():
    view = rows((1, 300))
    io = 4 * 16 + 4 * 4

    # 150 compressed entries before the query, each one product of width 16
    # per head and read once in fp8.
    assert work(scorer(2, None), view, TP, "bf16") == Work(
        {"fp8": 2 * 4 * 16 * 150}, 150 * 16 + io
    )
    # Its last 100, where an earlier layer chose 100 candidates.
    assert work(scorer(2, 100), view, TP, "bf16") == Work(
        {"fp8": 2 * 4 * 16 * 100}, 100 * 16 + io
    )


@pytest.mark.parametrize(
    "op, missing",
    [
        (sparse(2, stated=None), "window_size, compress_ratio, index_topk"),
        (scorer(2, None, stated=False), "compress_ratio, index_candidates"),
        (
            layer_post(stated=False),
            ", ".join([*GEOMETRY, "index_candidates", "expert_width"]),
        ),
    ],
    ids=["attention", "scorer", "mono"],
)
def test_a_leaf_that_states_no_geometry_is_refused_by_name(op, missing):
    with pytest.raises(CostRefused, match=f"{op['op']}: the leaf states no {missing}"):
        work(op, rows((1, 300), (1, 5)), TP, "bf16")


@pytest.mark.parametrize("fold", [True, False])
def test_mono_k1_runs_its_rows_through_each_weight_and_reads_no_table(fold):
    # 2 rows through 192 mix weights at bf16, and 128 and 512 at fp8. Each
    # weight and row read once; with nothing to fold the residual stays put.
    want = Work({"bf16": 4 * 192, "fp8": 4 * 640}, 768 + 128 + 1 + 512 + 64)

    got = work(attn_pre(fold), rows((1, 300), (1, 5)), TP, "bf16")

    assert got == want + Work({}, 32 + 64 if fold else 0)


def test_mono_k2_attends_scores_and_routes_the_view_s_rows():
    view = rows((1, 300), (1, 5))
    # The attention's 13 entries and the scorer's 102 of its 100 candidates,
    # as the sparse attention's and the scorer's leaves count them.
    attended = Work(
        {"bf16": 4 * 2 * 8 * 13, "fp8": 2 * 2 * 16 * 102}, 13 * 16 + 102 * 16
    )
    # 2 rows through 128 output-projection and 32 router weights; 4 routes
    # through the real half of an expert's 96 weights at fp8. 3 of 4 experts
    # read, each half its 50 bytes of weights and gate-up scales and its 1
    # byte of down scales whole.
    products = Work({"bf16": 4 * 32, "fp8": 4 * 128 + 4 * 2 * 48}, 3 * 26)
    # q, iq, the two products' weights and the output read once.
    io = Work({}, 64 + 64 + 128 + 64 + 32)
    reduced = allreduce(TP, 2 * 8 * 2)

    got = work(layer_post(), view, TP, "bf16")

    assert got == attended + products + io + reduced + reduced


def test_the_mono_mailbox_reset_zeroes_its_buffers_whole_whatever_the_rows():
    op = leaf(
        STEP_BEGIN,
        key=build("StepBegin", tp=TP, debug=False),
        scratch=[t([1024], U8), t([512], U8)],
        peers=t([TP], I64),
        fence_off=256,
        rank=0,
        state=t([137], I64),
    )
    # Then each rank's epoch word to every other: the fence.
    want = Work({}, 1024 + 512 + 256) + allgather(TP, 8 * TP)

    assert work(op, rows((1, 9)), TP, "bf16") == want
    assert work(op, rows(*[(1, 9)] * 3), TP, "bf16") == want


@pytest.mark.parametrize(
    "op",
    [attn_pre(True, tok_par=True), layer_post(ep=2), layer_post(moe_tp=2)],
    ids=["token_parallel", "ep", "moe_tp"],
)
def test_a_mono_build_whose_rows_cross_ranks_is_refused_by_name(op):
    with pytest.raises(CostRefused, match=f"{op['op']}: a token-parallel or DP-"):
        work(op, rows((1, 9)), TP, "bf16")


def test_the_delayed_hyper_connection_projects_each_row_s_widened_residual():
    op = leaf(
        "aiter.v41_mhc_pre_delayed.default",
        [[1, 3, 8], [1, 3, 4], [1, 3, 4], [1, 3, 4, 4]],
        residual=t([1, 3, 4, 8]),
        pre_mix=t([1, 3, 4], F32),
        hc_fn=t([24, 32], F32, role="weight"),
        hc_scale=t([3], F32, role="weight"),
        hc_base=t([24], F32, role="weight"),
        norm_weight=t([8], role="weight"),
        rms_eps=1e-6,
    )

    # 3 rows of 4 copies of width 8 onto 24 mixes; every argument read once,
    # the 96 output elements written at a byte each.
    read = 96 * 2 + 12 * 4 + 768 * 4 + 3 * 4 + 24 * 4 + 8 * 2
    assert work(op, rows((3, 3), decode=False), TP, "bf16") == Work(
        {"bf16": 2 * 3 * 4 * 8 * 24}, read + 96
    )


@pytest.mark.parametrize("dequantize, written", [(False, 128 * 33 / 32), (True, 256)])
def test_the_group_32_quant_writes_fp8_and_a_scale_byte_per_32(dequantize, written):
    op = leaf(
        "atom.model_ops.blockscale:quantize_fp8", x=t([2, 64]), dequantize=dequantize
    )

    assert work(op, rows((2, 2)), TP, "bf16") == Work({}, 256 + written)


def test_the_per_group_quant_reads_its_rows_and_writes_fp8_with_a_scale_per_group():
    # 2 rows of 64 bf16 read once and written in fp8, with an e8m0 scale for
    # each group of 32: a DP-attention rank's quant before its MoE dispatch.
    op = leaf(
        "aiter.dynamic_per_group_scaled_quant.default",
        out=t([2, 64], FP8),
        input=t([2, 64]),
        scales=t([2, 2], E8M0),
        group_size=32,
        shuffle_scale=False,
    )

    assert work(op, rows((2, 2)), TP, "bf16") == Work({}, 256 + 128 + 4)


def test_the_dual_norm_quant_writes_both_sides_in_mxfp8():
    op = leaf(
        "aiter.ops.triton.quant.fused_mxfp8_quant:fused_dual_rmsnorm_mxfp8_quant",
        q=t([2, 64]),
        k=t([2, 32]),
        q_weight=t([64], role="weight"),
        k_weight=t([32], role="weight"),
        yq=None,
    )

    read = 256 + 128 + 128 + 64
    assert work(op, rows((2, 2)), TP, "bf16") == Work({}, read + 192 * 33 / 32)


def test_a_block_scaled_fp8_gemm_reads_its_scales_once():
    op = leaf(
        "aiter.gemm_a8w8_blockscale.default",
        [[16, 32]],
        XQ=t([16, 64], FP8),
        WQ=t([32, 64], FP8, role="weight"),
        x_scale=t([16, 2], E8M0),
        w_scale=t([1, 2], E8M0, role="weight"),
        dtype=BF16,
    )

    want = gemm(16, 32, 64, "fp8", "fp8", "bf16") + Work({}, 32 + 2)
    assert work(op, rows(*[(1, 9)] * 16), TP, "bf16") == want


def test_a_row_gather_moves_each_row_of_each_tensor_in_and_out():
    op = leaf(
        "atom.model_ops.attentions.token_layout.gather:gather_rows",
        rows=t([3], I32),
        dim=1,
        out=None,
        tensors=[t([1, 10, 4]), t([1, 10, 2], F32)],
    )

    assert work(op, rows((10, 10), decode=False), TP, "bf16") == Work(
        {}, 2 * 3 * (4 * 2 + 2 * 4) + 3 * 4
    )


def test_an_index_fill_writes_the_slices_its_index_names():
    # Two of ten 6-wide fp32 rows reset, not the whole tensor.
    op = leaf(
        "aten.index_fill_.int_Scalar",
        [[10, 6]],
        self=t([10, 6], F32, role="KV"),
        dim=0,
        index=t([2], I64),
        value=0,
    )

    assert work(op, rows((1, 9)), TP, "bf16") == Work({}, 2 * 6 * 4 + 2 * 8)


def _view(shape: str, context=1024) -> BatchView:
    """The capture's rows, a decode row over `context` tokens; a prefill's
    final chunk samples."""
    sizes = dict(part.split("=") for part in shape.split()[1:])
    if shape.startswith("decode"):
        held = (RequestShape(1, context, True),) * int(sizes["B"])
        return BatchView(held, produces_output=True)
    tokens = int(sizes["T"])
    return BatchView((RequestShape(tokens, tokens, False),), produces_output=True)


#: Each capture, and the kinds its decode attends through.
CAPTURES = [
    (SHAPES + ARGS, "gfx1250", 1, {V41_ATTENTION, V41_SCORERS[0]}),
    (SHAPES + ARGS + EP4, "gfx1250", TP, {V41_ATTENTION, V41_SCORERS[0]}),
    ("--width 16" + ARGS + EP4, "gfx950", TP, {V41_ATTENTION, COMPILED}),
    (MONO, "gfx950", 1, MONOS),
]


@pytest.mark.parametrize(
    "argv, arch, ep, attends",
    CAPTURES,
    ids=["gfx1250_ep_off", "gfx1250_ep4", "gfx950_ep4", "gfx950_ep_off"],
)
def test_every_v41_flash_leaf_prices(argv, arch, ep, attends, tmp_path_factory):
    graphs = capture(tmp_path_factory, argv, arch)
    kinds = set()
    for shape, (graph, _) in graphs.items():
        view = _view(shape)
        for op, _ in leaves(graph["region"]).values():
            kinds.add(op["op"])
            w = work(op, view, TP, "bf16", ep)
            assert w is None or price(w, MI455) > 0

    assert attends <= kinds

    # The op-level backend prices each step, a decode over a longer context
    # higher: its scorers read more entries.
    for shape, (graph, _) in graphs.items():
        backend = OpGraphBackend(
            MI455, lambda *s, g=graph: g["region"], tp=TP, kv="bf16", ep=ep
        )
        assert backend.price(_view(shape)).seconds > 0
    graph, _ = graphs["decode B=16"]
    backend = OpGraphBackend(MI455, lambda *s: graph["region"], tp=TP, kv="bf16", ep=ep)
    near, far = (backend.price(_view("decode B=16", c)).seconds for c in (1024, 65536))
    assert far > near

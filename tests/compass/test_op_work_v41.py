# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's op kinds, priced from each leaf's own arguments and what it
states.

Each extractor matches a hand count on one shape; the sparse attention and the
indexer's scorer read their entries at the geometry their leaf states, over
the queries the leaf holds, and are refused by name where it states none;
every leaf of the V4.1-Flash TP4 captures, EP off and EP4, and gfx950's EP4
decode prices; and the op-level backend prices a V4.1-Flash decode step and a
prefill step on MI455 A0's spec, a decode step over a longer context higher.
"""

import json

import pytest
from test_capture_v41_flash import ARGS, COMPILED, EP4, SHAPES, TP, TREE, capture
from test_op_work import BF16, F32, I32, leaf, rows, t

from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.op_graph import OpGraphBackend
from atom.compass.backends.op_work import work
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.work import Work, gemm, price
from atom.compass.bench.ops import leaves
from atom.compass.capture.forward import V41_ATTENTION, V41_SCORERS
from atom.compass.spec import MachineSpec

FP8, E8M0, I64 = "torch.float8_e4m3fn", "torch.float8_e8m0fnu", "torch.int64"
MI455 = MachineSpec.from_mapping(
    json.loads((TREE / "scripts/compass/machines/mi455-a0.json").read_text())
)
#: Window 4, every 2 tokens compressed, the top 3 read.
GEOMETRY = {"window_size": 4, "compress_ratio": 2, "index_topk": 3}


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
    ],
    ids=["attention", "scorer"],
)
def test_a_leaf_that_states_no_geometry_is_refused_by_name(op, missing):
    with pytest.raises(CostRefused, match=f"{op['op']}: the leaf states no {missing}"):
        work(op, rows((1, 300), (1, 5)), TP, "bf16")


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


CAPTURES = [
    (SHAPES + ARGS, "gfx1250", 1),
    (SHAPES + ARGS + EP4, "gfx1250", TP),
    ("--width 16" + ARGS + EP4, "gfx950", TP),
]


@pytest.mark.parametrize(
    "argv, arch, ep", CAPTURES, ids=["gfx1250_ep_off", "gfx1250_ep4", "gfx950_ep4"]
)
def test_every_v41_flash_leaf_prices(argv, arch, ep, tmp_path_factory):
    graphs = capture(tmp_path_factory, argv, arch)
    kinds = set()
    for shape, (graph, _) in graphs.items():
        view = _view(shape)
        for op, _ in leaves(graph["region"]).values():
            kinds.add(op["op"])
            w = work(op, view, TP, "bf16", ep)
            assert w is None or price(w, MI455) > 0

    assert V41_ATTENTION in kinds
    assert COMPILED in kinds if arch == "gfx950" else V41_SCORERS[0] in kinds

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

# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's sparse attention over its FP8 pool under `compass measure
ops`: its operator states its layer's geometry under capture and takes the
lists and row pointers its index build writes for its pass, as the BF16
pool's does, and a prefill's own rows as one buffer, as ATOM passes them.
"""

import dataclasses
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from test_bench_ops import _graph, _leaf, _Runner, _t
from test_capture_ir import _decoder
from test_capture_v41_flash import LAYERS, capture, geometry

from atom.compass.bench import ops
from atom.compass.capture.forward import decode
from atom.utils import forward_context as fctx

#: The FP8 pool's sparse attention.
FP8 = "aiter.mla_sparse_prefill_v41.default"
I32 = torch.int32
#: The tables V4.1's cache fills each ratio's row pointers into.
TABLES = "attn_metadata_builder.cache.indptr_buffers"


def test_the_fp8_pool_s_attention_states_each_layer_s_geometry(tmp_path_factory):
    args = "--width 16 -- --kv_cache_dtype fp8 --index-cache-dtype fp8"
    graph, _ = capture(tmp_path_factory, args)["decode B=16"]

    layers = _decoder(graph)
    assert len(layers) == LAYERS
    for index, layer in enumerate(layers):
        attention = [op for op in layer if op["op"] == FP8]
        assert [op.get("stated") for op in attention] == [geometry(index)]


def test_the_fp8_pool_s_attention_takes_its_pass_s_lists_and_reads_the_rows_they_name():
    # V4.1's attention at ratio 2, top-k 8, its first layer there layer 2,
    # over a prefill's 4 rows of one request from position 1199, seeing 600
    # and 601 rows. ATOM's index build returns lists `100 k` on, its row
    # pointers the cache's tables at ratio 2, as the BF16 pool's attention
    # takes them; the bench spreads each row's 8 over its own visible rows, 12
    # of the pool's rows of 16 bytes
    from atom.model_ops.attentions.deepseek_v41.metadata import BatchStep, RequestSpan
    from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec

    runner = _Runner()
    runner.forward_vars |= {
        f"{TABLES}.{r}.{i}": torch.zeros(5, dtype=I32) for r in (2, 4) for i in (0, 1)
    }
    runner.config.kv_cache_block_size = 32
    runner.model.append(torch.nn.Module())
    runner.model[-1].spec = LayerAttentionSpec(2, 2, AttentionMode.FULL, 2, 2)
    visible = {2: torch.tensor([600, 600, 601, 601], dtype=I32)}
    spans = (RequestSpan(0, 1199, 0, 4, 0),)
    step = BatchStep(spans, *(torch.zeros(4, dtype=I32),) * 4, None, visible=visible)
    step.decoder = SimpleNamespace(
        tail=step, tail_rows=torch.zeros(3), last=torch.zeros(1), filled_gpu=None,
        finishing_gpu=None, query_ids=torch.tensor([-1, -1, -1, 0], dtype=I32),
        query_ptrs=torch.tensor([3, 0], dtype=I32),
    )  # fmt: skip
    lists = [torch.arange(4, dtype=I32) + 100 * k for k in (1, 2, 3, 4)]
    cache = SimpleNamespace(attention_indices=lambda *_: lists)
    cache.geometry = SimpleNamespace(batch_topk=lambda ratio: 8)
    prepare = runner.prepare_model

    def prepare_model(batch):
        out = prepare(batch)
        vars(fctx._forward_context.attn_metadata).update(step=step, cache=cache)
        return out

    indices = [(f"kv_indices_{s}", [4], "activation") for s in ("prefix", "extend")]
    pointers = [(f"kv_indptr_{s}", [5], "metadata") for s in ("prefix", "extend")]
    args = [(s, _t(n, role, dtype="torch.int32")) for s, n, role in indices + pointers]
    q, kv = _t([4, 2, 4], "activation"), _t([999, 4], "kv")
    leaf = _leaf(FP8, ("q", q), ("unified_kv", kv), *args)
    top = ("window_size", 128), ("compress_ratio", 2), ("index_topk", 8)
    leaf = dataclasses.replace(leaf, stated=top)
    runner.prepare_model, stub = prepare_model, ops.Stub(runner, 10_000)
    stub.leaves = ops.leaves(_graph(leaf, layers=1)["region"])
    stub._prepare((0, 0))

    [(sig, (op, instances))] = stub.leaves.items()
    [(_, bound)] = stub.placed
    assert {p: name for (s, p), (name, _) in bound.items() if s == sig} == {
        "kv_indices_prefix": "attention_indices.2.0",
        "kv_indices_extend": "attention_indices.2.2",
        "kv_indptr_prefix": f"{TABLES}.2.0",
        "kv_indptr_extend": f"{TABLES}.2.1",
    }
    assert bound[sig, "kv_indices_prefix"][1].tolist() == [100, 101, 102, 103]
    assert bound[sig, "kv_indices_extend"][1].tolist() == [300, 301, 302, 303]
    assert stub._plan(op, instances, {}, 99, 1, None, sig)[3][0] == 12 * 16


@pytest.mark.parametrize("stride", [528, 544])
def test_a_prefill_s_values_and_scales_filling_their_rows_are_one_buffer_a_copy(
    monkeypatch, stride
):
    # The attention's extend source, 4 rows of 512 E4M3 values and 16 E8M0
    # scales at one row stride, beside a query of 4 * 8 floats; 2 copies pass
    # twice a 2000-byte cache. At 528 they fill each row, as ATOM's
    # `rope_quant_window` passes them: a copy builds one buffer, its scales
    # 512 bytes past its values, and a call counts its rows once. At 544 they
    # do not, and stay two activations.
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    e4m3, e8m0 = torch.float8_e4m3fn, torch.float8_e8m0fnu
    kv = _t([4, 512], "activation", [stride, 1], str(e4m3))
    scale = _t([4, 16], "activation", [stride, 1], str(e8m0))
    q = _t([4, 8], "activation")
    leaf = _leaf(FP8, ("q", q), ("kv", kv), ("kv_scale", scale))
    found = ops.leaves(_graph(leaf)["region"])
    placed = SimpleNamespace(block_size=16, batch=lambda view, b, s: (b, s))
    graphs = []

    def replay(fn, calls, flush=None):
        graphs.append(calls)
        return [1.0] * ops.REPEATS

    with (
        patch.object(ops, "_function", return_value=lambda **kwargs: None),
        patch.object(ops, "_kernels", return_value=(["k"], [None], 1.0)),
        patch.object(ops, "_replay", side_effect=replay),
    ):
        [record] = ops.Stub(_Runner(), 2000).point(placed, decode(2), 20, found)

    copies = [(kw["kv"], kw["kv_scale"]) for kw, _ in graphs[0][:2]]
    assert copies[0][0].data_ptr() != copies[1][0].data_ptr()
    joined = stride == 528
    for values, scales in copies:
        assert (values.dtype, scales.dtype) == (e4m3, e8m0)
        assert values.stride() == scales.stride() == (stride, 1)
        stores = values.untyped_storage(), scales.untyped_storage()
        held = {(s.data_ptr(), s.nbytes()) for s in stores}
        if joined:  # one buffer, its values from its start, its scales 512 on
            assert held == {(values.data_ptr(), 4 * 528)}
            assert scales.data_ptr() - values.data_ptr() == 512
        else:
            assert len(held) == 2
    extend = 4 * 528 if joined else 2 * 3 * stride + 512 + 16
    assert record["footprint_bytes"] == 2 * (4 * 8 * 4 + extend)

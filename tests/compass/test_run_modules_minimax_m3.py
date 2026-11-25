# SPDX-License-Identifier: MIT
"""MiniMax-M3's meta build: its dense attention layers are priced as dense
attention, and its sparse ones as attention over the blocks each query
selects, with their indexer's scoring, so the step roofline prices an M3 step;
the sparse attention is checked against a count by hand, once memory-bound
and once compute-bound. Its MoE layers are priced as their routed experts with
the shared one fused beside them, though its MoE block holds no TP width.
Above TP1 each sub-layer's output is all-reduced once, inside the next norm.

The build owns its process because the driverless declarations are
process-global.
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import pytest
from test_backend_step_roofline_mla import BUILD, MI355X, TREE, _build
from test_backend_step_roofline_v4 import BOUND, _module, _price, view

from atom.compass.backends.step_roofline import (
    Module,
    StepRooflineBackend,
    _block_keys,
)
from atom.compass.backends.work import Work
from atom.compass.spec import MachineSpec

#: The published MiniMax-M3 text config at its widths, cut to its first dense
#: and first sparse attention layer, both with a dense MLP. Its text config
#: names no type and loads as Mixtral's.
M3 = {
    "architectures": ["MiniMaxM3SparseForCausalLM"],
    "model_type": "mixtral",
    "hidden_size": 6144,
    "dense_intermediate_size": 12288,
    "intermediate_size": 3072,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "num_hidden_layers": 2,
    "moe_layer_freq": [0, 0],
    "num_local_experts": 128,
    "num_experts_per_tok": 4,
    "hidden_act": "swigluoai",
    "swiglu_alpha": 1.702,
    "swiglu_limit": 7.0,
    "partial_rotary_factor": 0.5,
    "rope_theta": 5000000,
    "rms_norm_eps": 1e-6,
    "vocab_size": 200064,
    "max_position_embeddings": 1048576,
    "sparse_attention_config": {
        "sparse_index_dim": 128,
        "sparse_num_index_heads": 4,
        "sparse_topk_blocks": 16,
        "sparse_block_size": 128,
        "sparse_score_type": "max",
        "sparse_local_block": 1,
        "sparse_attention_freq": [0, 1],
    },
    "torch_dtype": "bfloat16",
}


def _attended(p):
    """The keys a query at position `p` attends over the top 2 blocks of 4
    tokens: one whole block and its own, which holds `(p - 1) % 4 + 1`."""
    return min(p, 4 + (p - 1) % 4 + 1)


#: 4 heads over 2 KV heads of 8, the top 2 blocks of 4 tokens, 2 index heads
#: of 4.
DIMS = (4, 2, 8, 2, 4, 2, 4)
PREFILL = [(3, 10, False), (5, 5, False)]
DECODE = [(1, 11, True), (2, 6, True)]


def _attention(rows):
    """The attention over `rows`' selected blocks, by hand: two products of
    8 a head and key in bf16; Q read and O written in bf16, and K and V of
    the blocks each request's last query selects at each KV head, in fp8."""
    new = [p for q, kv, _ in rows for p in range(kv - q + 1, kv + 1)]
    keys, read = sum(map(_attended, new)), sum(_attended(kv) for _, kv, _ in rows)
    tokens = sum(q for q, _, _ in rows)
    return Work({"bf16": 4 * 4 * 8 * keys}, tokens * 4 * 8 * 4 + read * 2 * 2 * 8)


def test_a_querys_keys_are_its_selected_blocks():
    attended = [_attended(p) for p in range(1, 40)]
    assert attended[7:12] == [8, 5, 6, 7, 8]  # one block, then its own
    assert list(_block_keys(np.arange(40), 2, 4)) == list(np.cumsum([0] + attended))


@BOUND
def test_block_sparse_attention_and_its_indexer_by_hand(spec):
    attn = Module("attn", "block_sparse_attention", DIMS, "fp8")
    # Prefill: keys 8 + 5 + 6 and 1 + 2 + 3 + 4 + 5, its requests' last
    # queries reading 6 and 5; 42 query-key pairs scored. Decode: keys 7 and
    # 5 + 6, reading 7 and 6; 22 pairs.
    for rows, keys, read, pairs, rate in [
        (PREFILL, 34, 11, 42, "fp8"),
        (DECODE, 18, 13, 22, "bf16"),
    ]:
        tokens, context = sum(r[0] for r in rows), sum(r[1] for r in rows)
        work = Work({"bf16": 4 * 4 * 8 * keys}, tokens * 4 * 8 * 4 + read * 32)
        assert work == _attention(rows)
        # The prefill scorer casts the bf16 index query to the fp8 cache's
        # format, the decode scorer lifts the fp8 keys to bf16: 2 products of
        # 4 a head and pair; the query and each request's index keys read once.
        work += Work({rate: 2 * 2 * 4 * pairs}, tokens * 2 * 4 * 2 + context * 4)
        assert _price([attn], view(*rows), spec)[
            "roofline.block_sparse_attention"
        ] == pytest.approx(_module(work, spec))


@BOUND
def test_a_reused_selection_scores_nothing_and_an_fp4_index_scores_at_fp8(spec):
    reuses = Module("attn", "block_sparse_attention", DIMS[:5] + (0, 4), "fp8")
    fp4 = Module("attn", "block_sparse_attention", DIMS, "fp8", index="fp4")
    for rows, pairs in [(PREFILL, 42), (DECODE, 22)]:
        tokens, context = sum(r[0] for r in rows), sum(r[1] for r in rows)
        assert _price([reuses], view(*rows), spec)[
            "roofline.block_sparse_attention"
        ] == pytest.approx(_module(_attention(rows), spec))
        # An e4m3 index query against fp4 keys in either phase.
        index = Work({"fp8": 2 * 2 * 4 * pairs}, tokens * 2 * 4 + context * 4 / 2)
        assert _price([fp4], view(*rows), spec)[
            "roofline.block_sparse_attention"
        ] == pytest.approx(_module(_attention(rows) + index, spec))


def test_a_sparse_attention_layer_prices_its_indexer_and_selected_blocks(
    monkeypatch,
):
    # A bare config leaves the CUDA graph mode unset, which M3's fused decode
    # path reads at build; it adds no module the walk sees.
    monkeypatch.setenv("ATOM_MONO_ENABLE", "0")
    walked = _build("tp4", M3)
    machine = MachineSpec.from_mapping(json.loads(MI355X.read_text()))

    assert not [m.name for m in walked if m.kind == "unclassified"]
    assert [(m.layer, m.dims) for m in walked if m.kind == "attention"] == [
        (0, (16, 1, 128))
    ]
    # A rank's 16 heads over its one KV head, the top 16 blocks of 128 tokens,
    # its one index head of 128, over the fp8 KV cache.
    sparse = [m for m in walked if m.kind == "block_sparse_attention"]
    assert [(m.layer, m.dims, m.fmt) for m in sparse] == [
        (1, (16, 1, 128, 16, 128, 1, 128), "fp8")
    ]
    for step in view((1, 4096, True)), view((2048, 2048, False)):
        terms = StepRooflineBackend(machine, walked).price(step).terms
        assert {"roofline.attention", "roofline.block_sparse_attention"} <= {
            t.name for t in terms
        }


def test_a_layer_reusing_the_selection_runs_no_indexer(monkeypatch):
    monkeypatch.setenv("ATOM_MONO_ENABLE", "0")
    # Both layers sparse, the second reusing the first's selection.
    sparse = {**M3["sparse_attention_config"], "sparse_attention_freq": [1, 1]}
    config = {**M3, "sparse_attention_config": sparse}
    config.update(use_index_cache=True, index_topk_freq=2)
    walked = _build("tp4", config)

    assert [
        (m.layer, m.dims[5]) for m in walked if m.kind == "block_sparse_attention"
    ] == [(0, 1), (1, 0)]
    index_norms = [m.name for m in walked if ".index_" in m.name]
    assert index_norms == [
        "model.layers.0.self_attn.index_q_norm",
        "model.layers.0.self_attn.index_k_norm",
    ]


def _walk_index(index: str) -> list:
    """`M3` walked as `_build` walks it at TP4, over an fp8 KV cache and an
    `index` index cache in pages of 128 tokens, as M3's recipe runs: the
    layer, dims and index format of each sparse attention module."""
    cache = f'index_cache_dtype="{index}", kv_cache_block_size=128,'
    script = BUILD.replace('kv_cache_dtype="fp8",', 'kv_cache_dtype="fp8", ' + cache)
    assert script != BUILD
    script += (
        "print(json.dumps([[m.layer, list(m.dims), m.index] for m in walk(tree)"
        " if m.kind == 'block_sparse_attention']))\n"
    )
    with tempfile.TemporaryDirectory() as root:
        (pathlib.Path(root) / "model").mkdir()
        (pathlib.Path(root) / "model" / "config.json").write_text(json.dumps(M3))
        done = subprocess.run(
            [sys.executable, "-c", script, "tp4", root],
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
            cwd=TREE,
            env={**os.environ, "PYTHONPATH": str(TREE)},
        )
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.splitlines()[-1])


def test_the_indexer_scores_at_its_own_caches_format(monkeypatch):
    monkeypatch.setenv("ATOM_MONO_ENABLE", "0")
    # An fp4 index cache beside the fp8 KV cache.
    assert _walk_index("fp4") == [[1, [16, 1, 128, 16, 128, 1, 128], "fp4"]]


def test_an_indexer_sharding_the_context_is_unclassified(monkeypatch):
    monkeypatch.setenv("ATOM_MONO_ENABLE", "0")
    # Indexer context parallelism, valid at TP4 over M3's 4 KV heads: each
    # rank scores every index head over a quarter of the blocks.
    monkeypatch.setenv("ATOM_M3_INDEXER_CP", "1")
    walked = _build("tp4", M3)

    assert [m.name for m in walked if m.kind == "unclassified"] == [
        "Attention at model.layers.1.self_attn.attn"
    ]


#: The published MXFP4 checkpoint's scheme: fp4 weights and activations, an
#: e8m0 scale for each 32.
MXFP4 = {
    "quant_method": "quark",
    "global_quant_config": {
        "weight": {"dtype": "fp4", "qscheme": "per_group", "group_size": 32},
        "input_tensors": {"dtype": "fp4", "qscheme": "per_group", "is_dynamic": True},
    },
}


@pytest.mark.parametrize("quant", ["bf16", "mxfp4"])
def test_a_moe_layer_prices_its_routed_and_fused_shared_experts(quant, monkeypatch):
    monkeypatch.setenv("ATOM_MONO_ENABLE", "0")
    # Its published MoE fields, the second layer an MoE one, both attending
    # densely.
    sparse = {**M3["sparse_attention_config"], "sparse_attention_freq": [0, 0]}
    config = {**M3, "moe_layer_freq": [0, 1], "sparse_attention_config": sparse}
    config.update(n_shared_experts=1, use_routing_bias=True, scoring_func="sigmoid")
    config.update(routed_scaling_factor=2.0)
    if quant == "mxfp4":
        config["quantization_config"] = MXFP4
    walked = _build("tp4", config)

    # Each expert's gate-up and down weights at 3072 / 4 = 768 a rank: two
    # bytes each in bf16, or half a byte and an e8m0 scale byte per 32 in fp4.
    weights = 3 * 768 * 6144
    held = 2 * weights if quant == "bf16" else weights // 2 + weights // 32
    fmt = "bf16" if quant == "bf16" else "fp4"
    # The 128 routed experts, 4 picked a row, and the shared one beside them,
    # at TP4 on one data-parallel rank, their output left to the next norm.
    dims = (128, 128, 4, 768, 6144, 4, 1, 0, held, 1)
    assert not [m for m in walked if m.kind == "unclassified"]
    assert not [m for m in walked if "shared_experts" in m.name]
    moe = [(m.layer, m.dims, m.fmt, m.act, m.ranks) for m in walked if m.kind == "moe"]
    assert moe == [(1, dims, fmt, fmt, 1)]


@pytest.mark.parametrize("tp", [1, 4])
def test_each_sub_layer_is_all_reduced_once_in_the_next_norm(tp, monkeypatch):
    monkeypatch.setenv("ATOM_MONO_ENABLE", "0")
    # A dense layer, then an MoE one, both attending densely.
    sparse = {**M3["sparse_attention_config"], "sparse_attention_freq": [0, 0]}
    config = {**M3, "moe_layer_freq": [0, 1], "sparse_attention_config": sparse}
    walked = _build(f"tp{tp}", config)

    # Beside the embedding's all-reduce and the LM head's all-gather, each
    # layer's attention output is reduced in its post-attention norm, and its
    # MLP's or MoE's in the next layer's input norm or the final norm; the
    # first layer's input norm reads the embedding, already reduced.
    norms = ("layers.0.post_attention_layernorm", "layers.1.input_layernorm")
    norms += ("layers.1.post_attention_layernorm", "norm")
    hand = [("model.embed_tokens", "embedding", 4, False)]
    hand += [(f"model.{n}", "norm", 4, True) for n in norms]
    hand += [("lm_head", "lm_head", 4, False)]
    found = [
        (m.name, m.kind, m.ranks, m.fused) for m in walked if m.ranks > 1 or m.fused
    ]
    assert found == (hand if tp > 1 else [])
    assert [m.ranks for m in walked if m.kind == "moe"] == [1]

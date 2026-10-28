# SPDX-License-Identifier: MIT
"""MiniMax-M3's meta build: its dense attention layers are priced as dense
attention, and the step roofline refuses its sparse ones by layer name, whose
top-k kernel reads a few blocks of the cache rather than every token. Its MoE
layers are priced as their routed experts with the shared one fused beside
them, though its MoE block holds no TP width. Above TP1 each sub-layer's
output is all-reduced once, inside the next norm.

The build owns its process because the driverless declarations are
process-global.
"""

import json

import pytest
from test_backend_step_roofline_mla import MI355X, _build
from test_backend_step_roofline_v4 import view

from atom.compass.backends import CostRefused
from atom.compass.backends.step_roofline import StepRooflineBackend
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


def test_a_sparse_attention_layer_refuses_every_step_naming_it(monkeypatch):
    # A bare config leaves the CUDA graph mode unset, which M3's fused decode
    # path reads at build; it adds no module the walk sees.
    monkeypatch.setenv("ATOM_MONO_ENABLE", "0")
    walked = _build("tp4", M3)
    machine = MachineSpec.from_mapping(json.loads(MI355X.read_text()))

    assert [(m.layer, m.dims) for m in walked if m.kind == "attention"] == [
        (0, (16, 1, 128))
    ]
    assert [m.name for m in walked if m.kind == "unclassified"] == [
        "Attention at model.layers.1.self_attn.attn"
    ]
    with pytest.raises(CostRefused, match=r"Attention at model\.layers\.1\.self_attn"):
        StepRooflineBackend(machine, walked).price(view((1, 64, True)))


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

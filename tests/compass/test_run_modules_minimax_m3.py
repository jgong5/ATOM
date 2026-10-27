# SPDX-License-Identifier: MIT
"""MiniMax-M3's meta build: its dense attention layers are priced as dense
attention, and the step roofline refuses its sparse ones by layer name, whose
top-k kernel reads a few blocks of the cache rather than every token.

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

# SPDX-License-Identifier: MIT
"""MXFP4 models' meta builds: each fp4 linear, and an fp4 MLA up-projection,
is walked at its unpacked input width, though its weight packs two of it to a
byte; each fp4 linear reads its input at fp4, which ATOM quantizes it to before
the product; and one fp4 linear's price is checked against a count by hand.

The build owns its process because the driverless declarations are
process-global.
"""

import functools
import json

import pytest
from test_backend_step_roofline_mla import CONFIG as V3
from test_backend_step_roofline_mla import _build
from test_backend_step_roofline_v4 import BOUND, _module, _price, view

from atom.compass.backends.work import Work

#: A dense model whose linears are all MXFP4, as Quark exports them.
MXFP4 = {"dtype": "fp4", "qscheme": "per_group", "group_size": 32, "ch_axis": -1}
FP4_DENSE = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "num_hidden_layers": 2,
    "vocab_size": 1024,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000,
    "torch_dtype": "bfloat16",
    "quantization_config": {
        "quant_method": "quark",
        "global_quant_config": {
            "weight": {**MXFP4, "is_dynamic": False},
            "input_tensors": {**MXFP4, "is_dynamic": True},
        },
        "exclude": ["lm_head"],
    },
}


@functools.cache
def _fp4(tp: int) -> tuple:
    return _build(f"tp{tp}", FP4_DENSE)


@pytest.mark.parametrize("tp", [1, 2])
def test_each_fp4_linear_is_walked_at_its_unpacked_input_width(tp):
    modules = _fp4(tp)
    gemms = {(m.name.split(".")[-1], m.dims, m.fmt) for m in modules if m.layer == 0}

    assert not [m for m in modules if m.kind == "unclassified"]
    # q, k and v of 4 + 2 + 2 heads of 64, and gate and up, from the hidden
    # 256; o from the 4 heads of 64 and down from the intermediate 512, each
    # split over the group.
    assert {g for g in gemms if g[2] == "fp4"} == {
        ("qkv_proj", (512 // tp, 256), "fp4"),
        ("o_proj", (256, 256 // tp), "fp4"),
        ("gate_up_proj", (1024 // tp, 256), "fp4"),
        ("down_proj", (256, 512 // tp), "fp4"),
    }
    assert [m.dims for m in modules if m.kind == "act"] == [(512 // tp,)] * 2


@pytest.mark.parametrize("tp", [1, 2])
def test_each_fp4_linear_reads_its_input_at_fp4(tp):
    gemms = [m for m in _fp4(tp) if m.kind == "gemm" and m.fmt == "fp4"]

    assert len(gemms) == 4 * 2
    assert {m.act for m in gemms} == {"fp4"}


def test_an_fp4_up_projection_is_walked_from_its_whole_latent():
    v3 = json.loads(V3.read_text())
    v3.update(num_hidden_layers=1, first_k_dense_replace=1)
    v3["quantization_config"] = FP4_DENSE["quantization_config"]
    (up,) = [m for m in _build("tp8", v3) if m.kind == "mla_up"]

    # 16 heads at TP8, no-rope and value widths of 128, from the 512 latent.
    assert (up.dims, up.fmt) == ((16, 128, 128, 512), "fp4")


@BOUND
def test_one_fp4_linear_by_hand(spec):
    (down,) = [m for m in _fp4(1) if m.name == "model.layers.0.mlp.down_proj"]
    # 8 prefill tokens from 512 to 256: 2 flops a multiply-add at the fp4
    # rate, the fp4 input and weight at half a byte an element, and the bf16
    # output.
    flops = 2 * 8 * 256 * 512
    moved = 8 * 512 / 2 + 512 * 256 / 2 + 8 * 256 * 2
    by_name = _price([down], view((8, 8, False)), spec)

    assert (flops, moved) == (2_097_152, 71_680)
    assert by_name["roofline.gemm"] == pytest.approx(
        _module(Work({"fp4": flops}, moved), spec)
    )

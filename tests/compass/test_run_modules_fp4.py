# SPDX-License-Identifier: MIT
"""MXFP4 models' meta builds: each fp4 linear, and an fp4 MLA up-projection,
is walked at its unpacked input width, though its weight packs two of it to a
byte; each fp4 linear reads its input at fp4, which ATOM quantizes it to before
the product; the up-projection at the cache format and prefill chunk size
ATOM's config states; and one fp4 linear's price, and an fp4 up-projection's
on a prefill cached and not, are checked against a count by hand.

The build owns its process because the driverless declarations are
process-global.
"""

import dataclasses
import functools
import json
import os
import subprocess
import sys

import pytest
from test_backend_step_roofline_mla import BUILD, TREE, _build
from test_backend_step_roofline_mla import CONFIG as V3
from test_backend_step_roofline_v4 import BOUND, _module, _price, view

from atom.compass.backends.step_roofline import Module
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

    # 16 heads at TP8, no-rope and value widths of 128, from the 512 latent,
    # which its prefill product reads at fp4.
    assert (up.dims, up.fmt, up.act) == ((16, 128, 128, 512), "fp4", "fp4")


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


#: The build at the KV cache format and prefill chunk size its third and
#: fourth arguments name, each module's cache format and chunk in place of its
#: fused flag and layer.
CACHE_BUILD = BUILD.replace(
    'kv_cache_dtype="fp8", **dpa)',
    "kv_cache_dtype=sys.argv[3], attn_prefill_chunk_size=int(sys.argv[4]), **dpa)",
).replace("m.fused, m.layer]", "m.kv, m.chunk]")


def test_an_fp4_up_projection_gathers_at_the_cache_and_chunk_atom_states(tmp_path):
    v3 = json.loads(V3.read_text())
    v3.update(num_hidden_layers=1, first_k_dense_replace=1)
    v3["quantization_config"] = FP4_DENSE["quantization_config"]
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text(json.dumps(v3))
    done = subprocess.run(
        [sys.executable, "-c", CACHE_BUILD, "tp8", str(tmp_path), "bf16", "4096"],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )
    assert done.returncode == 0, done.stderr[-4000:]
    walked = json.loads(done.stdout.splitlines()[-1])

    assert [m[1:] for m in walked if m[1] == "mla_up"] == [
        ["mla_up", [16, 128, 128, 512], "fp4", "fp4", 1, "bf16", 4096]
    ]


#: 16 heads at TP8, no-rope and value widths of 128, from the 512 latent: an
#: fp4 weight of 4096 by 512 at half a byte, over an fp8 cache. Each row it
#: gathers reads its latent at a byte, each the linear runs on at half one,
#: and each writes 4096 bf16 outputs.
UP = Module("up", "mla_up", (16, 128, 128, 512), "fp4", act="fp4", kv="fp8")
WEIGHT, GATHERED, LINEAR = 4096 * 512 // 2, 512 + 4096 * 2, 512 // 2 + 4096 * 2


@BOUND
@pytest.mark.parametrize(
    "context, chunk, flops, moved",
    [
        # No cached tokens: the linear over the 1024 new rows, at fp4.
        (1024, 16384, {"fp4": 2 * 1024 * 4096 * 512}, 1024 * LINEAR + WEIGHT),
        # 2048 cached, within the chunk: the whole context of 3072 gathered,
        # the product bf16 by fp4.
        (3072, 16384, {"bf16": 2 * 3072 * 4096 * 512}, 3072 * GATHERED + WEIGHT),
        # Past a chunk of 1024: the 2048 cached rows gathered, the 1024 new
        # ones through the linear, each reading the weight.
        (
            3072,
            1024,
            {"bf16": 2 * 2048 * 4096 * 512, "fp4": 2 * 1024 * 4096 * 512},
            2048 * GATHERED + 1024 * LINEAR + 2 * WEIGHT,
        ),
    ],
    ids=["uncached", "cached", "chunked"],
)
def test_an_fp4_up_projection_prefill_by_hand(spec, context, chunk, flops, moved):
    up = dataclasses.replace(UP, chunk=chunk)
    by_name = _price([up], view((1024, context, False)), spec)

    assert by_name["roofline.mla_up"] == pytest.approx(
        _module(Work(flops, moved), spec)
    )

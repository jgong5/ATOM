# SPDX-License-Identifier: MIT
"""The step roofline over MLA: the absorbed decode, the prefill over each
request's up-projected context, and the routed experts beside a fused shared
expert; DeepSeek-V3's meta build at TP8 and under DP attention; and an
attention layer with a sliding window, refused by name.

Each price is checked against a count by hand, on a machine with no kernel
floor, once memory-bound and once compute-bound. The builds own their
processes because the driverless declarations are process-global.
"""

import functools
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import pytest
import torch
from test_backend_step_roofline_v4 import BOUND, EP_OFF, LINK, _module, _price, view

from atom.compass.backends import BatchView, CostRefused, RequestShape
from atom.compass.backends.step_roofline import Module, StepRooflineBackend
from atom.compass.backends.work import Work, activated_experts, gemm
from atom.compass.runner.overrides import FactoriesOnMeta
from atom.compass.spec import MachineSpec

TREE = pathlib.Path(__file__).resolve().parents[2]
CONFIG = pathlib.Path(__file__).with_name("deepseek_v3_config.json")
MI355X = TREE / "scripts" / "compass" / "machines" / "mi355x-gpu45.json"
#: Heads, the decode kernel's padded heads, latent, rope, no-rope and value.
MLA = (2, 4, 8, 4, 6, 5)


def _causal_pairs(rows):
    """Each query attends every key up to its own position."""
    return sum(p for q, kv, _ in rows for p in range(kv - q + 1, kv + 1))


@BOUND
def test_an_absorbed_decode_step_by_hand(spec):
    attn = Module("attn", "mla_attention", MLA, "fp8")
    up = Module("up", "mla_up", (2, 6, 5, 8), "fp8", act="fp8")
    rows = [(1, 10, True), (2, 7, True)]
    step = BatchView(tuple(RequestShape(*r) for r in rows), capture_rung=4)
    # Four kernel heads score each pair at 8 + 4 and sum values at 8: Q of 3
    # tokens and the 10 + 7 entries at a byte, the latent output in bf16.
    pairs = _causal_pairs(rows)
    moved = 3 * 4 * 12 + 17 * 12 + 3 * 4 * 8 * 2
    kernel = Work({"fp8": pairs * 4 * (2 * 12 + 2 * 8)}, moved)
    # The graph's 4 rows of 2 tokens: 7 rows, one 6-to-8 and one 8-to-5 GEMM
    # a head.
    head = gemm(7, 8, 6, "fp8", "fp8") + gemm(7, 5, 8, "fp8", "fp8")
    by_name = _price([attn, up], step, spec)

    assert pairs == 23
    assert by_name["roofline.mla_attention"] == pytest.approx(_module(kernel, spec))
    assert by_name["roofline.mla_up"] == pytest.approx(_module(head + head, spec))


@BOUND
def test_a_prefill_step_up_projects_each_context_by_hand(spec):
    attn = Module("attn", "mla_attention", MLA, "fp8")
    up = Module("up", "mla_up", (2, 6, 5, 8), "fp8", act="fp8")
    rows = [(4, 4, False), (3, 9, False)]
    # Two heads, QK at 6 + 4 and PV at 5, in bf16: Q and O of 7 tokens, K and
    # V of the 4 + 9 context tokens, cached ones included.
    pairs = _causal_pairs(rows)
    moved = (7 * 2 * (10 + 5) + 13 * 2 * (10 + 5)) * 2
    kernel = Work({"bf16": pairs * 2 * (2 * 10 + 2 * 5)}, moved)
    by_name = _price([attn, up], view(*rows), spec)

    assert pairs == 34
    assert by_name["roofline.mla_attention"] == pytest.approx(_module(kernel, spec))
    assert by_name["roofline.mla_up"] == pytest.approx(
        _module(gemm(13, 22, 8, "fp8", "fp8"), spec)
    )


@BOUND
def test_a_fused_shared_expert_takes_every_row_by_hand(spec):
    group = Module("experts", "moe", EP_OFF + (1,), "fp4", ranks=2)
    deferred = Module("experts", "moe", EP_OFF[:6] + (1, 0, 816, 1), "fp4")
    # The gathered 10 rows each reach 2 routed experts and the shared one,
    # with three top-k ids and weights a row.
    hit = activated_experts(8, 2, 10) + 1
    experts = Work({"bf16": 2 * 30 * (32 + 16) * 32}, hit * 816 + 10 * (128 + 24))
    gather = 2e-6 + 10 * (32 + 8) * 2 / 2 / LINK
    scatter = 2e-6 + 10 * 32 * 2 / 2 / LINK
    step = view((3, 3, False), moe_rows=10)
    assert _price([group], step, spec)["roofline.moe"] == pytest.approx(
        _module(experts, spec) + gather + scatter
    )
    # One data-parallel rank whose block leaves the reduce to the next norm:
    # its graph's 8 rows, 3 decode tokens and the pad reaching 2 routed
    # experts each, every row the shared one, and no all-reduce.
    hit = activated_experts(8, 2, 4) + 1
    experts = Work({"bf16": 2 * 24 * (32 + 16) * 32}, hit * 816 + 8 * (128 + 24))
    one = BatchView((RequestShape(1, 4, True),) * 3, capture_rung=8)
    assert _price([deferred], one, spec)["roofline.moe"] == pytest.approx(
        _module(experts, spec)
    )


def test_a_meta_tensor_moved_to_the_gpu_stays_on_meta_and_not_to_the_host():
    with FactoriesOnMeta():
        scale = torch.tensor(1.0)
        moved = [scale.to(0), scale.to("cuda", non_blocking=True)]
        moved += [scale.to(device=torch.device("cuda", 0), dtype=torch.float16)]
        with pytest.raises(NotImplementedError, match="meta tensor"):
            scale.to("cpu")

    assert [(t.device.type, t.dtype) for t in moved] == [
        ("meta", torch.float32),
        ("meta", torch.float32),
        ("meta", torch.float16),
    ]


BUILD = """
import json, pathlib, sys, torch
mode, root = sys.argv[1], pathlib.Path(sys.argv[2])
from atom.compass.capture import driverless
driverless.install(str(root), "gfx950")
tp = int(mode[2:]) if mode.startswith("tp") else 1
driverless.group(tp, 1)
from atom.config import Config, set_current_atom_config
dpa = {} if mode.startswith("tp") else dict(
    enable_dp_attention=True, dp_logical_size=8, enable_expert_parallel=mode == "ep8"
)
config = Config(model=str(root / "model"), tensor_parallel_size=tp, load_dummy="empty",
                kv_cache_dtype="fp8", **dpa)
set_current_atom_config(config)
from atom.model_engine.model_runner import support_model_arch_dict
from atom.utils import resolve_obj_by_qualname
from atom.compass.runner.overrides import FactoriesOnMeta, process_weights_after_loading
from atom.compass.run.modules import walk
model = support_model_arch_dict[config.hf_config.architectures[0]]
torch.set_default_dtype(config.torch_dtype)
with FactoriesOnMeta():
    tree = resolve_obj_by_qualname(model)(config)
    process_weights_after_loading(tree)
walked = [[m.name, m.kind, list(m.dims), m.fmt, m.act, m.ranks, m.fused, m.layer]
          for m in walk(tree)]
print(json.dumps(walked))
"""


def _build(mode: str, config: dict) -> tuple:
    """`config`'s model walked in a process of its own at `mode`: `tp<N>`,
    or DP attention over 8 ranks, `dpa` or, with expert parallelism, `ep8`."""
    with tempfile.TemporaryDirectory() as root:
        (pathlib.Path(root) / "model").mkdir()
        (pathlib.Path(root) / "model" / "config.json").write_text(json.dumps(config))
        done = subprocess.run(
            [sys.executable, "-c", BUILD, mode, root],
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
            cwd=TREE,
            env={**os.environ, "PYTHONPATH": str(TREE)},
        )
    assert done.returncode == 0, done.stderr[-4000:]
    return tuple(
        Module(n, k, tuple(d), f, r, x, a, layer)
        for n, k, d, f, a, r, x, layer in json.loads(done.stdout.splitlines()[-1])
    )


@functools.cache
def _v3(mode: str) -> tuple:
    return _build(mode, json.loads(CONFIG.read_text()))


def _counts(modules):
    found = {}
    for m in modules:
        key = (m.kind, m.dims, m.fmt, m.act, m.ranks)
        found[key] = found.get(key, 0) + 1
    return found


@pytest.mark.parametrize("mode", ["tp8", "dpa"])
def test_the_v3_build_classifies_every_module(mode):
    found = _counts(_v3(mode))
    heads, dp = (16, 1) if mode == "tp8" else (128, 8)

    assert not [m for m in found if m[0] == "unclassified"]
    # Each layer's MLA at its rank's heads over the fp8 latent cache, and its
    # up-projection, the fp8 block-scaled kv_b_proj, priced as one.
    attn = ("mla_attention", (heads, heads, 512, 64, 128, 128), "fp8", "bf16", 1)
    assert found[attn] == 61
    assert found[("mla_up", (heads, 128, 128, 512), "fp8", "fp8", 1)] == 61
    assert not [m for m in found if m[0] == "gemm" and m[1][1] == 512]
    # The fp8 experts read fp8 activations, the shared expert fused beside the
    # 256 routed ones, an eighth of each expert's 2048 on a rank: its fp8
    # weights and an fp32 scale per 128 by 128 block. At TP8 the next norm
    # all-reduces their output.
    held = 3 * 256 * 7168 + 4 * 56 * (2 * 256 // 128 + 256 // 128)
    moe = (256, 256, 8, 256, 7168, 8, dp, 0, held, 1)
    assert found[("moe", moe, "fp8", "fp8", 8 if dp > 1 else 1)] == 58


def test_the_v3_ep8_build_refuses_its_fused_shared_expert_by_name():
    walked = _v3("ep8")

    assert [m.name for m in walked if m.kind == "unclassified"] == [
        "FusedMoE at model.layers.3.mlp.experts"
    ]
    assert sum(m.kind == "mla_attention" for m in walked) == 61


def test_the_v3_tp8_decode_step_at_batch_64_and_context_4096_by_hand():
    machine = MachineSpec.from_mapping(json.loads(MI355X.read_text()))
    step = BatchView((RequestShape(1, 4096, True),) * 64, capture_rung=64)
    cost = StepRooflineBackend(machine, _v3("tp8")).price(step)
    by_name = {t.name: t.seconds for t in cost.terms}
    # 16 heads, each of 64 queries scoring 4096 keys at 512 + 64 and summing
    # values at 512, at fp8; every request's 4096 entries of 576 bytes read
    # once, Q at 576 bytes a head, the output at 1024.
    attention = Work(
        {"fp8": 64 * 4096 * 16 * (2 * 576 + 2 * 512)},
        64 * 4096 * 576 + 64 * 16 * 576 + 64 * 16 * 1024,
    )
    # Per head, 64 rows from 128 to 512 and 512 to 128 against fp8 weights.
    head = gemm(64, 512, 128, "fp8", "fp8") + gemm(64, 128, 512, "fp8", "fp8")
    up = Work({"fp8": 16 * head.flops_by_format["fp8"]}, 16 * head.bytes)

    assert attention.flops_by_format["fp8"] == pytest.approx(9.127e9, rel=1e-3)
    assert attention.bytes == pytest.approx(152.6e6, rel=1e-3)
    assert by_name["roofline.mla_attention"] == pytest.approx(
        61 * _module(attention, machine)
    )
    assert by_name["roofline.mla_up"] == pytest.approx(61 * _module(up, machine))


#: A dense model whose second layer attends a sliding window.
WINDOWED = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "num_hidden_layers": 2,
    "layer_types": ["full_attention", "sliding_attention"],
    "sliding_window": 128,
    "vocab_size": 1024,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000,
    "torch_dtype": "bfloat16",
}


def test_a_windowed_attention_layer_refuses_every_step_naming_it():
    modules = _build("tp1", WINDOWED)
    attention = [m.layer for m in modules if m.kind == "attention"]

    assert attention == [0]
    with pytest.raises(CostRefused, match=r"Attention at model\.layers\.1\.self_attn"):
        StepRooflineBackend(
            MachineSpec.from_mapping(json.loads(MI355X.read_text())), modules
        ).price(view((1, 64, True)))

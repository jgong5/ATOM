# SPDX-License-Identifier: MIT
"""Kimi-K3's meta build: every module classified, its KDA layer priced as
ATOM runs it, and a step priced by the step roofline alone, the KDA
recurrence by hand.

The build owns its process because the driverless declarations are
process-global.
"""

import functools
import json
import pathlib

import pytest
import test_backend_step_roofline_mla as mla
from test_backend_step_roofline_v4 import _module

from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.step_roofline import StepRooflineBackend
from atom.compass.backends.work import Work
from atom.compass.spec import MachineSpec

#: The published Kimi-K3 config (`kimi_k3_config.json`).
CONFIG = json.loads(pathlib.Path(__file__).with_name("kimi_k3_config.json").read_text())
CONFIGURED = "set_current_atom_config(config)\n"
#: The vision tower's config, which ATOM reads through the checkpoint's own
#: configuration code, taken from config.json, as `test_pd_kda_slot.py` takes it.
VISION = """
from types import SimpleNamespace
raw = json.loads((root / "model" / "config.json").read_text())
config.multimodal_config = SimpleNamespace(
    vision_config=SimpleNamespace(**raw["vision_config"])
)
"""
ATTENTION = "language_model.model.layers.0.self_attn"


@functools.cache
def _k3() -> tuple:
    """Kimi-K3 cut to its first two layers, a KDA one then a full attention
    one, built at TP8 with its vision tower."""
    assert mla.BUILD.count(CONFIGURED) == 1
    built = mla.BUILD.replace(CONFIGURED, CONFIGURED + VISION)
    text = CONFIG["text_config"]
    kda = {**text["linear_attn_config"], "kda_layers": [1], "full_attn_layers": [2]}
    text = {**text, "num_hidden_layers": 2, "linear_attn_config": kda}
    saved, mla.BUILD = mla.BUILD, built
    try:
        return mla._build("tp8", {**CONFIG, "text_config": text})
    finally:
        mla.BUILD = saved


def test_every_module_is_classified_and_the_kda_layer_is_priced_as_atom_runs_it():
    walked = _k3()
    layer = {
        (m.name.removeprefix(ATTENTION), m.kind, m.dims, m.fmt, m.ranks)
        for m in walked
        if m.name.startswith(ATTENTION)
    }

    assert not [m.name for m in walked if m.kind == "unclassified"]
    # 12 of the 96 heads of 128 a rank, the state in fp32: no speculative
    # decoding, so no ReplaySSM. One convolution over q, k and v; one
    # in-projection of q, k, v, the output gate, beta and the forget gate's
    # first factor; the norm over each token's 12 heads.
    assert layer == {
        ("", "kda", (12, 128, 0), "fp32", 1),
        (".conv", "conv", (3 * 1536,), "bf16", 1),
        (".in_proj", "gemm", (4 * 1536 + 12 + 128, 7168), "bf16", 1),
        (".f_b_proj", "gemm", (1536, 128), "bf16", 1),
        (".o_norm", "gated_norm", (1536,), "bf16", 1),
        (".o_proj", "gemm", (7168, 1536), "bf16", 8),
    }
    # SiTU before the dense MLP's padded 4224 and the shared experts' 768.
    assert sorted(m.dims for m in walked if m.kind == "act") == [(768,), (4224,)]
    tower = ("vision_tower", "mm_projector")
    assert not [m for m in walked if m.name.startswith(tower)]


def test_a_decode_step_prices_the_kda_recurrence_by_hand():
    machine = MachineSpec.from_mapping(json.loads(mla.MI355X.read_text()))
    step = BatchView((RequestShape(1, 4096, True),) * 64, capture_rung=64)
    cost = StepRooflineBackend(machine, _k3()).price(step)
    by_name = {t.name: t.seconds for t in cost.terms}
    print(f"\n{by_name}")
    # 64 tokens read q, k, the gate and v at 12 x 128 and beta at 12, and
    # write o, in bf16; A_log and dt_bias once in fp32; each row reads its
    # 12 x 128 x 128 fp32 state and writes it back. Three products a token
    # and head.
    state = 12 * 128 * 128 * 4
    moved = 64 * 12 * (5 * 128 + 1) * 2 + 12 * 129 * 4 + 2 * 64 * state
    recurrence = Work({"bf16": 6 * 64 * 12 * 128 * 128}, moved)

    assert moved == 101_654_064
    assert by_name["roofline.kda"] == pytest.approx(_module(recurrence, machine))
    assert {"roofline.conv", "roofline.gated_norm", "roofline.act"} <= set(by_name)

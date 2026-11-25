# SPDX-License-Identifier: MIT
"""GPT-OSS-120B's meta build: the step roofline prices its windowed layers'
attention over their sliding window and its full attention layers' over the
whole context, and refuses no step.

Its config is `gpt_oss_120b_config.json`, openai/gpt-oss-120b's config.json
byte for byte at revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a. The build
owns its process because the driverless declarations are process-global.
"""

import json
import pathlib

import pytest
import test_backend_step_roofline_mla as mla
from test_backend_step_roofline_v4 import _module, view

from atom.compass.backends.step_roofline import StepRooflineBackend
from atom.compass.backends.work import Work, attention
from atom.compass.spec import MachineSpec

CONFIG = json.loads(
    pathlib.Path(__file__).with_name("gpt_oss_120b_config.json").read_text()
)


def test_a_windowed_layer_prices_its_attention_over_its_window_by_hand():
    # Cut to its first two layers: a windowed one, then a full attention one.
    types = CONFIG["layer_types"][:2]
    walked = mla._build("tp1", {**CONFIG, "num_hidden_layers": 2, "layer_types": types})
    machine = MachineSpec.from_mapping(json.loads(mla.MI355X.read_text()))
    attn = [(m.layer, m.dims, m.fmt) for m in walked if m.kind == "attention"]
    # A 2048-token chunk at context 4096 and four decodes there.
    step = view((2048, 4096, False), *[(1, 4096, True)] * 4)
    terms = {
        t.name: t.seconds
        for t in StepRooflineBackend(machine, walked).price(step).terms
    }
    # 64 heads over 8 KV heads of 64 on the fp8 cache; the window, 128 keys a
    # query, every chunk query past it; Q read and O written in bf16, the
    # chunk's whole context read once, each decode's last 128.
    pairs = (2048 + 4) * 128
    keys = 4096 + 4 * 128
    windowed = Work(
        {"bf16": 4 * 64 * 64 * pairs},
        2052 * 2 * 64 * 64 * 2 + keys * 2 * 8 * 64 * 1,
    )
    full = attention([2048, 1, 1, 1, 1], [4096] * 5, 64, 64, 8, kv_fmt="fp8")

    assert not [m.name for m in walked if m.kind == "unclassified"]
    assert attn == [(0, (64, 8, 64, 128, 1), "fp8"), (1, (64, 8, 64), "fp8")]
    assert terms["roofline.attention"] == pytest.approx(
        _module(windowed, machine) + _module(full, machine), rel=1e-12
    )

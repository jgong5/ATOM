# SPDX-License-Identifier: MIT
"""GPT-OSS-120B's meta build: the step roofline refuses every step by name,
naming the sliding window of its windowed layers and the issue that would
price it, and prices its full attention layers as attention.

Its config is `gpt_oss_120b_config.json`, openai/gpt-oss-120b's config.json
byte for byte at revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a. The build
owns its process because the driverless declarations are process-global.
"""

import json
import pathlib

import pytest
import test_backend_step_roofline_mla as mla
from test_backend_step_roofline_v4 import view

from atom.compass.backends import CostRefused
from atom.compass.backends.step_roofline import StepRooflineBackend
from atom.compass.spec import MachineSpec

CONFIG = json.loads(
    pathlib.Path(__file__).with_name("gpt_oss_120b_config.json").read_text()
)
REFUSED = "a sliding window; no roofline class prices it before #3965"


def test_a_windowed_layer_refuses_every_step_naming_its_window_and_issue():
    # Cut to its first two layers: a windowed one, then a full attention one.
    types = CONFIG["layer_types"][:2]
    walked = mla._build("tp1", {**CONFIG, "num_hidden_layers": 2, "layer_types": types})
    machine = MachineSpec.from_mapping(json.loads(mla.MI355X.read_text()))

    refused = [m.name for m in walked if REFUSED in m.name]
    assert refused == [f"Attention at model.layers.0.self_attn.attn ({REFUSED})"]
    assert [m.layer for m in walked if m.kind == "attention"] == [1]
    with pytest.raises(CostRefused, match=r"a sliding window; .* before #3965"):
        StepRooflineBackend(machine, walked).price(view((1, 64, True)))

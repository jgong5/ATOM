# SPDX-License-Identifier: MIT
"""Kimi-K3's meta build: the step roofline refuses every step by name, naming
the KDA recurrence and the issue that would price it, since the recurrence
runs inside `KimiKDAAttention`'s forward, in no module the walk reads.

The build owns its process because the driverless declarations are
process-global.
"""

import json
import pathlib

import pytest
import test_backend_step_roofline_mla as mla
from test_backend_step_roofline_v4 import view

from atom.compass.backends import CostRefused
from atom.compass.backends.step_roofline import StepRooflineBackend
from atom.compass.spec import MachineSpec

#: The published Kimi-K3 config (`kimi_k3_config.json`).
CONFIG = json.loads(pathlib.Path(__file__).with_name("kimi_k3_config.json").read_text())
REFUSED = "Kimi-K3's KDA recurrence; no roofline class prices it before #3963"
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


def test_a_kda_layer_refuses_every_step_naming_its_recurrence_and_issue(monkeypatch):
    assert mla.BUILD.count(CONFIGURED) == 1
    built = mla.BUILD.replace(CONFIGURED, CONFIGURED + VISION)
    monkeypatch.setattr(mla, "BUILD", built)
    # Cut to its first two layers: a KDA one, then a full attention one.
    text = CONFIG["text_config"]
    kda = {**text["linear_attn_config"], "kda_layers": [1], "full_attn_layers": [2]}
    text = {**text, "num_hidden_layers": 2, "linear_attn_config": kda}
    walked = mla._build("tp8", {**CONFIG, "text_config": text})
    machine = MachineSpec.from_mapping(json.loads(mla.MI355X.read_text()))

    attention = "language_model.model.layers.0.self_attn"
    refused = [m.name for m in walked if REFUSED in m.name]
    assert refused == [f"KimiKDAAttention at {attention} ({REFUSED})"]
    assert not [m for m in walked if m.name.startswith(attention + ".")]
    with pytest.raises(CostRefused, match=r"KDA recurrence; .* before #3963"):
        StepRooflineBackend(machine, walked).price(view((1, 64, True)))

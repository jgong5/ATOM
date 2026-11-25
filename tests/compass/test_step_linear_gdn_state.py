# SPDX-License-Identifier: MIT
"""Qwen3.8-Flash-Next's gated delta-net layers state their state work as KDA layers.

Its 36 GDN layers keep their recurrent state in ATOM's state slots and
checkpoints (`GDNStateMixin`), which the scheduler saves, forks and moves as it
does Kimi-K3's KDA states, so its step lines state the same four counts and
its laws may fit the KDA prefill columns over them. Other GDN models (Qwen3.5,
Qwen3.8 dense) keep stating none: their records and laws are unchanged.
"""

import json
from types import SimpleNamespace

from test_calib_v1 import _record_dir

from atom.compass.backends.step_linear import GDN_STATE_MODELS, kda_layers
from atom.compass.calib import deploy

LAYERS = ["linear_attention"] * 3 + ["full_attention"]
FLASH_NEXT = SimpleNamespace(model_type="qwen4_exp_text", layer_types=LAYERS * 12)


def test_qwen38_flash_next_counts_its_gdn_layers():
    assert GDN_STATE_MODELS == ("qwen4_exp_text",)
    assert kda_layers(FLASH_NEXT) == 36
    assert kda_layers(SimpleNamespace(text_config=FLASH_NEXT)) == 36


def test_other_gdn_models_state_no_state_work():
    for model_type in ("qwen3_5_text", "qwen3_next", None):
        dense = SimpleNamespace(model_type=model_type, layer_types=LAYERS * 12)
        assert kda_layers(dense) == 0


def test_a_kda_models_layers_are_still_its_linear_attention_configs():
    kimi = SimpleNamespace(
        model_type="kimi_k3", linear_attn_config={"kda_layers": [1, 2, 3]}
    )
    assert kda_layers(kimi) == 3


def test_a_qwen38_flash_next_record_states_kda_state(tmp_path):
    rec = _record_dir(tmp_path / "q38")
    model = tmp_path / "q38" / "model" / "config.json"
    text = {"model_type": "qwen4_exp_text", "layer_types": LAYERS * 12}
    text |= {"num_experts": 512, "num_experts_per_tok": 10}
    model.write_text(json.dumps({"model_type": "qwen4_exp", "text_config": text}))
    got = deploy.from_record(rec)
    assert got["kda_state"] is True
    assert got["moe_routing"] == [512, 10]
    dense = _record_dir(tmp_path / "dense")
    config = tmp_path / "dense" / "model" / "config.json"
    config.write_text(json.dumps({"text_config": text | {"model_type": "qwen3_5_text"}}))
    assert "kda_state" not in deploy.from_record(dense)

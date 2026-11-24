# SPDX-License-Identifier: MIT
"""A key computed from a config ATOM has not built is the one its runner
computes once built. `ModelRunner.__init__` remaps the quantization config's
layer names to the model's modules before it builds, which moves the config's
hash wherever its checkpoint names a layer the model packs or renames.
`compass run --check`, `compass measure steps` and a server's frontend key the
config unbuilt; `compass measure memory` and a record key it in the worker.

The worker's remap runs here in a process that declares a device, since the
model class's module imports aiter.

A model whose config class is its checkpoint's own code keys alike once the
check's tokenizer load runs that code again, giving the class a new object.

No device: aiter's arch probe answers a declared arch.
"""

import copy
import json
import pathlib
import sys
import tempfile

import pytest
from test_key_index_cache_dtype import served
from test_memory_readings import DOCUMENT, publish_readings
from test_run_modules_minimax_m3 import M3, MXFP4
from tokenizers import Tokenizer, models

from atom.compass.bench.ops import _child
from atom.compass.run import memory
from atom.compass.run.step_grid import keyed
from atom.compass.spec import MachineSpec

QWEN35 = pathlib.Path(__file__).with_name("qwen3_5_35b_a3b_fp8_config.json")
#: Kimi-K2.5-MXFP4's config.json, its excludes cut to the text model's first
#: layer and the projector: its model class holds its packed modules in a
#: property, which its text model type replaces in the remap.
KIMI_K25 = pathlib.Path(__file__).with_name("kimi_k2_5_mxfp4_config.json")
#: MiniMax-M3's published MXFP4 checkpoint as ATOM serves it, cut as
#: test_run_modules_minimax_m3 cuts it: the VL architecture, whose excludes
#: name the text model under ``language_model.`` and its projections unpacked,
#: as the checkpoint's first ones do.
M3_VL = {
    **M3,
    "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
    "quantization_config": {
        **MXFP4,
        "exclude": ["language_model.lm_head"]
        + [
            f"language_model.model.layers.0.{name}"
            for name in (
                "mlp.gate_proj",
                "mlp.up_proj",
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
            )
        ],
    },
}
#: MiniMax-M3's config module as its checkpoint ships it, cut to the class its
#: config.json's ``auto_map`` names, its text model a Mixtral config.
REMOTE_M3 = """\
from transformers import MixtralConfig, PretrainedConfig


class MiniMaxM3VLConfig(PretrainedConfig):
    model_type = "minimax_m3_vl"

    def __init__(self, text_config=None, **kwargs):
        self.text_config = MixtralConfig(**(text_config or {}))
        super().__init__(**kwargs)
"""


def as_its_runner(quant, hf_config):
    """`quant` with its layer names remapped as `ModelRunner.__init__` remaps
    them before it builds the model class `hf_config` names."""
    from atom.compass.capture import driverless

    with tempfile.TemporaryDirectory() as tmpdir:
        driverless.install(tmpdir, "gfx950")
        from atom.model_engine.model_runner import support_model_arch_dict
        from atom.utils import resolve_obj_by_qualname

        model_class = resolve_obj_by_qualname(
            support_model_arch_dict[hf_config.architectures[0]]
        )
        quant.remap_layer_name(
            hf_config,
            packed_modules_mapping=getattr(model_class, "packed_modules_mapping", {}),
            quant_exclude_name_mapping=getattr(
                model_class, "quant_exclude_name_mapping", {}
            ),
        )
    return quant


def built(config):
    """`config` as its runner holds it once remapped, built in its own process."""
    held = copy.copy(config)
    held.quant_config = _child(as_its_runner, config.quant_config, config.hf_config)
    return held


@pytest.fixture(params=["MiniMax-M3", "Qwen3.5", "Kimi-K2.5"])
def unbuilt(request, tmp_path, monkeypatch):
    """A TP4 server's config of MiniMax-M3's MXFP4 checkpoint, Qwen3.5-35B-A3B's
    FP8 one or Kimi-K2.5's MXFP4 one, as `compass run` resolves it, unbuilt."""
    model = {"Qwen3.5": QWEN35, "Kimi-K2.5": KIMI_K25}.get(request.param)
    if model is None:
        model = tmp_path / "m3.json"
        model.write_text(json.dumps(M3_VL))
    return served(tmp_path, monkeypatch, "gfx950", model, "-tp", "4")


def test_readings_the_worker_keys_load_for_the_unbuilt_config(unbuilt, tmp_path):
    worker = built(unbuilt)
    publish_readings(tmp_path / "store", worker)
    machine = MachineSpec.from_mapping(json.loads(json.dumps(DOCUMENT)))

    memory.load(str(tmp_path / "store"), unbuilt, machine, 0)
    assert keyed(unbuilt) == keyed(worker)
    hashes = (unbuilt.quant_config.compute_hash(), worker.quant_config.compute_hash())
    assert hashes[0] != hashes[1]


def test_a_config_its_runner_remapped_keys_without_a_second_remap(
    tmp_path, monkeypatch
):
    # Qwen3.5's remap renames a remapped name again: ``in_proj_ba`` holds
    # ``.in_proj_b``, which it packs into ``in_proj_ba``.
    unbuilt = served(tmp_path, monkeypatch, "gfx950", QWEN35, "-tp", "4")
    once = built(unbuilt)
    twice = built(once)
    assert once.quant_config.compute_hash() != twice.quant_config.compute_hash()

    assert keyed(once) == keyed(unbuilt)


def keyed_across_its_tokenizer(model: str) -> tuple:
    """`keyed` of the config `compass run --check` builds for a TP4 server of
    `model`, before and after the check loads its tokenizer, and whether that
    load gave the config's class a new object in its module."""
    from atom.compass.drive.run import _config
    from atom.model_engine.llm_engine import load_tokenizer

    config = _config(["--model", model, "--trust-remote-code", "-tp", "4"], "gfx950")
    before = keyed(config)
    load_tokenizer(model, True)
    held = type(config.hf_config)
    moved = getattr(sys.modules[held.__module__], held.__qualname__) is not held
    return before, keyed(config), moved


def test_a_remote_code_config_keys_alike_once_its_tokenizer_runs_its_module_again(
    tmp_path, monkeypatch
):
    model = tmp_path / "MiniMax-M3"
    model.mkdir()
    vl = {k: M3_VL[k] for k in ("architectures", "quantization_config")}
    vl |= {"model_type": "minimax_m3_vl", "text_config": M3}
    vl["auto_map"] = {"AutoConfig": "configuration_minimax_m3_vl.MiniMaxM3VLConfig"}
    (model / "config.json").write_text(json.dumps(vl))
    (model / "configuration_minimax_m3_vl.py").write_text(REMOTE_M3)
    vocab = models.WordLevel({"[UNK]": 0, "a": 1}, unk_token="[UNK]")
    Tokenizer(vocab).save(str(model / "tokenizer.json"))
    fast = {"tokenizer_class": "PreTrainedTokenizerFast"}
    (model / "tokenizer_config.json").write_text(json.dumps(fast))
    # A process of its own, as the check is: transformers places its module
    # cache as it imports, and the model's config module is written there.
    monkeypatch.setenv("HF_MODULES_CACHE", str(tmp_path / "modules"))

    before, after, moved = _child(keyed_across_its_tokenizer, str(model))

    assert moved
    assert after == before

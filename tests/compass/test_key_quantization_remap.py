# SPDX-License-Identifier: MIT
"""A key computed from a config ATOM has not built is the one its runner
computes once built. `ModelRunner.__init__` remaps the quantization config's
layer names to the model's modules before it builds, which moves the config's
hash wherever its checkpoint names a layer the model packs or renames.
`compass run --check`, `compass measure steps` and a server's frontend key the
config unbuilt; `compass measure memory` and a record key it in the worker.

The worker's remap runs here in a process that declares a device, since the
model class's module imports aiter.

No device: aiter's arch probe answers a declared arch.
"""

import copy
import json
import pathlib
import tempfile

import pytest
from test_key_index_cache_dtype import served
from test_memory_readings import DOCUMENT, publish_readings
from test_run_modules_minimax_m3 import M3, MXFP4

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

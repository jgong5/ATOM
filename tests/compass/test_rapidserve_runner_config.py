# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""`enable_rapidserve` refuses a runner that cannot answer its engine cores."""

from types import SimpleNamespace

import pytest
import test_runner_rpc_surface as surface
from transformers import PretrainedConfig

import atom.config as config_module
from atom.config import CompilationConfig, Config

RAPID = "atom.model_engine.model_runner.RapidServeModelRunner"
SUBCLASS = "my.pkg.FastRapidServeRunner"  # a RapidServeModelRunner subclass


def _config(monkeypatch, **kwargs):
    hf = PretrainedConfig(architectures=["LlamaForCausalLM"])
    monkeypatch.setattr(config_module, "get_hf_config", lambda *_a, **_k: hf)
    monkeypatch.setattr(config_module, "get_generation_config", lambda _m: None)
    quant = SimpleNamespace(exclude_layers=[])
    monkeypatch.setattr(config_module, "QuantizationConfig", lambda *_a, **_k: quant)
    compilation = CompilationConfig(level=0, use_cudagraph=False)
    return Config(model="test-model", compilation_config=compilation, **kwargs)


@pytest.mark.parametrize(
    "qualname",
    [
        "atom.compass.runner.model_runner.CompassModelRunner",
        "atom.rollout.model_runner_ext.RLHFModelRunner",
        "some.other.module.RapidServeModelRunner",
        SUBCLASS,  # not listed in RAPIDSERVE_RUNNERS
    ],
)
def test_a_runner_that_is_not_a_rapidserve_runner_is_refused(monkeypatch, qualname):
    with pytest.raises(ValueError, match=f"runner_qualname='{qualname}': ") as err:
        _config(monkeypatch, enable_rapidserve=True, runner_qualname=qualname)
    assert all(rpc in str(err.value) for rpc in config_module.RAPIDSERVE_RPCS)
    assert _config(monkeypatch, runner_qualname=qualname).runner_qualname == qualname


@pytest.mark.parametrize("qualname", [None, RAPID, SUBCLASS])
def test_a_rapidserve_runner_is_accepted(monkeypatch, qualname):
    listed = config_module.RAPIDSERVE_RUNNERS | {SUBCLASS}
    monkeypatch.setattr(config_module, "RAPIDSERVE_RUNNERS", listed)
    named = {} if qualname is None else {"runner_qualname": qualname}
    config = _config(monkeypatch, enable_rapidserve=True, **named)
    assert config.runner_qualname == (qualname or RAPID)


def test_the_rpcs_named_are_the_waits_only_the_rapidserve_cores_make():
    """`RAPIDSERVE_RPCS` is the waited names `PrefillEngineCore` and
    `DecodeEngineCore` broadcast that `ModelRunner` lacks, and also the
    broadcast names `RapidServeModelRunner` defines and `ModelRunner` lacks."""
    classes = surface._classes(surface.ENGINE / "engine_core.py")
    cores = [classes["PrefillEngineCore"], classes["DecodeEngineCore"]]
    waited = {
        name
        for name, sites in surface.SITES.items()
        for s in sites
        if s.waits
        and s.file == "atom/model_engine/engine_core.py"
        and any(c.lineno <= s.line <= c.end_lineno for c in cores)
    }
    rapid_only = (set(surface.SITES) & surface.RAPID) - surface.BASE
    assert set(config_module.RAPIDSERVE_RPCS) == waited - surface.BASE == rapid_only

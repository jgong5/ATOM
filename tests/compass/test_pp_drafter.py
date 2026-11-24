# SPDX-License-Identifier: MIT
"""A pipeline with a drafter: ATOM builds the drafter on the last stage alone
(`ModelRunner.__init__`), so a simulated stage builds as one stage does, the
last charges the draft and an earlier one charges none. A config that declares
no acceptance is still refused.

Every stage builds ATOM's aux relay over its target (`PPAuxRelay`), which finds
the decoder layers through the model's `.model` wrappers; a simulated runner's
model is an `UnbuiltModel`. That check imports ATOM's drafter module, which
needs aiter declared on a device, so it owns a process.
"""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import MockConfig
from test_dp_step_max import prefill, runner
from test_speculative_step import _speculative

from atom.compass.backends.step_stub import Coefficients, StepStubBackend
from atom.compass.runner import overrides
from atom.compass.runner.overrides import (
    NonAllocatingRunner,
    RunnerRefusal,
    _group_step_seconds,
    _installed_backend,
)

TREE = Path(__file__).resolve().parents[2]
#: 10 s of prefill for the whole stack and a 4 s draft.
STUB = dataclasses.replace(Coefficients.constant(10.0, 1.0), draft_step=4.0)
#: ATOM's decoder-layer lookup on an unbuilt model whose tree wraps its stack.
RESOLVE = """
from atom.compass.capture.driverless import import_aiter_on
import_aiter_on("gfx950")
from torch import nn
from atom.compass.runner.overrides import UnbuiltModel
from atom.spec_decode.drafter import _resolve_decoder_layers
held = UnbuiltModel(nn.Module)
held.tree = nn.Module()
held.tree.model = nn.Module()
held.tree.model.layers = nn.ModuleList([nn.Identity()])
assert _resolve_decoder_layers(held) is held.tree.model.layers
"""


def _stage(rank: int):
    """Stage `rank` of two, priced for half the stack, its drafts declared."""
    stage = _speculative(runner(StepStubBackend(STUB, stage_share=(0.5, 0.5))))
    stage.config.pipeline_parallel_size = 2
    stage.config.parallel_config.pipeline_parallel_rank = rank
    return stage


@pytest.mark.parametrize("rates", [(0.5,), None], ids=["declared", "undeclared"])
def test_a_stage_builds_its_drafter_on_meta_only_on_a_declared_acceptance(
    monkeypatch, rates
):
    class Built(Exception):
        pass

    def built():
        raise Built

    monkeypatch.setattr(overrides, "drafts_on_meta", built)
    stage = NonAllocatingRunner()
    spec = SimpleNamespace(synthetic_acceptance_rates=rates)
    stage.config = MockConfig(pipeline_parallel_size=2, speculative_config=spec)
    with pytest.raises(RunnerRefusal if rates is None else Built) as raised:
        stage._build_and_load_model(object)
    assert rates is not None or "declares none" in str(raised.value)


def test_the_last_stage_alone_charges_the_draft():
    first, last = _stage(0), _stage(1)
    seconds = [
        _group_step_seconds(s, prefill(8), _installed_backend(s)) for s in (first, last)
    ]
    assert seconds == [5.0, 9.0]
    assert first.refused_commands() == last.refused_commands() == ()


def test_atom_finds_the_decoder_layers_of_an_unbuilt_model():
    env = dict(os.environ, PYTHONPATH=str(TREE))
    done = subprocess.run(
        [sys.executable, "-c", RESOLVE],
        capture_output=True,
        text=True,
        check=False,
        cwd=TREE,
        env=env,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-4000:]

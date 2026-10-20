# SPDX-License-Identifier: MIT
"""A RapidServe deployment's memory readings: taken by a RapidServe start, from
its prefill engine, and keyed apart from a plain engine's."""

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_measure_memory import MACHINE
from test_memory_readings import READINGS, publish_readings
from test_rapidserve_runner_config import _config
from test_run_step_grid import deployment

from atom.compass.artifacts import ArtifactRefusal
from atom.compass.drive import measure_memory
from atom.compass.run import memory

DRIVE = Path(measure_memory.__file__).parent


def test_a_plain_engines_readings_refuse_a_rapidserve_run(tmp_path):
    r = deployment(tmp_path)
    r.config.enable_rapidserve = True

    with pytest.raises(ArtifactRefusal) as refused:
        memory.load(tmp_path / "store", r.config, MACHINE, 0)
    assert '"enable_rapidserve": true' in str(refused.value)
    assert "the nearest entry differs in engine_config" in str(refused.value)

    publish_readings(tmp_path / "store", r.config, readings=READINGS[2])
    assert memory.load(tmp_path / "store", r.config, MACHINE, 0).values == READINGS[2]


@pytest.mark.parametrize("rapid", [False, True])
def test_a_start_names_a_runner_atom_accepts_under_its_flags(monkeypatch, rapid):
    named = []
    engine = types.ModuleType("atom.model_engine.llm_engine")
    engine.LLMEngine = lambda model, runner_qualname, **kw: named.append(
        runner_qualname
    ) or SimpleNamespace(close=lambda: None)
    monkeypatch.setitem(sys.modules, engine.__name__, engine)

    measure_memory.start(["--model", "m"] + ["--enable-rapidserve"] * rapid)

    _config(monkeypatch, enable_rapidserve=rapid, runner_qualname=named[0])
    runner = measure_memory.RAPIDSERVE_RUNNER if rapid else measure_memory.RUNNER
    assert named == [runner]


def _load(file, name):
    spec = importlib.util.spec_from_file_location(name, DRIVE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rapidserve_runner(monkeypatch, warmups):
    """The RapidServe reading runner, on stand-ins for ATOM's two runners that
    bind and warm up as theirs do."""

    class ModelRunner:
        def forward(self, batch):
            raise AssertionError("prefill warms up through prefill_forward")

    class RapidServeModelRunner(ModelRunner):
        def __init__(self):
            self.forward = self.prefill_forward

        def prefill_forward(self, batch):
            return batch

        def _maybe_warmup(self):
            warmups.append(self.forward(SimpleNamespace(total_tokens_num=8)))

    engine = types.ModuleType("atom.model_engine.model_runner")
    engine.ModelRunner = ModelRunner
    engine.RapidServeModelRunner = RapidServeModelRunner
    monkeypatch.setitem(sys.modules, engine.__name__, engine)
    reading = _load("memory_runner.py", "reading_runner")
    monkeypatch.setitem(sys.modules, "atom.compass.drive.memory_runner", reading)
    module = _load("rapidserve_memory_runner.py", "rapidserve_reading_runner")
    return module.MemoryReadingRapidServeModelRunner()


def test_prefill_keeps_its_forward_after_the_counted_warmup(monkeypatch):
    warmups = []
    runner = _rapidserve_runner(monkeypatch, warmups)
    runner._read = {}

    runner._maybe_warmup()

    assert len(warmups) == 1 and runner._read == {"warmup_tokens": 8}
    assert runner.forward == runner.prefill_forward


@pytest.mark.parametrize("decode", [False, True])
def test_only_prefill_writes_readings(tmp_path, monkeypatch, decode):
    runner = _rapidserve_runner(monkeypatch, [])
    runner.rank, runner.config = 0, deployment(tmp_path).config
    vars(runner.config).update(enable_rapidserve=True, disagg_is_decode=decode)
    runner._read = dict.fromkeys(
        ["total", "non_torch", "loaded", "weights", "current", "peak"], 0
    ) | {"warmup_tokens": 0, "num_kvcache_blocks": 1}
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv(measure_memory.OUT_ENV, str(out))

    runner._write(None)

    written = [json.loads(p.read_text())["key"] for p in out.iterdir()]
    expected = [] if decode else [memory.key_fields(runner.config)]
    assert written == expected

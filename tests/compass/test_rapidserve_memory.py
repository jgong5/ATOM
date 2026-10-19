# SPDX-License-Identifier: MIT
"""A RapidServe deployment's memory readings: taken by a RapidServe start, one
entry for each engine's key, and keyed apart from a plain engine's."""

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import test_measure_memory
from test_measure_memory import MACHINE, a_start, started
from test_memory_readings import READINGS, STACK, publish_readings
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

        def import_model_weight_ipc_handles(self, paths):
            return True

        def _read_device_memory(self):
            return SimpleNamespace(total=100, non_torch=7)

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
def test_decode_writes_its_readings_beside_prefills(tmp_path, monkeypatch, decode):
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

    written = out / measure_memory.DECODE if decode else out
    (held,) = written.glob("*.json")
    assert json.loads(held.read_text())["key"] == memory.key_fields(runner.config)


def test_decode_reads_the_device_once_it_holds_prefills_weights(monkeypatch):
    # ATOM's decode sizes no pool, so nothing in it reads the device; and it
    # warms up nothing, so its peak is where it stands.
    runner = _rapidserve_runner(monkeypatch, [])
    reading = sys.modules["atom.compass.drive.memory_runner"]
    stats = {"current": 5, "peak": 9}
    monkeypatch.setattr(reading, "_allocated", stats.get)
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda: stats.update(peak=5)
    )
    runner._read = {}

    assert runner.import_model_weight_ipc_handles(["handles"]) is True

    assert runner._read == {"total": 100, "non_torch": 7, "current": 5, "peak": 5}


#: A RapidServe decode rank's readings: no warmup, so no activations, and the
#: pool its capture took.
DECODE = dict(
    READINGS[1], activation_bytes=0, warmup_tokens=0, cudagraph_pool_bytes=1 << 20
)


@pytest.mark.parametrize("eager", [False, True])
def test_decode_publishes_its_entry_unless_its_key_is_prefills(tmp_path, eager):
    # ATOM's RapidServe makes prefill eager; decode keeps its own ladder.
    prefill = deployment(tmp_path, enable_rapidserve=True, enforce_eager=True).config
    decode = SimpleNamespace(**vars(prefill) | {"enforce_eager": eager})
    out = tmp_path / "out"
    record = started(out, prefill, [READINGS[1]])
    (out / measure_memory.READINGS / measure_memory.DECODE).mkdir()
    held = {k: v for k, v in DECODE.items() if k != "record"}
    name = next((out / measure_memory.READINGS).glob("*.json")).name
    (out / measure_memory.READINGS / measure_memory.DECODE / name).write_text(
        json.dumps({"key": memory.key_fields(decode), "readings": held})
    )

    measure_memory.publish(out, MACHINE, tmp_path / "s", "c")

    read = memory.load(tmp_path / "s", prefill, MACHINE, 0)
    assert read.values == dict(READINGS[1], record=record)
    read = memory.load(tmp_path / "s", decode, MACHINE, 0)
    assert read.values == dict(READINGS[1] if eager else DECODE, record=record)
    assert (out / "memory_readings.decode.json").exists() is not eager


def test_a_decode_start_that_missed_a_graph_starts_again(tmp_path):
    r = deployment(tmp_path)
    out = tmp_path / "out"
    started(out, r.config, [READINGS[1]])
    decode = out / measure_memory.READINGS / measure_memory.DECODE
    decode.mkdir()
    (decode / "readings.0.json").write_text(json.dumps({"fxgraph_cache_misses": 3}))

    assert measure_memory._cold(out) == ["decode/readings.0.json"]


def test_a_rapidserve_run_is_identified_by_prefills_engine_file(tmp_path, monkeypatch):
    r, argv = a_start(tmp_path, monkeypatch, STACK)
    wrote = test_measure_memory.ranks_wrote

    def and_decode(out, config, readings):
        wrote(out, config, readings)
        files = out / measure_memory.READINGS
        (files / measure_memory.DECODE).mkdir()
        for held in files.glob("*.json"):
            decode = json.loads(held.read_text()) | {"engine": {"decode": True}}
            (files / measure_memory.DECODE / held.name).write_text(json.dumps(decode))

    monkeypatch.setattr(test_measure_memory, "ranks_wrote", and_decode)

    measure_memory.main(argv)

    assert json.loads((tmp_path / "out" / "engine.json").read_text()) == {}

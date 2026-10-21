# SPDX-License-Identifier: MIT
"""MI455's machine spec, for a device no engine has run on, and the assumed
memory_readings entry a run on it reads.

Each field of `machines/mi455.json` is MI455X's datasheet, or assumed: a ratio
mi355-gpu-45 probed on a datasheet peak, or MI355X's value where no datasheet
states one. A run on that spec reads the entry `mi455_readings.py` copies from
MI355X's, through `run.runner` as `test_run_graph_pool` installs one.
"""

import json

import pytest
import test_run_graph_pool as graph_pool
from test_memory_readings import READINGS, publish_readings, with_readings
from test_probe_device import SCRIPTS, _load
from test_spec_origin import labels

from atom.compass.artifacts import ArtifactRefusal, Axis, Reading
from atom.compass.run import memory
from atom.compass.spec import MachineSpec

spec_script = _load("mi455_spec")
readings_script = _load("mi455_readings")
MI355X, MI455 = spec_script.MI355X, SCRIPTS / "machines" / "mi455.json"


def _spec(path):
    return MachineSpec.from_mapping(json.loads(path.read_text()))


def test_the_mi455_spec_rebuilds_byte_for_byte(tmp_path):
    out = tmp_path / "mi455.json"
    assert spec_script.main(["--out", str(out)]) == 0
    assert out.read_bytes() == MI455.read_bytes()


def test_each_field_is_the_datasheet_or_assumed_its_peaks_at_mi355x_ratios():
    mi455, mi355x = _spec(MI455), _spec(MI355X)
    origins = {label: mi455.origin(label)["origin"] for label in labels(mi455)}
    assert set(origins.values()) == {"datasheet", "assumed"}
    assert {p for p, o in origins.items() if o == "datasheet"} == {
        "device.name",
        "device.count_per_node",
        "device.memory.capacity_bytes",
        "device.memory.bandwidth_bytes_per_s",
        "device.compute.bf16_flops",
        "device.compute.fp8_flops",
        "device.compute.fp4_flops",
        "interconnect.intra_node.topology",
        "interconnect.intra_node.domain_devices",
        "interconnect.intra_node.link_bandwidth_bytes_per_s",
        "interconnect.inter_node.link_bandwidth_bytes_per_s",
    }
    assert mi455.value("device.memory.capacity_bytes") == 432 * 2**30
    # Each attainable rate is MI455X's peak times the ratio MI355X probed.
    for fmt in ("bf16", "fp8", "fp4"):
        derate = f"device.compute.derate.{fmt}"
        assert mi355x.origin(derate)["origin"] == "probed"
        assert mi455.value(derate) == mi355x.value(derate)
    peak = "device.memory.bandwidth_bytes_per_s"
    curve = "device.memory.bandwidth_curve"
    for (size, rate), (probed, at) in zip(mi455.value(curve), mi355x.value(curve)):
        assert size == probed
        assert rate / mi455.value(peak) == pytest.approx(at / mi355x.value(peak))
    # No datasheet states a kernel floor or a link latency: MI355X's probed ones.
    for path in ("device.kernel_floor_s", "interconnect.intra_node.link_latency_s"):
        assert mi355x.origin(path)["origin"] == "probed"
        assert (mi455.value(path), origins[path]) == (mi355x.value(path), "assumed")


def _copied_run(monkeypatch, tmp_path, readings, revision=None):
    """A worker on MI455 after `run.runner` installed its readings: the entry
    measure memory published on mi355-gpu-45 holding `readings`, at the ATOM
    `revision` when one is given, copied by `mi455_readings.py`. Returns the
    runner and both entries' keys."""
    keys, taken = [], memory.conditions

    def elsewhere(*args, **kwargs):
        """The conditions here, the ATOM source read the same way at `revision`."""
        here = taken(*args, **kwargs)
        kind = here.reading(Axis.ATOM_SRC).kind
        return here.with_reading(Axis.ATOM_SRC, Reading(kind, revision))

    def copied(runner, store, readings, machine):
        with_readings(runner, store, None)
        mi355x = json.loads(MI355X.read_text())
        with monkeypatch.context() as m:
            if revision:
                m.setattr(memory, "conditions", elsewhere)
            keys.append(publish_readings(store, runner.config, mi355x, readings))
        keys.append(readings_script.publish(str(store), keys[0].token(), "test"))
        return runner

    monkeypatch.setattr(graph_pool, "with_readings", copied)
    mi455 = json.loads(MI455.read_text())
    runner = graph_pool._installed(
        monkeypatch, tmp_path, eager=False, readings=readings, machine=mi455
    )
    return runner, *keys


def test_a_run_on_mi455_reads_the_copied_entry_and_declares_each_term_on_it(
    monkeypatch, tmp_path
):
    readings = dict(READINGS[1], cudagraph_pool_bytes=420_000_000)
    runner, original, _ = _copied_run(monkeypatch, tmp_path, readings)
    assumed = f"copied from mi355x-gpu45's memory_readings entry {original.token()}"
    assert runner.compass_memory["assumed"] == assumed
    terms = [t for r in runner.compass_readings.as_dict().values() for t in r.terms]
    # Only the capacity, MI455X's datasheet, and the model's own bytes are not
    # the entry's.
    stand = {(t.name, t.method.value) for t in terms if t.method.value != "declared"}
    assert stand == {
        ("capacity", "analytical"),
        ("weights", "analytical"),
        ("buffers", "analytical"),
        ("total", "analytical"),
    }
    (pool,) = runner.compass_memory["predicts"]
    assert (pool["nbytes"], pool["method"]) == (420_000_000, "declared")
    assert runner.compass_readings.total.total == 432 * 2**30


def test_the_copy_answers_only_at_the_atom_tree_of_its_original(monkeypatch, tmp_path):
    with pytest.raises(ArtifactRefusal, match="`ATOM src`: recorded '0{40}'"):
        _copied_run(monkeypatch, tmp_path, READINGS[1], "0" * 40)


def test_an_entry_measure_memory_did_not_take_on_mi355x_is_refused(
    monkeypatch, tmp_path
):
    _, _, copy = _copied_run(monkeypatch, tmp_path, READINGS[1])
    with pytest.raises(ValueError, match="is not keyed by mi355x-gpu45.json"):
        readings_script.publish(str(tmp_path / "store"), copy.token(), "test")

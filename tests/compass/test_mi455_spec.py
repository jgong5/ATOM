# SPDX-License-Identifier: MIT
"""MI455's two machine specs: A0's from probes on an MI455 node, each device
field probed there or MI455X's datasheet; B0's, for machines no card here
reads, the datasheet's peaks at MI355X's probed ratios, with nothing A0's probe
read."""

import json

import pytest
from test_probe_device import SCRIPTS, _load
from test_spec_origin import labels

from atom.compass.artifacts import machine_platform
from atom.compass.spec import MachineSpec

MACHINES = SCRIPTS / "machines"
A0, B0, MI355X = (
    MACHINES / f"{name}.json" for name in ("mi455-a0", "mi455-b0", "mi355x-gpu45")
)


def _spec(path):
    return MachineSpec.from_mapping(json.loads(path.read_text()))


def _script():
    """`mi455_spec.py`, loaded here: loading it names the machine in the
    shared `mi308x_spec`, which loading another machine's script renames."""
    return _load("mi455_spec")


@pytest.mark.parametrize("machine", [A0, B0])
def test_each_mi455_spec_rebuilds_byte_for_byte(tmp_path, machine):
    out = tmp_path / machine.name
    stepping = machine.stem.removeprefix("mi455-")
    assert _script().main([stepping, "--out", str(out)]) == 0
    assert out.read_bytes() == machine.read_bytes()


def test_each_a0_device_field_is_probed_or_the_datasheet_and_the_rest_assumed():
    a0, peaks = _spec(A0), _script().PEAKS
    origins = {label: a0.origin(label)["origin"] for label in labels(a0)}
    assert {p for p, o in origins.items() if o == "assumed"} == {
        "interconnect.inter_node.link_latency_s",
        "interconnect.inter_node.derate",
        "interconnect.router_relay_s",
    }
    device = {o for p, o in origins.items() if p.startswith("device.")}
    assert device == {"probed", "datasheet"}
    # The probe's record holds the readings, the device and the stack.
    record = f"sha256:{peaks[3]}"
    for path in (
        "device.arch",
        "device.memory.capacity_bytes",
        "device.software_pinned_to.aiter",
        "device.compute.derate.bf16",
        "device.compute.derate.fp4",
        "device.memory.bandwidth_curve",
        "interconnect.intra_node.derate",
    ):
        assert a0.origin(path)["record"] == record
    for path in ("device.kernel_floor_s", "interconnect.intra_node.link_latency_s"):
        band = a0.origin(path)
        assert band["record"] == record
        assert band["p10"] <= a0.value(path) <= band["p90"]
    found, peak = peaks[2], a0.value("device.compute.fp4_flops")
    assert a0.value("device.compute.derate.fp4") == found["fp4"] / peak


def test_b0_shares_a0_s_peaks_and_carries_mi355x_s_ratios_never_a0_s_readings():
    a0, b0, mi355x = _spec(A0), _spec(B0), _spec(MI355X)
    origins = {label: b0.origin(label)["origin"] for label in labels(b0)}
    assert set(origins.values()) == {"datasheet", "assumed"}
    peaks = {p for p, o in origins.items() if o == "datasheet"}
    assert {p for p in peaks if p.startswith("device.")} == {
        "device.count_per_node",
        "device.memory.capacity_bytes",
        "device.memory.bandwidth_bytes_per_s",
        "device.compute.bf16_flops",
        "device.compute.fp8_flops",
        "device.compute.fp4_flops",
    }
    # A0's capacity is what torch reads on a sample card; B0's the datasheet's.
    for path in peaks - {"device.memory.capacity_bytes"}:
        assert b0.value(path) == a0.value(path)
    for path in (
        "device.compute.derate.bf16",
        "device.compute.derate.fp8",
        "device.compute.derate.fp4",
        "device.memory.derate",
        "device.kernel_floor_s",
        "interconnect.intra_node.derate",
        "interconnect.intra_node.link_latency_s",
        "device.software_pinned_to.aiter",
        "device.software_pinned_to.rocm",
    ):
        assert b0.value(path) == mi355x.value(path) != a0.value(path)
    a0_key, b0_key = (machine_platform(s) for s in (a0, b0))
    assert (a0_key["device"], b0_key["device"]) == (
        "MI455X gfx1250",
        "MI455X B0 gfx1250",
    )
    assert a0_key["software_stack"] != b0_key["software_stack"]


def test_a_run_on_a0_charges_the_intake_the_node_s_bursts_fitted(monkeypatch, tmp_path):
    """The frontend loop's intake a simulated run on A0 charges: its law file's
    host entry's."""
    from atom.compass import run as compass_run

    run = tmp_path / "run.json"
    law = SCRIPTS / "laws" / "deepseek-v4.1-flash-tp4-dspark5-mi455-a0.json"
    host = json.loads(law.read_text())["host"]
    terms = host["http_intake"]
    run.write_text(json.dumps({"bound_s": 600.0, "host": host}))
    monkeypatch.setenv(compass_run.ENV, str(run))
    fixed = terms["fixed_s"]
    # No lone request was timed on the node, so the fixed term stands for idle.
    assert compass_run.http_intake() == (fixed, terms["per_byte_s"], fixed)

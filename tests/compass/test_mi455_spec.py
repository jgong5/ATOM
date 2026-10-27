# SPDX-License-Identifier: MIT
"""MI455's machine spec, from probes on an MI455 node: each device field
probed there or MI455X's datasheet, and assumed only what no probe on the node
fills yet."""

import json

from test_probe_device import SCRIPTS, _load
from test_spec_origin import labels

from atom.compass.artifacts import machine_spec_key
from atom.compass.drive import identity
from atom.compass.spec import MachineSpec

MI455 = SCRIPTS / "machines" / "mi455.json"


def _spec(path):
    return MachineSpec.from_mapping(json.loads(path.read_text()))


def _script():
    """`mi455_spec.py`, loaded here: loading it names the machine in the
    shared `mi308x_spec`, which loading another machine's script renames."""
    return _load("mi455_spec")


def test_the_mi455_spec_rebuilds_byte_for_byte(tmp_path):
    out = tmp_path / "mi455.json"
    assert _script().main(["--sweep-from", str(MI455), "--out", str(out)]) == 0
    assert out.read_bytes() == MI455.read_bytes()


def test_each_device_field_is_probed_or_the_datasheet_and_the_rest_assumed():
    mi455, peaks = _spec(MI455), _script().PEAKS
    origins = {label: mi455.origin(label)["origin"] for label in labels(mi455)}
    assert {p for p, o in origins.items() if o == "assumed"} == {
        "host.ipc.zmq_roundtrip_s",
        "host.ipc.shm_broadcast_s",
        "host.http_intake.fixed_s",
        "host.http_intake.per_byte_s",
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
        assert mi455.origin(path)["record"] == record
    for path in ("device.kernel_floor_s", "interconnect.intra_node.link_latency_s"):
        band = mi455.origin(path)
        assert band["record"] == record
        assert band["p10"] <= mi455.value(path) <= band["p90"]
    found, peak = peaks[2], mi455.value("device.compute.fp4_flops")
    assert mi455.value("device.compute.derate.fp4") == found["fp4"] / peak
    assert origins["host.tokenizers[deepseek-v4.1-flash fast]"] == "probed"


#: rocm-smi's listing of GPU 0 on the MI455 node, 2026-10-27: the series is no
#: brand, so the PCI device id names the model.
MI455_CARD = {
    "Card Series": "AMD Radeon Graphics",
    "Card Model": "0x75c7",
    "Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
    "Card SKU": "M4500001",
    "Subsystem ID": "0x75c1",
    "Device Rev": "0x00",
    "Node ID": "2",
    "GUID": "25919",
    "GFX Version": "gfx1250",
}


def test_an_mi455_card_is_the_device_its_spec_states(monkeypatch):
    """What `compass measure memory` compares before it starts an engine."""
    listing = json.dumps({"card0": MI455_CARD})
    monkeypatch.setattr(identity, "smi", lambda *args: listing)
    (card,) = identity._device((0,))
    assert f"{card['name']} {card['arch']}" == (
        machine_spec_key(_spec(MI455)).values["device"]
    )

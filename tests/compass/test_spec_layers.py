# SPDX-License-Identifier: MIT
"""A device datasheet and a network file each hold their own layer's fields,
with each value's origin, and state what the committed machine specs state."""

import json
import pathlib

import pytest

from atom.compass.spec import MachineSpec, SpecRefusal
from atom.compass.spec.layers import DATASHEET, NETWORK, datasheet, network
from atom.compass.spec.rules import Rule

ROOT = pathlib.Path(__file__).resolve().parents[2]
MACHINES = ROOT / "scripts" / "compass" / "machines"
#: Each committed spec, by file stem: the datasheet and the network it states.
LAYERS = {
    "mi308x-hjbog18": ("mi308x", "nic-400g"),
    "mi308x-hjbog39": ("mi308x", "nic-400g"),
    "mi355x-crs005": ("mi355x", "nic-400g"),
    "mi355x-gpu40": ("mi355x", "nic-400g"),
    "mi355x-gpu45": ("mi355x", "nic-400g"),
    "mi455-a0": ("mi455x-a0", "mi455x-scale-out"),
    "mi455-b0": ("mi455x-b0", "mi455x-scale-out"),
}


def test_every_committed_spec_names_its_layers():
    assert {p.stem for p in MACHINES.glob("*.json")} == set(LAYERS)


@pytest.mark.parametrize("stem", sorted(LAYERS))
def test_a_spec_states_its_layers_values_and_origins(stem):
    spec = MachineSpec.from_mapping(json.loads((MACHINES / f"{stem}.json").read_text()))
    sheet, net = LAYERS[stem]
    for layer, paths in ((datasheet(sheet), DATASHEET), (network(net), NETWORK)):
        for path in paths:
            assert layer.values.get(path) == spec.values.get(path), path
            if path in spec.values:
                # A layer file states no subject where the nodes' starts differ.
                assert layer.origins[path].items() <= spec.origin(path).items(), path


def _document(**extra):
    document = {
        "schema_version": 1,
        "name": "test-net",
        "provenance": {
            "authored_by": "test",
            "date": "2026-11-03",
            "origin": "assumed",
            "notes": "chosen for the test",
        },
        "interconnect": {
            "inter_node": {
                "link_bandwidth_bytes_per_s": 5.0e10,
                "link_latency_s": 5.0e-6,
                "derate": 0.8,
            }
        },
    }
    for path, value in extra.items():
        _set(document, path, value)
    return document


def _set(document, path, value):
    *blocks, leaf = path.split(".")
    for block in blocks:
        document = document.setdefault(block, {})
    document[leaf] = value


def _written(tmp_path, document):
    path = tmp_path / "layer.json"
    path.write_text(json.dumps(document))
    return path


def test_a_network_file_reads_by_path(tmp_path):
    layer = network(_written(tmp_path, _document()))
    assert layer.name == "test-net"
    assert layer.origins["interconnect.inter_node.derate"] == {"origin": "assumed"}


@pytest.mark.parametrize(
    "path, value",
    [
        ("device.name", "MI355X"),
        ("interconnect.router_relay_s", 1.5e-3),
        ("interconnect.intra_node.link_latency_s", 2.0e-6),
    ],
)
def test_a_network_file_refuses_another_layers_field_by_name(tmp_path, path, value):
    with pytest.raises(SpecRefusal) as refused:
        network(_written(tmp_path, _document(**{path: value})))
    assert refused.value.rule is Rule.LAYER
    assert f"`{path}`" in refused.value.what


@pytest.mark.parametrize(
    "path, value",
    [
        ("device.compute.derate.bf16", 0.6),
        ("device.kernel_floor_s", 1.5e-6),
        ("interconnect.inter_node.link_latency_s", 5.0e-6),
    ],
)
def test_a_datasheet_refuses_another_layers_field_by_name(tmp_path, path, value):
    document = json.loads((ROOT / "atom/compass/devices/mi355x.json").read_text())
    _set(document, path, value)
    with pytest.raises(SpecRefusal) as refused:
        datasheet(_written(tmp_path, document))
    assert refused.value.rule is Rule.LAYER
    assert f"`{path}`" in refused.value.what


def test_a_value_with_no_origin_is_refused_by_name(tmp_path):
    document = _document(
        **{
            "provenance.origin": "mixed",
            "provenance.origins": {
                "interconnect.inter_node.link_bandwidth_bytes_per_s": {
                    "origin": "assumed"
                },
                "interconnect.inter_node.derate": {"origin": "assumed"},
            },
        }
    )
    with pytest.raises(SpecRefusal) as refused:
        network(_written(tmp_path, document))
    assert refused.value.rule is Rule.TOTALITY
    assert "`interconnect.inter_node.link_latency_s`" in refused.value.what


def test_a_document_with_no_notes_is_refused(tmp_path):
    document = _document()
    del document["provenance"]["notes"]
    with pytest.raises(SpecRefusal) as refused:
        network(_written(tmp_path, document))
    assert "`provenance.notes`" in refused.value.what

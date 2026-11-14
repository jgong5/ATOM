# SPDX-License-Identifier: MIT
"""The layers a person writes: a device datasheet per SKU, and a network file
per scale-out network; and `composed`, the one machine spec a command reads
from a datasheet, a `platform_readings` entry's document and, across nodes, a
network file.

Each is a document holding its own layer's fields at their machine-spec paths,
under the header every spec document has. A field of another layer is refused
by name, as is a value whose origin the document does not record, and a
document with no notes to say where its values came from. A datasheet is named
by its file's stem under `atom/compass/devices/`, a network file by its stem
under `atom/compass/networks/`, or either by path.

`composed` takes the datasheet's fields as they stand and the readings
entry's as follows: a measured entry's rate over the datasheet peak it attains is
that peak's derate, an assumed entry's ratio is the derate as it stands, and
the bandwidth curve is a measured entry's as it stands, or an assumed entry's
ratio curve times the datasheet's bandwidth; the memory derate is the curve's
largest point over that bandwidth. A peak the entry gives no derate for is
left out. The entry's key states the stack, torch and PTL state; the spec
pins them, so a reader keys what it priced on by the spec it read.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fields import BAND, BY_PATH, SCHEMA, check
from .machine import (
    MachineSpec,
    _check_domain,
    _stated,
    _version,
    _walk,
    check_subject,
    field_origin,
)
from .rules import Rule, SpecRefusal

ROOT = Path(__file__).resolve().parents[1]
FORMATS = ("bf16", "fp8", "fp4")
CURVE = "device.memory.bandwidth_curve"
BANDWIDTH = "device.memory.bandwidth_bytes_per_s"
#: Each rate a measured entry holds, to the datasheet peak it attains and the
#: derate it composes into, which an assumed entry holds in its place.
RATES = {
    **{
        f"device.compute.attainable_flops.{f}": (
            f"device.compute.{f}_flops",
            f"device.compute.derate.{f}",
        )
        for f in FORMATS
    },
    "interconnect.intra_node.attainable_bandwidth_bytes_per_s": (
        "interconnect.intra_node.link_bandwidth_bytes_per_s",
        "interconnect.intra_node.derate",
    ),
}
#: An assumed entry's bandwidth curve, over its source's datasheet bandwidth.
RATIO_CURVE = "device.memory.bandwidth_derate_curve"
#: What a readings document states about itself.
READINGS_HEADER = ("schema_version", "name", "provenance")
#: What every layer document states about itself.
HEADER = (
    "schema_version",
    "name",
    "provenance.authored_by",
    "provenance.date",
    "provenance.origin",
    "provenance.subject",
    "provenance.origins",
    "provenance.notes",
)
#: What varies with the SKU alone.
DATASHEET = (
    "device.name",
    "device.arch",
    "device.count_per_node",
    "device.memory.capacity_bytes",
    "device.memory.bandwidth_bytes_per_s",
    "device.compute.bf16_flops",
    "device.compute.fp8_flops",
    "device.compute.fp4_flops",
    "interconnect.intra_node.topology",
    "interconnect.intra_node.domain_devices",
    "interconnect.intra_node.link_bandwidth_bytes_per_s",
)
#: What varies with the scale-out network alone.
NETWORK = (
    "interconnect.inter_node.link_bandwidth_bytes_per_s",
    "interconnect.inter_node.link_latency_s",
    "interconnect.inter_node.link_setup_s",
    "interconnect.inter_node.derate",
)


@dataclass(frozen=True, slots=True)
class Layer:
    """One layer document, checked: each field it states by dotted path, and
    that field's origin, with a probed field's subject and band."""

    name: str
    notes: str
    values: Mapping[str, Any]
    origins: Mapping[str, dict]


def datasheet(name_or_path: str | Path) -> Layer:
    """A device datasheet, by SKU or by path, or a refusal."""
    sheet = _read("device datasheet", DATASHEET, ROOT / "devices", name_or_path)
    _check_domain(sheet.values)
    return sheet


def network(name_or_path: str | Path) -> Layer:
    """A network file, by the network's name or by path, or a refusal."""
    return _read("network file", NETWORK, ROOT / "networks", name_or_path)


def _locate(directory: Path, name_or_path: str | Path) -> Path:
    path = Path(name_or_path)
    if isinstance(name_or_path, Path) or "/" in name_or_path or path.suffix == ".json":
        return path
    return directory / f"{name_or_path}.json"


def _read(kind: str, fields: tuple, directory: Path, name_or_path) -> Layer:
    path = _locate(directory, name_or_path)
    document = json.loads(path.read_text())
    holder = f"{kind} {path}"
    _version(document)
    found: dict[str, Any] = {}
    _walk(document, "", found)
    for stated in found:
        if stated not in HEADER and stated not in fields:
            raise SpecRefusal(
                Rule.LAYER,
                f"{holder} states `{stated}`, which is not a {kind}'s field",
                f"state it in the layer that names what it varies with; a "
                f"{kind} holds {', '.join(f'`{f}`' for f in fields)}",
            )
    values = {}
    for stated in (*HEADER, *fields):
        if _stated(BY_PATH[stated], found):
            values[stated] = check(BY_PATH[stated], found[stated], stated)
    if "provenance.notes" not in values:
        raise SpecRefusal(
            Rule.TOTALITY,
            f"{holder} states no `provenance.notes`",
            "say where its values came from: the document read and when, the "
            "probe and what it ran on, or why an assumed value was chosen",
        )
    check_subject(values, holder)
    own = {f: values[f] for f in fields if f in values}
    origins = {f: field_origin(values, f, holder) for f in own}
    return Layer(values["name"], values["provenance.notes"], own, origins)


def stack_of(fields: Mapping) -> str:
    """The stack and torch a key's `fields` state, as `--stack` names them."""
    return f"{fields['software_stack']} / torch {fields['torch']}"


def composed(
    sheet: Layer, key: Any, document: Mapping, net: Layer | None = None
) -> MachineSpec:
    """The machine spec of the datasheet `sheet`, the readings entry at `key`,
    an artifact store's `Key`, whose document is `document`, and the network
    file `net`."""
    stated = document["provenance"]["origins"]
    values, origins = dict(sheet.values), dict(sheet.origins)
    for path, value in document.items():
        if path in READINGS_HEADER:
            continue
        origin = stated[path]
        if path in RATES:
            peak, path = RATES[path]
            if peak not in sheet.values:
                continue
            value = value / sheet.values[peak]
            origin = {
                k: v / sheet.values[peak] if k in BAND else v for k, v in origin.items()
            }
        elif path == RATIO_CURVE:
            path = CURVE
            value = [[size, r * sheet.values[BANDWIDTH]] for size, r in value]
        elif BY_PATH[path].of and BY_PATH[path].of not in sheet.values:
            continue  # an assumed derate of a peak this datasheet omits
        values[path], origins[path] = value, origin
    if CURVE in values:
        values["device.memory.derate"] = values[CURVE][-1][1] / values[BANDWIDTH]
        origins["device.memory.derate"] = origins[CURVE]
    for field in SCHEMA:
        if field.of and field.path not in values:
            values.pop(field.of, None)
            origins.pop(field.of, None)
    pinned = dict(p.split(" ", 1) for p in key.values["software_stack"].split(" / "))
    entry = {"origin": document["provenance"]["origin"]}
    for component, version in (pinned | {"torch": key.values["torch"]}).items():
        values[f"device.software_pinned_to.{component}"] = version
        origins[f"device.software_pinned_to.{component}"] = entry
    values["device.ptl"], origins["device.ptl"] = key.values["ptl"], entry
    notes = (
        f"composed from the device datasheet {sheet.name} and {key} as {key.token()}"
    )
    if net is not None:
        values |= net.values
        origins |= net.origins
        notes += f", across nodes on the network file {net.name}"
    values |= {
        "schema_version": 1,
        "name": f"{sheet.name} at {key.token()}",
        "provenance.authored_by": "compass",
        "provenance.date": document["provenance"]["date"],
        "provenance.origin": "mixed",
        "provenance.origins": origins,
        "provenance.notes": notes,
    }
    return MachineSpec.from_mapping(MachineSpec(values).echo())

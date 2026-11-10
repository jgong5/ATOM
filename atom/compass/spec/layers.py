# SPDX-License-Identifier: MIT
"""The layers a person writes: a device datasheet per SKU, and a network file
per scale-out network.

Each is a document holding its own layer's fields at their machine-spec paths,
under the header every spec document has. A field of another layer is refused
by name, as is a value whose origin the document does not record, and a
document with no notes to say where its values came from. A datasheet is named
by its file's stem under `atom/compass/devices/`, a network file by its stem
under `atom/compass/networks/`, or either by path.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fields import BY_PATH, check
from .machine import (
    _check_domain,
    _stated,
    _survey,
    _version,
    check_subject,
    field_origin,
)
from .rules import Rule, SpecRefusal

ROOT = Path(__file__).resolve().parents[1]
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
    for refusal in _survey(document, "", found):
        raise refusal
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

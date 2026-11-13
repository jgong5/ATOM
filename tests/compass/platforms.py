# SPDX-License-Identifier: MIT
"""A machine spec document's platform, as the commands that price read one:
its datasheet fields written as a datasheet file, and the rest published as
a `platform_readings` entry of one node: a probed derate as the rate it is of
its peak, as a measured entry holds it, and an assumed one as the ratio, as
an assumed entry holds it. A spec that states no bandwidth curve gets one point
at its memory derate, which the composition derives the derate from."""

import json
from pathlib import Path

from test_artifact_invalidation import STANZA

from atom.compass.artifacts import (
    ArtifactStore,
    Conditions,
    GateState,
    Key,
    Kind,
    Topology,
    machine_platform,
    member_name,
)
from atom.compass.spec import MachineSpec
from atom.compass.spec.fields import BAND
from atom.compass.spec.layers import BANDWIDTH, CURVE, DATASHEET, NETWORK, RANK, RATES

#: What a spec states that no readings entry holds: the document's header, the
#: stack and PTL state its key states, the derates the composition makes, the
#: router's hop.
NOT_READINGS = (
    "schema_version",
    "name",
    "provenance.",
    "device.software_pinned_to.",
    "device.ptl",
    "device.memory.derate",
    "interconnect.router_relay_s",
    *DATASHEET,
    *NETWORK,
)
#: Each derate a readings entry holds as a rate: the rate, and its peak.
DERATES = {derate: (rate, peak) for rate, (peak, derate) in RATES.items()}


def flags(store, document, directory, at=None) -> list[str]:
    """Write `document`'s datasheet in `directory` and publish its readings in
    the store at `store`, once, at the stack and torch `at`, an entry key's
    fields, states, else at its own; the `--device` and `--ptl` that name them."""
    spec = MachineSpec.from_mapping(document)
    v = spec.values
    readings, origins = {}, {}
    for field, value in v.items():
        if field.startswith(NOT_READINGS):
            continue
        origin = spec.origin(field)
        path, scale = field, 1.0
        if field in DERATES and origin["origin"] != "assumed":
            path, peak = DERATES[field]
            scale = v[peak]
        readings[path] = value * scale if path != field else value
        origins[path] = {k: x * scale if k in BAND else x for k, x in origin.items()}
    if CURVE not in readings:
        readings[CURVE] = [[1 << 32, v["device.memory.derate"] * v[BANDWIDTH]]]
        origins[CURVE] = spec.origin("device.memory.derate")
    sheet = _layer(Path(directory) / f"{v['name']}.datasheet.json", spec, DATASHEET)
    platform = machine_platform(spec)
    if at:
        platform |= {k: at[k] for k in ("software_stack", "torch")}
    key = Key.of(Kind.PLATFORM_READINGS, **platform)
    held = ArtifactStore(store)
    if not held.directory_for(key).exists():
        stated = {o["origin"] for o in origins.values()}
        document = {
            "schema_version": 1,
            "name": f"{v['name']} readings",
            "provenance": {
                "authored_by": "tests/compass/platforms.py",
                "date": v["provenance.date"],
                "origin": "assumed" if stated == {"assumed"} else "probed",
                "origins": origins,
            },
            **readings,
        }
        held.publish(
            key,
            provenance=STANZA,
            topology=Topology(),
            conditions=Conditions.of(
                **platform,
                atom_src="no ATOM engine code",
                model="no model",
                host_cpu="no host CPU term",
                engine_config="no engine",
            ),
            gates=GateState.of(),
            members={
                member_name("readings", RANK, "json"): json.dumps(document).encode()
            },
        )
    return ["--device", str(sheet), "--ptl", platform["ptl"]]


def network(document, directory) -> list[str]:
    """Write `document`'s inter-node link as a network file in `directory`; the
    `--network` that names it."""
    spec = MachineSpec.from_mapping(document)
    path = Path(directory) / f"{spec.values['name']}.network.json"
    return ["--network", str(_layer(path, spec, NETWORK))]


def _layer(path: Path, spec: MachineSpec, fields) -> Path:
    """`spec`'s `fields` written at `path` as a layer document, each with the
    origin `spec` records for it."""
    stated = [p for p in fields if p in spec.values]
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": spec.values["name"],
                "provenance": {
                    "authored_by": "tests/compass/platforms.py",
                    "date": spec.values["provenance.date"],
                    "origin": "mixed",
                    "origins": {p: spec.origin(p) for p in stated},
                    "notes": "a test's",
                },
                **{p: spec.values[p] for p in stated},
            }
        )
    )
    return path

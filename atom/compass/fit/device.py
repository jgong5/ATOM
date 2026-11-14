# SPDX-License-Identifier: MIT
"""`compass fit device`: a platform's readings, folded from its nodes' probe
records, or assumed from another device's for a device never run.

    compass fit device RECORD... --store S --out D
    compass fit device --device DATASHEET --assume-from DEVICE --ptl P
        --store S --out D

Each RECORD is a `probe_record` in S, named by the token its key prints. The
`platform_readings` entry published in S, keyed by the records' device, stack,
PTL state and torch, holds each reading `measure device` took as the median
over the records stating it, its p10 and p90 the lowest p10 and the highest
p90 among them, and its subject those records; the bandwidth curve as the
median at each size; a PP carrier's term as the median over the records
holding one, its band their lowest and highest value. Rates stay rates, and
the entry names no datasheet. Records of two devices, stacks, PTL states or
torches refuse, naming each record's value, and so does a key S already holds.

With `--assume-from`, DEVICE is a device name a datasheet under
`atom/compass/devices/` states. The one measured `platform_readings` entry S
holds for it at `--ptl` is taken by `assumed`, and published for the device
`--device` names, a datasheet by SKU or path, keyed by the source's stack and
torch, every field assumed. None, or several, refuse, listing them.

`--out` gets a readable copy of the entry, ``platform_readings.json``, with
its key and token.

A command that prices reads its platform through `compose`: the datasheet
it names, the one entry S holds for that device (`readings`), and, across
nodes, a network file, composed by `layers.composed`.
"""

import argparse
import datetime
import json
import statistics
from collections.abc import Mapping
from pathlib import Path

from atom.compass.artifacts import (
    PLATFORM,
    ArtifactStore,
    Conditions,
    GateState,
    Key,
    Kind,
    Provenance,
    Topology,
    member_name,
    roots_for,
)
from atom.compass.spec import MachineSpec
from atom.compass.spec.layers import (
    BANDWIDTH,
    CURVE,
    RATES,
    RATIO_CURVE,
    READINGS_HEADER,
    ROOT,
    composed,
    datasheet,
    network,
    stack_of,
)

PROG = "compass fit device"
#: The rank a readings entry's one member is written at.
RANK = Topology().ranks()[0]
PP = "interconnect.intra_node.pp"
#: What a spec at another device's ratios takes from that device's spec as it
#: stands, no datasheet stating it.
CARRIED = (
    "device.kernel_floor_s",
    "device.software_pinned_to.",
    "interconnect.intra_node.link_latency_s",
    "interconnect.inter_node.link_latency_s",
)


def fold(records: Mapping[str, dict]) -> tuple[dict, dict]:
    """The readings and their origins folded from `records`, each a probe
    record's document by its token."""
    flat = {token: _flat(r["readings"]) for token, r in records.items()}
    values, origins = {}, {}
    for path in sorted({p for held in flat.values() for p in held}):
        stating = {t: r[path] for t, r in flat.items() if path in r}
        origins[path] = {"origin": "probed", "record": ", ".join(stating)}
        if path == CURVE:
            at: dict[int, list] = {}
            for curve in stating.values():
                for size, rate in curve:
                    at.setdefault(size, []).append(rate)
            values[path] = [[s, statistics.median(r)] for s, r in sorted(at.items())]
            continue
        medians = [m for m, _, _ in stating.values()]
        # A byte count stays one a record measured.
        middle = statistics.median_low if type(medians[0]) is int else statistics.median
        values[path] = middle(medians)
        origins[path]["p10"] = min(lo for _, lo, _ in stating.values())
        origins[path]["p90"] = max(hi for _, _, hi in stating.values())
    return values, origins


def _flat(readings: Mapping) -> dict:
    """A record's readings by field path, each PP carrier's term as
    [value, value, value], since its fit states no band."""
    flat = {p: v for p, v in readings.items() if p != PP}
    for carrier, terms in readings.get(PP, {}).items():
        for name, value in terms.items():
            if not isinstance(value, list):  # the fit's residual spread
                flat[f"{PP}.{carrier}.{name}"] = [value] * 3
    return flat


def assumed(readings: Mapping, sheet: Mapping) -> dict:
    """A measured entry's `readings` as a device never run takes them: each
    rate over the peak `sheet`, its datasheet, states (`RATES`), and the
    bandwidth curve over its bandwidth; every other reading as it stands. A
    rate whose peak `sheet` does not state is left out."""
    out = {}
    for path, value in readings.items():
        if path == CURVE:
            out[RATIO_CURVE] = [[s, r / sheet[BANDWIDTH]] for s, r in value]
        elif path in RATES:
            peak, ratio = RATES[path]
            if peak in sheet:
                out[ratio] = value / sheet[peak]
        else:
            out[path] = value
    return out


def _records(store: ArtifactStore, tokens: list[str]) -> tuple[dict, dict]:
    """The probe records `tokens` name, by token, and their one platform."""
    records, platforms = {}, {}
    for token in tokens:
        key = store.key_from_token(Kind.PROBE_RECORD, token)
        body = store.read(key).read_member("record", RANK, "json")
        records[token] = json.loads(body)
        platforms[token] = {a.field: key.values[a.field] for a in PLATFORM}
    for axis in PLATFORM:
        seen = {p[axis.field] for p in platforms.values()}
        if len(seen) > 1:
            each = "; ".join(f"{t} {p[axis.field]!r}" for t, p in platforms.items())
            raise ValueError(
                f"the records are of {len(seen)} {axis.field} values: {each}; "
                "a platform's readings fold records of that one platform"
            )
    return records, platforms[tokens[0]]


def readings(
    store: ArtifactStore, device: str, ptl: str, stack: str | None = None
) -> tuple[Key, dict]:
    """The one `platform_readings` entry `store` holds for `device`, `<name>
    <arch>`, at PTL state `ptl` and, when given, at `stack` (`stack_of`), and
    its document; none or several refuse, listing each."""
    folder = store.root / Kind.PLATFORM_READINGS.value
    keys = [
        Key.of(Kind.PLATFORM_READINGS, **json.loads(p.read_text())["key"])
        for p in sorted(folder.glob("[!.]*/entry.json"))
    ]
    found = [
        k
        for k in keys
        if (k.values["device"], k.values["ptl"]) == (device, ptl)
        and stack in (None, stack_of(k.values))
    ]
    if len(found) != 1:
        at = f" at {stack}" if stack else ""
        listed = "".join(f"; {stack_of(k.values)} as {k.token()}" for k in found)
        remedy = (
            "--stack names one"
            if found
            else "compass measure device on each node and compass fit device "
            "publish one"
        )
        raise ValueError(
            f"{store.root} holds {len(found)} platform_readings entries for "
            f"{device} at PTL {ptl}{at}{listed}; {remedy}"
        )
    body = store.read(found[0]).read_member("readings", RANK, "json")
    return found[0], json.loads(body)


def compose(
    device: str | Path,
    store: ArtifactStore,
    ptl: str,
    stack: str | None = None,
    network_file: str | Path | None = None,
) -> MachineSpec:
    """The machine spec of the datasheet `device` names, by SKU or path, the
    one readings entry `store` holds for it at `ptl` and `stack` (`readings`),
    and the network file `network_file` names, when it names one."""
    sheet = datasheet(device)
    key, document = readings(store, _device(sheet), ptl, stack)
    net = None if network_file is None else network(network_file)
    return composed(sheet, key, document, net)


def assumed_entry(
    store: ArtifactStore, device: str, ptl: str, stack: str | None = None
) -> tuple[Key, dict, object]:
    """The one measured `platform_readings` entry `store` holds for the device
    named `device` at `ptl` and `stack` (`readings`), its document as a
    device never run takes it, `assumed`'s readings every one assumed, and the
    datasheet stating `device`."""
    source = _named(device)
    found, document = readings(store, _device(source), ptl, stack)
    if document["provenance"]["origin"] != "probed":
        raise ValueError(
            f"{found} as {found.token()} is assumed itself; assume from a "
            "measured entry, whose rates its datasheet divides"
        )
    measured = {k: v for k, v in document.items() if k not in READINGS_HEADER}
    values = assumed(measured, source.values)
    provenance = document["provenance"] | {
        "origin": "assumed",
        "origins": {p: {"origin": "assumed"} for p in values},
    }
    header = {k: document[k] for k in READINGS_HEADER}
    return found, header | values | {"provenance": provenance}, source


def _named(device: str):
    """The datasheet under `atom/compass/devices/` stating device name `device`."""
    sheets = [datasheet(p) for p in sorted((ROOT / "devices").glob("*.json"))]
    named = [s for s in sheets if s.values["device.name"] == device]
    if len(named) != 1:
        names = ", ".join(s.values["device.name"] for s in sheets)
        raise ValueError(
            f"{len(named)} datasheets state device {device!r}; they state {names}"
        )
    return named[0]


def _device(sheet) -> str:
    return f"{sheet.values['device.name']} {sheet.values['device.arch']}"


def main(argv: list[str]) -> None:
    args = _parser().parse_args(argv)
    store = ArtifactStore(args.store)
    date = datetime.datetime.now(datetime.timezone.utc)
    if args.assume_from:
        if args.records or not (args.device and args.ptl):
            raise ValueError("--assume-from takes --device and --ptl and no RECORD")
        found, document, source = assumed_entry(store, args.assume_from, args.ptl)
        values = {k: v for k, v in document.items() if k not in READINGS_HEADER}
        target = datasheet(args.device)
        platform = found.values | {"device": _device(target)}
        origins = document["provenance"]["origins"]
        notes = (
            f"no {platform['device']} card ran: {found} as {found.token()} over "
            f"the {source.name} datasheet; the stack and torch are that entry's"
        )
    else:
        if not args.records or args.device or args.ptl:
            raise ValueError("name the probe records, or --assume-from a device")
        records, platform = _records(store, args.records)
        values, origins = fold(records)
        hosts = ", ".join(f"{r['host']} as {t}" for t, r in records.items())
        notes = f"folded from the probe records of {hosts}"
    key = Key.of(Kind.PLATFORM_READINGS, **platform)
    document = {
        "schema_version": 1,
        "name": f"{platform['device']} readings",
        "provenance": {
            "authored_by": PROG,
            "date": date.date().isoformat(),
            "origin": "assumed" if args.assume_from else "probed",
            "origins": origins,
            "notes": notes,
        },
        **values,
    }
    store.publish(
        key,
        provenance=Provenance(
            " ".join([PROG, *argv]),
            date.isoformat(),
            roots_for({"atom": "atom", "aiter": "aiter"}),
        ),
        topology=Topology(),
        # A fit runs no model, engine or ATOM engine code: no row reads them.
        conditions=Conditions.of(
            **platform,
            atom_src="no ATOM engine code",
            model="no model",
            host_cpu="no host CPU term",
            engine_config="no engine",
        ),
        gates=GateState.of(),
        members={member_name("readings", RANK, "json"): json.dumps(document).encode()},
        notes=notes,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    copy = {"key": key.values, "token": key.token(), **document}
    (out / "platform_readings.json").write_text(json.dumps(copy, indent=1))
    print(f"published {key} as {key.token()}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    p.add_argument(
        "records", nargs="*", metavar="RECORD", help="a probe record's token"
    )
    p.add_argument("--device", help="with --assume-from: the datasheet, SKU or path")
    p.add_argument("--assume-from", help="the device name whose readings to assume")
    p.add_argument("--ptl", help="with --assume-from: the PTL state to take")
    p.add_argument("--store", required=True, help="the artifact store's root")
    p.add_argument("--out", required=True, help="a directory for the readable copy")
    return p

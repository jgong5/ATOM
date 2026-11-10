# SPDX-License-Identifier: MIT
"""Move artifacts keyed by a machine-spec digest to keys that state their platform.

    python3 scripts/compass/migrate_keys.py [--store OLD NEW] [--laws DIR]

A key used to state `machine_spec`, the digest of one spec's device, software
stack and PTL state; it now states each platform field its kind's row marks,
by value. A digest is read back as the platform of the spec in `MACHINES` it
names, and an entry or law whose digest names none is listed and left as it
is.

`--store OLD NEW` republishes every entry of the store OLD into NEW at its key
of today, its members, provenance, gates and fingerprints as they are, its
notes stating the key and the entry digest it came from. An entry that states
no digest takes the platform fields its key lacks from the readings its
fingerprints recorded. An entry listed is one whose platform is not known, whose
fingerprints are not the ones its kind's rows record today, or that NEW
refuses, each with the reason.

`--laws DIR` rewrites each law file there in place: its `machine_spec` becomes
its `platform`, and its host key's the platform fields a `host_laws` key states.

Each entry and law prints one line, MOVED or LISTED, and the counts end it.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom.compass.artifacts import (
    KEY_FIELDS,
    ArtifactRefusal,
    ArtifactStore,
    Axis,
    Conditions,
    GateState,
    Key,
    Kind,
    Provenance,
    Reading,
    Topology,
    fingerprint,
    keyed_platform,
    machine_platform,
    rows_for,
)
from atom.compass.spec import MachineSpec

MACHINES = Path(__file__).resolve().parent / "machines"
#: The key field that stated the digest.
OLD = "machine_spec"


def platforms(machines: Path = MACHINES) -> dict[str, dict]:
    """Each spec's platform in `machines`, by the digest a key stated for it:
    the sha256 of its canonical `machine_spec` key, its device, software stack
    and PTL state."""
    found = {}
    for path in sorted(machines.glob("*.json")):
        platform = machine_platform(
            MachineSpec.from_mapping(json.loads(path.read_text()))
        )
        old = {f: platform[f] for f in ("device", "software_stack", "ptl")}
        canonical = json.dumps(
            {"kind": OLD, "key": old}, sort_keys=True, separators=(",", ":")
        )
        found["sha256:" + hashlib.sha256(canonical.encode()).hexdigest()] = platform
    return found


def _swap(mapping: dict, new: dict) -> dict:
    """`mapping` with its `OLD` field replaced, where it stood, by `new`'s fields."""
    out = {}
    for name, value in mapping.items():
        out |= new if name == OLD else {name: value}
    return out


def law(path: Path, known: dict) -> str:
    """Rewrite the law file `path` with the platforms `known` by digest."""
    document = json.loads(path.read_text())
    digests = {document.get(OLD)}
    if "host" in document:
        digests.add(document["host"]["key"].get(OLD))
    if not digests <= known.keys():
        return (
            f"LISTED {path.name}: its digests {sorted(map(str, digests))} name no spec"
        )
    platform = known[document[OLD]]
    document = _swap(
        document, {"platform": keyed_platform(Kind.STEP_LINEAR_LAW, platform)}
    )
    if "host" in document:
        key = document["host"]["key"]
        document["host"]["key"] = _swap(
            key, keyed_platform(Kind.HOST_LAWS, known[key[OLD]])
        )
    path.write_text(json.dumps(document) + "\n")
    return f"MOVED {path.name}"


def entry(directory: Path, known: dict, into: ArtifactStore) -> str:
    """Republish the entry in `directory` into `into` at its key of today."""
    raw = (directory / "entry.json").read_bytes()
    document = json.loads(raw)
    kind, stated = Kind(document["kind"]), document["key"]
    where = f"{kind} {directory.name}"
    cells = {
        name: Reading.from_json(cell)
        for recorded in document["fingerprints"].values()
        for name, cell in recorded["cells"].items()
    }
    if OLD in stated:
        if stated[OLD] not in known:
            return f"LISTED {where}: digest {stated[OLD]} names no spec in {MACHINES}"
        platform = known[stated[OLD]]
    else:
        platform = {name: read.value for name, read in cells.items()}
    values = {k: v for k, v in stated.items() if k != OLD}
    missing = [f for f in KEY_FIELDS[kind] if f not in values and f not in platform]
    if missing:
        return f"LISTED {where}: no reading of {', '.join(missing)}"
    values |= {f: platform[f] for f in KEY_FIELDS[kind] if f not in values}
    unread = Reading("unread", "no row of this kind depends on it")
    conditions = Conditions.of(**{a.field: cells.get(a.field, unread) for a in Axis})
    prints = {r.field: fingerprint(r, conditions).as_json() for r in rows_for(kind)}
    if prints != document["fingerprints"]:
        return f"LISTED {where}: its fingerprints are not its kind's rows today"
    key = Key.of(kind, **values)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    origin = (
        f"migrated from {kind} key {json.dumps(stated, sort_keys=True)}, entry {digest}"
    )
    try:
        into.publish(
            key,
            provenance=Provenance.from_json(document["provenance"]),
            topology=Topology.from_mapping(document["topology"]),
            conditions=conditions,
            gates=GateState.from_json(document["gates"]),
            members={n: (directory / n).read_bytes() for n in document["members"]},
            notes="\n".join(filter(None, (document["notes"], origin))),
        )
    except ArtifactRefusal as refused:
        return f"LISTED {where}: {refused}"
    return f"MOVED {where} -> {key.token()}"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", nargs=2, metavar=("OLD", "NEW"))
    p.add_argument("--laws", type=Path, metavar="DIR")
    args = p.parse_args(argv)
    known, lines = platforms(), []
    if args.store:
        old, into = Path(args.store[0]), ArtifactStore(args.store[1])
        for directory in sorted(old.glob("*/[!.]*")):
            if (directory / "entry.json").is_file():
                lines.append(entry(directory, known, into))
    for path in sorted(args.laws.glob("*.json")) if args.laws else ():
        lines.append(law(path, known))
    for line in lines:
        print(line)
    moved = sum(line.startswith("MOVED") for line in lines)
    print(f"{moved} moved, {len(lines) - moved} listed")


if __name__ == "__main__":
    main()

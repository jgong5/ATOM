# SPDX-License-Identifier: MIT
"""Publish MI455's memory_readings entry of one configuration, every reading
assumed, copied from MI355X's entry of the same configuration.

    python3 scripts/compass/mi455_readings.py --store S TOKEN

No engine starts on a device that never ran, so `compass measure memory`
cannot take MI455's readings. TOKEN names an entry it published in the store
S, keyed by `machines/mi355x-gpu45.json`. This publishes that entry's members
again in S, at the key `machines/mi455.json` gives the same configuration,
each stating `assumed`: the entry it was copied from. A run on MI455 reads it,
and every term built on it is declared. The copy states MI455's device, stack
and PTL state and keeps everything else the original was published under, the
ATOM tree included, so it answers where the original does.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from atom.compass.artifacts import (
    ArtifactStore,
    Conditions,
    Key,
    Kind,
    Provenance,
    Row,
    machine_spec_digest,
    machine_spec_key,
    member_name,
)
from atom.compass.run import host_cpu
from atom.compass.run.memory import EXTENSION, STEM
from atom.compass.spec import MachineSpec

MACHINES = Path(__file__).resolve().parent / "machines"
MI355X, MI455 = MACHINES / "mi355x-gpu45.json", MACHINES / "mi455.json"


def _spec(path: Path) -> MachineSpec:
    return MachineSpec.from_mapping(json.loads(path.read_text()))


def publish(store: str, token: str, command: str) -> Key:
    """Publish the members of MI355X's entry `token` in `store` again at
    MI455's key of its configuration, each stating where it was copied from."""
    store = ArtifactStore(store)
    found = store.read(store.key_from_token(Kind.MEMORY_READINGS, token))
    fields = dict(found.key.values)
    if fields.pop("machine_spec") != machine_spec_digest(_spec(MI355X)):
        raise ValueError(
            f"{found.key} is not keyed by {MI355X.name}; name an entry "
            "compass measure memory took on mi355-gpu-45"
        )
    machine = _spec(MI455)
    key = Key.of(
        Kind.MEMORY_READINGS, machine_spec=machine_spec_digest(machine), **fields
    )
    assumed = f"copied from {MI355X.stem}'s memory_readings entry {token}"
    members = {}
    for coords in found.topology.ranks():
        member = json.loads(found.read_member(STEM, coords, EXTENSION))
        member["assumed"] = assumed
        members[member_name(STEM, coords, EXTENSION)] = json.dumps(member).encode()
    held = {a.field: r for a, r in found.fingerprints[Row.MEMORY_READINGS].cells}
    platform = machine_spec_key(machine).values | {"host_cpu": host_cpu()}
    platform["torch"] = machine.value("device.software_pinned_to.torch")
    store.publish(
        key,
        provenance=Provenance(
            command,
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            found.provenance.source_roots,
        ),
        topology=found.topology,
        conditions=Conditions.of(**held | platform),
        gates=found.gate_state,
        members=members,
    )
    print(f"published {key} as {key.token()}")
    return key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", required=True, help="the artifact store's root")
    parser.add_argument("token", help="MI355X's entry, as measure memory printed it")
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    publish(args.store, args.token, " ".join(["mi455_readings.py", *argv]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

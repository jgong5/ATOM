# SPDX-License-Identifier: MIT
"""Probe one GPU host's attainable peaks and print them as a spec fragment.

    HIP_VISIBLE_DEVICES=a,b python3 scripts/compass/probe_device.py SPEC OUT

SPEC is the machine file the peaks belong to. Card 0 runs the one-card probes,
cards 0 and 1 the link; `atom/compass/drive/device_probe.py` holds the probes,
what each one samples, and the cards they refuse. OUT is the record: the cards,
the stack (ROCm, aiter's commit, RCCL and torch), the capacity torch reads on
card 0, every sample and the text of this file and of the probes' module; the
fragment printed names its sha256.
"""

import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from atom.compass.artifacts.keys import NO_PTL
from atom.compass.drive import device_probe
from atom.compass.drive.device_probe import (
    BANDED,
    FORMATS,
    cards,
    probe,
    readings,
    refuse_unfit,
)
from atom.compass.drive.gpu import kfd_pid
from atom.compass.spec import Fragment, MachineSpec


def fragment(
    name: str, found: dict, peaks: dict, author: str, date: str, subject: dict
) -> dict:
    """The fragment `readings` make on machine `name`, whose spec-peak numbers
    `peaks` holds by path, every field probed on `subject`; a banded field
    the readings lack, as a record taken before its probe does, is left out."""
    curve = found["bandwidth_curve"]
    banded = [path for path in BANDED if path in found]
    values = {
        **{
            f"device.compute.derate.{f}": found[f] / peaks[f"device.compute.{peak}"]
            for f, peak in FORMATS.items()
            if f in found
        },
        "device.memory.bandwidth_curve": curve,
        "device.memory.derate": curve[-1][1]
        / peaks["device.memory.bandwidth_bytes_per_s"],
        "interconnect.intra_node.derate": found["link"]
        / peaks["interconnect.intra_node.link_bandwidth_bytes_per_s"],
        **{path: found[path][0] for path in banded},
    }
    origins = {path: {"origin": "probed", **subject} for path in values}
    for path in banded:
        origins[path] |= {"p10": found[path][1], "p90": found[path][2]}
    document = {
        "schema_version": 1,
        "name": name,
        "provenance": {
            "authored_by": author,
            "date": date,
            "origin": "mixed",
            "origins": origins,
            "notes": "probe_device.py: GEMM, read, write, copy, graph and link samples",
        },
    }
    for path, value in values.items():
        *blocks, leaf = path.split(".")
        node = document
        for block in blocks:
            node = node.setdefault(block, {})
        node[leaf] = value
    return document


def main(argv: list[str] | None = None) -> int:
    import torch

    from atom.compass.artifacts.provenance import git_tree_root
    from atom.compass.drive.identity import _stack

    spec_path, out = (Path(a) for a in (argv or sys.argv[1:]))
    kfd_pid()  # this process's KFD entry, taken before torch opens a card
    spec = MachineSpec.from_mapping(json.loads(spec_path.read_text()))
    ptl = spec.values.get("device.ptl", NO_PTL)
    props = [torch.cuda.get_device_properties(i) for i in (0, 1)]
    bdfs = [
        f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
        for p in props
    ]
    record = {"cards": cards(bdfs), "arch": props[0].gcnArchName, "stack": _stack()}
    refuse_unfit(record["cards"], ptl)
    stated = [
        f for f, peak in FORMATS.items() if f"device.compute.{peak}" in spec.values
    ]
    record |= probe(torch, stated)
    for name, error in record.get("failed", {}).items():
        print(f"{name} failed: {error}", file=sys.stderr)
    record["source"] = "".join(
        Path(f).read_text() for f in (__file__, device_probe.__file__)
    )
    out.write_text(json.dumps(record) + "\n")
    root = git_tree_root("atom", str(ROOT))
    subject = {"source_root": f"{root.revision_kind}:{root.revision}"}
    subject["record"] = f"sha256:{hashlib.sha256(out.read_bytes()).hexdigest()}"
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    document = fragment(
        spec.value("name"),
        readings(record),
        spec.values,
        "probe_device.py",
        today,
        subject,
    )
    Fragment.from_mapping(document, "probe")  # refuses a derate above 1
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

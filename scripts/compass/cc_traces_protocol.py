"""Register, verify and stamp the cc-traces acceptance protocol.

`atom/compass/CC_TRACES_PROTOCOL.md` governs the final end-to-end acceptance
matrix. `atom/compass/PROTOCOL.md` governs the prepare-then-measure cells that
came before it and keeps its own registration, its own lock and its own stamped
results -- none of which this touches. Two protocols, two locks, and a cell
carries whichever one it ran under.

The registration mechanism is not reimplemented here. `protocol.py` already
decides what a registration is: the digest of the document, the instant stated
rather than read, and the lock it replaces kept inside the new one so the
history of what was claimed stays readable. Restating that would give two
implementations to keep in agreement, and the one that drifted would be the one
nobody was reading. So this loads `protocol.py` as a private module instance
and points it at the cc-traces document and lock.

Private on purpose: the instance is loaded under its own name, so rebinding its
paths cannot reach a `protocol.py` some other caller in the same process is
using for the legacy registration.

The stamp is this file's own, for one reason: a cell stamped by both would have
two files called `protocol.json` and the second would overwrite the first. A
cc-traces cell gets `cc_traces_protocol.json` beside it.

    python scripts/compass/cc_traces_protocol.py register --at 2026-09-11T16:00:00Z
    python scripts/compass/cc_traces_protocol.py verify
    python scripts/compass/cc_traces_protocol.py stamp <cell-dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "atom" / "compass" / "CC_TRACES_PROTOCOL.md"
LOCK = ROOT / "atom" / "compass" / "cc_traces_protocol.lock.json"
#: The client counts the matrix offers as an independent workload axis.
CLIENT_COUNTS = (1, 2, 4, 8)
#: The two classes of the clients matrix: subagent bursts whose every prompt is
#: at or below 4096 tokens, and bounded large-prompt bursts. There is one
#: workload file per (class, client count), so the eight below are what the
#: matrix actually replays.
CLIENT_CLASSES = ("clients_short", "clients_large")
#: The workloads the protocol registers. Their digests are written into the
#: document, so the document's own digest covers them: an edited workload and an
#: unedited protocol cannot both be current.
#:
#: `cc_traces_long` and `cc_traces_short` stay registered. They are what the
#: earlier accepted cells ran against, and removing them here would leave that
#: evidence describing a workload this protocol no longer names.
WORKLOADS = ("cc_traces_long", "cc_traces_short") + tuple(
    f"cc_traces_{klass}_c{clients}"
    for klass in CLIENT_CLASSES
    for clients in CLIENT_COUNTS
)


def _mechanism():
    """A private instance of `protocol.py`, pointed at the cc-traces files."""
    path = Path(__file__).resolve().parent / "protocol.py"
    spec = importlib.util.spec_from_file_location("compass_cc_traces_mechanism", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.PROTOCOL, module.LOCK = ROOT, PROTOCOL, LOCK
    return module


def digest(path: Path | None = None) -> str:
    return _mechanism().digest(path or PROTOCOL)


def register(when: str, revision: str | None = None) -> int:
    return _mechanism().register(when, revision)


def verify() -> int:
    return _mechanism().verify()


def workload_digests() -> dict:
    """What each registered workload hashes to right now, read from disk."""
    out = {}
    for name in WORKLOADS:
        path = ROOT / "atom" / "compass" / f"{name}.jsonl"
        out[name] = digest(path) if path.exists() else None
    return out


def stamp(where: str) -> int:
    """Write the cell's copy of what it ran under, including the workloads.

    The workload digests are stamped beside the protocol digest because they
    are the other half of what a cell is: a result produced against a workload
    that has since been re-emitted is not a result about the registered one,
    and reading the protocol digest alone would not show it.
    """
    rc = verify()
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    Path(where).mkdir(parents=True, exist_ok=True)
    (Path(where) / "cc_traces_protocol.json").write_text(
        json.dumps(
            {
                "protocol": str(PROTOCOL.relative_to(ROOT)),
                "sha256": digest(),
                "registered_sha256": lock.get("sha256"),
                "registered_at": lock.get("registered_at"),
                "matches_registration": rc == 0,
                "workloads": workload_digests(),
            },
            indent=1,
        )
        + "\n"
    )
    return rc


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    reg = sub.add_parser("register")
    reg.add_argument(
        "--at", required=True, help="the instant of registration, UTC ISO-8601"
    )
    reg.add_argument(
        "--revision",
        default=None,
        help="the revision of the tree this protocol governs, "
        "when it cannot be read here; say so if it is dirty",
    )
    sub.add_parser("verify")
    st = sub.add_parser("stamp")
    st.add_argument("dir")
    args = ap.parse_args(argv)
    if args.cmd == "register":
        return register(args.at, args.revision)
    if args.cmd == "verify":
        return verify()
    return stamp(args.dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

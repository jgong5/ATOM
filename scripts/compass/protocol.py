"""Register, verify and stamp the prepare-then-measure protocol.

`atom/compass/PROTOCOL.md` says how an acceptance cell is run. This makes that
checkable rather than remembered: the file's digest is registered once, before
the runs it governs, and every cell verifies the digest before it starts and
writes it beside its own artifacts.

The point is not that the file cannot be edited. It is that an edit cannot be
made to apply retroactively -- a cell carries the digest it actually ran under,
so a result produced under one protocol can never be read as though it were
produced under a later one.

    python scripts/compass/protocol.py register       # once, before the runs
    python scripts/compass/protocol.py verify         # non-zero if it changed
    python scripts/compass/protocol.py stamp <dir>    # write the cell's copy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "atom" / "compass" / "PROTOCOL.md"
LOCK = ROOT / "atom" / "compass" / "protocol.lock.json"


def digest(path: Path | None = None) -> str:
    # Resolved at call time, not bound as a default: a default would capture
    # the module-level path at import and quietly ignore any later
    # redirection, so the digest could name a file other than the one read.
    return hashlib.sha256((path or PROTOCOL).read_bytes()).hexdigest()


def _revision() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance, not correctness
        return None


def register(when: str, revision: str | None = None) -> int:
    """Write the lock, keeping any lock it replaces inside it.

    `when` is passed in rather than read from the clock, so the caller records
    the instant it means and a re-registration cannot be silently backdated by
    a machine whose clock disagrees.
    """
    previous = None
    if LOCK.exists():
        previous = json.loads(LOCK.read_text())
        if previous.get("sha256") == digest():
            print(f"already registered: {previous['sha256'][:16]} "
                  f"at {previous.get('registered_at')}")
            return 0
    lock = {
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": digest(),
        "registered_at": when,
        # Supplied by the caller when the tree the digest describes is not the
        # tree the tool is running in -- the node copies are rsynced, not
        # cloned, so `git` there names nothing. A dirty tree is stated as such:
        # a bare revision from a working tree with uncommitted changes points
        # at code that was not what ran.
        "revision": revision or _revision(),
        "superseded": previous,
    }
    LOCK.write_text(json.dumps(lock, indent=1) + "\n")
    print(f"registered {lock['sha256'][:16]} at {when}"
          + (f", superseding {previous['sha256'][:16]}" if previous else ""))
    return 0


def verify() -> int:
    if not LOCK.exists():
        print("ATOMCompass WARNING: no protocol registration, so a cell run "
              "now could not say which protocol it ran under; register before "
              "the acceptance runs")
        return 2
    lock = json.loads(LOCK.read_text())
    have = digest()
    if have != lock["sha256"]:
        print(f"ATOMCompass WARNING: {PROTOCOL.name} has changed since it was "
              f"registered ({lock['sha256'][:16]} -> {have[:16]}); cells run "
              f"now are under a different protocol and the ones already run "
              f"are not covered by it")
        return 1
    print(f"protocol {have[:16]} registered {lock['registered_at']}")
    return 0


def stamp(where: str) -> int:
    rc = verify()
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    Path(where).mkdir(parents=True, exist_ok=True)
    (Path(where) / "protocol.json").write_text(json.dumps({
        "sha256": digest(),
        "registered_sha256": lock.get("sha256"),
        "registered_at": lock.get("registered_at"),
        "matches_registration": rc == 0,
    }, indent=1) + "\n")
    return rc


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    reg = sub.add_parser("register")
    reg.add_argument("--at", required=True,
                     help="the instant of registration, UTC ISO-8601")
    reg.add_argument("--revision", default=None,
                     help="the revision of the tree this protocol governs, "
                          "when it cannot be read here; say so if it is dirty")
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

"""Which run produced this -- the one definition, for everyone who asks.

A digest says what is in a file. It does not say which execution produced it,
and two repeats of one acceptance cell can produce byte-identical artifacts and
still be independent runs. That is exactly the distinction a reader has to make
between a residual left behind by the source and an independent repeat, so the
identity cannot be derived from the payload.

It is derived from the *launch* instead: host, cell, side, repeat, the server's
pid and the moment it started. Derived rather than drawn at random so it can be
checked -- everything it is made of is recorded beside it, and
`verify_execution_id` recomputes it. Neither the pid nor the timestamp is the id
on its own: the kernel reuses pids and a clock can be set backwards, but the
six together identify one launch on one machine.

This module is stdlib-only and imports nothing from ATOM, so any reader --
harness, classifier, a notebook opening one artifact -- can verify an id with
the same code that minted it. A second implementation of this rule is a second
scheme, however carefully it is copied.

    from scripts.compass.execution_id import verify_execution_id
    assert verify_execution_id(json.load(open(artifact))["execution"])
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: Bump when the record's shape changes in a way a reader must notice.
EXECUTION_SCHEMA = "compass.execution/1"

#: The inputs, in the order they are joined. Order is part of the rule.
ID_FIELDS = ("host", "cell", "side", "repeat", "server_pid", "launched_at_ns")

ID_RULE = (
    "sha256 of the id_inputs values joined by NUL, in the order "
    "host, cell, side, repeat, server_pid, launched_at_ns; first 16 hex "
    "characters, prefixed 'cx-'"
)

#: What a stamp carried inside an artifact must have for a reader to check it.
STAMP_FIELDS = (
    "schema",
    "execution_id",
    "id_rule",
    "id_inputs",
    "cell",
    "side",
    "repeat",
)


def derive_execution_id(host, cell, side, repeat, server_pid, launched_at_ns) -> str:
    """The one identifier, from the facts of the launch."""
    parts = [
        str(host),
        str(cell),
        str(side),
        str(repeat),
        str(server_pid),
        str(launched_at_ns),
    ]
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"cx-{digest[:16]}"


def verify_execution_id(record) -> bool:
    """Does this record's id follow from its own recorded inputs?

    Takes the record itself or an artifact's `execution` stamp -- they carry
    the same inputs, which is what makes a copied artifact still checkable.
    """
    if not isinstance(record, dict):
        return False
    inputs = record.get("id_inputs")
    if not isinstance(inputs, dict):
        return False
    try:
        expected = derive_execution_id(*(inputs[name] for name in ID_FIELDS))
    except KeyError:
        return False
    return expected == record.get("execution_id")


def stamp_of(record: dict) -> dict:
    """The part of a record that travels inside an artifact.

    Enough to verify the id and to say which cell, side and repeat it was --
    not the process, source or artifact detail, which belong to the manifest
    and would go stale the moment the file was copied.
    """
    stamp = {name: record.get(name) for name in STAMP_FIELDS}
    if isinstance(stamp.get("id_inputs"), dict):
        stamp["id_inputs"] = dict(stamp["id_inputs"])
    if isinstance(stamp.get("cell"), dict):
        stamp["cell"] = dict(stamp["cell"])
    return stamp


def read_stamp(blob) -> dict | None:
    """The stamp an artifact carries, if it carries one."""
    if not isinstance(blob, dict):
        return None
    stamp = blob.get("execution")
    return stamp if isinstance(stamp, dict) else None


def file_digest(path):
    """A file's digest and size, or None if it is not there."""
    path = Path(path)
    if not path.exists():
        return None
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

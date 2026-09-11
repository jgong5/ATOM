"""Which run produced this -- the one definition, for everyone who asks.

A digest says what is in a file. It does not say which execution produced it,
and two repeats of one acceptance cell can produce byte-identical artifacts and
still be independent runs. That is exactly the distinction a reader has to make
between a residual left behind by the source and an independent repeat, so the
identity cannot be derived from the payload. It comes apart in the other
direction too: re-serialising one record with a different indent changes its
hash without there having been a second execution at all.

The identity is derived from the *launch* instead: host, cell, side, repeat,
the server's pid and the moment it started. Derived rather than drawn at random
so it can be checked -- everything it is made of is recorded beside it, and
`verify_execution_id` recomputes it. Neither the pid nor the timestamp is the
id on its own: the kernel reuses pids and a clock can be set backwards, but the
six together identify one launch on one machine.

This module is stdlib-only, imports nothing from ATOM and touches no
filesystem, so every reader can hold it -- the acceptance harness that mints an
id, the memory classifier that reads one, a notebook opening a single artifact.
That is the whole reason it lives in `core` rather than beside either caller:
`scripts/compass/cc_traces_run.py` loads sibling modules at import time and so
cannot be imported for two functions, and a second implementation of this rule
is a second scheme however carefully it is copied. `scripts/compass/
execution_id.py` re-exports these names rather than restating them.

`compass.execution/1` is the schema CC minted in `cc_traces_run.py` at
f4e06b0c. The derivation here is byte-compatible with it and is pinned against
a vector taken from that implementation (`tests/compass/test_execution_id.py`),
because records already carry ids under it. Changing the rule is a schema
change: bump the version.

    from atom.compass.core.execution_id import verify_execution_id
    assert verify_execution_id(json.load(open(artifact))["execution"])
"""

from __future__ import annotations

import hashlib

__all__ = [
    "EXECUTION_SCHEMA",
    "ID_INPUTS",
    "ID_FIELDS",
    "ID_RULE",
    "STAMP_FIELDS",
    "derive_execution_id",
    "verify_execution_id",
    "stamp_of",
    "read_stamp",
]

#: Bump when the record's shape changes in a way a reader must notice.
EXECUTION_SCHEMA = "compass.execution/1"

#: The inputs, in the order they are joined. Order is part of the rule -- the
#: values are joined, so a different order is a different id for one launch.
ID_INPUTS = ("host", "cell", "side", "repeat", "server_pid", "launched_at_ns")

#: The same tuple under the name CC's harness and its tests already use.
ID_FIELDS = ID_INPUTS

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


def derive_execution_id(host, cell, side, repeat, server_pid,
                        launched_at_ns) -> str:
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

    False means "this record cannot say which run it was", never "a different
    run". That is the answer for every record written before the schema
    existed, and callers must not read an id out of one that fails here: an id
    that does not follow from its inputs is either damaged or transplanted, and
    both are worse than unidentified.
    """
    if not isinstance(record, dict):
        return False
    inputs = record.get("id_inputs")
    if not isinstance(inputs, dict):
        return False
    try:
        expected = derive_execution_id(*(inputs[name] for name in ID_INPUTS))
    except (KeyError, TypeError):
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

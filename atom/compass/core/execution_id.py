"""The one producer identity, shared rather than reimplemented.

An execution is identified by the facts of its launch, not by the bytes it
wrote. A content hash answers "are these the same bytes", which comes apart
from "was this the same run" in both directions: three phase C repeats of the
27B source configuration wrote byte-identical memory records, and
re-serialising one record with a different indent changes its hash without
there having been a second execution at all.

`compass.execution/1` is CC's schema, minted in `cc_traces_run.py` at the
instant the server process is launched. This module is that definition and
nothing else -- stdlib only, no engine imports, no file system -- so the
acceptance harness that writes an id and the memory classifier that reads one
cannot drift apart. It exists because `cc_traces_run.py` loads sibling modules
at import time and so cannot be imported for its two functions alone.

The derivation is byte-compatible with `cc_traces_run.py` at f4e06b0c and is
pinned against a vector taken from that implementation
(`test_execution_id.py`). Changing it is a schema change: bump the version,
because records already carry ids under this one.
"""

from __future__ import annotations

import hashlib

__all__ = [
    "EXECUTION_SCHEMA",
    "ID_INPUTS",
    "ID_RULE",
    "derive_execution_id",
    "verify_execution_id",
]

EXECUTION_SCHEMA = "compass.execution/1"

#: In order. The order is part of the rule -- the values are joined, so a
#: different order is a different id for the same launch.
ID_INPUTS = ("host", "cell", "side", "repeat", "server_pid", "launched_at_ns")

ID_RULE = (
    "sha256 of the id_inputs values joined by NUL, in the order "
    "host, cell, side, repeat, server_pid, launched_at_ns; first 16 hex "
    "characters, prefixed 'cx-'"
)


def derive_execution_id(host, cell, side, repeat, server_pid,
                        launched_at_ns) -> str:
    """The one identifier, from the facts of the launch.

    Derived rather than random so that it can be checked: everything it is made
    of is recorded beside it, and `verify_execution_id` re-computes it. A pid is
    reused by the kernel eventually and a clock can be set backwards, which is
    why neither is the id on its own.
    """
    parts = [str(host), str(cell), str(side), str(repeat), str(server_pid),
             str(launched_at_ns)]
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return "cx-%s" % digest[:16]


def verify_execution_id(record) -> bool:
    """Does this record's id follow from its own recorded inputs?

    False for a record that carries no `id_inputs`, which is the answer for
    every record written before the schema existed: not "a different run", but
    "this record cannot say which run it was". Callers must not read an id out
    of one that fails here -- an id that does not follow from its inputs is
    either damaged or transplanted, and both are worse than unidentified.
    """
    inputs = (record or {}).get("id_inputs") or {}
    try:
        expected = derive_execution_id(*[inputs[name] for name in ID_INPUTS])
    except (KeyError, TypeError):
        return False
    return expected == (record or {}).get("execution_id")

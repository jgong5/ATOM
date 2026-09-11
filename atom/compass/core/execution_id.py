"""The one producer identity, shared rather than reimplemented.

An execution is identified by the facts of its launch, not by the bytes it
wrote. A content hash answers "are these the same bytes", which comes apart
from "was this the same run" in both directions: three phase C repeats of the
27B source configuration wrote byte-identical memory records, and
re-serialising one record with a different indent changes its hash without
there having been a second execution at all.

`compass.execution/1` is the schema, minted by the acceptance harness at the
instant the server process is launched. **This module is the canonical
implementation**: the rule, the field order, the stamp shape and the two
functions live here and nowhere else. Everything else delegates. Two copies
that agree today are still two implementations, and the failure they produce is
not loud -- a drifted verifier answers "unidentified" for every execution,
which reads as caution and is actually the silent loss of the only
reproducibility evidence a calibration has.

Stdlib only, and no ATOM import: importing this module loads no torch, no
AITER and nothing from the engine. Importing it *by package path* does pull
`atom/__init__.py` and `atom/compass/__init__.py`, which are cheap but not
nothing. A reader that wants neither can load this file directly, which is
supported and costs no ATOM import at all:

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "execution_id", "atom/compass/core/execution_id.py")
    execution_id = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(execution_id)

The derivation is byte-compatible with `cc_traces_run.py` at f4e06b0c and with
`scripts/compass/execution_id.py` at 7a5df610, and is pinned against a vector
taken from that implementation (`test_execution_id.py`). Changing it is a
schema change: bump the version, because records already carry ids under this
one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

__all__ = [
    "EXECUTION_SCHEMA",
    "ID_FIELDS",
    "ID_INPUTS",
    "ID_RULE",
    "STAMP_FIELDS",
    "derive_execution_id",
    "read_stamp",
    "stamp_of",
    "verify_execution_id",
]

EXECUTION_SCHEMA = "compass.execution/1"

#: In order. The order is part of the rule -- the values are joined, so a
#: different order is a different id for the same launch.
ID_FIELDS = ("host", "cell", "side", "repeat", "server_pid", "launched_at_ns")

#: The name the memory side first used for the same tuple. Kept so that neither
#: caller has to change at the moment the two implementations become one; they
#: are the same object, so they cannot come apart.
ID_INPUTS = ID_FIELDS

#: What a stamp carried inside an artifact must have for a reader to check it.
STAMP_FIELDS = ("schema", "execution_id", "id_rule", "id_inputs", "cell",
                "side", "repeat")

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

    Takes the execution record itself or an artifact's `execution` stamp --
    they carry the same inputs, which is what makes a copied artifact still
    checkable.

    False for a record that carries no `id_inputs`, which is the answer for
    every record written before the schema existed: not "a different run", but
    "this record cannot say which run it was". Callers must not read an id out
    of one that fails here -- an id that does not follow from its inputs is
    either damaged or transplanted, and both are worse than unidentified.
    """
    if not isinstance(record, Mapping):
        return False
    inputs = record.get("id_inputs")
    if not isinstance(inputs, Mapping):
        return False
    try:
        expected = derive_execution_id(*(inputs[name] for name in ID_FIELDS))
    except KeyError:
        return False
    return expected == record.get("execution_id")


def stamp_of(record) -> dict:
    """The part of an execution record that travels inside an artifact.

    Enough to verify the id and to say which cell, side and repeat it was --
    not the process, source or artifact detail, which belong to the manifest
    and would go stale the moment the file was copied.
    """
    record = record if isinstance(record, Mapping) else {}
    stamp = {name: record.get(name) for name in STAMP_FIELDS}
    for nested in ("id_inputs", "cell"):
        if isinstance(stamp.get(nested), Mapping):
            stamp[nested] = dict(stamp[nested])
    return stamp


def read_stamp(blob):
    """The stamp an artifact carries, if it carries one.

    `None` rather than an empty stamp: "this file names no execution" and "this
    file names one that does not check out" are different answers and only the
    caller knows what to do with each.
    """
    if not isinstance(blob, Mapping):
        return None
    stamp = blob.get("execution")
    return stamp if isinstance(stamp, Mapping) else None

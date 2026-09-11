"""The execution identity, for callers that run as scripts.

Every name that defines the identity is `atom.compass.core.execution_id`'s.
This module re-exports them so a script can keep importing
`scripts.compass.execution_id` without there being two copies of the rule:
there was briefly one here and one in core, minting the same ids by
coincidence of careful copying rather than by construction, which is a second
scheme waiting to drift.

What is genuinely this module's is `file_digest`, which is about bytes rather
than identity -- the thing an execution id exists to be distinguished from --
and which touches the filesystem, so it does not belong in a core module that
every reader must be able to hold.

    from scripts.compass.execution_id import verify_execution_id
    assert verify_execution_id(json.load(open(artifact))["execution"])
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

if __package__ in (None, ""):  # loaded by path, as the harness loads it
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom.compass.core.execution_id import (  # noqa: E402
    EXECUTION_SCHEMA,
    ID_FIELDS,
    ID_INPUTS,
    ID_RULE,
    STAMP_FIELDS,
    derive_execution_id,
    read_stamp,
    stamp_of,
    verify_execution_id,
)

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
    "file_digest",
]


def file_digest(path):
    """A file's digest and size, or None if it is not there.

    A digest answers "are these the same bytes", which is not "was this the
    same run" -- see `derive_execution_id` for the identity that is.
    """
    path = Path(path)
    if not path.exists():
        return None
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

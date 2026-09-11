"""The script side of the one identity rule -- which is defined elsewhere.

The rule itself lives in `atom/compass/core/execution_id.py`, in the runtime
package, and this module does not restate any of it: `derive_execution_id`,
`verify_execution_id`, `EXECUTION_SCHEMA`, `ID_RULE` and `ID_INPUTS` are that
module's, re-exported. Two byte-compatible copies of a rule are still two
implementations, and the day one of them is corrected is the day ids minted by
the acceptance harness stop verifying in the classifier that reads them.

What is left here is the one part that is about files rather than identity: a
digest helper. That has no place in the runtime package, which touches no file
system. `stamp_of` and `read_stamp` were here too, on the reading that a stamp
is a file concern -- but the classifier reads stamps out of artifacts as well,
so they moved to the rule and are re-exported below.

The canonical module is loaded *by path*, not by importing `atom.compass.core`.
`atom/__init__.py` imports the sglang plugin, so a package import would pull in
the engine -- and on a machine with no device that fails, which would make an
id unverifiable for reasons that have nothing to do with the id. The file it
loads imports only the standard library, so this stays a standard-library
operation.

    from scripts.compass.execution_id import verify_execution_id
    assert verify_execution_id(json.load(open(artifact))["execution"])
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

#: The runtime package's copy -- the only one.
CANONICAL = (
    Path(__file__).resolve().parents[2]
    / "atom"
    / "compass"
    / "core"
    / "execution_id.py"
)


def _canonical():
    """Load the rule without importing the package that contains it."""
    if not CANONICAL.exists():
        raise ImportError(
            f"the canonical execution identity is missing: {CANONICAL}. It "
            f"lives in the runtime package (atom/compass/core/execution_id.py) "
            f"and this module deliberately keeps no copy of it, because a "
            f"second copy is a second implementation. Nothing here can mint or "
            f"verify an id until that module is present."
        )
    name = "atom_compass_core_execution_id"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, CANONICAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_core = _canonical()

EXECUTION_SCHEMA = _core.EXECUTION_SCHEMA
ID_RULE = _core.ID_RULE

#: The canonical name for the ordered inputs. `ID_FIELDS` is the name this
#: script helper used before the rule moved into the package; it is kept as an
#: alias so existing readers do not break, and is not the name to write new
#: code against.
ID_INPUTS = _core.ID_INPUTS
ID_FIELDS = ID_INPUTS

derive_execution_id = _core.derive_execution_id
verify_execution_id = _core.verify_execution_id

#: The stamp an artifact carries, and the two functions that build and find
#: one. These were written here on the reading that a stamp is a file concern
#: -- but the classifier reads stamps out of artifacts too, so both ends need
#: them, and a second copy of the shape is a second place for it to drift.
#: The rule owns the stamp; what stays here is the digest, which touches disk.
STAMP_FIELDS = _core.STAMP_FIELDS
stamp_of = _core.stamp_of
read_stamp = _core.read_stamp

__all__ = [
    "CANONICAL",
    "EXECUTION_SCHEMA",
    "ID_FIELDS",
    "ID_INPUTS",
    "ID_RULE",
    "STAMP_FIELDS",
    "derive_execution_id",
    "file_digest",
    "read_stamp",
    "stamp_of",
    "verify_execution_id",
]


def file_digest(path):
    """A file's digest and size, or None if it is not there.

    About bytes, deliberately not about identity: two executions can write the
    same bytes, which is the whole reason an execution id exists.
    """
    import hashlib

    path = Path(path)
    if not path.exists():
        return None
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

# SPDX-License-Identifier: MIT
"""The aiter JIT modules a real server built while it served, which run.json
names as ``aiter_jit_built`` and grading refuses.

aiter builds a module, a ``.so``, the first time a process meets a kernel
variant its image did not prebuild, and keeps it for every later process of
that aiter tree; the step that meets it waits on the build, about a minute for
`mha_varlen_fwd`'s. Its modules are found from the installed aiter, which is
not imported: under its package's ``jit`` or ``AITER_JIT_DIR``, under
``~/.aiter``, which holds ``jit`` when the package's is read-only, and under
``AITER_ROOT_DIR``, where its templated modules land, ``~/.aiter`` unless set.
A module is named by its path, and counts when its file was written at or
after serving began: the modules a start builds, before ``/health`` answers,
do not. One another server of the same aiter tree built meanwhile counts too,
since this server may have waited on it.

    python -m atom.compass.drive.aiter_jit RUN_JSON SINCE_NS

adds the entry to RUN_JSON for the modules written at or after SINCE_NS, wall
clock nanoseconds, as `cctraces_sim.sh` does for a real cell.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

from atom.compass.run import _write_whole

#: run.json's entry naming them; a run whose server built none has no entry.
BUILT = "aiter_jit_built"


def _roots() -> set[Path]:
    try:
        spec = importlib.util.find_spec("aiter")
    except ValueError:  # an `aiter` in sys.modules with no spec, as a stub is
        spec = None
    package = (spec and spec.submodule_search_locations) or ()
    env = {os.environ.get(k) for k in ("AITER_JIT_DIR", "AITER_ROOT_DIR")} - {None, ""}
    return {Path.home() / ".aiter", *map(Path, env), *(Path(p, "jit") for p in package)}


def built_since(since_ns: int) -> dict:
    """run.json's entry naming each aiter JIT module whose file was written at
    or after `since_ns`, wall clock nanoseconds; {} when none was."""
    built = {
        str(path)
        for root in _roots()
        for path in root.rglob("*.so")
        if path.lstat().st_mtime_ns >= since_ns
    }
    return {BUILT: sorted(built)} if built else {}


def main(argv=None) -> None:
    run_json, since_ns = sys.argv[1:] if argv is None else argv
    run = json.loads(Path(run_json).read_text()) | built_since(int(since_ns))
    _write_whole(Path(run_json), json.dumps(run))


if __name__ == "__main__":
    main()

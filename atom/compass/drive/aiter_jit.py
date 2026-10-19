# SPDX-License-Identifier: MIT
"""The aiter JIT modules a real server built, or began to, while it served,
which run.json names as ``aiter_jit_builds`` and grading refuses.

aiter builds a module the first time a process meets a kernel variant its
image did not prebuild, and keeps it for every later process of that aiter
tree; the step that meets it waits on the build, about a minute for
`mha_varlen_fwd`'s. Each module builds in its own directory under a
``build`` directory of a JIT root: its package's ``jit`` or ``AITER_JIT_DIR``,
``~/.aiter``, which holds ``jit`` when the package's is read-only, and
``AITER_ROOT_DIR``, ``~/.aiter`` unless set, where its templated modules
build. The roots are found from the installed aiter, which is not imported. A
module counts, named by its directory, when a path inside that directory was
written at or after serving began, so a build the server's stop cut short
counts too, and the modules a start builds, before ``/health`` answers, do
not. One another server of the same aiter tree built meanwhile counts as well,
since this server may have waited on it.

    python -m atom.compass.drive.aiter_jit RUN_JSON SINCE_NS

adds the entry to RUN_JSON for the paths written at or after SINCE_NS, wall
clock nanoseconds, as `cctraces_sim.sh` does for a real cell.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

from atom.compass.run import _write_whole

#: run.json's entry naming them; a run whose server built none has no entry.
BUILDS = "aiter_jit_builds"


def _roots() -> list[Path]:
    try:
        spec = importlib.util.find_spec("aiter")
    except ValueError:  # an `aiter` in sys.modules with no spec, as a stub is
        spec = None
    package = [Path(p, "jit") for p in (spec and spec.submodule_search_locations) or ()]
    env = [
        Path(os.environ[k])
        for k in ("AITER_JIT_DIR", "AITER_ROOT_DIR")
        if os.environ.get(k)
    ]
    return sorted({Path.home() / ".aiter", *env, *package})


def _module(parts: tuple[str, ...]) -> str | None:
    """The module whose build directory holds the path `parts` names under a
    root: the directory after its first ``build``, when the path is inside it."""
    i = parts.index("build") if "build" in parts else len(parts)
    return parts[i + 1] if len(parts) > i + 2 else None


def builds_since(since_ns: int) -> dict:
    """run.json's entry naming each aiter JIT module a path of whose build
    directory was written at or after `since_ns`, wall clock nanoseconds; {}
    when none was."""
    built = {
        _module(path.relative_to(root).parts)
        for root in _roots()
        for path in root.rglob("*")
        if path.lstat().st_mtime_ns >= since_ns
    } - {None}
    return {BUILDS: sorted(built)} if built else {}


def main(argv=None) -> None:
    run_json, since_ns = sys.argv[1:] if argv is None else argv
    run = json.loads(Path(run_json).read_text()) | builds_since(int(since_ns))
    _write_whole(Path(run_json), json.dumps(run))


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: MIT
"""The aiter JIT modules a real server built, or began to, while it served,
which run.json names as ``aiter_jit_builds`` and grading, the fits and
`validate.cost` refuse (`step_labels.refusal`).

aiter builds a module the first time a process meets a kernel variant its
image did not prebuild, and keeps it for every later process of that aiter
tree; the step that meets it waits on the build, about a minute for
`mha_varlen_fwd`'s. Each module builds in its own directory under a
``build`` directory of a JIT root: its package's ``jit`` or ``AITER_JIT_DIR``,
``~/.aiter``, which holds ``jit`` when the package's is read-only, and
``AITER_ROOT_DIR``, ``~/.aiter`` unless set, where its templated modules
build. The roots are found from the installed aiter, which is not imported. A
module counts, named by its directory, when a file inside that directory was
written at or after serving began, so a build the server's stop cut short
counts too, and the modules a start builds, before ``/health`` answers, do
not. A module another server of the same aiter tree built meanwhile counts as
well, since this server may have waited on it.

So does each kernel compiled the same way: `compile_hsaco`'s, in
``build/<gfx arch>``, FlyDSL's, in its own directory under
``FLYDSL_RUNTIME_CACHE_DIR``, the package's ``jit/flydsl_cache`` unless set,
and Triton's and Inductor's, in their cache directories and ATOM's compile
cache, ``~/.cache/atom``; each named by its file, FlyDSL's by its directory.

    python -m atom.compass.drive.aiter_jit RUN_JSON SINCE_NS

adds the entry to RUN_JSON for the paths written at or after SINCE_NS, wall
clock nanoseconds, as `cctraces_sim.sh` does for a real cell.
"""

import importlib.util
import json
import os
import sys
from contextlib import suppress
from pathlib import Path

from torch._inductor.runtime.cache_dir_utils import default_cache_dir
from triton import knobs

from atom.compass.backends.step_labels import BUILDS, refusal  # noqa: F401
from atom.compass.run import _write_whole


def _roots() -> dict[Path, object]:
    try:
        spec = importlib.util.find_spec("aiter")
    except ValueError:  # an `aiter` in sys.modules with no spec, as a stub is
        spec = None
    package = [Path(p, "jit") for p in (spec and spec.submodule_search_locations) or ()]
    env = [os.environ.get(k) for k in ("AITER_JIT_DIR", "AITER_ROOT_DIR")]
    flydsl = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    flydsl = [flydsl] if flydsl else [p / "flydsl_cache" for p in package]
    inductor = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or default_cache_dir()
    kernels = (knobs.cache.dir, inductor, Path.home() / ".cache" / "atom")
    return (
        {Path(p): _module for p in (Path.home() / ".aiter", *env, *package) if p}
        | {Path(p): lambda parts: parts[0] if len(parts) > 1 else None for p in flydsl}
        | {Path(p): lambda parts: _file(parts[-1]) for p in kernels}
    )


def _file(name: str) -> str:
    """A compiled kernel's name, from the name of one of its files."""
    return name.split(".")[0].removeprefix("__grp__")


def _module(parts: tuple[str, ...]) -> str | None:
    """The module whose build directory holds the path `parts` names under a
    root: the directory after its first ``build``, when the path is inside it,
    or the kernel `compile_hsaco` wrote there."""
    i = parts.index("build") if "build" in parts else len(parts)
    if len(parts) > i + 2 and parts[i + 1].startswith("gfx"):
        return _file(parts[i + 2])
    return parts[i + 1] if len(parts) > i + 2 else None


def builds_since(since_ns: int) -> dict:
    """run.json's entry naming each aiter JIT module or compiled kernel with a
    file written at or after `since_ns`, wall clock nanoseconds; {} when none
    was."""
    built = set()
    for root, name in _roots().items():
        for top, _, files in os.walk(root):
            for path in (Path(top, f) for f in files):
                with suppress(FileNotFoundError):  # a temporary since renamed
                    if path.lstat().st_mtime_ns >= since_ns:
                        built.add(name(path.relative_to(root).parts))
    built.discard(None)
    return {BUILDS: sorted(built)} if built else {}


def main(argv=None) -> None:
    run_json, since_ns = sys.argv[1:] if argv is None else argv
    run = json.loads(Path(run_json).read_text()) | builds_since(int(since_ns))
    _write_whole(Path(run_json), json.dumps(run))


if __name__ == "__main__":
    main()

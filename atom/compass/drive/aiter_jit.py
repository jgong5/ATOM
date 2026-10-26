# SPDX-License-Identifier: MIT
"""The JIT compiles a real server ran, or began, while it served, which run.json
names as ``aiter_jit_builds`` and grading, the fits and `validate.cost` refuse
(`step_labels.refusal`).

A process compiles a kernel the first time it meets a variant no cache on disk
holds, and keeps it there for every later process; the step that meets it waits
on the compile, about a minute for aiter's `mha_varlen_fwd` module, 53.6 s and
56.4 s for the first two prefills a fresh container's Inductor compiled. Each
compiler writes its own cache root, found here without importing aiter:

- aiter's modules, each in its own directory under a ``build`` directory of its
  package's ``jit``, ``AITER_JIT_DIR``, ``~/.aiter``, which holds ``jit`` when
  the package's is read-only, or ``AITER_ROOT_DIR``, ``~/.aiter`` unless set,
  where its templated modules build: named by that directory. aiter's
  `compile_hsaco` writes a kernel's files straight into ``build/<gfx arch>``:
  named by the file.
- FlyDSL's kernels, each in its own directory under ``FLYDSL_RUNTIME_CACHE_DIR``,
  which aiter points at its package's ``jit/flydsl_cache``: named by that
  directory.
- Triton's and Inductor's, in Triton's cache directory, Inductor's and ATOM's
  compile cache, ``~/.cache/atom``, which redirects both: named by the file, a
  Triton kernel's name or an Inductor hash.

A compile counts when a file of it was written at or after serving began, so
one the server's stop cut short counts too, and those a start ran, before
``/health`` answers, do not. One another server of the same caches ran meanwhile
counts as well, since this server may have waited on it.

    python -m atom.compass.drive.aiter_jit RUN_JSON SINCE_NS

adds the entry to RUN_JSON for the files written at or after SINCE_NS, wall
clock nanoseconds, as `cctraces_sim.sh` does for a real cell.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

from torch._inductor.runtime.cache_dir_utils import default_cache_dir
from triton import knobs

from atom.compass.backends.step_labels import BUILDS, refusal  # noqa: F401
from atom.compass.run import _write_whole


def _module(parts: tuple[str, ...]) -> str | None:
    """The aiter module whose build directory holds the file `parts` names
    under a root, the directory after its first ``build``, or the kernel
    `compile_hsaco` wrote there."""
    i = parts.index("build") if "build" in parts else len(parts)
    if len(parts) < i + 3:
        return None
    return Path(parts[i + 2]).stem if parts[i + 1].startswith("gfx") else parts[i + 1]


def _flydsl(parts: tuple[str, ...]) -> str | None:
    return parts[0] if len(parts) > 1 else None


def _kernel(parts: tuple[str, ...]) -> str:
    return parts[-1].split(".")[0].removeprefix("__grp__")


def _roots() -> dict[Path, object]:
    """Each cache root, with how a file under it names its compile."""
    try:
        spec = importlib.util.find_spec("aiter")
    except ValueError:  # an `aiter` in sys.modules with no spec, as a stub is
        spec = None
    package = [Path(p, "jit") for p in (spec and spec.submodule_search_locations) or ()]
    aiter = [os.environ.get(k) for k in ("AITER_JIT_DIR", "AITER_ROOT_DIR")]
    flydsl = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    flydsl = [flydsl] if flydsl else [p / "flydsl_cache" for p in package]
    inductor = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or default_cache_dir()
    kernels = (knobs.cache.dir, inductor, Path.home() / ".cache" / "atom")
    return (
        {Path(p): _module for p in (Path.home() / ".aiter", *package, *aiter) if p}
        | {Path(p): _flydsl for p in flydsl}
        | {Path(p): _kernel for p in kernels}
    )


def builds_since(since_ns: int) -> dict:
    """run.json's entry naming each compile a file of which was written at or
    after `since_ns`, wall clock nanoseconds; {} when none was."""
    built = set()
    for root, name in _roots().items():
        for top, _, files in os.walk(root):
            for file in files:
                path = Path(top, file)
                try:
                    if path.lstat().st_mtime_ns >= since_ns:
                        built.add(name(path.relative_to(root).parts))
                except FileNotFoundError:  # a compiler's temporary, renamed
                    pass
    built.discard(None)
    return {BUILDS: sorted(built)} if built else {}


def main(argv=None) -> None:
    run_json, since_ns = sys.argv[1:] if argv is None else argv
    run = json.loads(Path(run_json).read_text()) | builds_since(int(since_ns))
    _write_whole(Path(run_json), json.dumps(run))


if __name__ == "__main__":
    main()

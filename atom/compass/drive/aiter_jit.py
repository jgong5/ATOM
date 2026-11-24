# SPDX-License-Identifier: MIT
"""The aiter JIT modules a real server built, or began to, while it served,
which run.json names as ``aiter_jit_builds`` and times as ``jit_compiles``, one
entry per compile with its artifact and its first and last file's mtime. The
fits and `validate.cost` cut the steps that overlap one, or refuse the record
(`step_labels.cut_refusal`); grading refuses a run that names one
(`step_labels.refusal`).

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
not. A module another server of the same aiter tree built meanwhile counts as
well, since this server may have waited on it.

So does each kernel compiled the same way: `compile_hsaco`'s, in
``build/<gfx arch>``, FlyDSL's, in its own directory under
``FLYDSL_RUNTIME_CACHE_DIR``, the package's ``jit/flydsl_cache`` unless set,
and Triton's and Inductor's, in their cache directories; each named by its
file, FlyDSL's by its directory. ATOM's compile cache, ``~/.cache/atom``, is
left out: each start derives a directory of its own there, points Triton and
Inductor into it, and the stop removes it, so every process pays its compiles.

    python -m atom.compass.drive.aiter_jit RUN_JSON SINCE_NS

adds the entries to RUN_JSON for the paths written at or after SINCE_NS, wall
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

from atom.compass.backends.step_labels import (  # noqa: F401
    BUILDS,
    COMPILES,
    cut_refusal,
    refusal,
)
from atom.compass.run import _write_whole


def _roots() -> dict[Path, str]:
    """Each cache root, mapped to the compiler `NAMES` keys it by."""
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
    flydsl = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    flydsl = [flydsl] if flydsl else [p / "flydsl_cache" for p in package]
    inductor = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or default_cache_dir()
    return (
        {p: "aiter" for p in (Path.home() / ".aiter", *env, *package)}
        | {Path(p): "flydsl" for p in flydsl}
        | {Path(knobs.cache.dir): "triton", Path(inductor): "inductor"}
    )


def _file(name: str) -> str:
    return name.split(".")[0].removeprefix("__grp__")


def _module(parts: tuple[str, ...]) -> tuple[str, tuple] | None:
    """The module whose build directory holds the path `parts` names under a
    root, the directory after its first ``build`` when the path is inside it,
    and that directory's parts; or `compile_hsaco`'s kernel there, and its
    files' parts less their suffix."""
    i = parts.index("build") if "build" in parts else len(parts)
    if len(parts) > i + 2 and parts[i + 1].startswith("gfx"):  # compile_hsaco's
        return _file(parts[i + 2]), (*parts[: i + 2], _file(parts[i + 2]))
    return (parts[i + 1], parts[: i + 2]) if len(parts) > i + 2 else None


def _flydsl(parts: tuple[str, ...]) -> tuple[str, tuple] | None:
    """A FlyDSL kernel, named by its directory, and one build of it: the key
    its ``<key>.lock`` and ``<key>.pkl`` share, as parts. A directory holds a
    build per key, each compiled on its own."""
    return (parts[0], (*parts[:-1], _file(parts[-1]))) if len(parts) > 1 else None


#: How each compiler names the compile that wrote a path, given its parts under
#: the compiler's root, with the parts naming its artifact: the directory or
#: the path less suffix that the compile's files, and no other's, share.
NAMES = {
    "aiter": _module,
    "flydsl": _flydsl,
    "triton": lambda parts: (_file(parts[-1]), parts[:-1]),
    "inductor": lambda parts: (_file(parts[-1]), parts[:-1]),
}


def builds_since(since_ns: int) -> dict:
    """run.json's entries for each aiter JIT module or compiled kernel with a
    file written at or after `since_ns`, wall clock nanoseconds: its name in
    `BUILDS`, and in `COMPILES` each compile, one per artifact (`NAMES`), with
    its first and last such file's mtime; {} when none was."""
    spans = {}
    for root, compiler in _roots().items():
        for top, _, files in os.walk(root):
            for path in (Path(top, f) for f in files):
                with suppress(FileNotFoundError):  # a temporary since renamed
                    ns = path.lstat().st_mtime_ns
                    parts = path.relative_to(root).parts
                    if ns >= since_ns and (named := NAMES[compiler](parts)):
                        key = (compiler, named[0], str(Path(root, *named[1])))
                        first, last = spans.get(key, (ns, ns))
                        spans[key] = (min(first, ns), max(last, ns))
    if not spans:
        return {}
    compiles = [
        {"compiler": c, "name": n, "artifact": a, "first_ns": f, "last_ns": t}
        for (c, n, a), (f, t) in sorted(spans.items(), key=lambda kv: kv[1])
    ]
    return {BUILDS: sorted({n for _, n, _ in spans}), COMPILES: compiles}


def main(argv=None) -> None:
    run_json, since_ns = sys.argv[1:] if argv is None else argv
    run = json.loads(Path(run_json).read_text()) | builds_since(int(since_ns))
    _write_whole(Path(run_json), json.dumps(run))


if __name__ == "__main__":
    main()

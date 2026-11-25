# SPDX-License-Identifier: MIT
"""A library of compiled JIT artifacts per stack, so a container compiles a key
the stack has already compiled once, in any container, never again.

A library is the directory `library` names under ``<store>/jit``, one per
`stack_key`: what a container can read of its stack, since an artifact is
reusable only on the stack that compiled it and each compiler's own key
separates the rest. It mirrors each cache root `aiter_jit` reads, by its
absolute path, under ``roots``. `restore` merges it into those roots before a
real server starts; `save` writes back, once the server has stopped, each
artifact the roots finished at or after a time. Both copy only finished
artifacts, judged per compiler (`_finished`), keep each file's mtime, so
`aiter_jit.builds_since` names nothing restored, and never replace a file:
each file is copied beside its place under a temporary name and linked into
it, an artifact's files in an order that puts last the one its compiler looks
for. Two saves at once leave the union of their artifacts.

    python -m atom.compass.drive.jit_cache restore STORE GPUS
    python -m atom.compass.drive.jit_cache save STORE GPUS SINCE_NS

as `cctraces_sim.sh` runs them around a real server: GPUS comma-separated, as
rocm-smi indexes them, and SINCE_NS wall clock nanoseconds.
"""

import os
import shutil
import sys
import tempfile
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import triton

from atom.compass.drive import aiter_jit
from atom.compass.drive.identity import _device, _stack

#: The libraries' directory under an artifact store's root.
ROOT = "jit"
#: Where a library mirrors the cache roots.
ROOTS = "roots"


def stack_key(stack: dict, device: list[dict]) -> str:
    """A library's name: `stack`, the stack run.json's ``stack`` states (ROCm's
    version, aiter's commit, RCCL's and torch's), this process's Triton and
    FlyDSL versions, and the gfx arch of `device`, run.json's ``device``."""
    try:
        flydsl = version("flydsl")
    except PackageNotFoundError:
        flydsl = "none"
    stack = {**stack, "triton": triton.__version__, "flydsl": flydsl}
    arch = sorted({gpu["arch"] for gpu in device})
    return "_".join([*(f"{k}-{v}" for k, v in stack.items()), *arch])


def library(store, stack: dict, device: list[dict]) -> Path:
    """The library `stack_key` names in the artifact store whose root is
    `store`."""
    return Path(store, ROOT, stack_key(stack, device))


def restore(library) -> None:
    """Merge `library`'s finished artifacts into the cache roots, leaving every
    file a root has."""
    for root, compiler in aiter_jit._roots().items():
        mirror = _mirror(library, root)
        for files in _finished(mirror, compiler):
            _place(files, mirror, root)


def save(library, since_ns: int) -> None:
    """Write into `library` each finished artifact of the cache roots with a
    file written at or after `since_ns`, wall clock nanoseconds."""
    for root, compiler in aiter_jit._roots().items():
        for files in _finished(root, compiler):
            if max(f.lstat().st_mtime_ns for f in files) >= since_ns:
                _place(files, root, _mirror(library, root))


def _mirror(library, root) -> Path:
    return Path(library, ROOTS, Path(root).absolute().relative_to("/"))


def _place(files, src: Path, dst: Path) -> None:
    """Copy `files`, under `src`, to the same paths under `dst`, in order,
    each staged beside its place and linked into it, unless one is there."""
    for f in files:
        to = dst / f.relative_to(src)
        if os.path.lexists(to):
            continue
        to.parent.mkdir(parents=True, exist_ok=True)
        fd, staged = tempfile.mkstemp(prefix=f".{to.name}.", dir=to.parent)
        os.close(fd)
        try:
            shutil.copy2(f, staged)
            os.link(staged, to)
        except FileExistsError:  # another save placed it meanwhile
            pass
        finally:
            os.unlink(staged)


def _finished(root, compiler: str):
    """Each finished artifact under `root`, a cache root of `compiler`, as the
    list of its files, the one its compiler looks for last. A dot file is a
    temporary, never an artifact's, as is a Triton ``tmp.`` directory."""
    for top, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith((".", "tmp."))]
        files = sorted(Path(top, n) for n in names if not n.startswith("."))
        yield from RULES[compiler](Path(top), files, Path(root))


def _aiter(top: Path, files: list, root: Path):
    """A module's ``<name>.so`` beside its ``build`` directory, unless that
    holds the build's lock ``lock_<name>``; a templated module's
    ``build/<folder>/lib.so``, unless its folder holds ``lock``; and
    `compile_hsaco`'s ``build/<gfx arch>/<f>.hsaco`` and its ``<f>.json``,
    unless ``<f>.lock`` is beside them. A build's other files are its own."""
    in_build = top.parent.name == "build"
    for f in files:
        if f.suffix == ".so" and "build" not in f.relative_to(root).parts:
            if not (top / "build" / f"lock_{f.stem}").exists():
                yield [f]
        elif f.name == "lib.so" and in_build and not (top / "lock").exists():
            yield [f]
        elif f.suffix == ".hsaco" and in_build and top.name.startswith("gfx"):
            meta = f.with_suffix(".json")
            if meta.exists() and not f.with_suffix(".lock").exists():
                yield [meta, f]


def _flydsl(top: Path, files: list, root: Path):
    """A kernel's ``<key>.pkl``, which FlyDSL writes whole, with its
    ``<key>.lock``."""
    for f in files:
        if f.suffix == ".pkl" and f.with_suffix(".lock").exists():
            yield [f.with_suffix(".lock"), f]


def _kernels(top: Path, files: list, root: Path, inductor=False):
    """A Triton key's directory once it holds its ``__grp__`` json, which
    Triton writes after the files it lists, and an autotune result's
    ``.autotune.json``, which it writes whole. Under Inductor's root, outside
    its ``triton`` directory, each file too, which Inductor writes whole."""
    group = [f for f in files if f.name.startswith("__grp__")]
    if group:
        yield [*(f for f in files if f not in group), *group]
    elif inductor and "triton" not in top.relative_to(root).parts:
        yield from ([f] for f in files)
    else:
        yield from ([f] for f in files if f.name.endswith(".autotune.json"))


RULES = {
    "aiter": _aiter,
    "flydsl": _flydsl,
    "triton": _kernels,
    "inductor": partial(_kernels, inductor=True),
}


def main(argv=None) -> None:
    action, store, gpus, *since = sys.argv[1:] if argv is None else argv
    path = library(store, _stack(), _device(tuple(int(g) for g in gpus.split(","))))
    if action == "restore" and not since:
        restore(path)
    elif action == "save" and len(since) == 1:
        save(path, int(since[0]))
    else:
        sys.exit("usage: jit_cache restore STORE GPUS | save STORE GPUS SINCE_NS")


if __name__ == "__main__":
    main()

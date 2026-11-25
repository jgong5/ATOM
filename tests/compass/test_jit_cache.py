# SPDX-License-Identifier: MIT
"""`jit_cache`: a per-stack library of compiled JIT artifacts, saved from and
restored into the cache roots `aiter_jit` reads, each of them a directory of
the test's own; and `compass record --store`, whose server's start restores the
library and whose stop saves into it, on `test_drive_record.py`'s planted
server and GPU."""

import json
import os
import shutil
import threading
import time

import pytest
from test_drive_identity import GIT, _git
from test_drive_record import BUILDS, SERVING, planted_gpu  # noqa: F401
from test_validate_runs import _aiter

from atom.compass import cli
from atom.compass.drive import aiter_jit, jit_cache

JIT = "site/aiter/jit"
#: The cache roots below the test's directory: the installed aiter's JIT
#: directory and those the environment names.
ROOTS = (JIT, "aiter_root", "flydsl", "triton", "inductor")
#: One finished artifact per compiler, and an autotune result, each file named
#: under the test's directory.
FINISHED = [
    f"{JIT}/mha_a.so",
    "aiter_root/build/pa_a/lib.so",
    "aiter_root/build/gfx942/k_a.json",
    "aiter_root/build/gfx942/k_a.hsaco",
    "flydsl/launch_a/01.lock",
    "flydsl/launch_a/01.pkl",
    "triton/K1/k1.hsaco",
    "triton/K1/__grp__k1.json",
    "triton/K4/fn.autotune.json",
    "inductor/fxgraph/ab/x",
]
#: One artifact per compiler that its build left unfinished, and a module's
#: build files, which the module's ``.so`` does not need.
UNFINISHED = [
    f"{JIT}/mha_b.so",
    f"{JIT}/build/lock_mha_b",
    f"{JIT}/build/mha_a/build/mha_a.so",
    "aiter_root/build/pa_b/lib.so",
    "aiter_root/build/pa_b/lock",
    "aiter_root/build/gfx942/k_b.json",
    "aiter_root/build/gfx942/k_b.hsaco",
    "aiter_root/build/gfx942/k_b.lock",
    "flydsl/launch_b/02.lock",
    "flydsl/launch_b/.02.pkl.x1.tmp",
    "triton/K2/k2.hsaco",
    "triton/K3/tmp.pid_1_2/k3.hsaco",
    "inductor/fxgraph/ab/.1.2.tmp",
    "inductor/triton/0/K5/k5.hsaco",
]


def _caches(tmp_path, monkeypatch):
    """Point every cache root at the test's directory."""
    _aiter(tmp_path, monkeypatch)
    for var, root in (
        ("AITER_ROOT_DIR", "aiter_root"),
        ("FLYDSL_RUNTIME_CACHE_DIR", "flydsl"),
        ("TRITON_CACHE_DIR", "triton"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
    ):
        monkeypatch.setenv(var, str(tmp_path / root))


def _write(tmp_path, paths, ns):
    """Write each of `paths` under `tmp_path`, holding its own name, at the
    wall clock time `ns`."""
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        os.utime(path, ns=(ns, ns))


def _files(top, roots=ROOTS):
    return sorted(
        str(p.relative_to(top))
        for root in roots
        for p in (top / root).rglob("*")
        if p.is_file()
    )


def test_a_library_saved_and_restored_into_empty_roots_holds_the_finished_artifacts_only(
    tmp_path, monkeypatch
):
    """The named result: not an unfinished artifact, nor a build's own files,
    nor a module the image held, written before the save's time; and the
    restore writes nothing `aiter_jit.builds_since` names."""
    _caches(tmp_path, monkeypatch)
    since = time.time_ns() - 10 * 10**9
    _write(tmp_path, FINISHED + UNFINISHED, since + 5 * 10**9)
    library = tmp_path / "store" / jit_cache.ROOT / "stack"
    jit_cache.save(library, since)
    for root in ROOTS:
        shutil.rmtree(tmp_path / root)

    restored = time.time_ns()
    jit_cache.restore(library)

    assert _files(tmp_path) == sorted(FINISHED)
    assert all((tmp_path / name).read_text() == name for name in FINISHED)
    assert aiter_jit.builds_since(restored) == {}


def test_a_restore_leaves_every_file_a_root_has(tmp_path, monkeypatch):
    _caches(tmp_path, monkeypatch)
    now = time.time_ns()
    _write(tmp_path, FINISHED, now)
    library = tmp_path / "library"
    jit_cache.save(library, now)
    for root in ROOTS:
        shutil.rmtree(tmp_path / root)
    _write(tmp_path, ["triton/K1/k1.hsaco"], now)
    (tmp_path / "triton" / "K1" / "k1.hsaco").write_text("the container's")

    jit_cache.restore(library)

    assert (tmp_path / "triton" / "K1" / "k1.hsaco").read_text() == "the container's"
    assert _files(tmp_path) == sorted(FINISHED)


def test_two_saves_at_once_leave_the_union_of_their_artifacts(tmp_path, monkeypatch):
    """Two servers whose caches finished different Triton keys, a third of them
    in both, save into one library at once."""
    _caches(tmp_path, monkeypatch)
    keys, now = range(300), time.time_ns()
    _write(
        tmp_path,
        [f"triton/K{i}/{f}" for i in keys for f in ("k.hsaco", "__grp__k.json")],
        now,
    )
    finished = jit_cache._finished
    holds = {"a": lambda i: i % 3 != 0, "b": lambda i: i % 3 != 1}

    def _held(root, compiler):
        held = holds[threading.current_thread().name]
        artifacts = finished(root, compiler) if compiler == "triton" else ()
        return (f for f in artifacts if held(int(f[0].parent.name[1:])))

    monkeypatch.setattr(jit_cache, "_finished", _held)
    library = tmp_path / "library"
    start = threading.Barrier(2)

    def _save():
        start.wait()
        jit_cache.save(library, now)

    savers = [threading.Thread(target=_save, name=name) for name in holds]
    for saver in savers:
        saver.start()
    for saver in savers:
        saver.join()

    mirror = jit_cache._mirror(library, tmp_path / "triton")
    held = sorted(str(p.relative_to(mirror)) for p in mirror.rglob("*") if p.is_file())
    assert held == sorted(
        f"K{i}/{f}" for i in keys for f in ("k.hsaco", "__grp__k.json")
    )
    assert all((mirror / f).read_text() == f"triton/{f}" for f in held)


@pytest.mark.skipif(GIT is None, reason="needs git")
def test_a_record_s_stop_saves_what_it_finished_and_the_next_container_restores_it(
    tmp_path, planted_gpu, monkeypatch  # noqa: F811
):
    """`compass record --store`: the first record's server.json names the
    library, and its server finishes a Triton key and leaves an aiter module's
    build locked; a second record in a fresh container of the same stack finds
    the key in its cache before it serves, and names no build."""
    _caches(tmp_path, monkeypatch)
    triton, jit = tmp_path / "triton", tmp_path / JIT
    built = [triton / "K1" / "k1.hsaco", triton / "K1" / "__grp__k1.json"]
    built += [jit / "mha_x.so", jit / "build" / "lock_mha_x"]
    monkeypatch.setenv("PLANTED_START", str(tmp_path / "start" / "inert"))
    monkeypatch.setenv("PLANTED_BUILT", os.pathsep.join(map(str, built)))
    server = tmp_path / "server.py"
    server.write_text(BUILDS + server.read_text() + SERVING)
    store = tmp_path / "store"
    record = f"record --gpu 7 --store {store} --out {{}} -- --model {planted_gpu}"

    cli.main(record.format(tmp_path / "first").split())
    first = json.loads((tmp_path / "first" / "run.json").read_text())
    assert first[aiter_jit.BUILDS] == ["k1"]
    [library] = (store / jit_cache.ROOT).iterdir()
    aiter = _git(tmp_path / "aiter", "rev-parse", "--short=9", "HEAD")
    assert library.name.startswith(f"rocm-7.2.4_aiter-{aiter}_rccl-")
    assert library.name.endswith("_gfx942")
    ready = json.loads((tmp_path / "first" / "server.json").read_text())
    assert ready["library"] == str(library)

    for root in (triton, jit):
        shutil.rmtree(root)
    monkeypatch.setenv("PLANTED_BUILT", str(tmp_path / "elsewhere" / "inert"))
    cli.main(record.format(tmp_path / "second").split())

    second = json.loads((tmp_path / "second" / "run.json").read_text())
    assert aiter_jit.BUILDS not in second
    assert _files(tmp_path, ("triton", JIT)) == [
        "triton/K1/__grp__k1.json",
        "triton/K1/k1.hsaco",
    ]

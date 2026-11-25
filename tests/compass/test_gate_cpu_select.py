# SPDX-License-Identifier: MIT
"""`gate_cpu.sh` with COMPASS_GATE_SELECT runs only the test files that ran a
function the diff since the map's commit changes, or the whole gate and why.

Each case makes the throwaway tree of ``test_gate_cpu_verdict_through_pipe.py``
a git repository holding `atom/m.py` and a test file per function there, each
recording that it ran; `test_g.py` runs its function in a subprocess with only
PYTHONPATH set, and `test_src.py`, where a case adds it, reads the source of
`atom/`. The map of the first commit is written from the tree's own
`gate_select.py` and the files' known coverage and source reads, or by
`gate_map.sh` when COMPASS_COVERAGE_WHEEL names a coverage.py wheel. A second commit makes the
change; `snapshot.sh` stages it and `gate_cpu.sh` runs on the extraction, or in
the checkout. Nothing needs a driver.
"""

import importlib.util
import json
import os
import subprocess
import textwrap

import pytest
from test_gate_cpu_verdict_through_pipe import _run, _tree

MOD = """K = 1


def f():
    return 1


def g():
    return 2


class C:
    def h(self):
        return 3
"""
# Indented: `regen_gpu_gate_triggers.sh` reads a column-0 `from atom.` line as
# this file's import, and `atom.m` exists only in the throwaway tree.
TEST = textwrap.dedent("""\
    import pathlib, subprocess, sys
    from atom.m import {name}


    def test_{name}():
        pathlib.Path({ran!r}, "{name}").touch()
        {call}
    """)
G = "subprocess.run([sys.executable, '-c', 'from atom.m import g; g()'], check=True, env={'PYTHONPATH': str(pathlib.Path(__file__).parents[2])})"
CALLS = {"f": "f()", "g": G, "C": "C().h()"}
FILES = {
    "f": "tests/compass/test_f.py",
    "g": "tests/compass/test_g.py",
    "C": "tests/test_c.py",
}
RAN = {
    "tests/compass/test_f.py": ["atom/m.py::f"],
    "tests/compass/test_g.py": ["atom/m.py::g"],
    "tests/test_c.py": ["atom/m.py::C.h"],
    "tests/test_ok.py": [],
}
ALL = ["C", "f", "g", "ok"]
READER = """import os, pathlib


def test_src():
    pathlib.Path({ran!r}, "src").touch()
    for d, _, names in os.walk(pathlib.Path(__file__).parents[1] / "atom"):
        for p in (pathlib.Path(d, n) for n in names if n.endswith(".py")):
            assert "monotonic" not in p.read_text(), p
"""
# A helper under tests/ that lists atom/ as it is imported, once per process,
# and two test files importing it.
ROOTS = """import os, pathlib

ROOTS = sorted(os.listdir(pathlib.Path(__file__).parents[1] / "atom"))
"""
USER = """import pathlib
from roots import ROOTS


def test_{name}():
    pathlib.Path({ran!r}, "{name}").touch()
    assert "n.py" not in ROOTS
"""
WHEEL = os.environ.get("COMPASS_COVERAGE_WHEEL")


def _git(root, *args):
    cmd = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    return subprocess.run(
        cmd + list(args), check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(root, edits):
    for path, text in edits.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).unlink() if text is None else (root / path).write_text(text)
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "c")
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    root = _tree(tmp_path / "ATOM", stamped=False)
    ran = tmp_path / "ran"
    ran.mkdir()
    (root / "tests" / "test_ok.py").write_text(
        f"import pathlib\ndef test_ok():\n    pathlib.Path({str(ran)!r}, 'ok').touch()\n"
    )
    (root / "scripts" / "compass" / "gpu_gate_triggers.txt").write_text("")
    # As the project's own .gitignore does: bytecode is no change to the tree.
    files = {".gitignore": "__pycache__/\n", "atom/m.py": MOD}
    files |= {
        p: TEST.format(name=n, ran=str(ran), call=CALLS[n]) for n, p in FILES.items()
    }
    _git(root, "init", "-q")
    sha = _commit(root, files)
    _git(root, "branch", "feature/atomcompass_new")
    return root, sha


def _map(root, sha, reads=None, older=False):
    """The map gate_map.sh writes of `root` at `sha`, its coverage given as RAN
    and its source reads as `reads`; `older`, as one recording no reads wrote it."""
    spec = importlib.util.spec_from_file_location(
        "gs", root / "scripts/compass/gate_select.py"
    )
    gs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gs)
    paths = [
        p.relative_to(root).as_posix()
        for p in root.rglob("*.py")
        if ".git" not in p.parts
    ]
    funcs = sorted({f for fs in RAN.values() for f in fs})
    ran = {t: [funcs.index(f) for f in fs] for t, fs in RAN.items()}
    ran |= {t: [] for t in reads or {}}
    tree = {p: gs.structure((root / p).read_bytes())[0] for p in paths}
    out = root.parent / "map.json"
    m = {"sha": sha, "funcs": funcs, "ran": ran, "tree": tree}
    out.write_text(json.dumps(m if older else m | {"reads": reads or {}}))
    return out


def _gate(root, mapfile, where="snapshot", stage=True):
    """The select lines, the test files that ran and the run of the gate with
    COMPASS_GATE_SELECT=mapfile, staged with it when `stage`, or in place."""
    ran = root.parent / "ran"
    if where == "snapshot":
        tars = root.parent / "tars"
        tars.mkdir(exist_ok=True)
        staged = f"COMPASS_GATE_SELECT={mapfile} " if mapfile and stage else ""
        snap = _run(root, f"{staged}scripts/compass/snapshot.sh {tars}")
        assert snap.returncode == 0, snap.stdout + snap.stderr
        into = root.parent / "out"
        into.mkdir()
        tar = tars / f"compass-{_git(root, 'rev-parse', 'HEAD')[:9]}.tar"
        subprocess.run(["tar", "-x", "-C", into, "-f", tar], check=True)
        root = into / "ATOM"
    out = _run(root, (f"COMPASS_GATE_SELECT={mapfile} " if mapfile else "") + "{gate}")
    lines = [line for line in out.stdout.splitlines() if line.startswith("select:")]
    return lines, sorted(p.name for p in ran.iterdir()), out


@pytest.mark.parametrize("where", ["snapshot", "checkout"])
def test_a_diff_inside_one_function_runs_exactly_the_files_that_ran_it(repo, where):
    root, sha = repo
    mapfile = _map(root, sha)
    _commit(root, {"atom/m.py": MOD.replace("return 2", "return 2 + 0")})
    lines, ran, out = _gate(root, mapfile, where)
    assert lines == [f"select: 1 of 4, map {sha[:9]}"], out.stdout
    assert ran == ["g"], out.stdout + out.stderr
    assert (
        out.returncode == 0 and out.stdout.splitlines()[-1] == "GATE_CPU_RC=0 PASSED"
    ), out.stdout


@pytest.mark.parametrize("path", ["atom/m.py", "atom/n.py"])
def test_a_file_that_read_the_changed_file_or_listed_its_directory_runs(repo, path):
    # A change no function body shows, in a file a test reads, or added where it lists.
    root, _ = repo
    sha = _commit(
        root, {"tests/test_src.py": READER.format(ran=str(root.parent / "ran"))}
    )
    reads = {"tests/test_src.py": ["atom/", "atom/__init__.py", "atom/m.py"]}
    mapfile = _map(root, sha, reads)
    text = (root / path).read_text() if (root / path).exists() else ""
    _commit(root, {path: "# monotonic\n" + text})
    lines, ran, out = _gate(root, mapfile)
    assert lines == [f"select: 1 of 5, map {sha[:9]}"] and ran == ["src"], out.stdout
    assert out.stdout.splitlines()[-1].startswith("GATE_CPU_RC=1 "), out.stdout


def test_a_changed_test_file_runs_and_a_design_doc_runs_nothing_more(repo):
    root, sha = repo
    mapfile = _map(root, sha)
    edits = {
        FILES["f"]: TEST.format(name="f", ran=str(root.parent / "ran"), call="f() + 0")
    }
    _commit(root, edits | {"atom/compass/design/x.md": "x\n"})
    lines, ran, out = _gate(root, mapfile)
    assert lines == [f"select: 1 of 4, map {sha[:9]}"] and ran == ["f"], out.stdout


def test_a_design_doc_alone_selects_nothing_and_passes(repo):
    root, sha = repo
    mapfile = _map(root, sha)
    _commit(root, {"atom/compass/design/x.md": "x\n"})
    lines, ran, out = _gate(root, mapfile)
    assert lines == [f"select: 0 of 4, map {sha[:9]}"] and ran == [], out.stdout
    assert out.stdout.splitlines()[-1] == "GATE_CPU_RC=0 PASSED", (
        out.stdout + out.stderr
    )


FALLBACKS = {
    "module-level code": (
        "atom/m.py",
        lambda t: t.replace("K = 1", "K = 2"),
        "atom/m.py changes or removes module-level code",
    ),
    "removed method": (
        "atom/m.py",
        lambda t: t.replace("def h", "def h2"),
        "atom/m.py removes C.h",
    ),
    "conftest": (
        "tests/conftest.py",
        lambda t: "",
        "tests/conftest.py is a conftest.py",
    ),
    "gate script": (
        "scripts/compass/gate_select.py",
        lambda t: t + "#\n",
        "gate_select.py is a conftest.py or the gate's own",
    ),
    "non-Python": ("README.md", lambda t: "x\n", "README.md is not Python"),
    "class gains a method": (
        "atom/m.py",
        lambda t: t + "\n    def __post_init__(self):\n        raise AssertionError\n",
        "atom/m.py adds C.__post_init__ to a class",
    ),
    "compass_harness": (
        "compass_harness/compass_harness/timesource.py",
        lambda t: "def sim_ns():\n    return 0\n",
        "the gate runs an installed copy of compass_harness/",
    ),
}


@pytest.mark.parametrize("case", sorted(FALLBACKS))
def test_each_fallback_runs_the_whole_gate_and_says_why(repo, case):
    root, sha = repo
    mapfile = _map(root, sha)
    path, edit, why = FALLBACKS[case]
    _commit(
        root, {path: edit((root / path).read_text() if (root / path).exists() else "")}
    )
    lines, ran, out = _gate(root, mapfile)
    assert (
        len(lines) == 1
        and lines[0].startswith("select: whole gate -- ")
        and why in lines[0]
    ), out.stdout
    assert ran == ALL, out.stdout + out.stderr


@pytest.mark.parametrize("staged", [True, False])
def test_a_map_not_behind_head_or_not_stamped_runs_the_whole_gate(repo, staged):
    root, _ = repo
    _git(root, "checkout", "-q", "-b", "side")
    mapfile = _map(
        root, _commit(root, {"atom/m.py": MOD.replace("return 1", "return 1 + 0")})
    )
    _git(root, "checkout", "-q", "-")
    _commit(root, {"atom/m.py": MOD.replace("return 2", "return 2 + 0")})
    lines, ran, out = _gate(root, mapfile, stage=staged)
    why = (
        "is not an ancestor of HEAD"
        if staged
        else "the snapshot has no .compass-select stamp"
    )
    assert len(lines) == 1 and why in lines[0] and ran == ALL, out.stdout


def test_a_map_without_source_reads_runs_the_whole_gate(repo):
    root, sha = repo
    mapfile = _map(root, sha, older=True)
    _commit(root, {"atom/m.py": MOD.replace("return 2", "return 2 + 0")})
    lines, ran, out = _gate(root, mapfile)
    why = f"map {sha[:9]} records no source reads"
    assert len(lines) == 1 and why in lines[0] and ran == ALL, out.stdout


def test_unset_the_run_is_unchanged(repo):
    root, _ = repo
    _commit(root, {"atom/m.py": MOD.replace("return 2", "return 2 + 0")})
    lines, ran, out = _gate(root, None)
    assert lines == [] and ran == ALL, out.stdout
    assert out.stdout.splitlines()[-1] == "GATE_CPU_RC=0 PASSED", (
        out.stdout + out.stderr
    )


def _built(root, sha):
    """The map this tree's gate_map.sh writes of a snapshot of `root` at `sha`."""
    tars = root.parent / "tars"
    tars.mkdir()
    assert _run(root, f"scripts/compass/snapshot.sh {tars}").returncode == 0
    into = root.parent / "mapped"
    into.mkdir()
    subprocess.run(
        ["tar", "-x", "-C", into, "-f", tars / f"compass-{sha[:9]}.tar"], check=True
    )
    mapfile = root.parent / "built.json"
    out = _run(
        into / "ATOM",
        f"COMPASS_COVERAGE_WHEEL={WHEEL} scripts/compass/gate_map.sh {mapfile}",
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert not (into / "ATOM" / "sitecustomize.py").exists()
    for p in (root.parent / "ran").iterdir():
        p.unlink()
    return mapfile, json.loads(mapfile.read_text())


NEEDS_WHEEL = pytest.mark.skipif(
    not WHEEL,
    reason="gate_map.sh installs the coverage.py wheel COMPASS_COVERAGE_WHEEL names",
)


@NEEDS_WHEEL
def test_the_map_holds_what_each_file_ran_its_subprocesses_included(repo):
    root, _ = repo
    sha = _commit(
        root, {"tests/test_src.py": READER.format(ran=str(root.parent / "ran"))}
    )
    mapfile, m = _built(root, sha)
    ran = {
        t: {m["funcs"][i] for i in ids if m["funcs"][i].startswith("atom/")}
        for t, ids in m["ran"].items()
    }
    assert m["sha"] == sha
    assert {t: ran.get(t) for t in RAN} == {t: set(fs) for t, fs in RAN.items()}, m[
        "ran"
    ]
    # What a test reads of the tree, os.walk's listing included, never what the
    # import system does.
    src = {"atom/", "atom/__init__.py", "atom/m.py"}
    assert src <= set(m["reads"].get("tests/test_src.py", ())), m["reads"]
    assert [t for t, rs in m["reads"].items() if "atom/m.py" in rs] == [
        "tests/test_src.py"
    ]
    _commit(root, {"atom/m.py": MOD.replace("return 2", "return 2 + 0")})
    lines, ran, out = _gate(root, mapfile)
    assert lines == [f"select: 2 of 5, map {sha[:9]}"] and ran == [
        "g",
        "src",
    ], out.stdout


@NEEDS_WHEEL
def test_an_added_file_runs_what_listed_its_directory_a_helper_s_importers_too(repo):
    root, _ = repo
    ran = str(root.parent / "ran")
    files = {"tests/test_src.py": READER.format(ran=ran), "tests/roots.py": ROOTS}
    files |= {f"tests/test_{n}.py": USER.format(name=n, ran=ran) for n in ("ra", "rb")}
    sha = _commit(root, files)
    mapfile, m = _built(root, sha)
    assert "atom/" in m["reads"].get("tests/roots.py", ()), m["reads"]
    _commit(root, {"atom/n.py": "# monotonic\n"})
    lines, ran, out = _gate(root, mapfile)
    assert lines == [f"select: 3 of 7, map {sha[:9]}"], out.stdout
    assert ran == ["ra", "rb", "src"], out.stdout + out.stderr
    assert out.stdout.splitlines()[-1].startswith("GATE_CPU_RC=1 "), out.stdout

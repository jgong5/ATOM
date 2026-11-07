# SPDX-License-Identifier: MIT
"""`regen_gpu_gate_baseline.sh` re-measures the GPU gate's baseline.

It runs the gate's suite twice over a `git archive` tarball and rewrites
`gpu_gate_baseline.env` and `gpu_gate_known_failures.txt` in the pair of the
first card's GPU target, or refuses and writes nothing. Each case runs the real
script over a throwaway tree, as `test_gate_gpu_verdict_through_pipe.py` runs
the gate: a stub pre-flight, stub `torch` and `aiter` packages, and a generated
suite. Nothing here reaches a GPU or its driver.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_gate_gpu_verdict_through_pipe import BASH, _preflight, _torch

REPO = Path(__file__).resolve().parents[2]
GIT = shutil.which("git")
COMPASS = "scripts/compass"
PAIRS = f"{COMPASS}/gpu_gate_baselines"
# A target the tree has no pair for, so the regenerated pair is the only one it
# can read.
ARCH = "gfx000"
WRITTEN = ("gpu_gate_baseline.env", "gpu_gate_known_failures.txt")
# Stands in for a file that stalls on the card; gpu_gate_skip.txt leaves it out.
STALL = (
    "import pytest\n\n\n@pytest.mark.parametrize('i', range(3))\n"
    "def test_stall(i):\n    assert False\n"
)

pytestmark = pytest.mark.skipif(
    BASH is None or GIT is None, reason="needs bash and git"
)

# With FLIP naming a file, the first run fails test_a and the second test_b:
# the same counts, different failing sets.
SUITE = """import os

import pytest

FLIP = os.environ.get("FLIP")
SECOND = bool(FLIP) and os.path.exists(FLIP)
if FLIP:
    open(FLIP, "a").close()


@pytest.mark.parametrize("i", range(3))
def test_ok(i):
    pass


@pytest.mark.parametrize("i", range(2))
def test_known(i):
    assert False


def test_a():
    assert SECOND or not FLIP


def test_b():
    assert not SECOND
"""


def _git(cwd, *argv):
    return subprocess.run(
        [GIT, "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t", *argv],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(root, files):
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def _setup(tmp_path):
    """A tarball of a base commit, and a Compass tree to regenerate in."""
    base = tmp_path / "base"
    suite = {"tests/test_suite.py": SUITE, "tests/test_stall.py": STALL}
    _write(base, {"atom/__init__.py": "", **suite})
    _git(base, "init", "-q")
    _git(base, "add", "-A")
    _git(base, "commit", "-qm", "base")
    _git(base, "archive", "--prefix=ATOM/", "-o", str(tmp_path / "base.tar"), "HEAD")

    root = tmp_path / "ATOM"
    # Another gate shard may be writing bytecode into __pycache__ as this copies.
    shutil.copytree(
        REPO / COMPASS, root / COMPASS, ignore=shutil.ignore_patterns("__pycache__")
    )
    artifacts = REPO / "atom" / "compass" / "artifacts"
    _write(
        root,
        {
            f"{COMPASS}/preflight.sh": _preflight(0),
            "atom/__init__.py": "",
            "atom/compass/__init__.py": "",
            "atom/compass/artifacts/__init__.py": "",
            **{
                f"atom/compass/artifacts/{name}": (artifacts / name).read_text()
                for name in ("provenance.py", "rules.py")
            },
            f"{COMPASS}/gpu_gate_skip.txt": "MI308X tests/test_stall.py\n",
            "torch/__init__.py": _torch(ARCH),
            "aiter/__init__.py": "raise RuntimeError('aiter was executed')\n",
            **suite,
        },
    )
    (root / COMPASS / "preflight.sh").chmod(0o755)
    _git(root / "aiter", "init", "-q")
    _git(root / "aiter", "add", "-A")
    _git(root / "aiter", "commit", "-qm", "aiter")
    _git(root / "aiter", "tag", "v0")
    return root, _git(base, "rev-parse", "HEAD")


def _run(tmp_path, root, argv, **env_over):
    env = {k: v for k, v in os.environ.items() if not k.startswith("COMPASS_")}
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env["PATH"]
    env["GIT_CEILING_DIRECTORIES"] = str(tmp_path)
    env["HIP_VISIBLE_DEVICES"] = "3"
    env["GPU_NAME"] = "AMD Instinct MI308X"
    env.update(env_over)
    return subprocess.run(
        [BASH, "-c", f"{argv} 2>&1"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def _pairs(root):
    """Every baseline file in the tree, by path."""
    return {p: p.read_bytes() for p in (root / PAIRS).rglob("*") if p.is_file()}


def test_the_gate_passes_against_the_baseline_it_regenerates(tmp_path):
    root, sha = _setup(tmp_path)
    before = _pairs(root)
    out = _run(
        tmp_path, root, f"{root / COMPASS}/regen_gpu_gate_baseline.sh ../base.tar"
    )
    assert out.returncode == 0, out.stdout
    pair = root / PAIRS / ARCH
    env = (pair / WRITTEN[0]).read_text().splitlines()
    assert [line for line in env if not line.startswith("#")] == [
        f"BASE_COMMIT={sha[:9]}",
        "BASE_PASSED=5",
        "BASE_FAILED=2",
        "BASE_COMPASS_TESTS=0",
        "BASE_LEFT_OUT=3",
        "BASE_TORCH=1",
        "BASE_ROCM=1",
        "BASE_AITER=v0",
    ]
    known = (pair / WRITTEN[1]).read_text().splitlines()
    assert [line for line in known if not line.startswith("#")] == [
        f"tests/test_suite.py::test_known[{i}]" for i in range(2)
    ]
    # The other targets' pairs are untouched.
    assert {p: b for p, b in _pairs(root).items() if pair not in p.parents} == before

    gate = _run(tmp_path, root, f"{root / COMPASS}/gate_gpu.sh")
    assert gate.stdout.splitlines()[-1] == "GATE_GPU_RC=0 PASSED", gate.stdout
    assert "toolchain differs" not in gate.stdout
    assert "left out on MI308X: tests/test_stall.py (3 tests)" in gate.stdout


# id: (exit status, text the refusal carries, how the setup changes)
REFUSALS = {
    "runs-disagree": (
        97,
        (
            "failed in run 1 only: tests/test_suite.py::test_a\n"
            "  failed in run 2 only: tests/test_suite.py::test_b\n"
        ),
        "flip",
    ),
    "not-an-archive": (94, "is not a git archive tarball naming its commit", "plain"),
    "aiter-unread": (96, "part of the stack reads UNKNOWN", "aiter-not-in-git"),
    "target-unread": (96, "torch reads no GPU target for card 3", "torch-without-cuda"),
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_a_refusal_names_its_reason_and_writes_nothing(tmp_path, case):
    rc, why, change = REFUSALS[case]
    root, _ = _setup(tmp_path)
    tar, env = "../base.tar", {}
    if change == "flip":
        env["FLIP"] = str(tmp_path / "flip")
    elif change == "plain":
        tar = "../plain.tar"
        subprocess.run(
            ["tar", "-cf", tmp_path / "plain.tar", "-C", tmp_path, "base"], check=True
        )
    elif change == "torch-without-cuda":
        _write(
            root,
            {"torch/__init__.py": "__version__ = 1\nclass version:\n    hip = 1\n"},
        )
    else:
        shutil.rmtree(root / "aiter" / ".git")
    before = _pairs(root)

    out = _run(
        tmp_path, root, f"{root / COMPASS}/regen_gpu_gate_baseline.sh {tar}", **env
    )
    assert out.returncode == rc, out.stdout
    assert why in out.stdout and "nothing was written" in out.stdout, out.stdout
    assert _pairs(root) == before

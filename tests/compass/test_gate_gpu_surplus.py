# SPDX-License-Identifier: MIT
"""`compass_compass_pass_count`: the surplus the GPU gate allows over its baseline.

The GPU gate judges an equality, not a floor: it expects its baseline pass count
plus whatever `tests/compass/` contributes on the tree in front of it, minus what
`tests/compass/` contributed at the baseline. That surplus therefore has to be
derived on any tree the gate can be pointed at, including the trees that carry no
Compass tests at all -- the integration branch, and every branch that has not
taken this phase's work yet. A gate that produces no verdict there cannot be used
to show that a branch is gate-neutral, which is most of what it is for.

The defect these tests pin: the derivation used to be
`pytest tests/compass --collect-only`, which exits 4 with `file or directory not
found` when the directory is absent, leaving the count empty and the gate
refusing with `GATE_GPU_RC=93` before a single test ran. An absent directory is a
count of zero. An unreadable one is still a refusal, and the two are told apart
by asking the filesystem rather than by parsing pytest's error text.

`python` is stubbed here, so these run on any box: the function's contract is
what it does with pytest's exit status and summary line, not what pytest finds.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "scripts" / "compass" / "_lib.sh"
GATE_GPU = REPO / "scripts" / "compass" / "gate_gpu.sh"

BASH = shutil.which("bash")


def _stub_python(bindir, rc, summary):
    """A `python` on PATH that prints `summary` and exits `rc`.

    The function under test reads pytest's exit status and its summary line and
    nothing else, so a stub is the whole contract. Nothing here invokes a real
    interpreter, which is what keeps these tests CPU-only and off the driver.
    """
    script = bindir / "python"
    script.write_text(
        "#!/bin/sh\n"
        "cat <<'STUB_EOF'\n"
        f"{summary}\n"
        "STUB_EOF\n"
        f"exit {rc}\n"
    )
    script.chmod(0o755)


def _count(tree, bindir):
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
    return subprocess.run(
        [BASH, "-c", f'. "{LIB}"; compass_compass_pass_count "$1"', "sh", str(tree)],
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "tree"
    (root / "scripts" / "compass").mkdir(parents=True)
    return root


@pytest.fixture
def bindir(tmp_path):
    d = tmp_path / "bin"
    d.mkdir()
    return d


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_absent_tests_compass_counts_zero(tree, bindir):
    """The case that broke every tree but this phase's own."""
    _stub_python(bindir, 4, "ERROR: file or directory not found: tests/compass")
    r = _count(tree, bindir)
    assert r.returncode == 0, r.stderr
    assert r.stdout == "0"


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_absent_directory_does_not_consult_pytest(tree, bindir):
    """Told apart by the filesystem, not by pytest's error text.

    The stub would report 99 passing tests if it were run at all; an absent
    directory must not reach it.
    """
    _stub_python(bindir, 0, "99 passed in 0.10s")
    r = _count(tree, bindir)
    assert r.stdout == "0"


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_present_directory_reports_the_pass_count(tree, bindir):
    (tree / "tests" / "compass").mkdir(parents=True)
    _stub_python(bindir, 0, "66 passed, 2 skipped in 1.20s")
    r = _count(tree, bindir)
    assert r.returncode == 0, r.stderr
    assert r.stdout == "66"


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_present_but_empty_directory_counts_zero(tree, bindir):
    """pytest rc=5 is `no tests collected`, which over a real directory is a
    real zero rather than an answer that could not be read."""
    (tree / "tests" / "compass").mkdir(parents=True)
    _stub_python(bindir, 5, "no tests ran in 0.01s")
    r = _count(tree, bindir)
    assert r.returncode == 0, r.stderr
    assert r.stdout == "0"


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_unreadable_count_still_refuses(tree, bindir):
    """A present directory whose tests do not complete has no derivable surplus.

    This is the half of the old behaviour that was right and has to stay: a
    delta judged against a surplus with no source is not a measurement.
    """
    (tree / "tests" / "compass").mkdir(parents=True)
    _stub_python(bindir, 2, "INTERNALERROR> something went wrong")
    r = _count(tree, bindir)
    assert r.returncode == 93
    assert "no source" in r.stderr


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_failing_compass_tests_refuse(tree, bindir):
    (tree / "tests" / "compass").mkdir(parents=True)
    _stub_python(bindir, 1, "64 passed, 2 failed in 1.30s")
    r = _count(tree, bindir)
    assert r.returncode == 93


def test_gate_gpu_derives_the_surplus_through_the_helper():
    """The gate must not carry a second copy of the derivation.

    A duplicate would drift from the one these tests cover, and the absent-
    directory case is exactly where a duplicate would be wrong.
    """
    text = GATE_GPU.read_text()
    assert "compass_compass_pass_count" in text
    assert "--collect-only" not in text


def test_gate_gpu_states_the_expectation_for_a_tree_without_compass_tests():
    """4779 + 0 - 49 = 4730, the figure a tree without this phase's tests
    measures. Stated in the gate so the verdict on such a tree is readable
    without re-deriving it."""
    text = GATE_GPU.read_text()
    assert "4730" in text

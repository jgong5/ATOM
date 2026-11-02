# SPDX-License-Identifier: MIT
"""`gate_gpu.sh` leaves a file `gpu_gate_skip.txt` names out on that GPU only.

Each case runs the real script over a throwaway tree, as
`test_gate_gpu_verdict_through_pipe.py` does, with a stub torch that names the
card. The listed file's tests fail when run, standing in for a stall, and the
baseline counts them as passes, as it counted the stalling file's.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_gate_gpu_verdict_through_pipe import (
    BASH,
    EXPECT,
    GATE,
    KNOWN,
    KNOWN_TXT,
    _preflight,
    _suite,
    _torch,
)

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")


def _run(tmp_path, card):
    root = tmp_path / "ATOM"
    # Another gate shard may be writing bytecode into __pycache__ as this copies.
    shutil.copytree(
        REPO / "scripts" / "compass",
        root / "scripts" / "compass",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    files = {
        "scripts/compass/preflight.sh": _preflight(0),
        KNOWN_TXT: "\n".join(KNOWN) + "\n",
        "scripts/compass/gpu_gate_skip.txt": "# a comment\nMI308X tests/test_stall.py\n",
        "atom/__init__.py": "",
        "torch/__init__.py": _torch(),
        "tests/test_suite.py": _suite(EXPECT - 3),
        "tests/test_stall.py": (
            "import pytest\n\n\n@pytest.mark.parametrize('i', range(3))\n"
            "def test_stall(i):\n    assert False\n"
        ),
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "scripts" / "compass" / "preflight.sh").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("COMPASS_")}
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env["PATH"]
    env["GIT_CEILING_DIRECTORIES"] = str(tmp_path)
    env["GPU_NAME"] = card
    out = subprocess.run(
        [BASH, "-c", f"{root / GATE} 2>&1"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    return out, out.stdout.splitlines()[-1]


def test_a_listed_file_is_left_out_on_its_gpu_and_off_the_expectation(tmp_path):
    out, verdict = _run(tmp_path, "AMD Instinct MI308X")
    assert verdict == "GATE_GPU_RC=0 PASSED", out.stdout[-3000:]
    assert "left out on MI308X: tests/test_stall.py (3 tests)" in out.stdout
    assert f"expected: {EXPECT - 3} passed" in out.stdout


def test_a_listed_file_runs_on_another_gpu(tmp_path):
    out, verdict = _run(tmp_path, "AMD Instinct MI355X")
    assert out.returncode == 1
    assert "first tests/test_stall.py::test_stall[0]" in verdict, verdict

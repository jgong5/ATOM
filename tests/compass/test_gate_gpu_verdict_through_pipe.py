# SPDX-License-Identifier: MIT
"""`gate_gpu.sh`'s verdict has to survive being piped, and say why.

A pipeline's status is its last command's, so `gate_gpu.sh | tail -6` hands the
caller tail's 0 whatever the gate exited. What the script controls is its last
line: the verdict, which has to carry the reason, because every reason goes to
stderr and the after-run pre-flight readout sits just above the verdict on
stdout. `2>/dev/null | tail -6` keeps only that readout and the verdict.

Each case runs the real script over a throwaway tree: a stub pre-flight, a
generated suite whose passes and failures match the script's own baseline
constants, and one change that sends the run down a single exit. Nothing here
reaches a GPU or its driver.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GATE = "scripts/compass/gate_gpu.sh"
BASH = shutil.which("bash")
BASE = dict(re.findall(r"^(BASE_\w+)=(\S+)$", (REPO / GATE).read_text(), re.MULTILINE))
KNOWN = [
    f"tests/test_suite.py::test_known[{i}]" for i in range(int(BASE["BASE_FAILED"]))
]

pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")


def _suite(passing, extra=""):
    return (
        "import pytest\n\n\n"
        f"@pytest.mark.parametrize('i', range({passing}))\n"
        "def test_ok(i):\n    pass\n\n\n"
        f"@pytest.mark.parametrize('i', range({len(KNOWN)}))\n"
        "def test_known(i):\n    assert False\n" + extra
    )


def _preflight(rc):
    return f"#!/bin/sh\nprintf 'pre-flight %s\\n' 1 2 3 4 5 6\nexit {rc}\n"


# id: (exit status, text the verdict must carry, files changed, gate arguments)
CASES = {
    "passed": (
        0,
        "",
        {
            "tests/test_suite.py": _suite(
                int(BASE["BASE_PASSED"]) - int(BASE["BASE_COMPASS_TESTS"])
            )
        },
        "",
    ),
    "new-failure": (
        1,
        "first tests/test_suite.py::test_new; 2 more",
        {"tests/test_suite.py": _suite(1, "\n\ndef test_new():\n    assert False\n")},
        "",
    ),
    "no-summary": (1, "no usable pytest summary", {"tests/test_suite.py": ""}, ""),
    "interrupted": (
        1,
        "pytest rc=2",
        {
            "tests/test_suite.py": (
                "def test_a():\n    pass\n\n\ndef test_b():\n    raise KeyboardInterrupt\n"
            )
        },
        "",
    ),
    "known-missing": (
        93,
        "known-failures list is missing",
        {"scripts/compass/gpu_gate_known_failures.txt": None},
        "",
    ),
    "known-disagrees": (
        93,
        "BASE_FAILED disagrees",
        {"scripts/compass/gpu_gate_known_failures.txt": ""},
        "",
    ),
    "compass-count": (
        93,
        "pass count has no source",
        {"tests/compass/test_c.py": "def test_c():\n    assert False\n"},
        "",
    ),
    "r-flag": (95, "refused a -r argument", {}, " -rE"),
    "preflight": (
        91,
        "pre-flight failed",
        {"scripts/compass/preflight.sh": _preflight(1)},
        "",
    ),
    "atom": (
        91,
        "atom does not import",
        {"atom/__init__.py": "raise ImportError\n"},
        "",
    ),
    "other-checkout": (99, "could not be resolved", {}, ""),
}


def _tree(root, case):
    shutil.copytree(REPO / "scripts" / "compass", root / "scripts" / "compass")
    files = {
        "scripts/compass/preflight.sh": _preflight(0),
        "scripts/compass/gpu_gate_known_failures.txt": "\n".join(KNOWN) + "\n",
        "atom/__init__.py": "",
        "tests/test_suite.py": _suite(1),
        **CASES[case][2],
    }
    for rel, text in files.items():
        path = root / rel
        path.unlink(missing_ok=True)
        if text is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    (root / "scripts" / "compass" / "preflight.sh").chmod(0o755)
    return root


def _run(tmp_path, case, pipe):
    root = _tree(tmp_path / "ATOM", case)
    cwd = _tree(tmp_path / "other", case) if case == "other-checkout" else root
    env = {k: v for k, v in os.environ.items() if not k.startswith("COMPASS_")}
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env["PATH"]
    env["GIT_CEILING_DIRECTORIES"] = str(tmp_path)
    command = f"{root / GATE}{CASES[case][3]} {pipe}"
    return subprocess.run(
        [BASH, "-c", command],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def _expected(case):
    rc, why = CASES[case][:2]
    return f"GATE_GPU_RC={rc} " + ("PASSED" if rc == 0 else "NOT PASSED -- "), why


@pytest.mark.parametrize("case", list(CASES))
def test_the_verdict_key_is_printed_exactly_once_and_last(tmp_path, case):
    out = _run(tmp_path, case, "2>&1")
    lines = out.stdout.splitlines()
    assert out.returncode == CASES[case][0], out.stdout
    assert [line for line in lines if "GATE_GPU_RC=" in line] == lines[-1:], lines
    head, why = _expected(case)
    assert lines[-1].startswith(head) and why in lines[-1], lines[-1]
    if CASES[case][0] == 0:
        assert lines[-1] == head, lines[-1]


@pytest.mark.parametrize(
    "pipe", ["", "| tail -6", "2>&1 | tail -6", "2>/dev/null | tail -6"]
)
@pytest.mark.parametrize("case", ["new-failure", "known-disagrees"])
def test_a_piped_run_still_reads_as_not_passed_and_says_why(tmp_path, case, pipe):
    out = _run(tmp_path, case, pipe)
    # Unpiped is the control: the status every piped run loses to tail.
    assert out.returncode == (0 if pipe else CASES[case][0]), out.stdout
    kept = out.stdout.splitlines()
    head, why = _expected(case)
    assert (
        kept and kept[-1].startswith(head) and why in kept[-1]
    ), f"`{pipe or 'unpiped'}` ends without a reason on the verdict: {kept!r}"

# SPDX-License-Identifier: MIT
"""`gate_gpu.sh` runs the driver-run tests, once, with their inputs.

The driver-run tests skip without the slice model, the image model, the machine
spec and a router built at the tree's commit, and a skip used to count as
accounted. On a tree whose tests read the slice model the gate now sets each
input, refuses by name when one is missing, runs tests/compass once on every card
named and the rest of the suite on the first card, and fails on a driver-run skip.

Each case runs the real script over a throwaway tree, as
`test_gate_gpu_verdict_through_pipe.py` does, with a stub router check.
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
    _preflight,
    _suite,
)

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")

RECORD = """
import os


def _record(what):
    with open(os.environ["RUNS"], "a") as f:
        f.write(f"{what} {os.environ.get('HIP_VISIBLE_DEVICES')} "
                f"{os.environ.get('ATOM_COMPASS_SLICE_MODEL')}\\n")
"""
DRIVER = (
    "import pytest\n"
    + RECORD
    + "\n\ndef test_slice():\n    # reads ATOM_COMPASS_SLICE_MODEL\n    _record('compass')\n"
)
# Records which compass_harness the harness Python imports.
SMOKE = (
    "import subprocess\n" + RECORD + "\n\ndef test_cctraces():\n"
    "    where = subprocess.run(\n"
    "        [os.environ['COMPASS_HARNESS_PYTHON'], '-c',\n"
    "         'import compass_harness; print(compass_harness.WHERE)'],\n"
    "        cwd='/', capture_output=True, text=True).stdout.strip()\n"
    "    _record(f'cctraces {where}')\n"
)
# Refuses a router whose path does not end in `atomesh`, as router_binary refuses
# one built at another commit, by raising with its reason.
ROUTER = (
    "def router_binary(path):\n"
    "    if not path.endswith('/atomesh'):\n"
    "        raise ValueError(f'router binary {path} names no commit')\n"
)


def _tree(tmp_path, driver=DRIVER, smoke=SMOKE):
    root = tmp_path / "ATOM"
    shutil.copytree(REPO / "scripts" / "compass", root / "scripts" / "compass")
    main = _suite(EXPECT - 1, RECORD + "\n\ndef test_main():\n    _record('main')\n")
    files = {
        "scripts/compass/preflight.sh": _preflight(0),
        "scripts/compass/gpu_gate_known_failures.txt": "\n".join(KNOWN) + "\n",
        "atom/__init__.py": "",
        "atom/compass/__init__.py": "",
        "atom/compass/run.py": ROUTER,
        "torch/__init__.py": "__version__ = 1\nclass version:\n    hip = 1\n",
        "tests/test_suite.py": main,
        "tests/compass/test_d.py": driver,
        "tests/compass/test_cctraces_sim_smoke.py": smoke,
        "compass_harness/compass_harness/__init__.py": "WHERE = 'tree'\n",
    }
    for rel, text in files.items():
        if text is None:
            continue
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "scripts" / "compass" / "preflight.sh").chmod(0o755)
    hub = tmp_path / "hf" / "hub"
    for name in ("Qwen3-0.6B", "Qwen3.5-9B", "Qwen3.8-27B"):
        (hub / f"models--Qwen--{name}" / "snapshots" / "abc").mkdir(parents=True)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "atomesh").write_text("")
    (tmp_path / "traces.jsonl").write_text("{}\n")
    return root


def _run(tmp_path, root, **env_over):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("COMPASS_", "ATOM_COMPASS_")) and k != "ATOMESH"
    }
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env["PATH"]
    env["GIT_CEILING_DIRECTORIES"] = str(tmp_path)
    env["HF_HOME"] = str(tmp_path / "hf")
    env["ATOMESH"] = str(tmp_path / "bin" / "atomesh")
    env["ATOM_COMPASS_CCTRACES"] = str(tmp_path / "traces.jsonl")
    env["COMPASS_HARNESS_PYTHON"] = env["COMPASS_AIPERF_PYTHON"] = sys.executable
    env["HIP_VISIBLE_DEVICES"] = "4,5,6,7"
    env["RUNS"] = str(root / "runs.txt")
    env.update(env_over)
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


def test_driver_run_tests_run_once_on_every_card_with_their_inputs(tmp_path):
    root = _tree(tmp_path)
    out, verdict = _run(tmp_path, root)
    assert out.returncode == 0, out.stdout[-3000:]
    assert verdict == "GATE_GPU_RC=0 PASSED"
    hub = f"{tmp_path}/hf/hub/models--Qwen--"
    snap, big = f"{hub}Qwen3-0.6B/snapshots/abc/", f"{hub}Qwen3.8-27B/snapshots/abc/"
    runs = (root / "runs.txt").read_text().splitlines()
    assert sorted(runs) == sorted(
        [
            f"compass 4,5,6,7 {snap}",
            f"cctraces tree 4,5,6,7 {big}",
            f"main 4 {snap}",
        ]
    )


@pytest.mark.parametrize(
    "env, named",
    [
        ({"HF_HOME": "/nonexistent"}, "ATOM_COMPASS_SLICE_MODEL, a Qwen3-0.6B"),
        ({"ATOM_COMPASS_IMAGE_MODEL": "/nonexistent"}, "ATOM_COMPASS_IMAGE_MODEL"),
        (
            {"ATOM_COMPASS_CCTRACES_MODEL": "/nonexistent"},
            "ATOM_COMPASS_CCTRACES_MODEL, a Qwen3.8-27B",
        ),
        (
            {"ATOM_COMPASS_CCTRACES": "/nonexistent"},
            "ATOM_COMPASS_CCTRACES /nonexistent",
        ),
        (
            {"COMPASS_HARNESS_PYTHON": "/nonexistent"},
            "COMPASS_HARNESS_PYTHON /nonexistent",
        ),
        (
            {"COMPASS_AIPERF_PYTHON": "/nonexistent"},
            "COMPASS_AIPERF_PYTHON /nonexistent",
        ),
        (
            {"ATOM_COMPASS_SLICE_MACHINE": "/nonexistent.json"},
            "ATOM_COMPASS_SLICE_MACHINE",
        ),
        (
            {"ATOMESH": "/x/atomesh-old"},
            "ATOMESH, ValueError: router binary /x/atomesh-old",
        ),
        (
            {"HIP_VISIBLE_DEVICES": "4,5"},
            "four cards in HIP_VISIBLE_DEVICES, which names 4,5",
        ),
    ],
)
def test_a_missing_input_refuses_by_name_before_any_test(tmp_path, env, named):
    root = _tree(tmp_path)
    out, verdict = _run(tmp_path, root, **env)
    assert out.returncode == 92, out.stdout[-3000:]
    assert verdict.startswith("GATE_GPU_RC=92 NOT PASSED -- the driver-run tests lack ")
    assert named in verdict, verdict
    assert not (root / "runs.txt").exists()


@pytest.mark.parametrize(
    "which, reason, where",
    [
        ("driver", "needs ATOMESH, atomesh", "test_d.py"),
        ("smoke", "needs COMPASS_AIPERF_PYTHON", "test_cctraces_sim_smoke.py"),
    ],
)
def test_a_driver_run_skip_fails_by_name(tmp_path, which, reason, where):
    files = {"driver": DRIVER, "smoke": SMOKE}
    files[which] = files[which].replace(
        "def test_",
        f"import pytest\n\n\n@pytest.mark.skip(reason='{reason}')\ndef test_",
    )
    root = _tree(tmp_path, **files)
    out, verdict = _run(tmp_path, root)
    assert out.returncode == 1, out.stdout[-3000:]
    assert (
        f"-- 1 driver-run skip line(s), first SKIPPED [1] tests/compass/{where}"
        in verdict
    )
    assert reason in verdict


# A tests/compass run that collected nothing (pytest rc 5) completed and passes
# none; one interrupted by a collection error (rc 2) did not complete. rc 1, tests
# failed, is the pipe test's `compass-failure` case.
@pytest.mark.parametrize(
    "driver, rc, verdict_is",
    [
        ("# ATOM_COMPASS_SLICE_MODEL\n", 0, "GATE_GPU_RC=0 PASSED"),
        (
            "# ATOM_COMPASS_SLICE_MODEL\nimport no_such_module\n",
            1,
            "GATE_GPU_RC=1 NOT PASSED -- pytest rc=2 over tests/compass; the run did not complete",
        ),
    ],
)
def test_which_tests_compass_exit_codes_are_a_completed_run(
    tmp_path, driver, rc, verdict_is
):
    root = _tree(tmp_path, driver, smoke=None)
    out, verdict = _run(tmp_path, root)
    assert out.returncode == rc, out.stdout[-3000:]
    assert verdict == verdict_is

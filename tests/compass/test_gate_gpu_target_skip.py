# SPDX-License-Identifier: MIT
"""`gpu_gate_skip.txt` lines keyed by a GPU target, and lines for tests/compass.

A card whose name torch reads without a model in it is matched by its target.
A tests/compass file left out runs in neither of the gate's runs and is not
taken off the expectation, which counts what tests/compass passes. Where every
file reading the slice model is left out, no driver-run test runs, so the gate
requires none of their inputs and not four cards.

Each case runs the real script over a throwaway tree, as
`test_gate_gpu_driver_run.py` does, with a stub torch whose card reads the
gfx950 target under a name without "MI".
"""

import shutil

import pytest
from test_gate_gpu_driver_run import _run, _tree
from test_gate_gpu_verdict_through_pipe import BASH, EXPECT, _suite

pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")

NAME = "AMD Radeon Graphics"
STALL = (
    "import pytest\n\n\n@pytest.mark.parametrize('i', range(3))\n"
    "def test_stall(i):\n    assert False\n"
)


def _skip(root, *lines):
    (root / "scripts/compass/gpu_gate_skip.txt").write_text("\n".join(lines) + "\n")


def test_a_target_keyed_line_leaves_a_suite_file_out_on_that_target(tmp_path):
    root = _tree(tmp_path)
    (root / "tests/test_suite.py").write_text(_suite(EXPECT - 3))
    (root / "tests/test_stall.py").write_text(STALL)
    _skip(root, "gfx950 tests/test_stall.py")
    out, verdict = _run(tmp_path, root, GPU_NAME=NAME)
    assert verdict == "GATE_GPU_RC=0 PASSED", out.stdout[-3000:]
    assert "left out on gfx950: tests/test_stall.py (3 tests)" in out.stdout


def test_a_target_keyed_line_does_not_reach_another_target(tmp_path):
    root = _tree(tmp_path)
    (root / "tests/test_suite.py").write_text(_suite(EXPECT - 3))
    (root / "tests/test_stall.py").write_text(STALL)
    _skip(root, "gfx942 tests/test_stall.py")
    out, verdict = _run(tmp_path, root, GPU_NAME=NAME)
    assert out.returncode == 1, out.stdout[-3000:]
    assert "first tests/test_stall.py::test_stall[0]" in verdict, verdict


def test_left_out_driver_run_files_waive_their_inputs_and_the_four_cards(tmp_path):
    root = _tree(tmp_path)
    shutil.rmtree(tmp_path / "hf/hub/models--Qwen--Qwen3-0.6B")
    _skip(
        root,
        "gfx950 tests/compass/test_d.py",
        "gfx950 tests/compass/test_cctraces_sim_smoke.py",
    )
    out, verdict = _run(
        tmp_path,
        root,
        GPU_NAME=NAME,
        HIP_VISIBLE_DEVICES="4",
        COMPASS_HARNESS_PYTHON="/nonexistent",
    )
    assert verdict == "GATE_GPU_RC=0 PASSED", out.stdout[-3000:]
    assert "left out on gfx950: tests/compass/test_d.py\n" in out.stdout
    runs = (root / "runs.txt").read_text().splitlines()
    assert [r.split()[0] for r in runs] == ["main"]


def test_an_input_a_running_file_reads_is_still_required(tmp_path):
    root = _tree(tmp_path)
    shutil.rmtree(tmp_path / "hf/hub/models--Qwen--Qwen3-0.6B")
    _skip(root, "gfx950 tests/compass/test_cctraces_sim_smoke.py")
    out, verdict = _run(tmp_path, root, GPU_NAME=NAME, HIP_VISIBLE_DEVICES="4")
    assert out.returncode == 92, out.stdout[-3000:]
    assert "ATOM_COMPASS_SLICE_MODEL, a Qwen3-0.6B" in verdict
    assert "four cards in HIP_VISIBLE_DEVICES, which names 4" in verdict

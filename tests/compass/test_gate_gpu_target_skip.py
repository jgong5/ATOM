# SPDX-License-Identifier: MIT
"""`gpu_gate_skip.txt`'s `tests/compass` lines and its `<target> driver-run`
line, on the first card's GPU target.

A `tests/compass` file left out runs in neither of the gate's runs and is not
taken off the expectation, which counts what `tests/compass` passes. On a
target marked `driver-run`, whose nodes hold no driver-run inputs, the gate
sets none, needs one card and accepts a driver-run skip; elsewhere a missing
input still refuses the run.

Each case runs the real script over a throwaway tree, as
`test_gate_gpu_driver_run.py` does, with a stub torch whose card reads gfx950.
"""

import shutil

import pytest
from test_gate_gpu_driver_run import DRIVER, _run, _tree
from test_gate_gpu_verdict_through_pipe import BASH

pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")

#: The driver-run test, skipping as one does without its slice model.
SKIPPING = DRIVER.replace(
    "def test_",
    "import pytest\n\n\n@pytest.mark.skip(reason='needs ATOM_COMPASS_SLICE_MODEL')\n"
    "def test_",
)


def _skip(root, *lines):
    (root / "scripts/compass/gpu_gate_skip.txt").write_text("\n".join(lines) + "\n")


def test_a_left_out_tests_compass_file_leaves_its_run_and_not_the_expectation(
    tmp_path,
):
    root = _tree(tmp_path)
    (root / "tests/compass/test_bad.py").write_text("def test_bad():\n    assert 0\n")
    _skip(root, "gfx950 tests/compass/test_bad.py")
    out, verdict = _run(tmp_path, root)
    assert verdict == "GATE_GPU_RC=0 PASSED", out.stdout[-3000:]
    assert "left out on gfx950: tests/compass/test_bad.py\n" in out.stdout


def test_a_driver_run_target_sets_no_input_needs_one_card_and_accepts_its_skips(
    tmp_path,
):
    root = _tree(tmp_path, SKIPPING)
    shutil.rmtree(tmp_path / "hf")
    _skip(root, "gfx950 driver-run")
    out, verdict = _run(tmp_path, root, HIP_VISIBLE_DEVICES="4", ATOMESH="/x/old")
    assert verdict == "GATE_GPU_RC=0 PASSED", out.stdout[-3000:]
    assert "driver-run inputs: none on gfx950\n" in out.stdout
    assert "driver-run: 1 skip line(s) accepted on gfx950" in out.stdout


def test_another_targets_driver_run_line_leaves_every_input_required(tmp_path):
    root = _tree(tmp_path, SKIPPING)
    shutil.rmtree(tmp_path / "hf")
    _skip(root, "gfx942 driver-run")
    out, verdict = _run(tmp_path, root, HIP_VISIBLE_DEVICES="4")
    assert out.returncode == 92, out.stdout[-3000:]
    assert "ATOM_COMPASS_SLICE_MODEL, a Qwen3-0.6B" in verdict
    assert "four cards in HIP_VISIBLE_DEVICES, which names 4" in verdict

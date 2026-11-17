# SPDX-License-Identifier: MIT
"""`scripts/compass/one_real.py`: one real run against its simulated runs, each
metric's relative error, and a refusal of runs given other client arguments."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_validate_curves import _run

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compass" / "one_real.py"
ROOT = SCRIPT.parents[2]


def _one_real(*runs):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, runs)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT)},
        check=False,
    )


def test_each_latency_error_is_the_simulated_run_s_scale_less_one(tmp_path):
    real = _run(tmp_path / "real1", 4)
    sim = _run(tmp_path / "step_linear", 4, scale=1.1, server=False)

    done = _one_real(real, sim)

    assert done.returncode == 0, done.stderr
    rows = json.loads(done.stdout.splitlines()[-1])
    for name in ("TTFT p50 ms", "TTFT p99 ms", "TPOT p50 ms", "E2E p50 ms"):
        assert rows["step_linear"][name] == pytest.approx(1.1 * rows["real"][name])
    table = {line[:34].strip(): line for line in done.stdout.splitlines()[1:-1]}
    assert table["TTFT p50 ms"].split()[-1] == "+10.0%"
    assert rows["real"]["requests"] == rows["step_linear"]["requests"] > 0


def test_runs_given_other_client_arguments_exit_2(tmp_path):
    real = _run(tmp_path / "real1", 4)
    sim = _run(tmp_path / "step_linear", 8, server=False)

    done = _one_real(real, sim)

    assert done.returncode == 2
    assert "other client arguments" in done.stderr

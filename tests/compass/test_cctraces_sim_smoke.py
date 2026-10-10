# SPDX-License-Identifier: MIT
"""``scripts/compass/cctraces_sim.sh``: one cc-traces session through
agentx-harness on a simulated TP1 run, twice, with different ``PYTHONHASHSEED``.

Each run is the script end to end: ATOM's API server with ``--compass-run``,
``aiperf profile`` with the AgentX scenario and compass-harness, ended by a
short simulated duration. The two step tables must be byte-identical and carry
the workload the script hashed into the run file.

Skips without a driver (the engine core and the TP1 worker import aiter), and
without ``ATOM_COMPASS_SLICE_MODEL`` (a model directory holding a config and a
tokenizer), ``ATOM_COMPASS_CCTRACES`` (a cc-traces ``traces.jsonl``) and
``COMPASS_HARNESS_PYTHON`` (a Python with agentx-harness 56a0cf70 and
compass-harness installed).
"""

import copy
import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch
from test_memory_readings import DOCUMENT

from atom.compass import run as compass_run
from atom.compass.backends.shape import Coefficients
from atom.compass.detect.determinism import CONFIGURATION_PREFIX

MODEL = os.environ.get("ATOM_COMPASS_SLICE_MODEL")
TRACES = os.environ.get("ATOM_COMPASS_CCTRACES")
HARNESS_PYTHON = os.environ.get("COMPASS_HARNESS_PYTHON")
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and MODEL and TRACES and HARNESS_PYTHON),
    reason="needs a driver, ATOM_COMPASS_SLICE_MODEL, ATOM_COMPASS_CCTRACES "
    "and COMPASS_HARNESS_PYTHON",
)

TREE = Path(__file__).resolve().parents[2]
SCRIPT = TREE / "scripts/compass/cctraces_sim.sh"


def _template(path: Path) -> Path:
    """A run file whose declared tokenizer also covers the model's architecture."""
    machine = copy.deepcopy(DOCUMENT)
    arch = json.loads((Path(MODEL) / "config.json").read_text())["architectures"][0]
    machine["host"]["tokenizers"][0]["applies_to"].append(arch)
    run = {
        "bound_s": 1e7,
        "admission_path": "serving",
        "ipc_s": 2.0**-14,
        "stream_s": 2.0**-12,
        "coefficients": dataclasses.asdict(Coefficients()),
        "machine": machine,
        "parameter_count": 8_000_000_000,
    }
    path.write_text(json.dumps(run))
    return path


@pytest.fixture(scope="module")
def cells(tmp_path_factory):
    root = tmp_path_factory.mktemp("cctraces")
    template = _template(root / "template.json")
    out = []
    for seed in ("1", "2"):
        cell = root / f"seed{seed}"
        env = dict(
            os.environ,
            PYTHONHASHSEED=seed,
            MODEL=MODEL,
            TRACES=TRACES,
            HARNESS_PYTHON=HARNESS_PYTHON,
            SESSIONS="1",
            DURATION_S="60",
        )
        done = subprocess.run(
            ["bash", str(SCRIPT), str(template), str(cell)],
            cwd="/",
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
        out.append((cell, done.stdout.strip().splitlines()[-1]))
    return out


def test_both_runs_answer_every_request_and_refuse_nothing(cells):
    for _, result in cells:
        fields = dict(f.split("=", 1) for f in result.split()[1:])
        assert int(fields["requests"]) > 0, result
        assert (fields["errors"], fields["refusals"]) == ("0", "0"), result
        assert fields["coverage_report"] == "False", result


def test_the_step_tables_are_byte_identical_and_name_the_workload(cells):
    (left, _), (right, _) = cells
    table = (left / compass_run.STEP_TABLE_FILE).read_bytes()
    assert table == (right / compass_run.STEP_TABLE_FILE).read_bytes()
    first = table.decode().splitlines()[0]
    workload = json.loads(first[len(CONFIGURATION_PREFIX) :])["workload"]
    trace = (left / "traces/00000.json").read_bytes()
    assert workload["traces_sha256"] == [hashlib.sha256(trace).hexdigest()]
    assert "--benchmark-duration" in workload["aiperf_args"]

# SPDX-License-Identifier: MIT
"""`compass measure ops --coverage` builds an FP8 config as a worker on its
plan's device parses it, wherever it runs. aiter resolves its fp8 dtype at
import from the device it finds (`test_check_fp8_arch`), ATOM's config holds
that dtype in its quantization config, and the coverage reads the op graphs and
points keyed under that config.

The coverage runs as a command of its own, a process that has not imported
aiter, its report replaced by the quantization of the config it was handed.

No device: the coverage declares its plan's.
"""

import json
import os
import subprocess
import sys

import pytest
from test_check_fp8_arch import FP8, TREE, worker
from test_key_quantization_remap import QWEN35

from atom.compass.run.step_grid import keyed

#: ATOM's arguments past the model: an eager TP4 server of Qwen3.5's FP8
#: checkpoint, as `measure ops` measures one.
TP4 = ["-tp", "4", "--enforce-eager"]
#: `compass measure ops`, its coverage report the quantization of the config
#: it builds, and the tokenizer, which no checkpoint here ships, its ids alone.
COVERAGE = """\
import sys
from types import SimpleNamespace
from atom.compass.bench import ops
from atom.compass.run.step_grid import keyed
from atom.model_engine import llm_engine

llm_engine._load_tokenizer = lambda *a: SimpleNamespace(bos_token_id=1, eos_token_id=2)
ops.coverage = lambda store, root, config, *a: keyed(config)["quantization"]
ops.main(sys.argv[1:])
"""


@pytest.mark.parametrize("arch", sorted(FP8))
def test_coverage_builds_an_fp8_config_as_its_plan_s_worker_parses_it(
    tmp_path, monkeypatch, arch
):
    config = worker(tmp_path, monkeypatch, arch, QWEN35, *TP4)
    out = tmp_path / "out"
    out.mkdir()
    (out / "plan.jsonl").write_text("")
    (out / "run.json").write_text(json.dumps({"device": [{"arch": arch}]}))
    argv = ["--coverage", "graded", "--store", str(tmp_path), "--out", str(out)]
    env = dict(os.environ, PYTHONPATH=str(TREE))

    done = subprocess.run(
        [sys.executable, "-c", COVERAGE, *argv, "--", "--model", config.model, *TP4],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )

    assert done.returncode == 0, done.stderr[-3000:]
    assert json.loads(done.stdout.splitlines()[-1]) == keyed(config)["quantization"]

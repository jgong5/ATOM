# SPDX-License-Identifier: MIT
"""ATOM's runner whose model refuses at construction exits with that refusal
alone: the exit handler it registers before building the model runs at
interpreter exit and releases what was built, with no model to release.

The runner is built driverless in a process of its own by
`test_qwen3_pp_stages`' script. Qwen3-0.6B refuses a layer partition that does
not sum to its depth while it builds its layers.
"""

import os
import shutil
import subprocess
import sys

from test_qwen3_pp_stages import CONFIG, STAGE, TREE


def test_a_runner_whose_model_refuses_exits_with_that_refusal_alone(tmp_path):
    (tmp_path / "model").mkdir()
    shutil.copy(CONFIG, tmp_path / "model" / "config.json")
    done = subprocess.run(
        [sys.executable, "-c", STAGE, str(tmp_path), "0"],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE), "VLLM_PP_LAYER_PARTITION": "27"},
    )
    assert done.returncode != 0, done.stdout
    refusal = "ValueError: sum(partitions)=27 does not match num_hidden_layers=28."
    assert refusal in done.stderr, done.stderr[-4000:]
    assert done.stderr.count("Traceback") == 1, done.stderr[-4000:]

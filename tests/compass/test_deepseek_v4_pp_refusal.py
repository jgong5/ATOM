# SPDX-License-Identifier: MIT
"""ATOM's DeepSeek-V4 at `-pp 2`: each stage's runner refuses pipeline
parallelism, naming the model, where it built every decoder layer and planned
the whole model's paged KV block on every stage.

Each stage's runner is built driverless in a process of its own on a fake
pipeline group standing at its stage, by `test_qwen3_pp_stages`' script. The
model is DeepSeek-V4-Pro (`deepseek_v4_pro_config.json`).
"""

import os
import pathlib
import shutil
import subprocess
import sys

import pytest
from test_qwen3_pp_stages import STAGE, TREE

CONFIG = pathlib.Path(__file__).with_name("deepseek_v4_pro_config.json")


@pytest.mark.parametrize("rank", [0, 1])
def test_each_stage_refuses_pipeline_parallelism_naming_the_model(tmp_path, rank):
    (tmp_path / "model").mkdir()
    shutil.copy(CONFIG, tmp_path / "model" / "config.json")
    done = subprocess.run(
        [sys.executable, "-c", STAGE, str(tmp_path), str(rank), "-pp", "2"],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )
    assert done.returncode != 0, done.stdout
    refusal = "ValueError: DeepSeek-V4 does not support pipeline parallelism"
    assert refusal in done.stderr, done.stderr[-4000:]

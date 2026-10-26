# SPDX-License-Identifier: MIT
"""Under inference mode the capture's fake mode builds `torch.tensor([])` under
a device default on the declared device without reaching a driver, as ATOM's
attention layers build their empty KV caches.

The declarations are process-global, so the call runs in a process of its own.
"""

import json
import os
import pathlib
import subprocess
import sys

TREE = pathlib.Path(__file__).resolve().parents[2]

CALL = """
import json, sys
from atom.compass.capture import driverless
driverless.install(sys.argv[1], "gfx942")
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv

torch.set_default_device("cuda")
try:
    with driverless.driverless_mode(ShapeEnv()), torch.inference_mode():
        t = torch.tensor([])
    print(json.dumps([type(t).__name__, str(t.device), list(t.shape)]))
except RuntimeError as e:
    print(json.dumps(str(e)))
"""


def test_an_empty_tensor_under_a_device_default_builds_under_inference_mode(
    tmp_path,
):
    done = subprocess.run(
        [sys.executable, "-c", CALL, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE), "AITER_LOG_LEVEL": "WARNING"},
    )
    assert done.returncode == 0, done.stderr[-4000:]
    assert json.loads(done.stdout.splitlines()[-1]) == ["FakeTensor", "cuda:0", [0]]

# SPDX-License-Identifier: MIT
"""A capture hides every GPU below HIP, whatever mask its process started with.

The declarations are process-global, so the capture's setup runs in a process
of its own and reports the mask HIP's runtime would start under.
"""

import os
import pathlib
import subprocess
import sys

TREE = pathlib.Path(__file__).resolve().parents[2]

SETUP = """
import os, sys
from atom.compass.capture import driverless
driverless.install(sys.argv[1], "gfx942")
driverless.group(1, 1)
print(repr(os.environ.get("ROCR_VISIBLE_DEVICES")), os.environ["HIP_VISIBLE_DEVICES"])
"""


def test_a_capture_s_setup_leaves_no_gpu_visible_below_hip(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(TREE), "ROCR_VISIBLE_DEVICES": "0"}
    done = subprocess.run(
        [sys.executable, "-c", SETUP, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        cwd=TREE,
        env=env,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    # HIP is still told of GPU 0; the mask below it is what keeps it unopened.
    assert done.stdout.splitlines()[-1] == "'' 0"

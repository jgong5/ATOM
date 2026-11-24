# SPDX-License-Identifier: MIT
"""`bench.ops._child` called from a script read on stdin, as `deploy.sh` runs its run-file builder (`python3 -`): the
spawned child must not try to re-import `<stdin>` as the main module (BrokenProcessPool: no such file '/<stdin>').
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_child_runs_from_a_script_read_on_stdin():
    script = "import operator\nfrom atom.compass.bench.ops import _child\nprint(_child(operator.add, 1, 2))\n"
    run = subprocess.run(
        [sys.executable, "-"],
        input=script,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=REPO,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-800:]
    assert run.stdout.strip().splitlines()[-1] == "3"

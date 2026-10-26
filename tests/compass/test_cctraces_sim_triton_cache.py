# SPDX-License-Identifier: MIT
"""A ``REAL=1`` cell of ``scripts/compass/cctraces_sim.sh`` hands its server the
``TRITON_CACHE_DIR`` a ``compass record`` server holds, and its run.json's
``env`` states it, with ``TRITON_CACHE_AUTOTUNING``.

The script runs end to end with stand-ins on ``PATH``: a ``python3`` whose API
server writes the two variables it sees, says it is ready and serves until
stopped, a harness Python whose aiperf does nothing, and a ``rocm-smi``
listing no process. Nothing needs a driver or aiperf.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_drive_gpu import SHOWHW

TREE = Path(__file__).resolve().parents[2]
SCRIPT = TREE / "scripts/compass/cctraces_sim.sh"
NAMES = ("TRITON_CACHE_AUTOTUNING", "TRITON_CACHE_DIR")


@pytest.mark.parametrize("caller", [None, "/caller/cache"])
def test_a_real_cell_s_server_and_run_json_hold_a_record_s_triton_cache_dir(
    tmp_path, caller
):
    seen = tmp_path / "seen"
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "python3").write_text(
        "#!/bin/sh\n"
        'if [ "$2" = atom.entrypoints.openai.api_server ]; then\n'
        f'    echo "$TRITON_CACHE_AUTOTUNING $TRITON_CACHE_DIR" > {seen}\n'
        '    echo "Uvicorn running"; exec sleep 600\n'
        "fi\n"
        f'exec {sys.executable} "$@"\n'
    )
    (bin_ / "harness-python").write_text(
        "#!/bin/sh\n"
        'case "$2" in\n'
        '*"import compass_harness"*) exec echo "${PYTHONPATH%%:*}/compass_harness/x.py" ;;\n'
        "esac\n"
    )
    (tmp_path / "showhw").write_text(SHOWHW)
    (bin_ / "rocm-smi").write_text(
        f'#!/bin/sh\ncase "$*" in *--showhw*) cat {tmp_path}/showhw;; esac\n'
    )
    for f in bin_.iterdir():
        f.chmod(0o755)
    traces = tmp_path / "traces.jsonl"
    traces.write_text('{"id": 0}\n')
    run_file = tmp_path / "run.json"
    run_file.write_text("{}")
    env = dict(
        {k: v for k, v in os.environ.items() if k not in NAMES},
        PATH=f"{bin_}:{os.environ['PATH']}",
        TRITON_HOME=str(tmp_path / "home"),
        REAL="1",
        MODEL=str(tmp_path),
        TRACES=str(traces),
        HARNESS_PYTHON=str(bin_ / "harness-python"),
        WAIT_S="30",
    )
    if caller:
        env["TRITON_CACHE_DIR"] = caller
    cell = tmp_path / "cell"
    done = subprocess.run(
        ["bash", str(SCRIPT), str(run_file), str(cell)],
        cwd="/",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    held = ["1", caller or str(tmp_path / "home" / ".triton" / "cache")]
    assert seen.read_text().split() == held, done.stdout + done.stderr
    assert json.loads((cell / "run.json").read_text())["env"] == dict(zip(NAMES, held))

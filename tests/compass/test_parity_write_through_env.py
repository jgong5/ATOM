# SPDX-License-Identifier: MIT
"""A launcher's real server keeps its step record's lines until its exit,
whatever the caller exports: under an exported
``ATOM_COMPASS_PARITY_WRITE_THROUGH=1``, ``cctraces_sim.sh`` (``REAL=1``),
``parity_one_node.sh`` and ``parity_pd.sh`` start it at 0, as ``compass
record`` does.

A stand-in ``python3`` on ``PATH`` takes the API server's place: it writes the
value it was started with and exits, so each script stops at its first server.
Nothing needs a driver or a model.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import platforms
import pytest
from test_drive_gpu import SHOWHW

from atom.compass.parity import WRITE_THROUGH

TREE = Path(__file__).resolve().parents[2]
#: Node 18's MI308X, whose readings the parity scripts compose a machine from.
NODE18 = json.loads(
    (Path(__file__).parent / "machines/mi308x-hjbog18.json").read_text()
)


@pytest.mark.parametrize(
    "script", ["cctraces_sim.sh", "parity_one_node.sh", "parity_pd.sh"]
)
def test_a_real_server_starts_at_0_under_an_exported_1(tmp_path, script):
    seen, bin_, model = tmp_path / "seen", tmp_path / "bin", tmp_path / "model"
    bin_.mkdir()
    model.mkdir()
    (model / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}')
    (tmp_path / "showhw").write_text(SHOWHW)
    (tmp_path / "traces.jsonl").write_text('{"id": 0}\n')
    (tmp_path / "run.json").write_text("{}")
    (bin_ / "python3").write_text(
        "#!/bin/sh\n"
        'case "$1 $2" in "-m atom.entrypoints.openai"*)\n'
        f'    echo "${{{WRITE_THROUGH}-unset}}" >>{seen}; exit 1;;\n'
        "esac\n"
        f'exec {sys.executable} "$@"\n'
    )
    (bin_ / "harness-python").write_text(
        "#!/bin/sh\n"  # the import probe; the plugin probe finds none
        'case "$2" in *"import compass_harness"*)\n'
        '    echo "${PYTHONPATH%%:*}/compass_harness/x.py";; esac\n'
    )
    (bin_ / "rocm-smi").write_text(
        f'#!/bin/sh\nif [ "$1" = --showhw ]; then cat {tmp_path}/showhw; fi\n'
    )
    for f in bin_.iterdir():
        f.chmod(0o755)
    _, device, _, ptl = platforms.flags(tmp_path / "readings", NODE18, tmp_path)
    dropped = ("HIP_VISIBLE_DEVICES", "STORE", "HTTP_INTAKE", "CONFIG", "SERVER_ARGS")
    env = {k: v for k, v in os.environ.items() if k not in dropped} | {
        "PATH": f"{bin_}:{os.environ['PATH']}",
        WRITE_THROUGH: "1",
        "MODEL": str(model),
        "WAIT_S": "30",
    }
    if script == "cctraces_sim.sh":
        env |= {"REAL": "1", "TRACES": str(tmp_path / "traces.jsonl")}
        env |= {"HARNESS_PYTHON": str(bin_ / "harness-python")}
        argv = [tmp_path / "run.json", tmp_path / "out"]
    else:
        env |= {"DEVICE": device, "PTL": ptl, "READINGS": str(tmp_path / "readings")}
        env |= {"STORE": str(tmp_path / "store")}
        argv = [tmp_path / "out"]
    done = subprocess.run(
        ["bash", TREE / "scripts/compass" / script, *argv],
        cwd="/",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert seen.exists(), done.stdout + done.stderr
    assert seen.read_text().split() == ["0"], done.stderr

# SPDX-License-Identifier: MIT
"""``client.sh`` and ``cctraces_sim.sh`` run aiperf with ``HF_HUB_OFFLINE`` and
``TRANSFORMERS_OFFLINE`` unset. Either switch puts aiperf 0.12.0's tokenizer on
its offline path, which looks ``--tokenizer`` up in the Hugging Face cache
alone and refuses ``MODEL``, a directory, before the profiling phase; a
recipe that exports one for its server must not stop its client.

Each script runs with stand-ins on ``PATH`` whose aiperf records the
environment it was started with. Nothing needs a driver, a server or aiperf.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_client_sh import GOOD, SIM, _client, _setup

TREE = Path(__file__).resolve().parents[2]
SWITCHES = ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]


@pytest.mark.parametrize("switch", SWITCHES)
def test_a_client_run_starts_aiperf_without_an_offline_switch(tmp_path, switch):
    out, env = _setup(tmp_path, SIM)
    done = _client(out, *GOOD, env=env | {switch: "1"})
    assert done.returncode == 0, done.stdout + done.stderr

    _, aiperf_env, _ = json.loads((tmp_path / "seen.json").read_text())
    assert switch not in aiperf_env


@pytest.mark.parametrize("switch", SWITCHES)
def test_a_cell_starts_aiperf_without_an_offline_switch_and_its_server_with_it(
    tmp_path, switch
):
    """The server keeps the caller's switch: it reads the model offline."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "python3").write_text(
        "#!/bin/sh\n"
        'if [ "$2" = atom.entrypoints.openai.api_server ]; then\n'
        f'    env > {tmp_path}/server.env; echo "Uvicorn running"; sleep 1; exit 0\n'
        "fi\n"
        f'exec {sys.executable} "$@"\n'
    )
    (bin_ / "harness-python").write_text(
        "#!/bin/sh\n"
        '[ "$1" = -c ] && exec echo "${PYTHONPATH%%:*}/compass_harness/x.py"\n'
        f"env > {tmp_path}/aiperf.env\n"
    )
    for f in bin_.iterdir():
        f.chmod(0o755)
    traces, run_file = tmp_path / "traces.jsonl", tmp_path / "run.json"
    traces.write_text('{"id": 0}\n')
    run_file.write_text("{}")
    env = os.environ | {
        "PATH": f"{bin_}:{os.environ['PATH']}",
        "MODEL": str(tmp_path),
        "TRACES": str(traces),
        "HARNESS_PYTHON": str(bin_ / "harness-python"),
        "WAIT_S": "30",
        switch: "1",
    }
    script = TREE / "scripts/compass/cctraces_sim.sh"
    done = subprocess.run(
        ["bash", script, run_file, tmp_path / "cell"],
        cwd="/",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert (tmp_path / "aiperf.env").exists(), done.stdout + done.stderr
    seen = {
        name: (tmp_path / f"{name}.env").read_text().splitlines()
        for name in ("server", "aiperf")
    }
    assert f"{switch}=1" in seen["server"]
    assert not [line for line in seen["aiperf"] if line.startswith(f"{switch}=")]

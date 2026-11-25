# SPDX-License-Identifier: MIT
"""``scripts/compass/cctraces_sim.sh`` removes the compiled-graph cache
directory its TP1 server's log names once the server has stopped.

The script runs end to end with the stand-ins of
``test_cctraces_sim_no_export``, its API server logging a cache directory
under ``HOME`` as ATOM's ranks do. Nothing needs a driver or aiperf.
"""

import os
import subprocess
import sys
from pathlib import Path

TREE = Path(__file__).resolve().parents[2]
SCRIPT = TREE / "scripts/compass/cctraces_sim.sh"


def test_a_cell_removes_its_servers_compiled_graph_cache_directory(tmp_path):
    caches = tmp_path / "home/.cache/atom/torch_compile_cache"
    mine, other = caches / "0123456789", caches / "9876543210"
    other.mkdir(parents=True)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "python3").write_text(
        "#!/bin/sh\n"
        'if [ "$2" = atom.entrypoints.openai.api_server ]; then\n'
        f"    mkdir -p {mine}/rank_0/backbone\n"
        f'    echo "Using cache directory: {mine}/rank_0/backbone for vLLM\'s torch.compile"\n'
        '    echo "Uvicorn running"; sleep 1; exit 0\n'
        "fi\n"
        f'exec {sys.executable} "$@"\n'
    )
    harness = bin_ / "harness-python"
    harness.write_text(
        "#!/bin/sh\n"
        '[ "$1" = -c ] && exec echo "${PYTHONPATH%%:*}/compass_harness/x.py"\n'
        "exit 0\n"
    )
    for f in bin_.iterdir():
        f.chmod(0o755)
    traces = tmp_path / "traces.jsonl"
    traces.write_text('{"id": 0}\n')
    run_file = tmp_path / "run.json"
    run_file.write_text("{}")
    env = dict(
        os.environ,
        HOME=str(tmp_path / "home"),
        PATH=f"{bin_}:{os.environ['PATH']}",
        MODEL=str(tmp_path),
        TRACES=str(traces),
        HARNESS_PYTHON=str(harness),
        WAIT_S="30",
    )
    done = subprocess.run(
        ["bash", str(SCRIPT), str(run_file), str(tmp_path / "cell")],
        cwd="/",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert "Traceback" not in done.stderr, done.stderr
    assert "Using cache directory" in (tmp_path / "cell/server.log").read_text()
    assert not mine.exists() and other.exists()

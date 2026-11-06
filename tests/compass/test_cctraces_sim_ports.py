# SPDX-License-Identifier: MIT
"""``scripts/compass/cctraces_sim.sh`` keeps every other port it picks out of
the write-request ports from KV_WRITE_REQ_PORT, one per prefill and decode DP
rank pair, and hands a 1P1D cell's pd_sim.sh a free DP master port per side.

The script runs end to end with the stand-ins of
``test_cctraces_sim_no_export``, and its port pick gets planted ports first:
PORT is one above KV_WRITE_REQ_PORT. The test holds the planted ports until the
script returns, so no other process takes one before the pick binds it. Nothing
needs a driver or aiperf.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parents[2]
SCRIPT = TREE / "scripts/compass/cctraces_sim.sh"
# A run file's DP widths and the write-request ports they take.
WIDTHS = [
    ({"data_parallel_size": 2}, 4),
    ({"prefill_data_parallel_size": 2}, 2),
    ({"data_parallel_size": 2, "prefill_data_parallel_size": 1}, 2),
]

# Binds to port 0 take the planted ports, in order, while any are left,
# sharing them with the sockets the test holds.
SITECUSTOMIZE = """\
import os, socket
planted = [int(p) for p in os.environ["PLANTED_PORTS"].split()]
bind = socket.socket.bind
def plant(s, a):
    if a[1] == 0 and planted:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        a = (a[0], planted.pop(0))
    bind(s, a)
socket.socket.bind = plant
"""


def _hold(port=0):
    """A socket bound to port, or to a free one for 0, that a planted bind can share.

    It binds every address, as the port pick does: a port free on loopback can
    be held on another of the host's addresses, where the pick's bind is refused.
    """
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    s.bind(("", port))
    return s


def _held_pair():
    """Sockets held on a free port and on its successor."""
    while True:
        a = _hold()
        try:
            return [a, _hold(a.getsockname()[1] + 1)]
        except OSError:
            a.close()


@pytest.mark.parametrize("widths,block", WIDTHS, ids=["dp2", "P2D1", "P1D2"])
def test_no_other_port_lands_in_the_write_request_block(tmp_path, widths, block):
    held = _held_pair() + [_hold() for _ in range(4)]
    kv, _, *free = (s.getsockname()[1] for s in held)
    # CLOCK_PORT PORT DECODE_PORT ROUTER_PORT KV_WRITE_REQ_PORT PROMETHEUS_PORT
    planted = [free[0], kv + 1, free[1], free[2], kv, free[3]]
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(SITECUSTOMIZE)
    picked = tmp_path / "picked"
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "python3").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = - ]; then\n'
        f'    PYTHONPATH={site}:$PYTHONPATH {sys.executable} "$@" | tee -a {picked}\n'
        "    exit\n"
        "fi\n"
        'if [ "$2" = atom.entrypoints.openai.api_server ]; then\n'
        '    echo "Uvicorn running"; sleep 1; exit 0\n'
        "fi\n"
        f'exec {sys.executable} "$@"\n'
    )
    harness = bin_ / "harness-python"
    harness.write_text(
        "#!/bin/sh\n"
        '[ "$1" = -c ] && exec echo "${PYTHONPATH%%:*}/compass_harness/x.py"\n'
        "exit 3\n"
    )
    for f in bin_.iterdir():
        f.chmod(0o755)
    traces = tmp_path / "traces.jsonl"
    traces.write_text('{"id": 0}\n')
    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps(widths))
    env = dict(
        os.environ,
        PATH=f"{bin_}:{os.environ['PATH']}",
        PLANTED_PORTS=" ".join(map(str, planted)),
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
    for s in held:
        s.close()
    # The port pick is the first of the two `python3 -` runs.
    lines = picked.read_text().splitlines()
    assert lines, f"the port pick wrote nothing to {picked}:\n{done.stderr}"
    ports = [int(p) for p in lines[0].split()[1:]]
    assert len(ports) == 6, done.stdout + done.stderr
    kv_port = ports.pop(4)
    # The planted picks held, so the block is clear because KV_WRITE_REQ_PORT moved.
    assert ports == planted[:4] + planted[5:], (planted, kv_port, ports)
    taken = range(kv_port, kv_port + block)
    assert not [p for p in ports if p in taken], (planted, kv_port, ports)


def test_a_1p1d_cell_hands_pd_sim_a_free_dp_master_port_for_each_side(tmp_path):
    """pd_sim.sh's fixed defaults collide when two cells share a host network."""
    seen = tmp_path / "seen"
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    bash = shutil.which("bash")
    (bin_ / "bash").write_text(
        "#!/bin/sh\n"
        'case "$1" in */pd_sim.sh)\n'
        f'    echo "$PREFILL_DP_PORT $DECODE_DP_PORT" >{seen}; exit 1;;\n'
        "esac\n"
        f'exec {bash} "$@"\n'
    )
    (bin_ / "harness-python").write_text(
        "#!/bin/sh\n"
        '[ "$1" = -c ] && exec echo "${PYTHONPATH%%:*}/compass_harness/x.py"\n'
        "exit 3\n"
    )
    for f in bin_.iterdir():
        f.chmod(0o755)
    traces = tmp_path / "traces.jsonl"
    traces.write_text('{"id": 0}\n')
    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps({"kv_write_req_s": 2.0**-10}))
    done = subprocess.run(
        [bash, str(SCRIPT), str(run_file), str(tmp_path / "cell")],
        cwd="/",
        env=dict(
            os.environ,
            PATH=f"{bin_}:{os.environ['PATH']}",
            MODEL=str(tmp_path),
            TRACES=str(traces),
            HARNESS_PYTHON=str(bin_ / "harness-python"),
            WAIT_S="30",
        ),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert seen.exists(), done.stdout + done.stderr
    ports = seen.read_text().split()
    assert len(set(ports)) == 2, (ports, done.stderr)

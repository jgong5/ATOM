# SPDX-License-Identifier: MIT
"""`scripts/compass` launchers hold each port they draw until they exit: a port
`compass_held_port` drew, and each port `cctraces_sim.sh` hands `pd_sim.sh`,
refuses another process's plain bind while a server's listener binds it.

Nothing needs a driver: `pd_sim.sh` is stubbed on `PATH`.
"""

import errno
import json
import os
import select
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

TREE = Path(__file__).resolve().parents[2]
SCRIPTS = TREE / "scripts/compass"
#: Another process, then a server, on each port it is given: a plain bind on
#: every address that listens, then `socket.create_server`, which sets
#: SO_REUSEADDR as uvicorn and libzmq do. Prints each port and each bind's
#: errno, 0 when it bound.
SQUAT = """
import socket, sys
def errno_of(bind):
    try:
        bind()
        return 0
    except OSError as e:
        return e.errno
for port in map(int, sys.argv[1:]):
    with socket.socket() as squat:
        print(port, errno_of(lambda: squat.bind(("", port)) or squat.listen()),
              errno_of(lambda: socket.create_server(("127.0.0.1", port)).close()))
"""
HELD = [str(errno.EADDRINUSE), "0"]
#: The ports `cctraces_sim.sh` hands a 1P1D cell's `pd_sim.sh`.
PD_SIM_PORTS = [
    "CLOCK_PORT",
    "PREFILL_PORT",
    "DECODE_PORT",
    "ROUTER_PORT",
    "KV_WRITE_REQ_PORT",
    "PROMETHEUS_PORT",
    "PREFILL_DP_PORT",
    "DECODE_DP_PORT",
]


def _squat(*ports) -> list[list[str]]:
    done = subprocess.run(
        [sys.executable, "-c", SQUAT, *ports],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.split()[1:] for line in done.stdout.splitlines()]


def test_a_drawn_port_is_held_from_another_process_until_the_script_exits():
    script = subprocess.Popen(
        [
            "bash",
            "-c",
            f'. {SCRIPTS}/_lib.sh; p=$(compass_held_port); echo "$p"; read -r _',
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert select.select([script.stdout], [], [], 30)[0], "no port in 30 s"
    port = script.stdout.readline().strip()
    assert _squat(port) == [HELD]
    script.communicate("\n", timeout=30)
    deadline = time.monotonic() + 10
    while True:
        with socket.socket() as s:
            try:
                s.bind(("", int(port)))
                break
            except OSError:
                assert time.monotonic() < deadline, "the port outlived its script"
                time.sleep(0.1)


def test_cctraces_sim_hands_pd_sim_ports_held_from_another_process(tmp_path):
    seen = tmp_path / "seen"
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    ports = " ".join(f"${name}" for name in PD_SIM_PORTS)
    (bin_ / "bash").write_text(
        "#!/bin/sh\n"
        'case "$1" in */pd_sim.sh)\n'
        f'    {sys.executable} -c "$SQUAT" {ports} >{seen}; exit 1;;\n'
        "esac\n"
        f'exec {shutil.which("bash")} "$@"\n'
    )
    (bin_ / "harness-python").write_text(
        "#!/bin/sh\n"
        '[ "$1" = -c ] && exec echo "${PYTHONPATH%%:*}/compass_harness/x.py"\n'
        "exit 3\n"
    )
    for f in bin_.iterdir():
        f.chmod(0o755)
    (tmp_path / "traces.jsonl").write_text('{"id": 0}\n')
    (tmp_path / "run.json").write_text(json.dumps({"kv_write_req_s": 2.0**-10}))
    done = subprocess.run(
        [
            "bash",
            SCRIPTS / "cctraces_sim.sh",
            tmp_path / "run.json",
            tmp_path / "cell",
        ],
        cwd="/",
        env=dict(
            os.environ,
            PATH=f"{bin_}:{os.environ['PATH']}",
            MODEL=str(tmp_path),
            TRACES=str(tmp_path / "traces.jsonl"),
            HARNESS_PYTHON=str(bin_ / "harness-python"),
            WAIT_S="30",
            SQUAT=SQUAT,
        ),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert seen.exists(), done.stdout + done.stderr
    squatted = [line.split() for line in seen.read_text().splitlines()]
    assert len({port for port, *_ in squatted}) == len(PD_SIM_PORTS), squatted
    assert [binds for _, *binds in squatted] == [HELD] * len(PD_SIM_PORTS), squatted

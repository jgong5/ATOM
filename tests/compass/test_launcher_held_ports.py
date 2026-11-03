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


#: pd_sim.sh's stand-in: another process, then a server, on each port pd_sim.sh
#: takes, read from its environment and its run file.
PD_SIM = """
import json, os, sys
run, e = json.load(open(sys.argv[1])), lambda name: int(os.environ[name])
dp = run.get("data_parallel_size", 1)
p, d = run.get("prefill_instances", 1), run.get("decode_instances", 1)
pairs = p * d * run.get("prefill_data_parallel_size", dp)
pairs *= run.get("decode_data_parallel_size", dp)
ports = [e("CLOCK_PORT"), e("ROUTER_PORT"), e("PROMETHEUS_PORT")]
for side, n in (("PREFILL", p), ("DECODE", d)):
    ports += [e(side + "_PORT") + k for k in range(n)]
    ports += [e(side + "_DP_PORT") + 16 * k for k in range(n)]
ports += [e("KV_WRITE_REQ_PORT") + k for k in range(pairs)]
sys.argv[1:] = map(str, ports)
""" + SQUAT


def _plant_pd_sim(bin_: Path, seen: Path) -> None:
    (bin_ / "bash").write_text(
        "#!/bin/sh\n"
        'case "$1" in */pd_sim.sh)\n'
        f'    {sys.executable} -c "$PD_SIM" "$2" >>{seen}; exit 1;;\n'
        "esac\n"
        f'exec {shutil.which("bash")} "$@"\n'
    )


def _squatted(seen: Path) -> list[list[str]]:
    return [line.split()[1:] for line in seen.read_text().splitlines()]


def test_cctraces_sim_holds_every_write_request_port_at_dp_above_one(tmp_path):
    seen, bin_ = tmp_path / "seen", tmp_path / "bin"
    bin_.mkdir()
    _plant_pd_sim(bin_, seen)
    (bin_ / "harness-python").write_text(
        "#!/bin/sh\n"
        '[ "$1" = -c ] && exec echo "${PYTHONPATH%%:*}/compass_harness/x.py"\n'
        "exit 3\n"
    )
    for f in bin_.iterdir():
        f.chmod(0o755)
    (tmp_path / "traces.jsonl").write_text('{"id": 0}\n')
    run = {"kv_write_req_s": 2.0**-10}
    run |= {"prefill_data_parallel_size": 2, "decode_data_parallel_size": 2}
    (tmp_path / "run.json").write_text(json.dumps(run))
    done = subprocess.run(
        ["bash", SCRIPTS / "cctraces_sim.sh", tmp_path / "run.json", tmp_path / "c"],
        cwd="/",
        env=dict(
            os.environ,
            PATH=f"{bin_}:{os.environ['PATH']}",
            MODEL=str(tmp_path),
            TRACES=str(tmp_path / "traces.jsonl"),
            HARNESS_PYTHON=str(bin_ / "harness-python"),
            WAIT_S="30",
            PD_SIM=PD_SIM,
        ),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert seen.exists(), done.stdout + done.stderr
    # The clock's, the router's and its metrics', each side's port and DP port,
    # and 2 * 2 write requests.
    assert _squatted(seen) == [HELD] * 11


#: A real server of `parity_pd.sh`: another process, then a server, on its
#: port, its DP master port and each rank's Mooncake handshake port, then
#: `/health` on its port.
PD_SERVER = """
import http.server, json, os, subprocess, sys
a = sys.argv[1:]
kv = json.loads(a[a.index("--kv-transfer-config") + 1])
ports = [a[a.index(f) + 1] for f in ("--server-port", "--data-parallel-master-port")]
ports += [str(kv["handshake_port"] + r) for r in range(int(a[a.index("-tp") + 1]))]
with open(os.environ["SEEN"], "a") as seen:
    subprocess.run([sys.executable, "-c", os.environ["SQUAT"], *ports], stdout=seen)
os.makedirs(os.environ["ATOM_COMPASS_PARITY_RECORD"])
class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
http.server.HTTPServer(("127.0.0.1", int(ports[0])), Health).serve_forever()
"""
#: The real router: the same on its port and its metrics' port, then
#: `/v1/models` on its port.
PD_ROUTER = """
import http.server, os, subprocess, sys
a = sys.argv[1:]
ports = [a[a.index(f) + 1] for f in ("--port", "--prometheus-port")]
with open(os.environ["SEEN"], "a") as seen:
    subprocess.run([sys.executable, "-c", os.environ["SQUAT"], *ports], stdout=seen)
class Models(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
http.server.HTTPServer(("127.0.0.1", int(ports[0])), Models).serve_forever()
"""


def test_parity_pd_holds_each_server_s_ports_and_the_ones_pd_sim_derives(tmp_path):
    seen, bin_, model = tmp_path / "seen", tmp_path / "bin", tmp_path / "model"
    bin_.mkdir()
    model.mkdir()
    (model / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}')
    _plant_pd_sim(bin_, seen)
    (tmp_path / "server.py").write_text(PD_SERVER)
    (bin_ / "python3").write_text(
        "#!/bin/sh\n"
        'case "$1 $2" in "-m atom.entrypoints.openai_server")\n'
        f'    shift 2; exec {sys.executable} {tmp_path}/server.py "$@";;\n'
        "esac\n"
        'case "$2" in *atom.compass.parity.traffic*)\n'
        '    mkdir -p "$5" && echo {} >"$5/arrivals.json"; exit 0;;\n'
        "esac\n"
        f'exec {sys.executable} "$@"\n'
    )
    (bin_ / "atomesh").write_text(f"#!{sys.executable}\n{PD_ROUTER}")
    for f in bin_.iterdir():
        f.chmod(0o755)
    done = subprocess.run(
        ["bash", SCRIPTS / "parity_pd.sh", tmp_path / "out"],
        cwd="/",
        env=dict(
            os.environ,
            PATH=f"{bin_}:{os.environ['PATH']}",
            MODEL=str(model),
            STORE=str(tmp_path / "store"),
            ATOMESH=str(bin_ / "atomesh"),
            PREFILL_GPUS="0 1",
            DECODE_GPUS="2",
            WAIT_S="60",
            no_proxy="127.0.0.1",
            NO_PROXY="127.0.0.1",
            SEEN=str(seen),
            SQUAT=SQUAT,
            PD_SIM=PD_SIM,
        ),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert done.stderr.endswith("parity_pd: pd_sim.sh exited\n"), done.stderr
    # Three real servers at DP 2 with four ports each, the real router's two,
    # and pd_sim.sh's: three, two prefill servers' two each, the decode's two,
    # and 2 * 1 * 2 * 2 write requests.
    assert _squatted(seen) == [HELD] * (3 * 4 + 2 + 3 + 2 * 2 + 2 + 8)

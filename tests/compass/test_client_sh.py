# SPDX-License-Identifier: MIT
"""``scripts/compass/client.sh`` drives one client run against the server a
``compass record`` or ``compass run`` ``--out`` names, and writes its export
with ``atom.compass.parity.client_run``.

Stand-ins on ``PATH``: a ``curl`` the server's ``/health`` answers, and a
harness Python whose probes answer as one holding this tree's compass-harness
and whose aiperf saves how it ran to ``$SEEN`` and writes the export and raw
records ``$SPEC`` holds, its dataset the trace files it was given. Each
breaks the step ``$BREAK`` names. With ``--stop``, a stand-in compass command
serves a ``sleep``. Nothing needs a driver, a server or aiperf.
"""

import json
import os
import signal
import subprocess
import sys
import time
from fnmatch import fnmatchcase
from pathlib import Path

import pytest
from test_parity_client_run import ARGUMENTS, CONFIG, PARAMETERS, raw, sse

from atom.compass.parity.client_run import NAME

TREE = Path(__file__).resolve().parents[2]
SCRIPT = TREE / "scripts/compass/client.sh"
HARNESS = r"""
import json, os, sys
from pathlib import Path
a, brk = sys.argv[1:], os.environ.get("BREAK")
if "import compass_harness" in a[1]:
    tree = "/elsewhere" if brk == "elsewhere" else os.environ["PYTHONPATH"].split(":")[0]
    print(tree + "/compass_harness/x.py")
if "entry_points" in a[1] and brk == "probe":
    sys.exit("no metadata")
if "entry_points" in a[1] and brk == "plugin":
    print("compass_harness.plugin:Plugin")
if "profile" not in a:
    sys.exit()
if brk == "aiperf":
    sys.exit(3)
Path(os.environ["SEEN"]).write_text(json.dumps([a, dict(os.environ), os.getcwd()]))
spec = json.loads(Path(os.environ["SPEC"]).read_text())
art = Path(a[a.index("--artifact-dir") + 1])
art.mkdir()
(art / "profile_export_raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in spec["raw"]))
dataset = {"name": "main", "type": "file", "format": "weka_trace"}
dataset |= {"random_seed": 20260707, "path": a[a.index("--input-file") + 1]}
config = spec["config"] | {"datasets": [dataset]}
export = {"input_config": config, "benchmark_id": "000000000000"}
(art / "profile_export_aiperf.json").write_text(json.dumps(export))
"""
#: Serves a `sleep` until SIGTERM, then writes run.json and exits.
COMMAND = """
import json, signal, subprocess, sys
from pathlib import Path
out, server = Path(sys.argv[1]), subprocess.Popen(["sleep", "600"])
def stop(*_):
    server.kill()
    (out / "run.json").write_text("{}")
    sys.exit()
signal.signal(signal.SIGTERM, stop)
(out / "s").write_text(json.dumps({"url": "http://h:1", "pid": server.pid}))
(out / "s").rename(out / "server.json")
signal.pause()
"""
GOOD = ["--traces", "1-2", "--concurrency", "3"]
SIM = {"url": "http://127.0.0.1:40165", "pid": 0, "run_file": "/sim/run_file.json"}


def _setup(tmp_path, server=None, ids=("a", "b")):
    bin_, out, traces = tmp_path / "bin", tmp_path / "out", tmp_path / "traces.jsonl"
    bin_.mkdir(), out.mkdir()
    (bin_ / "curl").write_text('#!/bin/sh\n[ "$BREAK" != health ]\n')
    (bin_ / "harness").write_text(f"#!{sys.executable}{HARNESS}")
    for f in bin_.iterdir():
        f.chmod(0o755)
    records = [
        raw([sse(("data", json.dumps({"id": i})))], f"t{k}") for k, i in enumerate(ids)
    ]
    (tmp_path / "spec.json").write_text(json.dumps({"config": CONFIG, "raw": records}))
    if server is not None:
        (out / "server.json").write_text(json.dumps(server))
    traces.write_text("".join(f'{{"id": "s{k}"}}\n' for k in range(4)))
    return out, os.environ | {
        "PATH": f"{bin_}:{os.environ['PATH']}",
        "MODEL": "/m",
        "TRACES": str(traces),
        "HARNESS_PYTHON": str(bin_ / "harness"),
        "SEEN": str(tmp_path / "seen.json"),
        "SPEC": str(tmp_path / "spec.json"),
    }


def _client(*args, env):
    return subprocess.run(
        ["bash", SCRIPT, *map(str, args)],
        cwd="/",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize(
    "args, unset, refusal",
    [
        ([], None, "no OUT"),
        (["--traces", "2-1", "--concurrency", "3"], None, "--traces takes A-B"),
        (["--traces", "1-2"], None, "--traces and --concurrency are required"),
        ([*GOOD, "--duration"], None, "--duration takes whole seconds"),
        ([*GOOD, "--no-warmup"], None, "unknown argument --no-warmup"),
        (GOOD, "TRACES", "TRACES is not set"),
    ],
)
def test_bad_arguments_refuse_before_anything_runs(tmp_path, args, unset, refusal):
    out, env = _setup(tmp_path, SIM)
    done = _client(*([out] if args else []), *args, env=env | {unset or "_": ""})
    assert done.returncode == 2, done.stdout + done.stderr
    assert done.stderr.startswith(f"client: {refusal}")
    assert not (out / "client").exists()


def test_an_out_with_no_server_json_names_it(tmp_path):
    out, env = _setup(tmp_path)
    done = _client(out, *GOOD, env=env)
    assert done.returncode == 1, done.stdout + done.stderr
    assert done.stderr.startswith(f"client: server: no {out}/server.json")


@pytest.mark.parametrize("simulated", [True, False])
def test_a_run_file_makes_the_client_simulated(tmp_path, simulated):
    server = SIM if simulated else {k: SIM[k] for k in ("url", "pid")}
    out, env = _setup(tmp_path, server)
    stale = {"ATOM_COMPASS_RUN": "/stale", "ATOM_COMPASS_CLOCK_ENDPOINT": "tcp://h:1"}
    stale["AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID"] = "false"
    done = _client(out, *GOOD, env=env | stale)
    assert done.returncode == 0, done.stdout + done.stderr

    argv, aiperf_env, cwd = json.loads((tmp_path / "seen.json").read_text())
    assert (argv[:2] == ["-m", "aiperf"]) == simulated
    assert simulated or "uuid.UUID(int=0)" in argv[1]
    assert aiperf_env.get("ATOM_COMPASS_RUN") == server.get("run_file")
    assert "ATOM_COMPASS_CLOCK_ENDPOINT" not in aiperf_env
    # aiperf then sends each request's X-Correlation-ID as its X-Session-ID too.
    assert aiperf_env["AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID"] == "true"
    assert aiperf_env["AIPERF_DATASET_MMAP_BASE_PATH"] == cwd == str(out / "client")
    line = " ".join(argv)
    assert "--url 127.0.0.1:40165 " in line and " --concurrency 3 " in line
    assert " --export-level raw " in line
    assert "--trajectory-start-" not in line and "--warmup-" not in line
    assert sorted(os.listdir(out / "client/traces")) == ["00001.json", "00002.json"]
    assert json.loads((out / "artifacts" / NAME).read_text()) == {
        "requests": {
            "a": {"trace": "t0", "sampling": PARAMETERS},
            "b": {"trace": "t1", "sampling": PARAMETERS},
        },
        "warmup": {},
        "unnamed": [],
        "traces": ["s1", "s2"],
        "sessions": {"s1": 1, "s2": 2},
        "arguments": ARGUMENTS,
    }


#: The nightly's client arguments and aiperf timeouts client.sh has no default for.
NIGHTLY = (
    "--trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75"
    " --warmup-requests-per-lane 10 --warmup-grace-period 1800"
    " --trace-idle-gap-cap-seconds 300"
)
TIMEOUTS = {
    "AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT": "300",
    "AIPERF_HTTP_TCP_USER_TIMEOUT": "900000",
    "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "1800",
    "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "1800",
}


@pytest.mark.parametrize(
    "seed", ["", " --random-seed 42", " --random-seed=42", "\n--random-seed 42"]
)
def test_extra_arguments_follow_the_clients_own_and_their_seed_replaces_its(
    tmp_path, seed
):
    out, env = _setup(tmp_path, SIM)
    extra = {"AIPERF_EXTRA_ARGS": NIGHTLY + seed}
    done = _client(out, *GOOD, env=env | TIMEOUTS | extra)
    assert done.returncode == 0, done.stdout + done.stderr

    argv, aiperf_env, _ = json.loads((tmp_path / "seen.json").read_text())
    assert argv[-len((NIGHTLY + seed).split()) :] == (NIGHTLY + seed).split()
    line = " ".join(argv)
    assert line.count("--random-seed") == 1
    assert ("--random-seed 20260707" in line) == (not seed)
    assert aiperf_env.items() >= TIMEOUTS.items()


def test_a_failed_export_names_its_step(tmp_path):
    out, env = _setup(tmp_path, SIM, ids=("a", "a"))
    done = _client(out, *GOOD, env=env)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "two requests got response id a" in done.stderr
    assert done.stderr.splitlines()[-1].startswith("client: export: ")
    assert not (out / "artifacts" / NAME).exists()


def test_stop_stops_the_command_once_the_export_is_written(tmp_path):
    out, env = _setup(tmp_path)
    (tmp_path / "compass_command.py").write_text(COMMAND)
    command = subprocess.Popen([sys.executable, tmp_path / "compass_command.py", out])
    while not (out / "server.json").exists():
        assert command.poll() is None
        time.sleep(0.1)
    done = _client(out, *GOOD, "--stop", env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert command.wait(timeout=10) == 0
    assert (out / "run.json").exists() and (out / "artifacts" / NAME).exists()


@pytest.mark.parametrize("writes_run", [True, False])
def test_stop_sets_the_exports_server_stopped_only_once_the_command_wrote_its_run(
    tmp_path, writes_run
):
    out, env = _setup(tmp_path)
    no_run = COMMAND.replace('(out / "run.json").write_text("{}")', "pass")
    (tmp_path / "compass_command.py").write_text(COMMAND if writes_run else no_run)
    command = subprocess.Popen([sys.executable, tmp_path / "compass_command.py", out])
    while not (out / "server.json").exists():
        assert command.poll() is None
        time.sleep(0.1)
    done = _client(out, *GOOD, "--stop", env=env)
    assert command.wait(timeout=10) == 0
    assert done.returncode == (0 if writes_run else 1), done.stdout + done.stderr
    export = json.loads((out / "artifacts" / NAME).read_text())
    assert export.get("server_stopped") is (True if writes_run else None)


#: The step `$BREAK` breaks, and the refusal it gets.
REFUSALS = {
    "health": "server: http://h:1/health does not answer",
    "server.json": "server: */server.json names no url and pid",
    "probe": "harness: * cannot list aiperf's plugins",
    "plugin": "harness: the server is real, and * has compass-harness installed",
    "elsewhere": "harness: compass_harness resolved to /elsewhere/*",
    "client": "out: */client exists; one client run per server --out",
    "traces": "traces: * holds 2 of traces 2 to 9",
    "unreadable": "traces: cannot read */none",
    "aiperf": "aiperf: exited 3; see */client/aiperf.log",
}
#: The server's parent's last argument, and the refusal `--stop` gets.
STOP_REFUSALS = {
    "sh": "stop: process *, the server's parent, is no compass command",
    "compass": "stop: the command left without writing */run.json",
}


@pytest.mark.parametrize("step", REFUSALS)
def test_each_refusal_exits_1_naming_its_step(tmp_path, step):
    out, env = _setup(tmp_path, {"url": "http://h:1", "pid": 0})
    args, env["BREAK"] = [*GOOD], step
    if step == "server.json":
        (out / step).write_text("{}")
    if step == "client":
        (out / step).mkdir()
    if step == "traces":
        args[1] = "2-9"
    if step == "unreadable":
        env["TRACES"] = str(tmp_path / "none")
    done = _client(out, *args, env=env)
    assert done.returncode == 1, done.stdout + done.stderr
    last = done.stderr.splitlines()[-1]
    assert fnmatchcase(last, "client: " + REFUSALS[step]), done.stderr


@pytest.mark.parametrize("parent", STOP_REFUSALS)
def test_stop_refuses_a_parent_that_is_no_compass_command_or_writes_no_run(
    tmp_path, parent
):
    """The server is a shell's `sleep`; the shell's cmdline ends in `parent`,
    and SIGTERM ends it without a run.json."""
    out, env = _setup(tmp_path)
    shell = ["sh", "-c", "sleep 600 & echo $!; wait", parent]
    with subprocess.Popen(shell, stdout=subprocess.PIPE, start_new_session=True) as sh:
        server = {"url": "http://h:1", "pid": int(sh.stdout.readline())}
        (out / "server.json").write_text(json.dumps(server))
        try:
            done = _client(out, *GOOD, "--stop", env=env)
        finally:
            os.killpg(sh.pid, signal.SIGKILL)
    assert done.returncode == 1, done.stdout + done.stderr
    last = done.stderr.splitlines()[-1]
    assert fnmatchcase(last, "client: " + STOP_REFUSALS[parent]), done.stderr


#: An API server that says it is ready and leaves, and any other Python as is.
PYTHON3 = f"""#!/bin/sh
[ "$2" = atom.entrypoints.openai.api_server ] && echo "Uvicorn running" && exit
exec {sys.executable} "$@"
"""


@pytest.mark.parametrize("extra", ["", NIGHTLY])
def test_its_workload_is_the_one_cctraces_sim_writes_for_the_same_cell(tmp_path, extra):
    out, env = _setup(tmp_path, SIM)
    env["AIPERF_EXTRA_ARGS"] = extra
    done = _client(out, *GOOD, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    argv = json.loads((tmp_path / "seen.json").read_text())[0]
    workload = json.loads((out / "artifacts" / "workload.json").read_text())
    assert workload["aiperf_args"] == argv[argv.index("raw") + 1 :]

    # cctraces_sim.sh hashes the workload into run.json before its server starts.
    (tmp_path / "bin" / "python3").write_text(PYTHON3)
    (tmp_path / "bin" / "python3").chmod(0o755)
    (tmp_path / "run_file.json").write_text("{}")
    cell = {"FIRST_SESSION": "1", "SESSIONS": "2", "CONCURRENCY": "3", "WAIT_S": "5"}
    subprocess.run(
        ["bash", TREE / "scripts/compass/cctraces_sim.sh", "run_file.json", "cell"],
        cwd=tmp_path,
        env=env | cell,
        capture_output=True,
        timeout=120,
        check=False,
    )
    run = json.loads((tmp_path / "cell" / "run.json").read_text())
    assert workload == run["workload"]

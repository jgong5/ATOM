# SPDX-License-Identifier: MIT
"""`smoke_cell.sh`: the quickstart's path in order on one idle GPU, each failed
step named, and the cell's verdict last.

It runs from a copy in a stub tree: `atom.compass.cli` logs each command, a
`record` or `run` writes server.json and the tool its environment loads and
waits for the SIGTERM `client.sh --stop` sends, then exits 3 when STOP_FAIL is
in its words, and a fit prints a store token, `fit steps` writing its law with
UNFITTED as its unfitted vectors; FAIL, matched against a command's words,
makes it refuse. `atom.compass.parity.blocked` logs the tool it builds.
`atom.compass.validate` prints a verdict row and a warmup row per run;
`client.sh` logs its arguments and its Python, and exits 1 without stopping its
server when CLIENT_FAIL is in its OUT. A command whose words hold HANG, or
`client.sh` when HANG is client.sh, writes its pid to `$LOG.hung` and sleeps.
`rocm-smi` logs its arguments and prints a GPU's use and VRAM.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from subprocess import PIPE

import pytest

COMPASS = Path(__file__).resolve().parents[2] / "scripts" / "compass"
RUNS = ("real1", "real2", "step_linear")
CLI = """import json, os, signal, sys, time
from pathlib import Path
def main():
    argv = sys.argv[1:]
    words = " ".join(argv[: argv.index("--")] if "--" in argv else argv)
    open(os.environ["LOG"], "a").write(words + "\\n")
    if os.environ.get("FAIL") and os.environ["FAIL"] in words:
        sys.exit("refused")
    if os.environ.get("HANG") and os.environ["HANG"] in words:
        Path(os.environ["LOG"] + ".t").write_text(str(os.getpid()))
        os.rename(os.environ["LOG"] + ".t", os.environ["LOG"] + ".hung")
        time.sleep(600)
    out = Path(argv[argv.index("--out") + 1])
    out.mkdir(parents=True)
    if argv[0] in ("record", "run"):
        rc = 3 if os.environ.get("STOP_FAIL") and os.environ["STOP_FAIL"] in words else 0
        stop = lambda *_: (out.joinpath("run.json").write_text("{}"), sys.exit(rc))
        signal.signal(signal.SIGTERM, stop)
        out.joinpath("loaded").write_text(os.environ.get("ROCP_TOOL_LIBRARIES", ""))
        out.joinpath("server.json").write_text(json.dumps({"pid": os.getpid()}))
        time.sleep(600)
    if argv[:2] == ["fit", "steps"]:
        law = {"params": {"unfitted": json.loads(os.environ.get("UNFITTED", "{}"))}}
        out.joinpath("step_linear_law.json").write_text(json.dumps(law))
    print("published key as tok-" + argv[1])
"""
BLOCKED = """import os, sys
from pathlib import Path
open(os.environ["LOG"], "a").write(f"blocked {sys.argv[1]}\\n")
"blocked" in os.environ.get("FAIL", "") and sys.exit("refused")
Path(sys.argv[1]).mkdir()
Path(sys.argv[1], "libcompass_blocked.so").touch()
"""
VALIDATE = """import os, sys
"validate" in os.environ.get("FAIL", "") and sys.exit("refused")
print("cell | result | metric | value | verdict")
print("cell | cell | verdict | 2 real runs | " + os.environ.get("VERDICT", "graded"))
for run in ("real1", "real2", "step_linear"):
    print(f"cell | warmup {run} | warmup | requests 1, steps 3 | not graded")
"""
CLIENT = """#!/bin/bash
echo "client $* $HARNESS_PYTHON" >> "$LOG"
[[ -n ${CLIENT_FAIL-} && $1 == *$CLIENT_FAIL* ]] && exit 1
[[ ${HANG-} == client.sh ]] && echo $$ >"$LOG.t" && mv "$LOG.t" "$LOG.hung" && exec sleep 600
mkdir -p "$1/artifacts" && touch "$1/artifacts/client_run.json"
pid=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["pid"])' "$1/server.json")
kill -TERM $pid && while kill -0 $pid 2>/dev/null; do sleep 0.1; done
"""


def _smoke(tmp_path, use="0", stamp=True, **env):
    tree = tmp_path / "ATOM"
    scripts = tree / "scripts" / "compass"
    scripts.mkdir(parents=True)
    for name in ("smoke_cell.sh", "_lib.sh"):
        shutil.copy(COMPASS / name, scripts / name)
    (scripts / "client.sh").write_text(CLIENT)
    (scripts / "client.sh").chmod(0o755)
    if stamp:
        (tree / ".compass-commit").write_text("cafe\n")
    stubs = {"cli": CLI, "validate/__main__": VALIDATE, "parity/blocked": BLOCKED}
    for module, text in stubs.items():
        path = tree / "atom" / "compass" / f"{module}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for d in ("atom", "atom/compass", "atom/compass/validate", "atom/compass/parity"):
        (tree / d / "__init__.py").write_text("")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "rocm-smi").write_text(
        '#!/bin/sh\necho "rocm-smi $*" >> "$LOG"\n'
        f"printf 'GPU[7]\\t\\t: GPU use (%%): {use}\\nGPU[7]\\t\\t: GPU Memory "
        "Allocated (VRAM%%): 0\\nGPU[6]\\t\\t: GPU use (%%): 9\\n'\n"
    )
    (bindir / "rocm-smi").chmod(0o755)
    env = (
        dict(
            os.environ,
            PATH=f"{bindir}:{Path(sys.executable).parent}:{os.environ['PATH']}",
            LOG=str(tmp_path / "log"),
            GPU="7",
            MACHINE=str(tmp_path / "spec.json"),
            TRACES="/cc/traces.jsonl",
            HARNESS_PYTHON="/sim/python",
            AIPERF_PYTHON="/real/python",
        )
        | env
    )
    cmd = ["bash", str(scripts / "smoke_cell.sh"), str(tmp_path / "m"), "o"]
    p = subprocess.Popen(
        cmd, cwd=tmp_path, env=env, stdout=PIPE, stderr=PIPE, text=True
    )
    if env.get("HANG"):
        deadline = time.monotonic() + 60
        while not (tmp_path / "log.hung").exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        p.terminate()
    try:
        out, err = p.communicate(timeout=120)
    finally:
        p.kill()
    r = subprocess.CompletedProcess(cmd, p.returncode, out, err)
    log = tmp_path / "log"
    return r, log.read_text().splitlines() if log.exists() else [], tmp_path / "o"


def test_the_path_runs_in_order_and_ends_with_the_cell_s_verdict(tmp_path):
    r, log, o = _smoke(tmp_path)
    assert r.returncode == 0, r.stderr
    machine = f"--machine {tmp_path}/spec.json --store {o}/store"
    run = {
        d: f"--client-run {o}/{d} {o}/{d}/artifacts/client_run.json"
        for d in ("calibration", "trace")
    }
    client = "--concurrency 1 --duration 600 --stop"
    assert log == [
        f"blocked {o}/tool",
        f"measure memory --gpu 7 {machine} --out {o}/memory",
        f"record --gpu 7 --out {o}/warm",
        f"client {o}/warm --traces 0-7 {client} /real/python",
        f"record --gpu 7 --out {o}/calibration",
        f"client {o}/calibration --traces 353-360 {client} /real/python",
        f"record --gpu 7 --trace --out {o}/trace",
        f"client {o}/trace --traces 353-360 {client} /real/python",
        f"fit steps --backend step-linear --form history {o}/calibration {o}/trace "
        f"{run['calibration']} {run['trace']} {machine} --out {o}/fit_steps",
        f"fit host {o}/calibration {run['calibration']} {machine} --out {o}/fit_host",
        f"record --gpu 7 --out {o}/cell/real1",
        f"client {o}/cell/real1 --traces 0-7 {client} /real/python",
        f"record --gpu 7 --out {o}/cell/real2",
        f"client {o}/cell/real2 --traces 0-7 {client} /real/python",
        f"run --step-linear tok-steps --host-laws tok-host {machine} --out {o}/cell/step_linear",
        f"client {o}/cell/step_linear --traces 0-7 {client} /sim/python",
    ]
    warmup = "; ".join(f"{run} requests 1, steps 3" for run in RUNS)
    assert r.stdout.endswith(f"warmup: {warmup}\n")
    assert "smoke_cell: graded in 0 min;" in r.stdout


def test_a_gpu_rocm_smi_reads_busy_runs_and_rocm_smi_is_never_read(tmp_path):
    # Another tenant's rocminfo makes every GPU read 1 to 7% use; each compass
    # command refuses a GPU a process is on by itself.
    r, log, _ = _smoke(tmp_path, use="3")
    assert r.returncode == 0, r.stderr
    assert [line for line in log if line.startswith("rocm-smi")] == []


@pytest.mark.parametrize(
    "fail, step, ran",
    [
        ("blocked", "tool", 1),
        ("measure memory", "memory", 2),
        ("warm", "warm", 3),
        ("calibration", "calibration", 5),
        ("--trace", "trace", 7),
        ("fit steps", "fit_steps", 9),
        ("fit host", "fit_host", 10),
        ("real2", "real2", 13),
        ("step_linear", "step_linear", 15),
        ("validate", "grade", 16),
    ],
)
def test_a_failed_step_is_named_and_ends_the_run(tmp_path, fail, step, ran):
    r, log, _ = _smoke(tmp_path, FAIL=fail)
    assert r.returncode == 1
    assert f"smoke_cell: step {step} failed:" in r.stderr
    assert len(log) == ran


def test_the_calibration_server_alone_loads_the_blocking_call_tool(tmp_path):
    r, _, o = _smoke(tmp_path)
    assert r.returncode == 0, r.stderr
    servers = ("warm", "calibration", "trace", *(f"cell/{run}" for run in RUNS))
    loaded = {d: (o / d / "loaded").read_text() for d in servers}
    tool = f"{o}/tool/libcompass_blocked.so"
    assert loaded == {d: "" for d in servers} | {"calibration": tool}


def test_trace_s_sets_the_trace_run_s_duration_alone(tmp_path):
    r, log, o = _smoke(tmp_path, TRACE_S="120")
    assert r.returncode == 0, r.stderr
    clients = [line.split() for line in log if line.startswith("client")]
    durations = {c[1]: c[c.index("--duration") + 1] for c in clients}
    assert durations.pop(f"{o}/trace") == "120"
    assert len(durations) == 5 and set(durations.values()) == {"600"}


def test_a_law_that_leaves_a_vector_unfitted_fails_before_the_real_runs(tmp_path):
    why = "no prefill step carries a trace run's busy time"
    r, log, _ = _smoke(tmp_path, UNFITTED=json.dumps({"prefill": why}))
    assert r.returncode == 1
    assert "smoke_cell: step fit_steps failed: the law leaves unfitted" in r.stderr
    assert why in r.stderr
    assert log[-1].startswith("fit steps")


def test_a_cell_that_is_a_coverage_report_fails_its_grade(tmp_path):
    r, _, _ = _smoke(tmp_path, VERDICT="coverage report")
    assert r.returncode == 1
    assert "step grade failed: the cell is coverage report, not graded" in r.stderr


@pytest.mark.parametrize("gpu", ["", "7,6", "x"])
def test_a_gpu_that_is_not_one_rocm_smi_index_runs_nothing(tmp_path, gpu):
    r, log, _ = _smoke(tmp_path, GPU=gpu)
    assert r.returncode == 1
    assert "smoke_cell: step gpu failed: GPU names one GPU" in r.stderr
    assert log == []


def test_a_tree_neither_git_nor_snapshot_runs_nothing(tmp_path):
    r, log, _ = _smoke(tmp_path, stamp=False)
    assert r.returncode == 1
    assert "step tree failed:" in r.stderr and "snapshot.sh" in r.stderr
    assert log == []


def test_an_out_that_holds_files_runs_nothing(tmp_path):
    (tmp_path / "o").mkdir()
    (tmp_path / "o" / "x").touch()
    r, log, _ = _smoke(tmp_path)
    assert r.returncode == 2
    assert "is not empty" in r.stderr
    assert log == []


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_failed_client_run_stops_its_server_and_names_its_step(tmp_path):
    r, log, o = _smoke(tmp_path, CLIENT_FAIL="real1")
    assert r.returncode == 1
    assert "smoke_cell: step real1 failed: client.sh exit 1" in r.stderr
    assert len(log) == 12
    pid = json.loads((o / "cell" / "real1" / "server.json").read_text())["pid"]
    assert not _alive(pid)


def test_a_server_that_exits_non_zero_names_its_step(tmp_path):
    r, log, _ = _smoke(tmp_path, STOP_FAIL="real2")
    assert r.returncode == 1
    assert "smoke_cell: step real2 failed: its server exit 3" in r.stderr
    assert len(log) == 14


@pytest.mark.parametrize("hang", ["measure memory", "fit host", "client.sh"])
def test_sigterm_during_a_step_leaves_none_of_it_running(tmp_path, hang):
    r, _, _ = _smoke(tmp_path, HANG=hang)
    pid = int((tmp_path / "log.hung").read_text())
    alive = _alive(pid)
    alive and os.kill(pid, signal.SIGKILL)
    assert r.returncode == -signal.SIGTERM, r.stderr
    assert not alive

# SPDX-License-Identifier: MIT
"""`calibrate.sh`: the quickstart's calibration in order on one GPU, a stopped
run resumed, and a resume under other inputs refused.

It runs from a copy in a stub tree, as `test_smoke_cell.py` drives
`smoke_cell.sh`: `atom.compass.cli` logs each command; a `record` writes
server.json and waits for the SIGTERM `client.sh --stop` sends; `measure
memory` and a fit write their readable copy with a token, the law with one
held-out band; FAIL, matched against a command's words, makes it refuse.
`client.sh` logs its arguments, its Python and its scenario, stops its server
and writes its export marked `server_stopped`, or writes it unmarked and exits 1
without stopping the server when CLIENT_FAIL is in its OUT.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

COMPASS = Path(__file__).resolve().parents[2] / "scripts" / "compass"
CLI = """import json, os, signal, sys, time
from pathlib import Path
COPY = {"memory": "memory_readings", "steps": "step_linear_law", "host": "host_laws"}
BAND = {"decode rung 1": {"steps": 9, "p10": -0.012, "p50": 0.001, "p90": 0.02}}
def main():
    argv = sys.argv[1:]
    words = " ".join(argv[: argv.index("--")] if "--" in argv else argv)
    open(os.environ["LOG"], "a").write(words + "\\n")
    if os.environ.get("FAIL") and os.environ["FAIL"] in words:
        sys.exit("refused")
    out = Path(argv[argv.index("--out") + 1])
    out.mkdir(parents=True)
    if argv[0] == "record":
        stop = lambda *_: (out.joinpath("run.json").write_text("{}"), sys.exit(0))
        signal.signal(signal.SIGTERM, stop)
        out.joinpath("server.json").write_text(json.dumps({"pid": os.getpid()}))
        time.sleep(600)
    law = {"unfitted": {}, "decode": {"1": {"held_out": {"bands": BAND}}}}
    copy = {"token": "tok-" + argv[1], "params": law}
    out.joinpath(COPY[argv[1]] + ".json").write_text(json.dumps(copy))
"""
CLIENT = """#!/bin/bash
echo "client $* $HARNESS_PYTHON [$AIPERF_EXTRA_ARGS]" >> "$LOG"
if [[ -n ${CLIENT_FAIL-} && $1 == *$CLIENT_FAIL* ]]; then
    mkdir -p "$1/artifacts" && echo {} >"$1/artifacts/client_run.json"
    exit 1
fi
pid=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["pid"])' "$1/server.json")
kill -TERM $pid && while kill -0 $pid 2>/dev/null; do sleep 0.1; done
mkdir -p "$1/artifacts" && echo '{"server_stopped": true}' >"$1/artifacts/client_run.json"
"""
FIRST_TURN = "8 --trajectory-start-min-ratio 0 --trajectory-start-max-ratio 0"


def _tree(tmp_path):
    scripts = tmp_path / "ATOM" / "scripts" / "compass"
    scripts.mkdir(parents=True)
    for name in ("calibrate.sh", "_lib.sh"):
        shutil.copy(COMPASS / name, scripts / name)
    (scripts / "client.sh").write_text(CLIENT)
    (scripts / "client.sh").chmod(0o755)
    (tmp_path / "ATOM" / ".compass-commit").write_text("cafe\n")
    (tmp_path / "ATOM" / "atom" / "compass").mkdir(parents=True)
    (tmp_path / "ATOM" / "atom" / "compass" / "cli.py").write_text(CLI)
    for d in ("atom", "atom/compass"):
        (tmp_path / "ATOM" / d / "__init__.py").write_text("")


def _calibrate(tmp_path, *records, **env):
    """One run of the stub tree's calibrate.sh into `o`, and the commands it ran."""
    if not (tmp_path / "ATOM").exists():
        _tree(tmp_path)
    log = tmp_path / "log"
    log.unlink(missing_ok=True)
    path = f"{Path(sys.executable).parent}:{os.environ['PATH']}"
    env = {
        **os.environ,
        "PATH": path,
        "LOG": str(log),
        "GPU": "7",
        "MODEL": str(tmp_path / "m"),
        "MACHINE": str(tmp_path / "spec.json"),
        "TRACES": "/cc/traces.jsonl",
        "AIPERF_PYTHON": "/real/python",
        **env,
    }
    script = tmp_path / "ATOM" / "scripts" / "compass" / "calibrate.sh"
    cmd = ["bash", str(script), "o", *records]
    r = subprocess.run(
        cmd,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return r, log.read_text().splitlines() if log.exists() else []


def test_the_steps_run_in_order_and_print_the_tokens_and_the_run_line(tmp_path):
    r, log = _calibrate(tmp_path, "1", FIRST_TURN)
    assert r.returncode == 0, r.stderr
    o, client = tmp_path / "o", "--duration 1800 --stop /real/python"
    machine = f"--machine {tmp_path}/spec.json --store {o}/store"
    cal = f"{o}/cal1-c1 {o}/cal2-c8"
    runs = " ".join(
        f"--client-run {o}/{d} {o}/{d}/artifacts/client_run.json"
        for d in ("cal1-c1", "cal2-c8")
    )
    assert log == [
        f"record --gpu 7 --out {o}/warm",
        f"client {o}/warm --traces 0-7 --concurrency 1 --duration 600 --stop /real/python []",
        f"measure memory --gpu 7 {machine} --out {o}/memory",
        f"record --gpu 7 --out {o}/cal1-c1",
        f"client {o}/cal1-c1 --traces 256-392 --concurrency 1 {client} []",
        f"record --gpu 7 --out {o}/cal2-c8",
        f"client {o}/cal2-c8 --traces 256-392 --concurrency 8 {client} [{FIRST_TURN[2:]}]",
        f"fit steps --backend step-linear --form history {cal} {runs} {machine} --out {o}/fit_steps",
        f"fit host {cal} {runs} {machine} --out {o}/fit_host",
    ]
    assert (
        "memory_readings tok-memory, step_linear_law tok-steps, host_laws tok-host"
        in r.stdout
    )
    assert "decode rung 1 | 9 | -1.2% | +0.1% | +2.0%" in r.stdout
    compass = tmp_path / "ATOM" / "scripts" / "compass" / "compass"
    run = f"{compass} run --step-linear tok-steps --host-laws tok-host {machine} --out DIR"
    assert f"{run} -- --model {tmp_path}/m --max-model-len 262144" in r.stdout


def test_a_run_stopped_after_its_first_record_records_nothing_twice(tmp_path):
    r, _ = _calibrate(tmp_path, "1", "4", CLIENT_FAIL="cal2-c4")
    assert r.returncode == 1
    assert "calibrate: step cal2-c4 failed: client.sh exit 1" in r.stderr
    r, log = _calibrate(tmp_path, "1", "4")
    assert r.returncode == 0, r.stderr
    assert "calibrate: cal2-c4 was stopped; taking it again" in r.stdout
    o = tmp_path / "o"
    client = f"client {o}/cal2-c4 --traces 256-392 --concurrency 4 --duration 1800"
    assert [line.split(" --stop")[0] for line in log[:2]] == [
        f"record --gpu 7 --out {o}/cal2-c4",
        client,
    ]
    assert [line[:8] for line in log[2:]] == ["fit step", "fit host"]


def test_a_resume_under_other_inputs_runs_nothing(tmp_path):
    _calibrate(tmp_path, "1", FAIL="measure")
    r, log = _calibrate(tmp_path, "1", "4")
    assert r.returncode == 2
    assert "was started with other inputs" in r.stderr and "> record 4" in r.stderr
    assert log == []
    (tmp_path / "o" / "inputs").unlink()
    r, log = _calibrate(tmp_path, "1")
    assert r.returncode == 2 and "holds files and no inputs" in r.stderr
    assert log == []


@pytest.mark.parametrize(
    "records, env, refusal",
    [
        (["x"], {}, "RECORD 'x' does not start with a concurrency"),
        ([], {}, "no OUT or no RECORD"),
        (["1"], {"GPU": ""}, "GPU is not set"),
    ],
)
def test_bad_arguments_run_nothing(tmp_path, records, env, refusal):
    r, log = _calibrate(tmp_path, *records, **env)
    assert r.returncode == 2
    assert refusal in r.stderr
    assert log == []


@pytest.mark.parametrize(
    "fail, step, ran", [("measure", "memory", 3), ("fit host", "fit_host", 7)]
)
def test_a_failed_step_is_named_and_ends_the_run(tmp_path, fail, step, ran):
    r, log = _calibrate(tmp_path, "1", FAIL=fail)
    assert r.returncode == 1
    assert f"calibrate: step {step} failed:" in r.stderr
    assert len(log) == ran


def test_a_record_left_after_a_fit_refuses_and_records_nothing(tmp_path):
    _calibrate(tmp_path, "1")
    shutil.rmtree(tmp_path / "o" / "cal1-c1")
    r, log = _calibrate(tmp_path, "1")
    assert r.returncode == 1
    law = tmp_path / "o" / "fit_steps" / "step_linear_law.json"
    assert f"step cal1-c1 failed: {law} was fitted without it" in r.stderr
    assert "calibrate into a new OUT" in r.stderr
    assert log == []

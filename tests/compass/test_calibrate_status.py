# SPDX-License-Identifier: MIT
"""`calibrate.sh`'s store, GPU list, status file and timing: STORE takes every
publish and is an input; GPU names every card; each step's start, end with its
wall seconds, or failure lands in OUT/status, which a resume appends to; and a
run that fits both laws prints each step's wall time and the node-hours, and
ends the status file with them. Under `setsid nohup` a calibration outlives
the shell that started it and a SIGHUP; a SIGTERM fails its step, which a
resume takes again, running no finished step twice; and a second calibrate.sh
into an OUT one still runs in refuses, removing nothing.

It runs in `test_calibrate`'s stub tree, whose `client.sh` here first waits
for the file GATE when the record is cal1-c1.
"""

import fcntl
import os
import re
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path

import pytest
from test_calibrate import CLIENT, _calibrate, _path, _tree

#: A status line, less its UTC stamp, its seconds read as N.
STAMP = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ ")


def _status(o: Path) -> list[str]:
    """OUT/status's lines, each stamp checked and dropped, seconds as N."""
    lines = (o / "status").read_text().splitlines()
    assert all(STAMP.match(line) for line in lines), lines
    return [re.sub(r" \d+ s\b", " N s", STAMP.sub("", line)) for line in lines]


def _until(held, what: str, timeout_s: float = 60):
    for _ in range(int(timeout_s / 0.1)):
        if held():
            return
        time.sleep(0.1)
    raise AssertionError(f"no {what} in {timeout_s} s")


def test_store_takes_every_publish_and_a_resume_into_another_refuses(tmp_path):
    shared, other = tmp_path / "shared", tmp_path / "other"
    shared.mkdir()
    other.mkdir()

    r, log = _calibrate(tmp_path, "1", STORE=str(shared))

    assert r.returncode == 0, r.stderr
    published = [line for line in log if "--store" in line]
    assert [line.split()[:2] for line in published] == [
        ["measure", "memory"],
        ["fit", "steps"],
        ["fit", "host"],
    ]
    assert all(f" --store {shared} " in line for line in published)
    assert f"--store {shared} --out DIR" in r.stdout
    o = tmp_path / "o"
    assert (o / "inputs").read_text().splitlines()[-1] == f"store {shared}"
    r, log = _calibrate(tmp_path, "1", STORE=str(other))
    assert r.returncode == 2
    assert "was started with other inputs" in r.stderr
    assert f"> store {other}" in r.stderr
    assert log == []


@pytest.mark.parametrize(
    "env, refusal",
    [
        ({"GPU": "0;1"}, "GPU '0;1' is not a comma-separated list of GPU indices"),
        ({"GPU": "0,"}, "GPU '0,' is not a comma-separated list of GPU indices"),
        ({"STORE": "/no/such/store"}, "STORE /no/such/store is not a directory"),
    ],
)
def test_a_bad_gpu_list_or_store_runs_nothing(tmp_path, env, refusal):
    r, log = _calibrate(tmp_path, "1", **env)

    assert r.returncode == 2
    assert f"calibrate: {refusal}" in r.stderr
    assert log == []
    assert not (tmp_path / "o").exists()


def test_every_gpu_command_takes_the_whole_gpu_list(tmp_path):
    r, log = _calibrate(tmp_path, "1", GPU="0,1,2,3")

    assert r.returncode == 0, r.stderr
    gpus = [line.split("--gpu ")[1].split()[0] for line in log if "--gpu" in line]
    assert gpus == ["0,1,2,3"] * 3
    assert "node-hours over every take of every step, on GPUs 0,1,2,3" in r.stdout


def test_status_holds_each_take_s_start_and_end_or_failure_and_the_end_times_each(
    tmp_path,
):
    r, _ = _calibrate(tmp_path, "1", "trace 4", CLIENT_FAIL="cal1-c1")
    assert r.returncode == 1
    o = tmp_path / "o"
    # The failed take ran half an hour, each stub take 0 or 1 s.
    said = (o / "status").read_text()
    (o / "status").write_text(re.sub(r"(fail cal1-c1) \d+ s", r"\1 1800 s", said))

    r, _ = _calibrate(tmp_path, "1", "trace 4")

    assert r.returncode == 0, r.stderr
    status = _status(o)
    failed = f"fail cal1-c1 N s: client.sh exit 1; see {o}/cal1-c1.client.log"
    assert status[:8] == [
        "start tool",
        "end tool N s",
        "start warm",
        "end warm N s",
        "start memory",
        "end memory N s",
        "start cal1-c1",
        failed,
    ]
    # The stub's tool is empty, so each run builds it again.
    assert status[8:-1] == [
        "start tool",
        "end tool N s",
        "start cal1-c1",
        "end cal1-c1 N s",
        "start trace2-c4",
        "end trace2-c4 N s",
        "start fit_steps",
        "end fit_steps N s",
        "start fit_host",
        "end fit_host N s",
    ]
    hours = "0.50 node-hours"
    assert status[-1] == f"end calibration {hours}"
    steps = ["tool", "warm", "memory", "cal1-c1", "trace2-c4", "fit_steps", "fit_host"]
    table = r.stdout.split("step | wall s\n")[1].splitlines()[: len(steps)]
    assert [re.sub(r" \d+$", " N", row) for row in table] == [f"{s} | N" for s in steps]
    assert f"calibrate: {hours} over every take of every step, on GPUs 7" in r.stdout


def test_a_failure_before_a_step_starts_is_its_failure_after_0_s(tmp_path):
    _calibrate(tmp_path, "1")
    (tmp_path / "o" / "cal1-c1").rename(tmp_path / "o" / "kept")

    r, _ = _calibrate(tmp_path, "1")

    assert r.returncode == 1
    last = (tmp_path / "o" / "status").read_text().splitlines()[-1]
    assert STAMP.sub("", last).startswith("fail cal1-c1 0 s: ")
    assert last.endswith("calibrate into a new OUT")


def _gated(tmp_path) -> Path:
    """The stub tree, its client.sh waiting for GATE before cal1-c1's run."""
    _tree(tmp_path)
    gate = "[[ $1 != *cal1-c1 ]] || until [[ -e $GATE ]]; do sleep 0.1; done\n"
    client = tmp_path / "ATOM" / "scripts" / "compass" / "client.sh"
    client.write_text(CLIENT.replace("#!/bin/bash\n", "#!/bin/bash\n" + gate))
    return tmp_path / "gate"


def _started(tmp_path, out: str, gate: Path, request) -> int:
    """calibrate.sh into `out` under `setsid nohup` from a shell that exits at
    once, as the guide starts one; its PID once its status shows cal1-c1. Its
    session is killed when the test ends, however it ends."""
    script = tmp_path / "ATOM" / "scripts" / "compass" / "calibrate.sh"
    env = {
        **os.environ,
        "PATH": _path(tmp_path),
        "LOG": str(tmp_path / "log"),
        "GPU": "7",
        "MODEL": str(tmp_path / "m"),
        "TRACES": "/cc/traces.jsonl",
        "AIPERF_PYTHON": "/real/python",
        "GATE": str(gate),
    }
    detached = f"setsid nohup bash {script} {out} 1 > {out}.log 2>&1 < /dev/null &"
    started = subprocess.run(
        ["bash", "-c", f"{detached} echo $!"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    pid = int(started.stdout)
    request.addfinalizer(lambda: _killed(pid))
    o = tmp_path / out
    _until(lambda: "start cal1-c1" in _read(o / "status"), "cal1-c1 started")
    return pid


def _killed(pid: int) -> None:
    """Kill the process group `pid` leads, setsid's, if it is still there."""
    with suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)


def _read(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _unheld(o: Path) -> bool:
    """Whether OUT's lock is free: no calibrate.sh, nor a step one started, runs."""
    with open(o / "inputs", "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


def test_under_setsid_nohup_it_outlives_its_shell_and_a_sighup(tmp_path, request):
    gate = _gated(tmp_path)
    pid = _started(tmp_path, "o", gate, request)

    os.kill(pid, signal.SIGHUP)
    time.sleep(0.5)
    gate.touch()

    o = tmp_path / "o"
    _until(lambda: _unheld(o), "end of the calibration")
    assert _status(o)[-1].startswith("end calibration ")
    assert not any(line.startswith("fail") for line in _status(o))
    assert "calibrate: calibrated in" in (tmp_path / "o.log").read_text()


def test_a_second_calibrate_sh_into_a_running_out_refuses_and_removes_nothing(
    tmp_path, request
):
    gate = _gated(tmp_path)
    _started(tmp_path, "o", gate, request)
    o = tmp_path / "o"
    # From its server's start, the first logs nothing until its client passes GATE.
    _until(lambda: (o / "cal1-c1" / "server.json").exists(), "cal1-c1's server")
    # Were the second not refused, its own client would not wait, and it would end.
    opened = tmp_path / "opened"
    opened.touch()

    r, log = _calibrate(tmp_path, "1", GATE=str(opened))

    assert r.returncode == 2
    held = f"{o} is held by another calibrate.sh, or a step one started, still running"
    assert f"calibrate: {held}" in r.stderr
    assert log == []
    assert (o / "cal1-c1" / "server.json").exists()
    gate.touch()
    _until(lambda: _unheld(o), "end of the first calibration")
    assert _status(o)[-1].startswith("end calibration ")


def test_a_sigterm_fails_its_step_which_a_resume_takes_again_and_nothing_before_it(
    tmp_path, request
):
    gate = _gated(tmp_path)
    pid = _started(tmp_path, "o", gate, request)
    o = tmp_path / "o"

    os.kill(pid, signal.SIGTERM)

    # The exit trap stops the step's server, which writes run.json as it stops;
    # the lock is free once it and the script have ended.
    _until(lambda: _unheld(o), "stopped calibration")
    assert (o / "cal1-c1" / "run.json").exists()
    assert _status(o)[-1] == "fail cal1-c1 N s: stopped by a signal"
    gate.touch()
    r, log = _calibrate(tmp_path, "1", GATE=str(gate))
    assert r.returncode == 0, r.stderr
    assert "calibrate: cal1-c1 was stopped; taking it again" in r.stdout
    assert log[0] == f"record --gpu 7 --out {o}/cal1-c1"
    assert [line[:8] for line in log[2:]] == ["fit step", "fit host"]
    assert _status(o)[-3:-1] == ["start fit_host", "end fit_host N s"]

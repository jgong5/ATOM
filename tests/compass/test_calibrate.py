# SPDX-License-Identifier: MIT
"""`calibrate.sh`: the quickstart's calibration in order on one GPU, a stopped
run resumed, and a resume under other inputs refused; a 1P1D configuration's,
each record one deployment of both sides, each side FIT names measured and
fitted, and two sharing a prefill side published into one store.

It runs from a copy in a stub tree, as `test_smoke_cell.py` drives
`smoke_cell.sh`: `atom.compass.cli` logs each command; a `record` writes
server.json and waits for the SIGTERM `client.sh --stop` sends, then with
`--trace` writes its busy lines, or exits 1 without them when NO_BUSY; `measure
memory` and a fit write their readable copy with a token, the law with one
held-out band, and publish into the store `--store` names an entry keyed by
the ATOM arguments of the start they measure or of their first record, and
the variables KEYED names it ran under, refusing one whose key holds one, as
the store refuses a second publish at a key, its file one no store reader
takes for an entry of its own; FAIL, matched against a command's
words, makes it refuse. `deploy.sh` logs its arguments and each side's
variables, and serves as a `record` does, each side's record in its own
directory, its prefill side's busy lines with `--trace` among that side's own
flags. `client.sh` logs its arguments, its Python and its scenario, stops its
server and writes its export marked `server_stopped`, or writes it unmarked and
exits 1 without stopping the server when CLIENT_FAIL is in its OUT.

The stub tree's `python3` runs without site imports (`-S`), first on the PATH
`_path` gives every test that runs a script in the tree. In an image whose site
imports torch at start, as the MI455 image's `.pth` does, rocprofiler-sdk would
load the empty tool `record` gets in ROCP_TOOL_LIBRARIES, and abort.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

COMPASS = Path(__file__).resolve().parents[2] / "scripts" / "compass"
CLI = """import hashlib, json, os, signal, sys, time
from pathlib import Path
COPY = {"memory": "memory_readings", "steps": "step_linear_law", "host": "host_laws"}
BAND = {"decode rung 1": {"steps": 9, "p10": -0.012, "p50": 0.001, "p90": 0.02}}
def started(atom):
    return {"args": atom, "env": {k: os.environ[k] for k in os.environ.get("KEYED", "").split() if k in os.environ}}
def main():
    argv = sys.argv[1:]
    words = " ".join(argv[: argv.index("--")] if "--" in argv else argv)
    open(os.environ["LOG"], "a").write(words + "\\n")
    if os.environ.get("FAIL") and os.environ["FAIL"] in words:
        sys.exit("refused")
    out = Path(argv[argv.index("--out") + 1])
    out.mkdir(parents=True)
    if argv[0] == "record":
        def stop(*_):
            out.joinpath("run.json").write_text("{}")
            if "--trace" in argv:
                os.environ.get("NO_BUSY") and sys.exit(1)
                out.joinpath("steps").mkdir()
                out.joinpath("steps", "dp0.busy.jsonl").write_text("{}")
            sys.exit(0)
        signal.signal(signal.SIGTERM, stop)
        out.joinpath("tool").write_text(os.environ.get("ROCP_TOOL_LIBRARIES", ""))
        out.joinpath("started.json").write_text(json.dumps(started(argv[argv.index("--") + 1 :])))
        out.joinpath("server.json").write_text(json.dumps({"pid": os.getpid()}))
        time.sleep(600)
    if argv[0] == "measure":
        entry = started(argv[argv.index("--") + 1 :])
    else:
        first = next(Path(a) for a in argv if Path(a, "started.json").is_file())
        entry = json.loads(first.joinpath("started.json").read_text())
    key = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:12]
    at = Path(argv[argv.index("--store") + 1], COPY[argv[1]], key)
    if at.exists():
        sys.exit(f"refused: {at} holds an entry")
    at.mkdir(parents=True)
    at.joinpath("stub_entry.json").write_text(json.dumps(entry))
    law = {"unfitted": {}, "decode": {"1": {"held_out": {"bands": BAND}}}}
    copy = {"token": "tok-" + argv[1], "params": law}
    out.joinpath(COPY[argv[1]] + ".json").write_text(json.dumps(copy))
"""
DEPLOY = """#!/usr/bin/env python3
import json, os, signal, sys, time
from pathlib import Path
argv = sys.argv[1:]
out, own = Path(argv[0]), {}
env = [os.environ.get(k, "") for k in ("PREFILL_ENV", "DECODE_ENV", "WAIT_S")]
said = " ".join(argv[1:]) + " [" + "|".join(env) + "]"
open(os.environ["LOG"], "a").write(f"deploy {out} {said}\\n")
out.mkdir(parents=True)
for side, end in (("prefill", "--decode"), ("decode", "--router")):
    at = argv.index("--" + side)
    split = argv.index("--", at)
    own[side] = argv[at + 1 : split]
    keyed = dict(w.split("=", 1) for w in os.environ.get(side.upper() + "_ENV", "").split())
    keyed = {k: v for k, v in (dict(os.environ) | keyed).items() if k in os.environ.get("KEYED", "").split()}
    out.joinpath(side).mkdir()
    started = {"args": argv[split + 1 : argv.index(end)], "env": keyed}
    out.joinpath(side, "started.json").write_text(json.dumps(started))
    out.joinpath(side, "tool").write_text(os.environ.get("ROCP_TOOL_LIBRARIES", ""))
def stop(*_):
    out.joinpath("run.json").write_text("{}")
    if "--trace" in own["prefill"]:
        os.environ.get("NO_BUSY") and sys.exit(1)
        out.joinpath("prefill", "steps").mkdir()
        out.joinpath("prefill", "steps", "dp0.busy.jsonl").write_text("{}")
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
out.joinpath("server.json").write_text(json.dumps({"pid": os.getpid()}))
time.sleep(600)
"""
BLOCKED = """import sys
from pathlib import Path
tool = Path(sys.argv[1], "libcompass_blocked.so")
tool.parent.mkdir(parents=True)
tool.write_text("")
print(tool)
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
#: The stub tree's `serve.TRACE_STOP_S`.
TRACE_STOP_S = 5
#: A 1P1D configuration, its decode side's keyed variable set apart; and its
#: router's arguments.
ROUTER = "--backend atom --disable-retries"
ONE_P_ONE_D = {
    "GPU": None,
    "PREFILL_GPU": "0,1",
    "DECODE_GPU": "2,3",
    "PREFILL_ARGS": "--enforce-eager",
    "DECODE_ARGS": "--max-num-seqs 4",
    "PREFILL_ENV": "ATOM_KV_OFFLOAD=mooncake_store",
    "DECODE_ENV": "GPU_MAX_HW_QUEUES=2",
    "ROUTER_ARGS": ROUTER,
    "KEYED": "ATOM_KV_OFFLOAD GPU_MAX_HW_QUEUES",
    "ATOM_KV_OFFLOAD": None,
    "GPU_MAX_HW_QUEUES": None,
}


def _tree(tmp_path):
    scripts = tmp_path / "ATOM" / "scripts" / "compass"
    scripts.mkdir(parents=True)
    for name in ("calibrate.sh", "_lib.sh"):
        shutil.copy(COMPASS / name, scripts / name)
    for name, text in (("client.sh", CLIENT), ("deploy.sh", DEPLOY)):
        (scripts / name).write_text(text)
        (scripts / name).chmod(0o755)
    (tmp_path / "ATOM" / ".compass-commit").write_text("cafe\n")
    (tmp_path / "ATOM" / "atom" / "compass").mkdir(parents=True)
    (tmp_path / "ATOM" / "atom" / "compass" / "cli.py").write_text(CLI)
    (tmp_path / "ATOM" / "atom" / "compass" / "parity").mkdir()
    (tmp_path / "ATOM" / "atom" / "compass" / "parity" / "blocked.py").write_text(
        BLOCKED
    )
    (tmp_path / "ATOM" / "atom" / "compass" / "drive").mkdir()
    (tmp_path / "ATOM" / "atom" / "compass" / "drive" / "serve.py").write_text(
        f"TRACE_STOP_S = {TRACE_STOP_S}\n"
    )
    for d in ("atom", "atom/compass", "atom/compass/parity", "atom/compass/drive"):
        (tmp_path / "ATOM" / d / "__init__.py").write_text("")
    shim = tmp_path / "bin" / "python3"
    shim.parent.mkdir()
    shim.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -S "$@"\n')
    shim.chmod(0o755)


def _path(tmp_path):
    """PATH for a script in the stub tree: its `python3`, then this Python's."""
    return f"{tmp_path / 'bin'}:{Path(sys.executable).parent}:{os.environ['PATH']}"


def _calibrate(tmp_path, *records, out="o", **env):
    """One run of the stub tree's calibrate.sh into `out`, and the commands it
    ran; a variable given None is unset."""
    if not (tmp_path / "ATOM").exists():
        _tree(tmp_path)
    log = tmp_path / "log"
    log.unlink(missing_ok=True)
    env = {
        **os.environ,
        "PATH": _path(tmp_path),
        "LOG": str(log),
        "GPU": "7",
        "MODEL": str(tmp_path / "m"),
        "TRACES": "/cc/traces.jsonl",
        "AIPERF_PYTHON": "/real/python",
        **env,
    }
    env = {k: v for k, v in env.items() if v is not None}
    script = tmp_path / "ATOM" / "scripts" / "compass" / "calibrate.sh"
    cmd = ["bash", str(script), out, *records]
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
    store = f"--store {o}/store"
    cal = f"{o}/cal1-c1 {o}/cal2-c8"
    runs = " ".join(
        f"--client-run {o}/{d} {o}/{d}/artifacts/client_run.json"
        for d in ("cal1-c1", "cal2-c8")
    )
    assert log == [
        f"record --gpu 7 --out {o}/warm",
        f"client {o}/warm --traces 0-7 --concurrency 1 --duration 600 --stop /real/python []",
        f"measure memory --gpu 7 {store} --out {o}/memory",
        f"record --gpu 7 --out {o}/cal1-c1",
        f"client {o}/cal1-c1 --traces 256-392 --concurrency 1 {client} []",
        f"record --gpu 7 --out {o}/cal2-c8",
        f"client {o}/cal2-c8 --traces 256-392 --concurrency 8 {client} [{FIRST_TURN[2:]}]",
        f"fit steps --backend step-linear --form history --model {tmp_path}/m {cal} {runs} {store} --out {o}/fit_steps",
        f"fit host {cal} {runs} {store} --out {o}/fit_host",
    ]
    assert (
        "memory_readings tok-memory, step_linear_law tok-steps, host_laws tok-host"
        in r.stdout
    )
    assert "decode rung 1 | 9 | -1.2% | +0.1% | +2.0%" in r.stdout
    compass = tmp_path / "ATOM" / "scripts" / "compass" / "compass"
    run = f"{compass} run --step-linear tok-steps --host-laws tok-host"
    run += f" --device DATASHEET --ptl PTL {store} --out DIR"
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
        (["1"], ONE_P_ONE_D | {"ROUTER_ARGS": None}, "ROUTER_ARGS is not set"),
        (["1"], ONE_P_ONE_D | {"GPU": "7"}, "in place of GPU and SERVER_ARGS"),
        (["1"], ONE_P_ONE_D | {"FIT": "decode prefill"}, "FIT 'decode prefill' is"),
        (
            ["1", "trace 4"],
            ONE_P_ONE_D | {"FIT": "decode"},
            "RECORD 'trace 4' traces the prefill side, which FIT does not fit",
        ),
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


def test_trace_runs_go_to_fit_steps_alone(tmp_path):
    r, log = _calibrate(tmp_path, "1", "trace 16", "trace 4")
    assert r.returncode == 0, r.stderr
    o, client = tmp_path / "o", "--stop /real/python []"
    store = f"--store {o}/store"
    dirs = ("cal1-c1", "trace2-c16", "trace3-c4")
    runs = [f"--client-run {o}/{d} {o}/{d}/artifacts/client_run.json" for d in dirs]
    inputs = " ".join([*(f"{o}/{d}" for d in dirs), *runs])
    assert log[3:] == [
        f"record --gpu 7 --out {o}/cal1-c1",
        f"client {o}/cal1-c1 --traces 256-392 --concurrency 1 --duration 1800 {client}",
        f"record --gpu 7 --trace --out {o}/trace2-c16",
        f"client {o}/trace2-c16 --traces 256-392 --concurrency 16 --duration 120 {client}",
        f"record --gpu 7 --trace --out {o}/trace3-c4",
        f"client {o}/trace3-c4 --traces 256-392 --concurrency 4 --duration 120 {client}",
        f"fit steps --backend step-linear --form history --model {tmp_path}/m {inputs} {store} --out {o}/fit_steps",
        f"fit host {o}/cal1-c1 {runs[0]} {store} --out {o}/fit_host",
    ]


def test_a_trace_run_refused_at_its_stop_is_taken_again(tmp_path):
    r, _ = _calibrate(tmp_path, "1", "trace 4", NO_BUSY="1")
    assert r.returncode == 1
    assert "calibrate: step trace2-c4 failed: its server exit 1" in r.stderr
    exported = tmp_path / "o" / "trace2-c4" / "artifacts" / "client_run.json"
    assert "server_stopped" in exported.read_text()
    r, log = _calibrate(tmp_path, "1", "trace 4")
    assert r.returncode == 0, r.stderr
    assert "calibrate: trace2-c4 was stopped; taking it again" in r.stdout
    assert log[0] == f"record --gpu 7 --trace --out {tmp_path}/o/trace2-c4"


def test_every_server_but_a_trace_run_s_loads_the_tool(tmp_path):
    r, _ = _calibrate(tmp_path, "1", "trace 4")
    assert r.returncode == 0, r.stderr
    o = tmp_path / "o"
    tool = f"{o}/tool/libcompass_blocked.so"
    loaded = [(o / d / "tool").read_text() for d in ("warm", "cal1-c1", "trace2-c4")]
    assert loaded == [tool, tool, ""]


def _deployed(tmp_path, out: str, name: str, trace: str = "") -> str:
    """The stub `deploy.sh` line of `ONE_P_ONE_D`'s record `name` into `out`,
    each side's variables and the seconds it waits for each instance and its
    stop: `calibrate.sh`'s default 1800, and a trace run `TRACE_STOP_S` more."""
    o, m = tmp_path / out, tmp_path / "m"
    waits = 1800 + (TRACE_STOP_S if trace else 0)
    return (
        f"deploy {o}/{name} record --prefill --gpu 0,1{trace} -- --model {m} "
        f"--enforce-eager --decode --gpu 2,3 -- --model {m} --max-num-seqs 4 "
        f"--router -- {ROUTER} "
        f"[ATOM_KV_OFFLOAD=mooncake_store|GPU_MAX_HW_QUEUES=2|{waits}]"
    )


def test_a_1p1d_calibration_records_deployments_and_fits_each_side_on_its_own(
    tmp_path,
):
    r, log = _calibrate(tmp_path, "1", "trace 4", **ONE_P_ONE_D)
    assert r.returncode == 0, r.stderr
    o, store = tmp_path / "o", f"--store {tmp_path / 'o'}/store"
    stop = "--stop /real/python []"
    fit = f"fit steps --backend step-linear --form history --model {tmp_path}/m"
    run = "--client-run {o}/{d}/{s} {o}/{d}/artifacts/client_run.json"
    prefill = [run.format(o=o, d=d, s="prefill") for d in ("cal1-c1", "trace2-c4")]
    decode = run.format(o=o, d="cal1-c1", s="decode")
    assert log == [
        _deployed(tmp_path, "o", "warm"),
        f"client {o}/warm --traces 0-7 --concurrency 1 --duration 600 {stop}",
        f"measure memory --gpu 0,1 {store} --out {o}/memory_prefill",
        f"measure memory --gpu 2,3 {store} --out {o}/memory_decode",
        _deployed(tmp_path, "o", "cal1-c1"),
        f"client {o}/cal1-c1 --traces 256-392 --concurrency 1 --duration 1800 {stop}",
        _deployed(tmp_path, "o", "trace2-c4", " --trace"),
        f"client {o}/trace2-c4 --traces 256-392 --concurrency 4 --duration 120 {stop}",
        (
            f"{fit} {o}/cal1-c1/prefill {o}/trace2-c4/prefill {' '.join(prefill)} "
            f"{store} --out {o}/fit_steps_prefill"
        ),
        f"fit host {o}/cal1-c1/prefill {prefill[0]} {store} --out {o}/fit_host_prefill",
        f"{fit} {o}/cal1-c1/decode {decode} {store} --out {o}/fit_steps_decode",
        f"fit host {o}/cal1-c1/decode {decode} {store} --out {o}/fit_host_decode",
    ]
    # Each side measured under its own variables, on the connector a real
    # start of it takes.
    m, kv = ["--model", str(tmp_path / "m")], "--kv-transfer-config"
    role = '{{"kv_role": "{}", "kv_connector": "mooncake"}}'.format
    assert _entries(o / "store", "memory_readings") == sorted(
        [
            (
                [*m, "--enforce-eager", kv, role("kv_producer")],
                {"ATOM_KV_OFFLOAD": "mooncake_store"},
            ),
            (
                [*m, "--max-num-seqs", "4", kv, role("kv_consumer")],
                {"GPU_MAX_HW_QUEUES": "2"},
            ),
        ],
        key=json.dumps,
    )
    # A trace run's deployment loads no tool.
    tool = f"{o}/tool/libcompass_blocked.so"
    loaded = [
        (o / d / side / "tool").read_text()
        for d in ("cal1-c1", "trace2-c4")
        for side in ("prefill", "decode")
    ]
    assert loaded == [tool, tool, "", ""]
    assert "calibrate: prefill: memory_readings tok-memory" in r.stdout
    assert "calibrate: the decode side's law\nheld out | steps" in r.stdout
    assert "a run priced by the law" not in r.stdout
    assert "on GPUs 0,1,2,3" in r.stdout


def test_two_1p1d_calibrations_sharing_a_prefill_side_publish_each_key_once(
    tmp_path,
):
    store = tmp_path / "S"
    store.mkdir()
    shared = ONE_P_ONE_D | {"STORE": str(store)}
    r, _ = _calibrate(tmp_path, "1", "trace 4", out="a", **shared)
    assert r.returncode == 0, r.stderr
    # B's decode differs in a keyed variable, and B borrows A's prefill side:
    # stopped at its second record, then resumed.
    b = shared | {"DECODE_ENV": "GPU_MAX_HW_QUEUES=4", "FIT": "decode"}
    r, _ = _calibrate(tmp_path, "1", "4", out="b", CLIENT_FAIL="cal2-c4", **b)
    assert r.returncode == 1
    assert "calibrate: step cal2-c4 failed: client.sh exit 1" in r.stderr
    r, log = _calibrate(tmp_path, "1", "4", out="b", **b)

    assert r.returncode == 0, r.stderr
    assert "calibrate: cal2-c4 was stopped; taking it again" in r.stdout
    # The resume measures nothing again, nor any prefill side.
    assert [line for line in log if line.startswith("measure")] == []
    statuses = {o: (tmp_path / o / "status").read_text().splitlines() for o in "ab"}
    assert [line.split()[1:3] for line in statuses["b"] if "fail" in line] == [
        ["fail", "cal2-c4"]
    ]
    assert all(
        lines[-1].split()[1:3] == ["end", "calibration"] for lines in statuses.values()
    )
    # Each key published once into one store: A's two sides and B's decode,
    # whose prefill side is A's.
    m = ["--model", str(tmp_path / "m")]
    prefill = (m + ["--enforce-eager"], {"ATOM_KV_OFFLOAD": "mooncake_store"})
    decode = m + ["--max-num-seqs", "4"]
    sides = [
        prefill,
        (decode, {"GPU_MAX_HW_QUEUES": "2"}),
        (decode, {"GPU_MAX_HW_QUEUES": "4"}),
    ]
    for kind in ("step_linear_law", "host_laws"):
        assert _entries(store, kind) == sorted(sides, key=json.dumps)
    measured = [(args[:-2], env) for args, env in _entries(store, "memory_readings")]
    assert measured == sorted(sides, key=json.dumps)


def _entries(store, kind: str) -> list:
    """The ATOM arguments and keyed variables of each of the stub store's
    `kind` entries."""
    held = [json.loads(e.read_text()) for e in store.glob(f"{kind}/*/stub_entry.json")]
    return sorted(((h["args"], h["env"]) for h in held), key=json.dumps)

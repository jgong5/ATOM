# SPDX-License-Identifier: MIT
"""`compass run --check` prints a lacking step linear law with the
`calibrate.sh` command that records and fits one for the configuration: a
first-turn record for each decode rung it schedules, at that rung's batch on
every DP rank, then the trace runs where `compass record --trace` takes one,
on the cards ATOM's ranks use; then its expected time, unknown.

The deployment is `test_run_step_grid`'s, at the ladder and bounds a test
sets, checked as `test_run_lacking_rungs` checks it; the printed command runs
in the stub tree `test_calibrate` runs `calibrate.sh` in.
"""

import json
import os
import re
import shlex
import subprocess
import sys

import platforms
import pytest
from test_calibrate import _tree
from test_memory_readings import DOCUMENT
from test_run_lacking import served
from test_run_lacking_rungs import DEFAULT_SIZES, DPA8, mtp
from test_run_step_grid import deployment

from atom.compass.drive import lacking
from atom.compass.drive import run as drive_run
from atom.compass.runner.overrides import schedulable_ladder

FIRST_TURN = "--trajectory-start-min-ratio 0 --trajectory-start-max-ratio 0"
TRACES = ["trace 16", "trace 4"]
#: The bounds of the DeepSeek-V4-Pro template's single-node configurations.
TEMPLATE = {"capture_sizes": DEFAULT_SIZES, "max_num_seqs": 512}
TEMPLATE |= {"max_num_batched_tokens": 16384, "speculative_config": mtp(3)}
TP8 = ["-tp", "8", "--max-num-seqs", "512", "--max-num-batched-tokens", "16384"]
#: A token no store entry has.
ABSENT = "0123456789ab"
UNKNOWN = "expected time: unknown, as calibrate.sh records no calibration's time"


def printed(tmp_path, monkeypatch, capsys, atom_args, *flags, law=None, **config):
    """The deployment's config with `config`, and the step linear law's lack
    `--check` prints under ATOM's arguments `atom_args` and `flags`, for the
    law file `law`, else a token the store holds none of: the command that
    makes it, as printed, and each line printed after it."""
    r = deployment(tmp_path, **config)
    served(monkeypatch, r.config)
    at = json.loads(law.read_text())["platform"] if law else None
    common = platforms.flags(tmp_path / "store", DOCUMENT, tmp_path, at=at)
    common += ["--store", str(tmp_path / "store"), *flags]
    given = ["--step-linear", str(law or ABSENT), *common, "--check", "--"]
    with pytest.raises(SystemExit):
        drive_run.main([*given, "--model", str(tmp_path), *atom_args])
    out = capsys.readouterr().out
    state = r": stale \(coverage\)" if law else f" {ABSENT}: missing"
    lack = re.search(
        rf"^lacking step linear law{state}\n  refused: [^\n]*\n  made by: (.*?)\n"
        r"(?=lacking |apart: |\Z)",
        out,
        re.MULTILINE | re.DOTALL,
    )
    command, *after = lack[1].split("\n  ")
    return r.config, command, after


def parts(command):
    """A printed command's variables, script, OUT and RECORD arguments."""
    words = shlex.split(command)
    env = dict(w.split("=", 1) for w in words if re.match(r"^[A-Z_]+=", w))
    script, out, *records = words[len(env) :]
    return env, script, out, records


def test_each_rung_of_a_tp_ladder_is_a_first_turn_record_then_the_trace_runs(
    tmp_path, monkeypatch, capsys
):
    # S1: no law resolves, so the configuration was never calibrated.
    config, command, after = printed(tmp_path, monkeypatch, capsys, TP8, **TEMPLATE)

    env, script, out, records = parts(command)
    ladder = schedulable_ladder(config)
    assert records == [f"{rung} {FIRST_TURN}" for rung in ladder] + TRACES
    assert ladder[-1] == 512
    assert env["GPU"] == "0,1,2,3,4,5,6,7"
    assert (env["MODEL"], env["SERVER_ARGS"]) == (str(tmp_path), " ".join(TP8))
    assert (script, out) == (str(lacking.CALIBRATE), os.path.abspath("cal"))
    assert after == [UNKNOWN]


def test_a_dp_rank_takes_each_rung_s_batch_and_no_trace_run_is_planned(
    tmp_path, monkeypatch, capsys
):
    # S2, against the shipped DPA8 decode law, which it schedules rungs past.
    dpa8 = [*TP8, "--enable-dp-attention"]
    config, command, after = printed(
        tmp_path, monkeypatch, capsys, dpa8, law=DPA8, **TEMPLATE
    )

    env, _, _, records = parts(command)
    assert records == [f"{8 * r} {FIRST_TURN}" for r in schedulable_ladder(config)]
    assert env["GPU"] == "0,1,2,3,4,5,6,7"
    untraced = (
        "no trace run, which compass record refuses: --trace reduces each TP "
        "rank's busy time, and ATOM's arguments run DP ranks or pipeline stages; "
        "so the law's prefill stays unfitted"
    )
    assert after == [untraced, UNKNOWN]


@pytest.mark.parametrize(
    "eager, ladder", [(False, ["1", "2", "4", "8", "16", "48"]), (True, ["48"])]
)
def test_the_eager_rung_is_recorded_at_the_scheduler_s_bound(
    tmp_path, monkeypatch, capsys, eager, ladder
):
    bounds = {"capture_sizes": [1, 2, 4, 8, 16], "max_num_seqs": 48}
    atom_args = ["--max-num-seqs", "48", *(["--enforce-eager"] * eager)]
    _, command, _ = printed(
        tmp_path, monkeypatch, capsys, atom_args, enforce_eager=eager, **bounds
    )

    env, _, _, records = parts(command)
    assert records == [f"{n} {FIRST_TURN}" for n in ladder] + TRACES
    assert env["GPU"] == "0"


def test_an_argument_calibrate_sh_would_split_prints_why_in_place_of_a_command(
    tmp_path, monkeypatch, capsys
):
    quant = ["--online_quant_config", '{"use_index_cache": true}']
    _, command, after = printed(tmp_path, monkeypatch, capsys, ["-tp", "2", *quant])

    assert command == (
        "no calibrate.sh command: it splits SERVER_ARGS on whitespace, so it "
        f"cannot pass ATOM's argument {quant[1]!r}"
    )
    assert after == []


def test_the_printed_command_runs_as_printed(tmp_path, monkeypatch, capsys):
    _tree(tmp_path)
    stub = tmp_path / "ATOM" / "scripts" / "compass" / "calibrate.sh"
    monkeypatch.setattr(lacking, "CALIBRATE", stub)
    monkeypatch.setenv("TRACES", "/cc/traces.jsonl")
    monkeypatch.setenv("AIPERF_PYTHON", "/real/python")
    o, atom_args = tmp_path / "o", ["-tp", "2", "--max-num-seqs", "2"]
    bounds = {"capture_sizes": [1, 2, 4], "max_num_seqs": 2}
    _, command, _ = printed(
        tmp_path, monkeypatch, capsys, atom_args, "--out", str(o), **bounds
    )
    # The command states the clients' variables: the shell it runs in has none.
    env = {k: v for k, v in os.environ.items() if k not in lacking.CLIENT_ENV}
    env |= {"LOG": str(tmp_path / "log")}
    env["PATH"] = f"{os.path.dirname(sys.executable)}:{env['PATH']}"

    r = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert r.returncode == 0, r.stderr
    log = (tmp_path / "log").read_text().splitlines()
    client = "--duration 1800 --stop /real/python"
    traced = "--duration 120 --stop /real/python []"
    assert [line for line in log if line.startswith(("record", "client"))][2:] == [
        f"record --gpu 0,1 --out {o}/cal1-c1",
        f"client {o}/cal1-c1 --traces 256-392 --concurrency 1 {client} [{FIRST_TURN}]",
        f"record --gpu 0,1 --out {o}/cal2-c2",
        f"client {o}/cal2-c2 --traces 256-392 --concurrency 2 {client} [{FIRST_TURN}]",
        f"record --gpu 0,1 --trace --out {o}/trace3-c16",
        f"client {o}/trace3-c16 --traces 256-392 --concurrency 16 {traced}",
        f"record --gpu 0,1 --trace --out {o}/trace4-c4",
        f"client {o}/trace4-c4 --traces 256-392 --concurrency 4 {traced}",
    ]
    inputs = (o / "inputs").read_text().splitlines()
    assert inputs[1:3] == [
        f"atom --model {tmp_path} -tp 2 --max-num-seqs 2",
        "traces /cc/traces.jsonl 256-392",
    ]

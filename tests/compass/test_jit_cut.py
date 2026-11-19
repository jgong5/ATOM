# SPDX-License-Identifier: MIT
"""A record whose server compiled while it served: each compile timed from its
cache files (`aiter_jit.builds_since`) and placed on the step record's clock by
run.json's clock pair, each rank's steps that overlap it cut whole as `jit` by
`validate.cost` and the fits, the record refused only when its compiles carry
no times or cut too many steps, and one-run grading counting them."""

import importlib.util
import json
import os
from pathlib import Path

import pytest
from test_fit_steps import (  # noqa: F401  planted_roots: autouse
    LINEAR,
    clients,
    fit,
    law_key,
    member,
    planted_roots,
    record,
)
from test_validate_cost import LAW, _record, _steps

from atom.compass.backends import step_labels
from atom.compass.backends.step_labels import (
    ACCEPT_BUILDS_ENV,
    BUILDS,
    CLOCK,
    COMPILES,
    JIT,
    JIT_SLACK_NS,
)
from atom.compass.drive import aiter_jit
from atom.compass.validate import stalls
from atom.compass.validate.cost import cost_accuracy

#: The wall clock when the step record's monotonic clock read 0.
WALL = 1_795_000_000_000_000_000
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compass" / "one_real.py"


def _compiled(run_dir, *spans):
    """run.json naming a kernel compiled over each of `spans`, (first, last)
    file written on the step record's clock, timed and with the clock pair."""
    compiles = [
        {"compiler": "triton", "name": "_k", "artifact": f"/c/{i}"}
        | {"first_ns": WALL + a, "last_ns": WALL + b}
        for i, (a, b) in enumerate(spans)
    ]
    run = {BUILDS: ["_k"], COMPILES: compiles, CLOCK: [WALL, 0]}
    (run_dir / "run.json").write_text(json.dumps(run))


def _windowed(record):
    """The lines of `record`'s rank 0, each exiting when its device work ends."""
    path = record / "dp0.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for s in lines:
        s["t_exit_ns"] = s["t_enter_ns"] + s["d_busy_ns"]
    path.write_text("".join(json.dumps(s) + "\n" for s in lines))
    return lines


def test_the_step_that_waited_on_a_compile_is_cut_and_the_record_costed(
    tmp_path, monkeypatch
):
    """Prefill step 4 ran ten times its price, a compile inside it; the rest
    are priced as the law prices them. One step of twenty is past the bound
    a record of thousands takes."""
    monkeypatch.delenv(ACCEPT_BUILDS_ENV, raising=False)
    monkeypatch.setattr(step_labels, "JIT_BOUND", 0.1)
    record = _record(tmp_path / "steps", _steps(), stretch={4: 10.0})
    step = _windowed(record)[4]
    start = step["t_enter_ns"] + JIT_SLACK_NS + 1
    _compiled(tmp_path, (start, start + 80_000_000))

    report = cost_accuracy(record, LAW)

    assert report["cut"][JIT] == 1
    prefill = report["validated"]["prefill"]
    assert prefill["steps"] == 9
    assert prefill["on_totals"] == pytest.approx(0.0, abs=1e-9)


def test_compiles_untimed_or_cutting_too_many_steps_refuse(tmp_path, monkeypatch):
    monkeypatch.delenv(ACCEPT_BUILDS_ENV, raising=False)
    record = _record(tmp_path / "steps", _steps())
    lines = _windowed(record)
    _compiled(
        tmp_path, (lines[2]["t_enter_ns"] + JIT_SLACK_NS + 1, lines[-1]["t_exit_ns"])
    )

    with pytest.raises(ValueError, match="overlap 18 of DP rank 0's 20 forwards"):
        cost_accuracy(record, LAW)

    run = json.loads((tmp_path / "run.json").read_text())
    del run[CLOCK]
    (tmp_path / "run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match=f"its run.json has no {COMPILES} and {CLOCK}"):
        cost_accuracy(record, LAW)

    monkeypatch.setenv(ACCEPT_BUILDS_ENV, "1")  # taken whole
    assert JIT not in cost_accuracy(record, LAW)["cut"]


def test_a_fit_takes_a_record_whose_compiles_are_timed_and_cuts_their_steps(
    tmp_path, monkeypatch
):
    """One compile inside step 100, with no slack: the record's steps are
    shorter than it, and more than 1% of them cut would refuse it."""
    monkeypatch.delenv(ACCEPT_BUILDS_ENV, raising=False)
    monkeypatch.setattr(step_labels, "JIT_SLACK_NS", 0)
    a = record(tmp_path / "a", 19)
    lines = [json.loads(x) for x in (a / "steps/dp0.jsonl").read_text().splitlines()]
    at = (lines[100]["t_enter_ns"] + lines[100]["t_exit_ns"]) // 2
    run = json.loads((a / "run.json").read_text())
    _compiled(a, (at, at))
    (a / "run.json").write_text(
        json.dumps(run | json.loads((a / "run.json").read_text()))
    )

    fit(tmp_path, "fit steps", *LINEAR, a, *clients(a))

    assert member(tmp_path / "store", law_key(), "params")["cut"][JIT] == 1


def test_builds_since_times_each_compile_by_its_artifact(tmp_path, monkeypatch):
    """Two compiles of one Triton kernel in two key directories, an aiter
    module's nested build files as one, two builds of one FlyDSL kernel apart
    in its directory, as a cell A record's ``launch_radix_topk_one_block`` built
    6 min apart, and no file written before."""
    triton, jit, fly = tmp_path / "triton", tmp_path / "jit", tmp_path / "fly"
    roots = {triton: "triton", jit: "aiter", fly: "flydsl"}
    monkeypatch.setattr(aiter_jit, "_roots", lambda: roots)
    since = WALL
    topk = fly / "launch_radix_topk_one_block"
    written = {
        triton / "A" / "_k.source": since + 1_000,
        triton / "A" / "__grp___k.json": since + 9_000,
        triton / "B" / "_k.source": since + 50_000,
        triton / "B" / "_k.json": since + 60_000,
        triton / "C" / "_old.json": since - 1,
        jit / "build" / "mha" / "blob" / "k.cpp": since + 1_500,
        jit / "build" / "mha" / "build" / "k.o": since + 2_000,
        topk / "407d.lock": since + 3_000,
        topk / "407d.pkl": since + 4_000,
        topk / "e7a7.lock": since + 400_000,
        topk / "e7a7.pkl": since + 401_000,
    }
    for path, ns in written.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        os.utime(path, ns=(ns, ns))

    got = aiter_jit.builds_since(since)

    assert got[BUILDS] == ["_k", "launch_radix_topk_one_block", "mha"]
    expected = [
        ("triton", "_k", triton / "A", 1_000, 9_000),
        ("aiter", "mha", jit / "build" / "mha", 1_500, 2_000),
        ("flydsl", topk.name, topk / "407d", 3_000, 4_000),
        ("triton", "_k", triton / "B", 50_000, 60_000),
        ("flydsl", topk.name, topk / "e7a7", 400_000, 401_000),
    ]
    assert got[COMPILES] == [
        {
            "compiler": c,
            "name": n,
            "artifact": str(d),
            "first_ns": since + f,
            "last_ns": since + t,
        }
        for c, n, d, f, t in expected
    ]


def test_one_real_counts_the_compiles_in_its_graded_window_and_the_steps_they_cut(
    tmp_path, monkeypatch
):
    """Two ranks of 1 s steps; one compile before the window, two in step 5."""
    spec = importlib.util.spec_from_file_location("one_real", SCRIPT)
    one_real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(one_real)
    real = tmp_path / "real"
    (real / "steps").mkdir(parents=True)
    for rank in (0, 1):
        lines = [
            {"step": i, "t_enter_ns": i * 10**9, "t_exit_ns": (i + 1) * 10**9 - 1}
            for i in range(10)
        ]
        (real / "steps" / f"dp{rank}.jsonl").write_text(
            "".join(json.dumps(s) + "\n" for s in lines)
        )
    _compiled(
        real,
        (1_500_000_000, 1_500_000_000),
        (5_500_000_000, 5_500_000_000),
        (5_600_000_000, 5_700_000_000),
    )
    monkeypatch.setattr(stalls, "window", lambda run: (2.0, 8.0))

    assert one_real.compiled(real) == (
        "real: 2 of 3 JIT compiles in the graded window; the steps they cut "
        "there: 2 forwards over its ranks, 1.000 s on the rank they cut most"
    )

# SPDX-License-Identifier: MIT
"""A record whose server compiled while it served: each compile timed from its
cache files (`aiter_jit.builds_since`) and placed on the step record's clock by
run.json's clock pair, each rank's steps that overlap it cut whole as `jit` by
`validate.cost` and the fits, a forward whose host segments overlap one left
without host labels, the record refused only when its compiles carry no times
or cut too many steps; and grading, by `validate.runs` or `one_real.py`,
refusing the run whole however few steps they cut."""

import importlib.util
import json
import os
import re
from pathlib import Path

import pytest
import test_backend_step_linear_fit as linear
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
from test_validate_runs import GRADED, _four, _recorded

from atom.compass.backends import step_labels, step_linear_fit
from atom.compass.backends.step_labels import (
    ACCEPT_BUILDS_ENV,
    BUILDS,
    CLOCK,
    COMPILES,
    JIT,
    JIT_SLACK_NS,
)
from atom.compass.drive import aiter_jit
from atom.compass.validate.cost import cost_accuracy
from atom.compass.validate.runs import grade

#: The wall clock when the step record's monotonic clock read 0.
WALL = 1_795_000_000_000_000_000
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compass" / "one_real.py"


def _timed(*spans) -> dict:
    """run.json's entries naming a kernel compiled over each of `spans`,
    (first, last) file written on the step record's clock, timed and with the
    clock pair."""
    compiles = [
        {"compiler": "triton", "name": "_k", "artifact": f"/c/{i}"}
        | {"first_ns": WALL + a, "last_ns": WALL + b}
        for i, (a, b) in enumerate(spans)
    ]
    return {BUILDS: ["_k"], COMPILES: compiles, CLOCK: [WALL, 0]}


def _compiled(run_dir, *spans):
    """run.json holding only `_timed`'s entries."""
    (run_dir / "run.json").write_text(json.dumps(_timed(*spans)))


def _windowed(record):
    """The lines of `record`'s rank 0, each exiting when its device work ends."""
    path = record / "dp0.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for s in lines:
        s["t_exit_ns"] = s["t_enter_ns"] + s["d_busy_ns"]
    path.write_text("".join(json.dumps(s) + "\n" for s in lines))
    return lines


@pytest.mark.parametrize("linked", [False, True], ids=["its path", "by a link"])
def test_the_step_that_waited_on_a_compile_is_cut_and_the_record_costed(
    tmp_path, monkeypatch, linked
):
    """Prefill step 4 ran ten times its price, a compile inside it; the rest
    are priced as the law prices them. One step of twenty is past the bound
    a record of thousands takes. Named by a link from a directory with no
    run.json, the record is cut by the run.json its check read."""
    monkeypatch.delenv(ACCEPT_BUILDS_ENV, raising=False)
    monkeypatch.setattr(step_labels, "JIT_BOUND", 0.1)
    (tmp_path / "run").mkdir()
    record = _record(tmp_path / "run/steps", _steps(), stretch={4: 10.0})
    step = _windowed(record)[4]
    start = step["t_enter_ns"] + JIT_SLACK_NS + 1
    _compiled(tmp_path / "run", (start, start + 80_000_000))
    if linked:
        (tmp_path / "elsewhere").mkdir()
        (tmp_path / "elsewhere/steps").symlink_to(record)
        record = tmp_path / "elsewhere/steps"

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

    assert (
        member(tmp_path / "store", law_key(tmp_path / "store"), "params")["cut"][JIT]
        == 1
    )


@pytest.mark.parametrize(
    "spelling",
    ["{out}/steps/dp0.jsonl", "dp0.jsonl", "{tmp}/elsewhere/link/dp0.jsonl"],
    ids=["its path", "from inside it", "by a link"],
)
def test_step_linear_fit_cuts_by_the_run_json_its_check_read(
    tmp_path, monkeypatch, spelling
):
    """`step_linear_fit` checks the run.json beside the steps directory a
    record's path resolves into; a compile at step 100's entry, timed there,
    cuts that step however the path names the record."""
    monkeypatch.delenv(ACCEPT_BUILDS_ENV, raising=False)
    monkeypatch.setattr(step_labels, "JIT_SLACK_NS", 0)
    steps = tmp_path / "out/steps"
    steps.mkdir(parents=True)
    path = linear._record(steps / "dp0.jsonl", linear._steps(6))
    at = json.loads(path.read_text().splitlines()[100])["t_enter_ns"]
    _compiled(tmp_path / "out", (at, at))
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere/link").symlink_to(steps)
    monkeypatch.chdir(steps)
    record = spelling.format(out=tmp_path / "out", tmp=tmp_path)

    step_linear_fit.main([record, "--out", str(tmp_path / "law.json"), *linear.SPEC])

    assert json.loads((tmp_path / "law.json").read_text())["cut"][JIT] == 1


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


def test_a_compile_between_two_forwards_leaves_the_later_one_no_host_label(
    tmp_path, monkeypatch
):
    """Decode steps 14 and 15 enter one device busy time apart; a compile in
    between overlaps neither's window, so neither is cut, and step 15, whose
    `E` holds it, loses its host labels."""
    monkeypatch.delenv(ACCEPT_BUILDS_ENV, raising=False)
    monkeypatch.setattr(step_labels, "JIT_SLACK_NS", 0)
    record = _record(tmp_path / "steps", _steps()) / "dp0.jsonl"
    lines = [json.loads(x) for x in record.read_text().splitlines()]
    at = (lines[14]["t_exit_ns"] + lines[15]["t_enter_ns"]) // 2
    spans = step_labels.cut_spans(_timed((at, at)))

    samples = {s.step: s for s in step_labels.label(record, spans=spans)[0]}

    assert samples[15].cut is None and samples[15].device_s is not None
    assert samples[15].host == {}
    assert "E" in samples[14].host and "E" in samples[16].host


def test_a_dummy_forward_that_overlaps_a_compile_gives_no_dummy_host(
    tmp_path, monkeypatch
):
    """Steps 3 and 7 are a DP rank's dummy forwards, each running one prefill
    row, as at a step its group prefills; a compile overlaps step 3's."""
    monkeypatch.delenv(ACCEPT_BUILDS_ENV, raising=False)
    monkeypatch.setattr(step_labels, "JIT_SLACK_NS", 0)
    steps = [(None, None) if i in (3, 7) else x for i, x in enumerate(_steps())]
    record = _record(tmp_path / "steps", steps, group=[steps, _steps()]) / "dp0.jsonl"
    lines = [json.loads(x) for x in record.read_text().splitlines()]
    for i in (3, 7):
        lines[i]["rows"] = [[1, 1, False]]
    record.write_text("".join(json.dumps(s) + "\n" for s in lines))
    spans = step_labels.cut_spans(_timed((lines[3]["t_enter_ns"],) * 2))

    samples = step_labels.label(record, spans=spans)[0]

    assert [d[0] for d in samples[0].dummy_host] == [7]


def test_grading_refuses_a_record_whose_timed_compiles_a_fit_takes(
    tmp_path, monkeypatch
):
    """A real run's steps timed 1 ms apart and a compile after them all: the
    fits and `validate.cost` cut nothing and take its record; grading, which
    takes the cell without it, refuses it whole."""
    monkeypatch.delenv(ACCEPT_BUILDS_ENV, raising=False)
    *real, simulated = _four(tmp_path)
    for cell in real:
        _recorded(cell)
    _recorded(simulated, **GRADED)
    grade(real, simulated)
    _recorded(real[1], **_timed((10**15, 10**15)))
    run = json.loads((real[1] / "run.json").read_text())
    path = real[1] / "steps" / "dp0.jsonl"
    lines = [
        json.loads(x) | {"t_enter_ns": i * 10**6, "t_exit_ns": (i + 1) * 10**6 - 1}
        for i, x in enumerate(path.read_text().splitlines())
    ]
    path.write_text("".join(json.dumps(s) + "\n" for s in lines))

    assert step_labels.cut_refusal(run, [lines]) is None
    named = re.escape("its server ran a JIT compile of ['_k'] while it served")
    with pytest.raises(ValueError, match=f"r1 is not graded: {named}, so a step"):
        grade(real, simulated)


def test_one_real_refuses_a_real_run_that_compiled_while_it_served(
    tmp_path, monkeypatch, capsys
):
    """However few steps its compiles cut, as `validate.runs` refuses it: of a
    deployment, naming the side whose server compiled."""
    monkeypatch.delenv(ACCEPT_BUILDS_ENV, raising=False)
    spec = importlib.util.spec_from_file_location("one_real", SCRIPT)
    one_real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(one_real)
    real, sim = tmp_path / "real", tmp_path / "sim"
    for run in (real, sim):
        (run / "artifacts").mkdir(parents=True)
        (run / "artifacts/client_run.json").write_text('{"arguments": {}}')
    for side in ("prefill", "decode"):
        (real / side).mkdir()
        (real / side / "run.json").write_text("{}")
    _compiled(real / "decode", (10**15, 10**15))

    assert one_real.main([str(real), str(sim)]) == 1
    named = "its server ran a JIT compile of ['_k'] while it served"
    assert capsys.readouterr().err == (
        f"{real / 'decode'} is not graded: {named}, so a step may have waited on it\n"
    )

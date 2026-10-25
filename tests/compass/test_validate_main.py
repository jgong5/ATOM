# SPDX-License-Identifier: MIT
"""Grading a cell: `python -m atom.compass.validate` on runs laid out as `cctraces_sim.sh` writes them,
and as `record` and `run` do (`_as_recorded`).

Every run's step record is `test_validate_cost`'s, periods at its planted law's
price; the export and the server log are as `test_validate_runs` and
`test_memory_check` write them. `validate.runs`, `validate.cost` and
`memory.check` read them as they read a real cell.
"""

import json
import re
import runpy
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from test_fit_ops import measured  # noqa: F401, a fixture
from test_memory_check import GIB, _log
from test_memory_readings import DOCUMENT
from test_op_graph import ATTENTION, GEMM, listed, pt
from test_op_graph import g as gemm_leaf
from test_validate_cost import LAW, _record, _steps
from test_validate_runs import ARGUMENTS, EXPORT, _named_as_deployed
from test_work import SPEC

from atom.compass import parity
from atom.compass import run as compass_run
from atom.compass.backends import json_file, step_labels
from atom.compass.backends.cost import GROUPS
from atom.compass.backends.op_graph import OpGraphBackend
from atom.compass.backends.work import gemm, price
from atom.compass.drive import run as drive_run
from atom.compass.memory.check import parts
from atom.compass.run import LP_FILE, OP_GRAPH, PRICE_LIST
from atom.compass.validate.__main__ import main
from atom.compass.validate.runs import CLIENT_RUN, SIDES

TERM = {"method": "measured", "source": "planted"}
SHARE = {"steps": 3, "fraction_of_predicted_seconds": 0.4}


def _run(
    d,
    ttft,
    *,
    refusals=(),
    non_torch=1.0,
    speed=True,
    workload=1,
    stretch=None,
    steps=None,
    idle_after=(),
):
    """One run. `ttft` is (uncached, cached, ms) per profiling request; a warmup
    request with 9000 uncached tokens precedes them. `refusals` None makes the
    run real, a list of (source, count) makes it simulated with those refusals."""
    d.mkdir(parents=True)
    _record(d / "steps", steps or _steps(), stretch=stretch, idle_after=idle_after)
    (d / "artifacts").mkdir()
    export = json.loads(json.dumps(EXPORT))
    export["time_to_first_token"]["avg"] = ttft[0][2]
    (d / "artifacts" / "profile_export_aiperf.json").write_text(json.dumps(export))
    lines = [("warmup", 9000, 0, 1.0)] + [("profiling", *r) for r in ttft]
    (d / "artifacts" / "profile_export.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "metadata": {"benchmark_phase": phase},
                    "metrics": {
                        "time_to_first_token": {"value": ms},
                        "usage_prompt_tokens": {"value": u + c},
                        "usage_prompt_cache_read_tokens": {"value": c},
                    },
                }
            )
            + "\n"
            for phase, u, c, ms in lines
        )
    )
    run = {"workload": {"traces": workload}, "step_linear": LAW}
    if refusals is not None:
        run["host"] = {"laws": {}}
    (d / "run.json").write_text(json.dumps(run))
    if refusals is None:
        _log(d / "server.log", peak=50 * GIB, non_torch=non_torch * GIB)
        return
    readings = {"peak_torch": 50 * GIB, "non_torch": GIB}
    memory = {
        "label": "planted",
        "tokens": 8192,
        "readings": {k: [TERM | {"name": k, "nbytes": v}] for k, v in readings.items()},
        "predicts": None,
    }
    hand_in = {"refusals": [], "priced": {}, "memory": memory}
    (d / LP_FILE.format("engine")).write_text(json.dumps(hand_in))
    summary = {
        "coverage_report": bool(refusals),
        "cost": {
            "speed_ratio": 20.0 if speed else None,
            "speed_refused": None if speed else "no wall time was measured",
            "simulated_seconds": 900.0,
            "wall_seconds": 45.0 if speed else 0.0,
            "meets_speed_target": True if speed else None,
            "speed_target_ratio": 5.0,
        },
        "schedule": {
            "refusals": {
                "count": sum(n for _, n in refusals),
                "by_source": [list(r) for r in refusals],
                "steps": 20,
                "fraction_of_steps": dict(refusals).get("cost", 0) / 20,
            },
            "seconds_by_method": {g: SHARE for g in GROUPS},
        },
    }
    (d / "summary.json").write_text(json.dumps(summary))


def _rows(capsys, tmp_path, law=(), **law_run):
    """The cell's rows; real0's step 5 ran ten times its price, `law` is the
    simulated run's refusals and `law_run` sets its other arguments."""
    for i, (ms, non_torch) in enumerate(((100, 1.0), (110, 1.5), (105, 1.0))):
        _run(
            tmp_path / "c4" / f"real{i}",
            [(100, 1000, ms), (600, 0, 1.0)],
            refusals=None,
            non_torch=non_torch,
            stretch={5: 10.0} if i == 0 else None,
        )
    _run(
        tmp_path / "c4" / "step_linear",
        [(100, 1000, 120.0)],
        refusals=law,
        **law_run,
    )
    main([str(tmp_path / "c4")])
    out = capsys.readouterr().out.splitlines()
    return {tuple(r.split(" | ")[1:3]): r.split(" | ")[3:] for r in out[1:]}


def test_a_cell_is_graded_from_every_source_and_its_method_shares_grade_nothing(
    capsys, tmp_path
):
    rows = _rows(capsys, tmp_path)
    assert rows["cell", "verdict"] == ["3 real runs", "graded"]
    # Uncached tokens only, graded against the real spread of 10%.
    assert rows["end to end", "ttft p50 0-512 uncached"] == [
        "spread 0.1; result 0.2 0.09091 0.1429",
        "outside",
    ]
    assert rows["end to end", "ttft p50 512-2048 uncached"][0] == (
        "no request in ['step_linear']"
    )
    # The warmup request is in no band.
    assert rows["end to end", "ttft p50 8192+ uncached"][0].startswith("no request")
    assert rows["end to end", "ttft mean"][1] == "outside"
    assert rows["cost real0", "decode rung 1 validated"][0].endswith(
        "within 10% 100.0%"
    )
    # Every step the cost rows leave out is named, a warm one with its real seconds.
    assert rows["cost real0", "left out"][0] == (
        'not labelled {"exposed": 0, "mixed": 0, "unsplit": 0, "paced": 0}; warm steps, real s {5: 0.81}'
    )
    assert rows["cost real0", "prefill validated"][0].startswith("9 steps")
    assert rows["cost real1", "left out"][0].endswith("warm steps, real s {}")
    assert rows["memory real0", "non_torch"][1] == "pass"
    assert rows["memory real1", "non_torch"] == ["-33.33% (-536,870,912 B)", "fail"]
    for g in GROUPS:
        assert rows["summary step_linear", g] == [json.dumps(SHARE), "not gated"]
    assert rows["summary step_linear", "speed"] == [
        "20.0x: 900 simulated s in 45 wall s",
        "meets 5x",
    ]
    assert rows["summary step_linear", "refusals"] == [
        "0 {}; cost refused 0.0% of 20 steps",
        "none",
    ]


def test_a_clock_refusal_makes_the_cell_a_coverage_report(capsys, tmp_path):
    # A clock refusal refuses no step; it is counted under its source, not as a step.
    rows = _rows(capsys, tmp_path, law=[("clock", 1)], speed=False)
    assert rows["cell", "verdict"] == ["3 real runs", "coverage report"]
    assert rows["summary step_linear", "refusals"] == [
        '1 {"clock": 1}; cost refused 0.0% of 20 steps',
        "coverage report",
    ]
    assert rows["summary step_linear", "speed"] == [
        "no wall time was measured: 900 simulated s in 0 wall s",
        "no result for 5x",
    ]


def test_runs_of_different_workloads_are_not_one_cell(capsys, tmp_path):
    with pytest.raises(ValueError, match="ran different workloads"):
        _rows(capsys, tmp_path, workload=2)


def test_a_cost_refusal_makes_the_cell_a_coverage_report(capsys, tmp_path):
    rows = _rows(capsys, tmp_path, law=[("cost", 2)])
    assert rows["cell", "verdict"] == ["3 real runs", "coverage report"]
    assert rows["summary step_linear", "refusals"] == [
        '2 {"cost": 2}; cost refused 10.0% of 20 steps',
        "coverage report",
    ]


def test_counts_but_preemptions_and_prefill_tokens_are_not_gated(capsys, tmp_path):
    # A decode step of another request splits the law run's prefill chunks: one
    # prefill streak and one decode token more than every real run.
    steps = _steps()
    steps.insert(5, ([(1, 2000, True)], 1))
    rows = _rows(capsys, tmp_path, steps=steps, idle_after={4, 5})
    assert rows["counts", "prefill streaks"] == [
        "spread 0; result 1 1 1",
        "outside, not gated",
    ]
    assert rows["counts", "decode tokens"][1] == "outside, not gated"
    assert rows["counts", "prefill tokens"][1] == "exact"
    assert rows["counts", "preemptions"][1] == "exact"


def test_a_run_that_ended_without_its_memory_reads_refused_and_the_cell_goes_on(
    capsys, tmp_path
):
    whole = _rows(capsys, tmp_path / "whole")
    _rows(capsys, tmp_path, law=[("ended", 1)])
    law = tmp_path / "c4" / "step_linear"
    ended = {"refusals": ["ended:engine raised RuntimeError('the step raised')"]}
    (law / LP_FILE.format("engine")).write_text(json.dumps(ended))
    summary = json.loads((law / "summary.json").read_text())
    summary["schedule"].update(seconds_by_method=None)
    summary["schedule"]["refusals"].update(steps=None, fraction_of_steps=None)
    (law / "summary.json").write_text(json.dumps(summary))

    main([str(tmp_path / "c4")])
    out = capsys.readouterr().out.splitlines()
    rows = {tuple(r.split(" | ")[1:3]): r.split(" | ")[3:] for r in out[1:]}

    for real in ("real0", "real1", "real2"):
        assert rows[f"memory {real}", "memory"] == [
            f"{law / LP_FILE.format('engine')} has no memory section",
            "refused",
        ]
    memory = {k for k in whole if k[0].startswith("memory ")}
    assert set(rows) - {k for k in rows if k[1] == "memory"} == set(whole) - memory
    assert rows["summary step_linear", "unvalidated"] == ["null", "not gated"]
    assert rows["summary step_linear", "refusals"] == [
        '1 {"ended": 1}; steps not counted',
        "coverage report",
    ]
    assert rows["summary step_linear", "speed"][1] == "meets 5x"


def test_a_cell_of_renamed_names_is_refused_naming_their_names_now(capsys, tmp_path):
    _rows(capsys, tmp_path)
    cell = tmp_path / "c4"
    run = cell / "real0" / "run.json"
    run.write_text(json.dumps(json.loads(run.read_text()) | {"law": LAW}))
    with pytest.raises(ValueError, match=r"renamed: law \(now step_linear\)"):
        main([str(cell)])
    (cell / "step_linear").rename(cell / "law")
    with pytest.raises(ValueError, match="now named step_linear/"):
        main([str(cell)])


def test_a_cell_of_one_real_run_refuses_naming_it_and_the_count(tmp_path):
    cell = tmp_path / "c4"
    _run(cell / "real0", [(100, 1000, 100)], refusals=None)
    _run(cell / "step_linear", [(100, 1000, 120.0)])

    validate = [sys.executable, "-m", "atom.compass.validate", str(cell)]
    done = subprocess.run(validate, capture_output=True, text=True, check=False)

    refusal = f"{cell} holds 1 of real*/; a cell holds two or more real runs\n"
    assert (done.returncode, done.stderr) == (1, refusal)


@pytest.mark.parametrize("bare", ["real1", "step_linear"])
def test_a_run_without_run_json_refuses_naming_it(tmp_path, bare):
    cell = tmp_path / "c4"
    for name in ("real0", "real1"):
        _run(cell / name, [(100, 1000, 100)], refusals=None)
    _run(cell / "step_linear", [(100, 1000, 120.0)])
    # A run `record` or `run` writes holds none until it stops.
    (cell / bare / "run.json").unlink()

    validate = [sys.executable, "-m", "atom.compass.validate", str(cell)]
    done = subprocess.run(validate, capture_output=True, text=True, check=False)

    refusal = f"{cell}'s ['{bare}'] hold no run.json; every run of a cell holds one\n"
    assert (done.returncode, done.stderr) == (1, refusal)


def _cell(cell):
    """`cell`, of real0/, real1/ and step_linear/."""
    for name in ("real0", "real1"):
        _run(cell / name, [(100, 1000, 100)], refusals=None)
    _run(cell / "step_linear", [(100, 1000, 120.0)])
    return cell


AIPERF = "artifacts/profile_export_aiperf.json"
NO_EXPORT = f"hold no {AIPERF}; every run of a cell holds one"
NO_SUMMARY = "summary.json or client-0/summary.json; a simulated run holds one"


@pytest.mark.parametrize(
    "missing, refusal",
    [
        (f"real1/{AIPERF}", f"['real1'] {NO_EXPORT}"),
        (f"step_linear/{AIPERF}", f"['step_linear'] {NO_EXPORT}"),
        ("step_linear/summary.json", f"step_linear holds no run summary, {NO_SUMMARY}"),
    ],
    ids=["real1 export", "step_linear export", "step_linear summary"],
)
def test_a_run_still_recording_refuses_naming_it(tmp_path, missing, refusal):
    """`cctraces_sim.sh` writes run.json before its server starts, and the
    export and run summary once its client and server have ended."""
    cell = _cell(tmp_path / "c4")
    (cell / missing).unlink()

    validate = [sys.executable, "-m", "atom.compass.validate", str(cell)]
    done = subprocess.run(validate, capture_output=True, text=True, check=False)

    assert (done.returncode, done.stderr) == (1, f"{cell}'s {refusal}\n")


def test_a_malformed_run_json_refuses_naming_it(tmp_path):
    cell = _cell(tmp_path / "c4")
    (cell / "real1/run.json").write_text("{")

    validate = [sys.executable, "-m", "atom.compass.validate", str(cell)]
    done = subprocess.run(validate, capture_output=True, text=True, check=False)

    what = "Expecting property name enclosed in double quotes: line 1 column 2"
    refusal = f"{cell}/real1/run.json is not JSON: {what}\n"
    assert (done.returncode, done.stderr) == (1, refusal)


@pytest.mark.parametrize(
    "command, runs, name",
    [
        ("atom.compass.validate", ["c4"], "real1/run.json"),
        ("atom.compass.validate", ["c4"], "real1/steps/dp0.jsonl"),
        (
            "atom.compass.validate.runs",
            ["c4/step_linear", "c4/real0", "c4/real1"],
            "real1/run.json",
        ),
    ],
)
def test_an_input_that_is_not_utf8_refuses_naming_it(tmp_path, command, runs, name):
    path = _cell(tmp_path / "c4") / name
    path.write_bytes(b"\xff" + path.read_bytes())

    argv = [sys.executable, "-m", command, *(str(tmp_path / r) for r in runs)]
    done = subprocess.run(argv, capture_output=True, text=True, check=False)

    what = "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte"
    assert (done.returncode, done.stderr) == (1, f"{path} is not JSON: {what}\n")


@pytest.mark.parametrize(
    "malformed",
    [
        "real1/steps/dp0.jsonl",
        "real1/artifacts/profile_export.jsonl",
        "real1/artifacts/profile_export_aiperf.json",
        "step_linear/summary.json",
        f"step_linear/{LP_FILE.format('engine')}",
    ],
)
def test_a_malformed_input_refuses_naming_its_path(tmp_path, malformed):
    path = _cell(tmp_path / "c4") / malformed
    path.write_text(path.read_text()[:-3])  # its last line cut

    with pytest.raises(ValueError, match=f"^{re.escape(str(path))} is not JSON: "):
        main([str(tmp_path / "c4")])


@pytest.mark.parametrize(
    "reader",
    [step_labels.label, lambda p: parity.read(p.parent), lambda p: parts(p, p)],
)
def test_the_readers_it_reaches_refuse_a_malformed_file_naming_it(tmp_path, reader):
    path = tmp_path / "dp0.jsonl"
    path.write_text('{"step": 0}\n{"step"\n')

    named = f"^{re.escape(str(path))} is not JSON: .*: line 2 "
    with pytest.raises(ValueError, match=named):
        reader(path)


def test_a_jsonl_string_holding_a_unicode_line_separator_is_read_whole(tmp_path):
    # aiperf's orjson writes U+2028, U+2029 and U+0085 inside a string unescaped.
    path = tmp_path / "profile_export.jsonl"
    path.write_text('{"error": "x\u2028y\u2029z\x85"}\n{}\n')

    assert json_file.read_jsonl(path) == [{"error": "x\u2028y\u2029z\x85"}, {}]


@pytest.mark.parametrize("line", [b"\x001", b"\x00" * 6])
def test_a_jsonl_line_led_by_a_nul_byte_refuses_naming_its_path(tmp_path, line):
    # json.loads on bytes would read a NUL-led line as UTF-16 or UTF-32.
    path = tmp_path / "profile_export.jsonl"
    path.write_bytes(b'{"a": 1}\n' + line + b"\n")

    named = f"^{re.escape(str(path))} is not JSON: .*: line 2 "
    with pytest.raises(ValueError, match=named):
        json_file.read_jsonl(path)


@pytest.mark.parametrize(
    "command, runs",
    [
        ("atom.compass.validate.__main__", ["c4"]),
        ("atom.compass.validate.runs", ["c4/step_linear", "c4/real0", "c4/real1"]),
    ],
)
def test_a_json_error_no_reader_names_keeps_its_traceback(
    tmp_path, monkeypatch, command, runs
):
    _cell(tmp_path / "c4")

    def read_json(path):
        raise json.JSONDecodeError("planted", "{", 1)

    monkeypatch.setattr(json_file, "read_json", read_json)
    monkeypatch.delitem(sys.modules, command)  # run fresh, on the planted reader
    monkeypatch.setattr(sys, "argv", [command, *(str(tmp_path / r) for r in runs)])
    with pytest.raises(json.JSONDecodeError, match="^planted"):
        runpy.run_module(command, run_name="__main__")


def _as_recorded(cell, warmup=()):
    """Lay `cell`'s runs out as `record` and `run` write them: each request
    named at its last prefill row, by key `k<id>` and response id `id-<id>`;
    an export naming them all, from trace `t<id>`, those in `warmup` as its
    warmup's, else request `w`, a one-step prefill before every other step,
    numbered below them, and the same traces and arguments given to every run's
    client; and run.json's identity,
    with `run`'s grading entries for `step_linear/`, its law and host laws in
    its run file and its refusals its summary's."""
    for d in sorted(cell.iterdir()):
        path = d / "steps" / "dp0.jsonl"
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        last = {}
        for i, line in enumerate(lines):
            for (r, *_), (_q, _c, decode) in zip(line["batch"], line["rows"]):
                last[r] = i if not decode else last.get(r, i)
        for r, i in last.items():
            lines[i]["named"].append([r, f"k{r}", f"id-{r}"])
        if not warmup:
            r, step = max(last) + 1, lines[0]["step"] - 1
            w = {"step": step, "batch": [[r, 1, 1]], "named": [[r, "kw", "id-w"]]}
            lines.insert(0, lines[0] | w | {"rows": [[1, 1, False]], "rung": None})
        path.write_text("".join(json.dumps(x) + "\n" for x in lines))
        export = {"requests": {}, "warmup": {}, "traces": ["t0", "t1", "tw"]}
        export["arguments"] = ARGUMENTS
        for r in last:
            part = "warmup" if r in warmup else "requests"
            export[part][f"id-{r}"] = {"trace": f"t{r}", "sampling": {}}
        if not export["warmup"]:
            export["warmup"]["id-w"] = {"trace": "tw", "sampling": {}}
        (d / CLIENT_RUN).write_text(json.dumps(export))
        run = {"format_version": 2}
        if d.name == "step_linear":
            summary = json.loads((d / "summary.json").read_text())
            run |= {
                "run_file": {"step_linear": LAW, "host": {"laws": {}}},
                "step_roofline": False,
                "skip": [],
                "refusals": summary["schedule"]["refusals"] | {"reasons": []},
                "coverage": None,
                "provenance": {"digest": "sha256:0", "provenance": None},
            }
        (d / "run.json").write_text(json.dumps(run))


def _graded(capsys, cell):
    main([str(cell)])
    out = capsys.readouterr().out.splitlines()
    return {tuple(r.split(" | ")[1:3]): r.split(" | ")[3:] for r in out[1:]}


def test_a_cell_record_and_run_wrote_grades_as_the_one_cctraces_sim_wrote(
    capsys, tmp_path
):
    rows = _rows(capsys, tmp_path)
    _as_recorded(tmp_path / "c4")

    graded = _graded(capsys, tmp_path / "c4")
    stated = [graded.pop(k) for k in list(graded) if k[0].startswith("warmup ")]
    assert graded == rows
    # Each run's warmup is request `w`'s step; a `cctraces_sim.sh` run reads none.
    assert stated == [["requests 1, steps 1", "not graded"]] * 4


def test_a_refusal_in_the_client_run_record_and_run_wrote_refuses_the_cell(
    capsys, tmp_path
):
    _rows(capsys, tmp_path, law=[("clock", 1)])
    _as_recorded(tmp_path / "c4")

    with pytest.raises(ValueError, match="step_linear is not graded: it refused 1"):
        main([str(tmp_path / "c4")])


def test_a_later_client_runs_refusal_leaves_a_run_written_cell_graded(capsys, tmp_path):
    _rows(capsys, tmp_path)
    _as_recorded(tmp_path / "c4")
    # Client run 0 refused nothing; a later one on its server refused once.
    law = tmp_path / "c4" / "step_linear"
    (law / "client-0").mkdir()
    (law / "summary.json").rename(law / "client-0" / "summary.json")
    summary = json.loads((law / "client-0" / "summary.json").read_text())
    summary["coverage_report"] = True
    summary["schedule"]["refusals"].update(count=1, by_source=[["clock", 1]])
    (law / "summary.json").write_text(json.dumps(summary))

    rows = _graded(capsys, tmp_path / "c4")

    assert rows["cell", "verdict"] == ["3 real runs", "graded"]
    assert rows["summary step_linear", "refusals"] == [
        "0 {}; cost refused 0.0% of 20 steps",
        "none",
    ]


def test_runs_record_and_run_wrote_that_were_given_other_traces_are_not_one_cell(
    capsys, tmp_path
):
    _rows(capsys, tmp_path)
    _as_recorded(tmp_path / "c4")
    export = tmp_path / "c4" / "real1" / CLIENT_RUN
    other = {"traces": ["tx"]}
    export.write_text(json.dumps(json.loads(export.read_text()) | other))

    with pytest.raises(ValueError, match=r"given different traces; .*real1': \['tx'\]"):
        main([str(tmp_path / "c4")])


def test_runs_record_and_run_wrote_at_another_concurrency_are_not_one_cell(
    capsys, tmp_path
):
    _rows(capsys, tmp_path)
    _as_recorded(tmp_path / "c4")
    export = tmp_path / "c4" / "real1" / CLIENT_RUN
    other = json.loads(export.read_text())
    other["arguments"] = ARGUMENTS | {"phases.profiling.concurrency": 16}
    export.write_text(json.dumps(other))

    concurrency = r"\{'phases.profiling.concurrency': \{.*real1': 16, .*\}\}$"
    with pytest.raises(ValueError, match=f"given different arguments: {concurrency}"):
        main([str(tmp_path / "c4")])


#: A workload as `cctraces_sim.sh` and `client.sh` state it.
WORKLOAD = {"aiperf_args": ["--concurrency", "4"], "traces_sha256": ["d0", "d1"]}


def _mixed(capsys, tmp_path, law=()):
    """Cell c4 of `cctraces_sim.sh` real runs and a simulated run `run` wrote,
    which `client.sh` drove, every run given `WORKLOAD`."""
    _rows(capsys, tmp_path, law=law)
    cell, sim = tmp_path / "c4", tmp_path / "sim"
    for d in cell.glob("real*"):
        run = json.loads((d / "run.json").read_text())
        (d / "run.json").write_text(json.dumps(run | {"workload": WORKLOAD}))
    sim.mkdir()
    shutil.move(cell / "step_linear", sim)
    _as_recorded(sim)
    shutil.move(sim / "step_linear", cell)
    (cell / "step_linear" / "artifacts/workload.json").write_text(json.dumps(WORKLOAD))
    return cell


def test_a_run_written_run_grades_whole_beside_cctraces_sim_s_real_runs(
    capsys, tmp_path
):
    cell = _mixed(capsys, tmp_path)
    rows = _graded(capsys, cell)

    # As the run would grade, written by cctraces_sim.sh: its warmup's step too.
    sim = cell / "step_linear"
    (sim / CLIENT_RUN).unlink()
    run = {"workload": WORKLOAD, "step_linear": LAW, "host": {"laws": {}}}
    (sim / "run.json").write_text(json.dumps(run))
    assert rows == _graded(capsys, cell)
    assert rows["counts", "prefill steps"][0].startswith("spread 0; result 1 1 1")
    assert not [k for k in rows if k[0].startswith("warmup ")]


@pytest.mark.parametrize(
    "field, value, named",
    [
        ("aiperf_args", ["--concurrency", "16"], r"\['--concurrency', '16'\]"),
        (
            "traces_sha256",
            ["d0", "d9"],
            "'real0': 1, 'real1': 1, 'real2': 1, 'step_linear': 1",
        ),
    ],
)
def test_a_run_written_run_given_another_workload_refuses_naming_it(
    capsys, tmp_path, field, value, named
):
    cell = _mixed(capsys, tmp_path)
    stated = WORKLOAD | {field: value}
    (cell / "step_linear/artifacts/workload.json").write_text(json.dumps(stated))

    with pytest.raises(
        ValueError, match=f"ran different workloads: {{'{field}': .*{named}"
    ):
        main([str(cell)])


def test_a_run_written_run_graded_whole_still_refuses_as_it_would_apart(
    capsys, tmp_path
):
    cell = _mixed(capsys, tmp_path, law=[("clock", 1)])
    with pytest.raises(ValueError, match="step_linear is not graded: it refused 1"):
        main([str(cell)])


def test_a_run_written_run_that_served_a_later_client_run_does_not_grade_whole(
    capsys, tmp_path
):
    cell = _mixed(capsys, tmp_path)
    path = cell / "step_linear/steps/dp0.jsonl"
    lines = path.read_text().splitlines()
    later = json.loads(lines[-1]) | {"step": 99, "batch": [[99, 1, 1]], "named": []}
    path.write_text("\n".join([*lines, json.dumps(later)]) + "\n")

    later = "steps of requests it did not send follow its own, 1 from step 99"
    with pytest.raises(ValueError, match=f"step_linear is not graded: {later}"):
        main([str(cell)])


def _warmed(capsys, cell, slow):
    """A cell `record` and `run` wrote, request 0 every run's warmup: two steps
    of its own before request 1's, `slow` times their price in every real run."""
    steps = [([(200, 200, False)], None), ([(1, 200, True)], 1), *_steps()]
    for i, ms in enumerate((100, 110, 105)):
        _run(
            cell / f"real{i}",
            [(100, 1000, ms), (600, 0, 1.0)],
            refusals=None,
            steps=steps,
            idle_after={1},
            stretch={0: slow, 1: slow},
        )
    _run(cell / "step_linear", [(100, 1000, 120.0)], steps=steps, idle_after={1})
    _as_recorded(cell, warmup={0})
    return _graded(capsys, cell)


def test_a_warmed_run_is_graded_on_its_profiling_steps_and_slow_warmup_moves_nothing(
    capsys, tmp_path
):
    rows = _warmed(capsys, tmp_path / "at price", 1.0)

    assert _warmed(capsys, tmp_path / "slow", 1000.0) == rows
    for run in ("real0", "real1", "real2", "step_linear"):
        assert rows[f"warmup {run}", "warmup"] == ["requests 1, steps 2", "not graded"]
    # Cost prices the profiling steps alone: ten prefill chunks, ten decode steps.
    assert rows["cost real0", "prefill validated"][0].startswith("10 steps")
    assert rows["cost real0", "decode validated"][0].startswith("10 steps")
    assert rows["cost real0", "left out"][0].endswith("warm steps, real s {}")


def test_a_run_file_of_a_law_per_engine_config_costs_real_runs_at_their_own(
    capsys, tmp_path
):
    cell = tmp_path / "c4"
    _rows(capsys, tmp_path)
    _as_recorded(cell)
    rows = _graded(capsys, cell)
    ours, theirs = {"enable_tbo": False}, {"enable_tbo": True}
    other = json.loads(json.dumps(LAW))
    other["prefill"]["coefficients"]["step"] *= 2
    run = json.loads((cell / "step_linear" / "run.json").read_text())
    laws = [other | {"engine_config": theirs}, LAW | {"engine_config": ours}]
    run["run_file"]["step_linear"] = laws
    (cell / "step_linear" / "run.json").write_text(json.dumps(run))
    for real in ("real0", "real1", "real2"):
        engine = ours | {"num_kvcache_blocks": 8, "max_num_seqs": 4}
        (cell / real / "steps" / "dp0.engine.json").write_text(json.dumps(engine))

    assert _graded(capsys, cell) == rows


def test_the_steps_extrapolated_or_priced_by_a_fallback_are_stated_and_graded(
    capsys, tmp_path
):
    _rows(capsys, tmp_path)
    law = tmp_path / "c4" / "step_linear"
    summary = json.loads((law / "summary.json").read_text())
    summary["schedule"].update(extrapolated_steps=3, fallback_steps=2)
    (law / "summary.json").write_text(json.dumps(summary))

    main([str(tmp_path / "c4")])
    out = capsys.readouterr().out.splitlines()
    rows = {tuple(r.split(" | ")[1:3]): r.split(" | ")[3:] for r in out[1:]}

    assert rows["cell", "verdict"] == ["3 real runs", "graded"]
    assert rows["summary step_linear", "extrapolated steps"] == ["3 of 20", "not gated"]
    assert rows["summary step_linear", "fallback steps"] == ["2 of 20", "not gated"]


def test_a_refused_cost_row_names_its_reasons(capsys, tmp_path):
    # A law with no decode vector refuses each real run's ten decode steps.
    _rows(capsys, tmp_path)
    law = tmp_path / "c4" / "step_linear" / "run.json"
    run = json.loads(law.read_text())
    law.write_text(json.dumps(run | {"step_linear": {**LAW, "decode": {}}}))

    main([str(tmp_path / "c4")])
    out = capsys.readouterr().out.splitlines()
    rows = {tuple(r.split(" | ")[1:3]): r.split(" | ")[3:] for r in out[1:]}

    for real in ("real0", "real1", "real2"):
        value, verdict = rows[f"cost {real}", "refused decode rung 1"]
        assert value.endswith(
            '; reasons {"step-linear: no decode vector for rung 1": 10}'
        )
        assert value.startswith("10 steps, real ") and verdict == "-"


def _deploy(cell):
    """Lay each run of `cell` out as `deploy.sh` writes a deployment, both its
    sides that run's server: its steps, run.json and server log in prefill/
    and decode/, each side under the engine config one of the run file's two
    laws was fitted under; its client's export beside them; and a simulated
    run's engine hand-in as each side's; their requests named as
    `_named_as_deployed` names them."""
    laws = [LAW | {"engine_config": {"enable_tbo": tbo}} for tbo in (True, False)]
    for d in sorted(cell.iterdir()):
        for side, tbo in zip(SIDES, (True, False)):
            shutil.copytree(d / "steps", d / side / "steps")
            for name in ("run.json", "server.log"):
                if (d / name).exists():
                    shutil.copy(d / name, d / side / name)
            engine = {"enable_tbo": tbo, "num_kvcache_blocks": 8, "max_num_seqs": 4}
            (d / side / "steps" / "dp0.engine.json").write_text(json.dumps(engine))
            hand_in = d / LP_FILE.format("engine")
            if hand_in.exists():
                shutil.copy(hand_in, d / LP_FILE.format(f"engine-{side[0].upper()}"))
        _named_as_deployed(d)
        shutil.rmtree(d / "steps")
        run = {"kv_link": "inter_node"}
        if d.name == "step_linear":
            run["run_file"] = {"step_linear": laws, "host": {"laws": {}}}
            (d / LP_FILE.format("engine")).unlink()
        else:
            (d / "server.log").unlink()
        (d / "run.json").write_text(json.dumps(run))


def test_a_deployment_cell_is_graded_per_side_as_each_side_alone_would_be(
    capsys, tmp_path
):
    cell = tmp_path / "c4"
    _rows(capsys, tmp_path)
    _as_recorded(cell)
    alone = _graded(capsys, cell)
    _deploy(cell)

    rows = _graded(capsys, cell)

    per_side = ("counts", "distributions", "one run", "cost ", "memory ", "warmup ")
    expected = {}
    for (result, metric), value in alone.items():
        if not result.startswith(per_side):
            expected[result, metric] = value
            continue
        for side in SIDES:
            expected[f"{result} {side}", metric] = value
    assert rows == expected
    assert rows["memory real1 decode", "non_torch"] == [
        "-33.33% (-536,870,912 B)",
        "fail",
    ]


def test_a_deployment_at_two_dp_ranks_checks_each_ranks_memory(capsys, tmp_path):
    cell = tmp_path / "c4"
    _rows(capsys, tmp_path)
    _as_recorded(cell)
    _deploy(cell)
    one = _graded(capsys, cell)
    # Each simulated rank hands in as <engine>.dp<r>; each real rank logs its lines.
    for engine in sorted(cell.glob(f"step_linear/{LP_FILE.format('engine-*')}")):
        for r in (0, 1):
            shutil.copy(engine, engine.with_suffix(f".dp{r}.json"))
        engine.unlink()
    for log in cell.glob("real*/*/server.log"):
        log.write_text(log.read_text() * 2)

    rows = _graded(capsys, cell)

    expected = {}
    for (result, metric), value in one.items():
        ranks = (" dp0", " dp1") if result.startswith("memory ") else ("",)
        expected |= {(result + r, metric): value for r in ranks}
    assert rows == expected
    assert rows["memory real1 decode dp1", "non_torch"][1] == "fail"


def test_an_op_graph_cell_grades_with_its_method_shares_term_gaps_and_warm_over_cold(
    capsys, tmp_path, monkeypatch, measured  # noqa: F811
):
    cell = tmp_path / "c4"
    _rows(capsys, tmp_path)
    _as_recorded(cell)
    sim = (cell / "step_linear").rename(cell / OP_GRAPH)
    # Two GEMM points timed warm, an attention point not; a point measured
    # under another source root is another price list's.
    points = [(GEMM, 0.5), (GEMM, 0.7), (ATTENTION, None)]
    records = [
        {"shape": "decode B=1", "context": i, "leaf": "x", "op": op}
        | {"warm_over_cold": w}
        for i, (op, w) in enumerate(points)
    ]
    _, key = measured("ops", records)
    measured("other", [records[0] | {"warm_over_cold": 9.0}], root="b" * 40)
    fields = {k: v for k, v in key("s", 0, "x").values.items() if k != "point"}
    run = json.loads((sim / "run.json").read_text())
    run["argv"] = ["-c", "run", "--op-graph", "--out", str(sim), "--", "--model", "m"]
    run["run_file"] = {
        OP_GRAPH: True,
        PRICE_LIST: fields,
        "store": str(tmp_path / "store"),
        "machine": DOCUMENT,
        "host": {"laws": {}},
    }
    (sim / "run.json").write_text(json.dumps(run))
    # Decode's small GEMM is listed at its measured 20 ms, a large one is its
    # roofline, about 30 ms; prefill of 190 tokens has no graph.
    large = gemm_leaf(4096, 4096, 4096)
    count = int(0.03 / price(gemm(4096, 4096, 4096), SPEC)) + 1
    region = {
        "seq": [
            gemm_leaf(1, 64, 64),
            {"repeat": large, "count": count, "index": "l0", "start": 0, "step": 1},
        ]
    }
    prices = listed(
        *[pt(gemm_leaf(1, 64, 64), "decode B=1", 20e6, c) for c in (100, 1000)]
    )
    built = []

    def op_graph(run_file, config, machine, ep):
        built.append((run_file, config.atom_args, ep))
        return OpGraphBackend(
            SPEC,
            lambda phase, tokens, *_: None if tokens == 190 else region,
            kv="bf16",
            price_list=prices,
        )

    monkeypatch.setattr(
        drive_run,
        "_config",
        lambda args: SimpleNamespace(atom_args=args, enable_expert_parallel=False),
    )
    monkeypatch.setattr(compass_run, "op_graph", op_graph)

    rows = _graded(capsys, cell)

    assert built == [(run["run_file"], ["--model", "m"], 1)]
    assert rows["cell", "verdict"] == ["3 real runs", "graded"]
    analytical = price(gemm(4096, 4096, 4096), SPEC) * count
    for real in ("real0", "real1", "real2"):
        where = f"cost {real}"
        decode, gated = rows[where, "decode by method"]
        share = 0.02 / (0.02 + analytical)
        assert decode.endswith(f"analytical {1 - share:.1%}, measured {share:.1%}")
        assert gated == "not gated"
        assert rows[where, "prefill by method"][0].endswith(": analytical 100.0%")
        # One op kind, its listed and roofline terms together: the composed step.
        on_totals, per_step = rows[where, f"decode {GEMM} over measured"][0].split(";")
        error = float(rows[where, "decode validated"][0].split()[10].rstrip("%,"))
        assert float(on_totals.split()[-1]) == pytest.approx(1 + error / 100, abs=1e-3)
        # Each decode step's composed seconds over its own measured ones.
        real_s = [0.02 + 1e-7 * (2000 + 5000 * i) for i in range(10)]
        ratios = [(0.02 + analytical) / s for s in real_s]
        expected = np.percentile(ratios, (10, 50, 90))
        assert [float(x) for x in per_step.split()[3::2]] == pytest.approx(
            expected, abs=1e-3
        )
        assert rows[where, "decode validated"][0].startswith("10 steps")
        reason = "op-graph: no op_graph holds a prefill step of 190 tokens over 1 sequences gathering 190 rows; compass trace ops captures one"
        value = rows[where, "refused prefill tokens 0-512"][0]
        assert value.startswith("1 steps") and value.endswith(json.dumps({reason: 1}))
    assert rows["warm over cold", GEMM] == [
        "2 of 2 points timed warm; min 0.500, median 0.600, max 0.700",
        "not gated",
    ]
    assert rows["warm over cold", ATTENTION][0] == "0 of 1 points timed warm"
    # A run file that composes the step roofline below the graph falls back.
    run["run_file"]["step_roofline"] = True
    (sim / "run.json").write_text(json.dumps(run))
    rows = _graded(capsys, cell)
    assert rows["cost real0", "fallback prefill tokens 0-512"][0] == value
    assert ("cost real0", "refused prefill tokens 0-512") not in rows


def test_cost_rows_gate_every_graded_step_and_print_the_breakdown_not_gated(
    capsys, tmp_path
):
    # A held-out hull to context 10000 leaves decode steps 2 to 9 unvalidated.
    _rows(capsys, tmp_path)
    law = json.loads(json.dumps(LAW))
    law["decode"]["1"]["held_out"]["hull"]["facets"] = [[-1, 100], [1, -10000]]
    sim = tmp_path / "c4" / "step_linear" / "run.json"
    sim.write_text(json.dumps(json.loads(sim.read_text()) | {"step_linear": law}))

    main([str(tmp_path / "c4")])
    out = capsys.readouterr().out.splitlines()
    rows = {tuple(r.split(" | ")[1:3]): r.split(" | ")[3:] for r in out[1:]}

    for metric, steps, verdict in (
        ("graded", 10, "-"),
        ("validated", 2, "not gated"),
        ("unvalidated", 8, "not gated"),
    ):
        value, gated = rows["cost real1", f"decode {metric}"]
        assert value.startswith(f"{steps} steps") and gated == verdict
        assert rows["cost real1", f"decode rung 1 {metric}"][1] == verdict

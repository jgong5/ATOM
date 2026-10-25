# SPDX-License-Identifier: MIT
"""Run grades: cells laid out as `cctraces_sim.sh` writes them, and as `record`
and `run` do.

Each cell is a step record in `steps/dp0.jsonl`, the JSON lines a runner
writes, read by `runs._steps`, and aiperf's export in
`artifacts/profile_export_aiperf.json`. `_recorded` adds what a `record` or
`run` directory holds beside them: run.json's identity and grading entries,
and the client run's export. `_ran` has `compass run` write a simulated cell's
grading entries, over the summaries a server that served its client runs
wrote (`test_client_run_summary._serve`).
"""

import collections
import json
import os
import re
import shutil
import subprocess
import sys
import time

import pytest
from test_client_run_summary import CLIENT_RUNS, _serve
from test_stop_between_client_runs import CLIENT_RUN as UNREFUSED

from atom.compass import run as compass_run
from atom.compass.drive import aiter_jit, serve
from atom.compass.drive import run as drive_run
from atom.compass.parity import ARRIVALS, TRANSFERS
from atom.compass.validate.runs import (
    ASSUMED,
    CLIENT_RUN,
    SIDES,
    _requests,
    grade,
    main,
)

EXPORT = {
    "request_count": {"unit": "requests", "avg": 3.0},
    "time_to_first_token": {"unit": "ms", "avg": 1000.0, "p50": 900.0},
    "inter_token_latency": {"unit": "ms", "avg": 25.0, "p50": 24.0},
    "request_throughput": {"unit": "requests/sec", "avg": 0.5},
    "output_token_throughput": {"unit": "tokens/sec", "avg": 100.0},
}


def _schedule(ids=(7, 8)):
    """Request `a` prefills in two chunks, `b` in one, both decode; `a` is preempted and prefills again."""
    a, b = ids
    return [
        [(a, 100, 100, False)],
        [(a, 100, 200, False), (b, 50, 50, False)],
        [(a, 1, 201, True), (b, 1, 51, True)],
        [(a, 1, 202, True), (b, 1, 52, True)],
        [(a, 203, 203, False)],
        [(a, 1, 204, True)],
    ]


def _cell(directory, schedule, keys, **export):
    """A cell directory: `keys` maps a request id to the key its final chunk is named by."""
    (directory / "steps").mkdir(parents=True)
    (directory / "artifacts").mkdir()
    lines, named = [], set()
    for i, step in enumerate(schedule):
        final = [r for r, _q, _c, d in step if not d and r not in named]
        named.update(final)
        lines.append(
            {
                "step": i,
                "dp_rank": 0,
                "is_dummy_run": False,
                "produces_output": True,
                "batch": [[r, q, c] for r, q, c, _ in step],
                "named": [[r, keys[r]] for r in final if r in keys],
                "rows": [[q, c, d] for _, q, c, d in step],
                "rung": None,
            }
        )
    lines.insert(2, {**lines[0], "is_dummy_run": True, "batch": [], "named": []})
    (directory / "steps" / "dp0.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in lines)
    )
    out = json.loads(json.dumps(EXPORT))
    for name, (stat, value) in export.items():
        out[name][stat] = value
    (directory / "artifacts" / "profile_export_aiperf.json").write_text(json.dumps(out))
    return directory


def test_counts_match_across_runs_whose_request_ids_differ(tmp_path):
    keys = {7: "ka", 8: "kb", 1: "ka", 2: "kb"}
    real = [_cell(tmp_path / f"r{i}", _schedule(), keys) for i in range(3)]
    simulated = _cell(tmp_path / "sim", _schedule((1, 2)), keys)
    report = grade(real, simulated)

    counts = report["counts"]
    assert {k: g["simulated"] for k, g in counts.items()} == {
        "prefill steps": 3,
        "decode steps": 3,
        "prefill tokens": 453,
        "decode tokens": 5,
        "preemptions": 1,
        "prefill streaks": 2,
        "requests completed": 3.0,
    }
    assert {g["verdict"] for g in counts.values()} == {"exact"}
    widths = report["distributions"]["batch width"]
    assert widths["spread"] == 0 and widths["result"] == [0, 0, 0]


def test_a_metric_is_graded_against_the_real_spread(tmp_path):
    keys = {7: "ka", 8: "kb"}
    ttft = [1000.0, 1050.0, 1100.0]  # spread 10%, inside the 15% target
    tpot = [25.0, 25.5, 30.0]  # spread 20%, beyond 10%
    real = [
        _cell(
            tmp_path / f"r{i}",
            _schedule(),
            keys,
            time_to_first_token=("avg", t),
            inter_token_latency=("avg", p),
        )
        for i, (t, p) in enumerate(zip(ttft, tpot))
    ]

    def simulated(name, t, schedule=None, tpot=25.0):
        return _cell(
            tmp_path / name,
            schedule or _schedule(),
            keys,
            time_to_first_token=("avg", t),
            inter_token_latency=("avg", tpot),
        )

    report = grade(real, simulated("at", 1040.0))
    g = report["end to end"]["ttft mean"]
    assert g["result"] == pytest.approx([0.04, 1 - 1040 / 1050, 1 - 1040 / 1100])
    assert g["spread"] == pytest.approx(0.1)
    assert g["verdict"] == "inside the real spread"
    assert report["end to end"]["tpot mean"]["verdict"] == "ungradeable"
    assert report["end to end"]["ttft p50"]["verdict"] == "inside the real spread"

    report = grade(real, simulated("high", 1140.0))
    assert report["end to end"]["ttft mean"]["verdict"] == "within target"
    report = grade(real, simulated("higher", 1300.0))
    assert report["end to end"]["ttft mean"]["verdict"] == "outside"

    # One more decode step: a count off by one where the real runs agree.
    longer = _schedule() + [[(7, 1, 205, True)]]
    report = grade(real, simulated("long", 1040.0, longer))
    assert report["counts"]["decode steps"]["verdict"] == "outside"
    assert (
        report["distributions"]["batch width"]["result"] == [pytest.approx(1 / 14)] * 3
    )


def test_the_command_prints_every_metric_and_needs_two_real_runs(tmp_path, capsys):
    keys = {7: "ka", 8: "kb"}
    ttft = [1000.0, 1000.0, 1000.0, 1200.0]  # the real spread, 20%, swamps 15%
    cells = [
        _cell(tmp_path / f"c{i}", _schedule(), keys, time_to_first_token=("avg", t))
        for i, t in enumerate(ttft)
    ]
    main([str(c) for c in cells])
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "family | metric | spread | result | verdict"
    assert "counts | preemptions | 0 | 0 0 0 | exact" in out
    assert "end to end | ttft mean | 0.2 | 0 0 0.1667 | ungradeable" in out
    assert any(row.startswith("end to end | output token throughput |") for row in out)
    with pytest.raises(ValueError, match="two or more real runs"):
        grade(cells[:1], cells[1])


def test_the_command_refuses_one_real_run_naming_it(tmp_path):
    sim, real = (str(_cell(tmp_path / c, _schedule(), {})) for c in ("sim", "real0"))
    runs = [sys.executable, "-m", "atom.compass.validate.runs", sim, real]
    done = subprocess.run(runs, capture_output=True, text=True, check=False)

    refusal = f"{[real]}: a real spread needs two or more real runs\n"
    assert (done.returncode, done.stderr) == (1, refusal)


def test_the_command_refuses_a_malformed_run_json_naming_it(tmp_path):
    cells = [_cell(tmp_path / c, _schedule(), {}) for c in ("sim", "real0", "real1")]
    (cells[2] / "run.json").write_text("{")
    runs = [sys.executable, "-m", "atom.compass.validate.runs", *map(str, cells)]
    done = subprocess.run(runs, capture_output=True, text=True, check=False)

    what = "Expecting property name enclosed in double quotes: line 1 column 2"
    refusal = f"{cells[2]}/run.json is not JSON: {what}\n"
    assert (done.returncode, done.stderr) == (1, refusal)


def test_prefill_tokens_grade_the_requests_both_runs_issued(tmp_path):
    def run(name, ids, hit=0, extra=False):
        """Requests of 100 and 50 prompt tokens, `hit` of the second cached;
        `extra` issues a third, of the first one's length, after them."""
        a, b, c = ids
        schedule = [
            [(a, 100, 100, False)],
            [(a, 1, 101, True), (b, 50 - hit, 50, False)],
            [(a, 1, 102, True), (b, 1, 51, True)],
        ]
        if extra:
            schedule.append([(c, 100, 100, False)])
        return _cell(tmp_path / name, schedule, {i: f"k{i}" for i in ids})

    real = [run("r0", (7, 8, 9)), run("r1", (7, 8, 9)), run("r2", (5, 6, 9), hit=10)]
    simulated = run("sim", (21, 22, 23), hit=10, extra=True)
    report = grade(real, simulated)

    g = report["counts"]["prefill tokens"]
    assert g["simulated"] == 240 and g["real"] == [150, 150, 140]
    assert g["result"] == [10, 10, 0] and g["spread"] == 10
    assert g["verdict"] == "inside the real spread"
    one = report["one run"]["prefill tokens"]
    assert one["result"] == [100, 100, 100] and one["spread"] == 0
    assert one["verdict"] == "not gated"


def test_requests_keys_by_the_final_chunks_context_not_a_later_decode_row():
    """A decode row after the final chunk must not move the request's key away
    from its prompt length (100), which `named` fixes at the prefill row."""
    steps = [
        {"named": [[1, "k1"]], "batch": [[1, 100, 100]], "rows": [[100, 100, False]]},
        {"named": [], "batch": [[1, 1, 101]], "rows": [[1, 101, True]]},
    ]
    assert _requests(steps) == collections.Counter({(100, 0): 100})


def test_requests_of_one_id_on_two_dp_ranks_are_two_requests():
    """Each DP rank numbers its own requests, so request 1 on rank 0 and
    request 1 on rank 1 are two prompts of 100 tokens, not one of 200."""
    steps = [
        {
            "dp_rank": rank,
            "named": [[1, f"k{rank}"]],
            "batch": [[1, 100, 100]],
            "rows": [[100, 100, False]],
        }
        for rank in (0, 1)
    ]
    assert _requests(steps) == collections.Counter({(100, 0): 100, (100, 1): 100})


def test_a_distribution_no_run_samples_is_left_out(tmp_path):
    """A schedule of decode steps alone, as a deployment's decode side runs,
    has no prefill step's cached fraction in any run."""
    decode = [[(7, 1, 101, True)], [(7, 1, 102, True)]]
    real = [_cell(tmp_path / f"r{i}", decode, {}) for i in range(2)]
    report = grade(real, _cell(tmp_path / "sim", decode, {}))

    distributions = report["distributions"]
    assert set(distributions) == {"batch width", "scheduled tokens", "context tokens"}
    assert {g["verdict"] for g in distributions.values()} == {"inside the real spread"}


#: A simulated run's run.json entries that grading reads, as `run` writes
#: them, for a run it grades: host laws on, no roofline, no refusal, and a
#: backend fitted on a trace no cell here sends.
GRADED = {
    "run_file": {"step_linear": {}, "host": {"laws": {}}},
    "step_roofline": False,
    "skip": [],
    "refusals": {"count": 0, "reasons": []},
    "coverage": None,
    "provenance": {
        "digest": "sha256:0",
        "provenance": {"inputs": [{"run": "fit.jsonl", "traces": ["trace-kz"]}]},
    },
}


#: The traces every client run here was given.
GIVEN = ["trace-ka", "trace-kb", "trace-kc", "trace-kw"]
#: The arguments every client run here was given, as its export names them.
ARGUMENTS = {
    "phases.profiling.concurrency": 4,
    "phases.profiling.duration": 1800.0,
    "phases.profiling.warmup_requests_per_lane": 1,
    "datasets.main.random_seed": 20260707,
    "benchmark_id": "000000000000",
}


def _recorded(
    cell, sent=None, warmup=("kw",), given=GIVEN, held=True, args=ARGUMENTS, **run
):
    """`cell` as `record` lays it out, or with `run`'s entries as `run` does:
    each named request also named by its response id `id-<key>`, and the
    export naming those whose keys are in `sent` (all but its warmup's by
    default), each from trace `trace-<key>`; its warmup naming those whose
    keys are in `warmup`, by default `kw`, each that no step holds served
    first in a prefill step of its own unless `held` is False; and the traces
    and arguments its client was given, `given` and `args`. With `warmup`,
    `given` or `args` None the export has no such entry."""
    path = cell / "steps" / "dp0.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for line in lines:
        line["named"] = [[r, k, f"id-{k}"] for r, k, *_ in line["named"]]
    keys = sorted({k for line in lines for _, k, *_ in line["named"]})
    # Request ids 0 and up arrived before every request a cell here holds.
    for r, k in enumerate(sorted(set(warmup or ()) - set(keys)) if held else ()):
        warm = {"batch": [[r, 30, 30]], "named": [[r, k, f"id-{k}"]]}
        lines.insert(r, lines[0] | warm | {"rows": [[30, 30, False]]})
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))

    def named(mine):
        return {f"id-{k}": {"trace": f"trace-{k}", "sampling": {}} for k in mine}

    sent = set(keys) - set(warmup or ()) if sent is None else sent
    export = {"requests": named(k for k in keys if k in sent)}
    if warmup is not None:
        export["warmup"] = named(warmup)
    if given is not None:
        export["traces"] = given
    if args is not None:
        export["arguments"] = args
    (cell / CLIENT_RUN).write_text(json.dumps(export))
    (cell / "run.json").write_text(json.dumps({"format_version": 2, **run}))
    return cell


def _four(tmp_path, schedule=None, keys=None):
    """Three real cells and a simulated one, as `cctraces_sim.sh` lays them out."""
    keys = keys or {7: "ka", 8: "kb"}
    names = ("r0", "r1", "r2", "sim")
    return [_cell(tmp_path / n, schedule or _schedule(), keys) for n in names]


def test_a_cell_record_and_run_wrote_grades_as_the_one_cctraces_sim_wrote(tmp_path):
    *real, simulated = _four(tmp_path)
    report = grade(real, simulated)
    for cell in real:
        _recorded(cell)
    _recorded(simulated, **GRADED)

    assert grade(real, simulated) == report


@pytest.mark.parametrize(
    "entries, reason",
    [
        ({"skip": ["E"]}, r"it skips host segments \['E'\], a what-if run"),
        (
            {"run_file": {"step_linear": {}}},
            "its run file has no host laws, so it prices no host time",
        ),
        ({"step_roofline": True}, "the step roofline was on"),
        (
            {"refusals": {"count": 1, "reasons": [["clock:late", 1]]}},
            r"it refused 1: \[\['clock:late', 1\]\]",
        ),
        (
            {"run_file": {"step_grid": {}, "host": {}}},
            "it ran the step grid without a coverage check",
        ),
        # The step grid prices below it by the roofline, switch or none, and a
        # check that counts steps extrapolated or falling to it refuses none.
        (
            {
                "run_file": {"step_grid": {}, "host": {}},
                "step_roofline": True,
                "coverage": {"extrapolated": 3, "fallback": 2, "declined": [{}] * 2},
            },
            None,
        ),
        (
            {
                "provenance": {
                    "provenance": {
                        "inputs": [{"run": "p", "traces": [], "profiled": True}]
                    }
                }
            },
            r"its backend was fitted on profiled inputs: \['p'\]",
        ),
        (
            {"run_file": {"step_grid": {}, "host": {}}, "coverage": {"declined": []}},
            None,
        ),
        ({"provenance": {"digest": "sha256:stub", "provenance": None}}, None),
    ],
)
def test_a_simulated_run_grading_does_not_take_is_refused_naming_why(
    tmp_path, entries, reason
):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    _recorded(simulated, **(GRADED | entries))
    if reason is None:
        grade(real, simulated)
        return
    with pytest.raises(ValueError, match=f"sim is not graded: {reason}$"):
        grade(real, simulated)


def test_a_simulated_directory_run_did_not_write_is_refused(tmp_path):
    """`record` and `trace steps` write a run.json with no run file: a run of
    no backend, which no simulated-run refusal would read."""
    *real, simulated = (_recorded(c) for c in _four(tmp_path))

    unrun = "its run.json holds no run_file, so no `run` wrote it"
    with pytest.raises(ValueError, match=f"sim is not graded: {unrun}$"):
        grade(real, simulated)


def test_a_real_directory_run_wrote_is_refused(tmp_path):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    _recorded(simulated, **GRADED)
    _recorded(real[1], **GRADED)

    ran = "its run.json holds a run_file, so `run` wrote it"
    with pytest.raises(ValueError, match=f"r1 is not graded: {ran}$"):
        grade(real, simulated)


def test_a_record_holding_arrivals_is_refused(tmp_path):
    """`parity.read` leaves out the engine's start, the steps before the first
    arrival, and renumbers the rest, so its steps no longer pair with the
    record's lines."""
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    _recorded(simulated, **GRADED)
    path = real[1] / "steps" / "dp0.jsonl"
    lines = [json.loads(x) | {"t_enter_ns": 1} for x in path.read_text().splitlines()]
    start = lines[0] | {"batch": [], "named": [], "rows": [], "t_enter_ns": 0}
    path.write_text("".join(json.dumps(x) + "\n" for x in [start, *lines]))
    (path.parent / ARRIVALS).write_text(json.dumps({"sent_ns": [1]}))

    with pytest.raises(ValueError, match=f"r1 is not graded: its {ARRIVALS} renum"):
        grade(real, simulated)


HOST = {"host": {"laws": {}}}
NO_HOST = "its run file has no host laws, so it prices no host time"
ROOFLINE = "the step roofline was on"


@pytest.mark.parametrize(
    "run, reason",
    [
        (HOST | {"step_linear": {}, "step_roofline": True}, ROOFLINE),
        (HOST | {"step_stub": {}, "step_roofline": True}, ROOFLINE),
        ({"step_linear": {}}, NO_HOST),
        ({"step_linear": {}, "step_roofline": True}, f"{NO_HOST}; {ROOFLINE}"),
        (HOST | {"step_linear": {}, "step_roofline": False}, None),
        # The step grid prices below it by the roofline, switch or none.
        (HOST | {"step_grid": {}, "step_roofline": True}, None),
    ],
    ids=["law-roofline", "stub-roofline", "no-host", "both", "law", "grid-roofline"],
)
def test_a_cctraces_sim_cell_grading_does_not_take_is_refused_naming_why(
    tmp_path, run, reason
):
    *real, simulated = _four(tmp_path)
    report = grade(real, simulated)
    (simulated / "run.json").write_text(json.dumps(run))

    if reason is None:
        assert grade(real, simulated) == report
        return
    with pytest.raises(ValueError, match=f"sim is not graded: {reason}$"):
        grade(real, simulated)


@pytest.mark.parametrize(
    "run",
    [
        {"step_linear": {}},
        HOST | {"step_linear": {}, "step_roofline": True},
        {"host": {"skip": ["E"]}},
    ],
    ids=["no-host", "law-roofline", "skip"],
)
def test_real_cctraces_sim_cells_grade_whatever_run_file_they_were_handed(
    tmp_path, run
):
    """A real cell's server never reads the run file its run.json holds; a
    simulated cell handed the same one is refused, named."""
    *real, simulated = _four(tmp_path)
    report = grade(real, simulated)
    for cell in real:
        (cell / "run.json").write_text(json.dumps(run))
    assert grade(real, simulated) == report

    (simulated / "run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="/sim (is not graded|skips host segments)"):
        grade(real, simulated)


def test_the_command_states_the_steps_extrapolated_and_priced_by_a_fallback(
    tmp_path, capsys
):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    _recorded(simulated, **GRADED)
    schedule = {"refusals": {"steps": 9}, "extrapolated_steps": 3, "fallback_steps": 2}
    first = simulated / compass_run.CLIENT_DIR.format(0)
    first.mkdir()
    (first / compass_run.SUMMARY_FILE).write_text(json.dumps({"schedule": schedule}))

    main([str(simulated), *map(str, real)])

    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "of 9 steps priced: extrapolated 3, fallback 2"


def test_a_simulated_run_fitted_on_a_trace_the_cell_served_is_refused_naming_it(
    tmp_path,
):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    inputs = [{"run": "fit.jsonl", "traces": ["trace-kb", "trace-kz"]}]
    leaked = {"digest": "sha256:0", "provenance": {"inputs": inputs}}
    _recorded(simulated, **(GRADED | {"provenance": leaked, "step_roofline": True}))

    fitted = r"its backend was fitted on traces it served: \['trace-kb'\]"
    with pytest.raises(ValueError, match=f"the step roofline was on; {fitted}$"):
        main([str(simulated), *map(str, real)])


@pytest.mark.parametrize("target", ["trace-kz", "trace-kb"])
def test_a_step_grid_planned_on_the_traces_a_run_served_grades_unless_a_target_did(
    tmp_path, target
):
    """The provenance `fit steps --backend step-grid` reports: a residual
    target, a replay that served `target`, and the plan, whose points were
    placed from shapes of every trace the cell served."""
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    plan = {
        "source": "plan",
        "run": "sha256:plan",
        "inputs": ["grid"],
        "traces": [*GIVEN, "trace-kz"],
    }
    replay = {"source": "replay", "run": "sha256:replay", "traces": [target]}
    provenance = {"digest": "sha256:0", "provenance": {"inputs": [replay, plan]}}
    grid = {"run_file": {"step_grid": {}, "host": {}}, "coverage": {"uncovered": []}}
    _recorded(simulated, **(GRADED | grid | {"provenance": provenance}))

    if target == "trace-kz":
        grade(real, simulated)
        return
    fitted = r"its backend was fitted on traces it served: \['trace-kb'\]"
    with pytest.raises(ValueError, match=f"sim is not graded: {fitted}$"):
        grade(real, simulated)


def test_a_client_run_its_server_served_after_another_is_refused(tmp_path):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    # Request `ka` prefills alone first, and this client run sent only `kb`.
    _recorded(real[1], sent={"kb"})

    with pytest.raises(ValueError, match="r1 is not graded: steps of requests it did"):
        grade(real, simulated)


#: Request `kw` warms the server up, two steps of its own before `_schedule`'s.
WARMED = [[(6, 30, 30, False)], [(6, 1, 31, True)]] + _schedule()


def test_a_client_run_s_warmup_steps_precede_its_own_and_are_not_graded(tmp_path):
    *real, simulated = (_recorded(c) for c in _four(tmp_path / "one_step"))
    _recorded(simulated, **GRADED)
    report = grade(real, simulated)
    keys = {6: "kw", 7: "ka", 8: "kb"}
    *real, simulated = (
        _recorded(c, sent={"ka", "kb"}, warmup={"kw"})
        for c in _four(tmp_path / "two_steps", WARMED, keys)
    )
    _recorded(simulated, sent={"ka", "kb"}, warmup={"kw"}, **GRADED)

    assert grade(real, simulated) == report


def test_a_cell_cctraces_sim_wrote_with_its_export_grades_its_profiling_steps_alone(
    tmp_path,
):
    *real, simulated = _four(tmp_path / "whole")
    report = grade(real, simulated)

    def cells(name, schedule):
        out = _four(tmp_path / name, schedule, {6: "kw", 7: "ka", 8: "kb"})
        for cell in out:
            _recorded(cell, warmup={"kw"})
            # run.json as `cctraces_sim.sh` writes it: a run file, no `format_version`
            (cell / "run.json").write_text(json.dumps({"client_runs": 1} | HOST))
        return out

    warmed = cells("warmed", WARMED)
    assert grade(warmed[:3], warmed[3]) == report
    # A warmup still running past its first own step refuses it.
    late = cells("late", LATE)
    with pytest.raises(ValueError, match="r0 is not graded: .* has not ended"):
        grade(late[:3], late[3])


@pytest.mark.parametrize(
    "step", [[(9, 30, 30, False)], [(6, 1, 31, True), (9, 30, 30, False)]]
)
def test_a_step_of_a_request_neither_its_own_nor_its_warmup_s_before_them_refuses(
    tmp_path, step
):
    keys = {6: "kw", 7: "ka", 8: "kb", 9: "kx"}
    schedule = WARMED[:1] + [step] + WARMED[2:]
    *real, simulated = (
        _recorded(c, sent={"ka", "kb"}, warmup={"kw"})
        for c in _four(tmp_path, schedule, keys)
    )

    with pytest.raises(ValueError, match="r0 is not graded: steps of requests it did"):
        grade(real, simulated)


@pytest.mark.parametrize("warmup", [None, ()])
def test_a_client_run_whose_export_names_no_warmup_is_refused(tmp_path, warmup):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    _recorded(real[2], warmup=warmup)

    cold = "its export names no warmup, so its first steps may be cold"
    with pytest.raises(ValueError, match=f"r2 is not graded: {cold}$"):
        grade(real, simulated)


def test_a_client_run_whose_warmup_no_step_before_its_first_holds_is_refused(
    tmp_path,
):
    cells = _four(tmp_path, _schedule(), {6: "kw", 7: "ka", 8: "kb"})
    *real, simulated = (_recorded(c, held=False) for c in cells)

    unheld = "no step before its own first holds its warmup, so its first steps"
    with pytest.raises(ValueError, match=f"r0 is not graded: {unheld} may be cold$"):
        grade(real, simulated)


#: Request `kw` prefills in the client run's first step.
SHARED = [[(6, 30, 30, False), *_schedule()[0]], *_schedule()[1:]]
#: Request `kw` prefills alone first, and decodes on into the client run's second step.
LONG = [WARMED[0], _schedule()[0], [WARMED[1][0], *_schedule()[1]], *_schedule()[2:]]
#: Request `kw` prefills alone first, and decodes alone after the client run's first step.
APART = [WARMED[0], _schedule()[0], WARMED[1], *_schedule()[1:]]
#: Request `kw` warms the server up only after the client run's first step.
LATE = [_schedule()[0], [(6, 30, 30, False)], *_schedule()[1:]]


@pytest.mark.parametrize(
    "schedule, unheld, step",
    [(SHARED, True, 0), (LONG, False, 2), (APART, False, 2), (LATE, True, 1)],
    ids=["first", "later", "warmup-only-after-first-own", "late"],
)
def test_a_client_run_whose_warmup_runs_on_past_its_first_own_step_is_refused(
    tmp_path, schedule, unheld, step
):
    keys = {6: "kw", 7: "ka", 8: "kb"}
    cells = [_recorded(c, warmup={"kw"}) for c in _four(tmp_path, schedule, keys)]

    cold = (
        "no step before its own first holds its warmup, so its first steps may be cold"
    )
    why = [cold] if unheld else []
    why.append(
        f"steps from its own first on hold a request of its warmup, 1 from step {step}, "
        "so its warmup has not ended when it starts"
    )
    with pytest.raises(ValueError, match=f"r0 is not graded: {'; '.join(why)}$"):
        grade(cells[:3], cells[3])


def test_a_client_run_naming_no_warmup_grades_whole_beside_a_whole_real_run(tmp_path):
    *real, simulated = _four(tmp_path)
    report = grade(real, simulated)
    # r2 stays as `cctraces_sim.sh` wrote it, so every run is graded whole.
    for cell in real[:2]:
        _recorded(cell, warmup=None)
    _recorded(simulated, warmup=None, **GRADED)

    assert grade(real, simulated) == report


def test_a_run_graded_whole_whose_backend_was_fitted_on_its_warmup_is_refused(
    tmp_path,
):
    *real, simulated = _four(tmp_path)
    for cell in real[:2]:
        _recorded(cell)
    inputs = [{"run": "fit.jsonl", "traces": ["trace-kw"]}]
    leaked = {"digest": "sha256:0", "provenance": {"inputs": inputs}}
    _recorded(simulated, **(GRADED | {"provenance": leaked}))

    fitted = r"its backend was fitted on traces it served: \['trace-kw'\]"
    with pytest.raises(ValueError, match=f"sim is not graded: {fitted}$"):
        grade(real, simulated)


def test_a_client_run_whose_warmup_runs_on_past_its_first_own_step_grades_whole(
    tmp_path,
):
    *real, simulated = _four(tmp_path, LONG, {6: "kw", 7: "ka", 8: "kb"})
    for cell in real[:2]:
        _recorded(cell, warmup={"kw"})
    _recorded(simulated, warmup={"kw"}, **GRADED)

    assert grade(real, simulated)["counts"]["decode steps"]["verdict"] == "exact"


@pytest.mark.parametrize("stopped", [True, False])
def test_a_run_graded_whole_takes_steps_after_its_own_only_once_its_server_stopped(
    tmp_path, stopped
):
    """Request 9 runs on after the client run's own and its export does not
    name it: aiperf cancelled it at its cutoff when `client.sh --stop` stopped
    the server after the client run, else a later client run may have sent it."""
    *real, _ = _four(tmp_path)
    # r2 stays as `cctraces_sim.sh` wrote it, so every run is graded whole.
    for cell in real[:2]:
        _recorded(cell)
    late = _schedule() + [[(9, 30, 30, False)], [(9, 1, 31, True)]]
    simulated = _cell(tmp_path / "late", late, {7: "ka", 8: "kb", 9: "kc"})
    _recorded(simulated, sent={"ka", "kb"}, **GRADED)
    path = simulated / CLIENT_RUN
    export = json.loads(path.read_text()) | {"server_stopped": stopped}
    path.write_text(json.dumps(export))

    if stopped:
        decode = grade(real, simulated)["counts"]["decode steps"]
        assert [decode["simulated"] - r for r in decode["real"]] == [1, 1, 1]
    else:
        later = "steps of requests it did not send follow its own, 2 from step 6"
        with pytest.raises(ValueError, match=f"late is not graded: {later}, and it"):
            grade(real, simulated)


def test_the_steps_of_a_later_client_run_are_not_graded(tmp_path):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    _recorded(simulated, **GRADED)
    report = grade(real, simulated)
    later = _schedule() + [[(9, 30, 30, False)], [(9, 1, 31, True)]]
    _cell(tmp_path / "later", later, {7: "ka", 8: "kb", 9: "kc"})
    real[0] = _recorded(tmp_path / "later", sent={"ka", "kb"})

    assert grade(real, simulated) == report


def test_runs_given_different_traces_are_not_one_cell(tmp_path):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    _recorded(simulated, **GRADED)
    _recorded(real[2], given=["trace-other", *GIVEN[1:]])

    beyond = f"'{real[2]}': ['trace-other'], '{simulated}': ['trace-ka']}}"
    with pytest.raises(ValueError, match="were given different traces") as refused:
        grade(real, simulated)
    assert str(refused.value).endswith(beyond)


def test_runs_given_the_same_traces_pair_though_their_finished_requests_differ(
    tmp_path,
):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    _recorded(simulated, **GRADED)
    report = grade(real, simulated)
    # Request `kb` was in flight at every real run's cutoff, and aiperf
    # records nothing of it: only the simulated run's export names it.
    for cell in real:
        _recorded(cell, sent={"ka"})

    assert grade(real, simulated) == report


def test_a_client_run_whose_export_names_no_traces_given_is_refused(tmp_path):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    _recorded(real[1], given=None)

    unknown = "its export names no traces its client was given"
    with pytest.raises(ValueError, match=f"r1 is not graded: {unknown}$"):
        grade(real, simulated)


@pytest.mark.parametrize(
    "other",
    [
        {"phases.profiling.concurrency": 16},
        {"phases.profiling.duration": 600.0, "datasets.main.random_seed": 1},
        {"phases.profiling.warmup_requests_per_lane": None},
        {"endpoint.timeout": 60.0},
        {"benchmark_id": "0f8c2b6a"},
    ],
)
def test_runs_given_the_same_traces_and_other_arguments_are_not_one_cell(
    tmp_path, other
):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    _recorded(simulated, **GRADED)
    # None: the export names no such argument.
    args = {k: v for k, v in (ARGUMENTS | other).items() if v is not None}
    _recorded(real[1], args=args)

    cells = [str(c) for c in (*real, simulated)]
    differ = {
        name: {c: ARGUMENTS.get(name) for c in cells} | {str(real[1]): value}
        for name, value in sorted(other.items())
    }
    with pytest.raises(ValueError, match="were given different arguments") as refused:
        grade(real, simulated)
    assert str(refused.value).endswith(f": {differ}")


def test_a_client_run_whose_export_names_no_arguments_given_is_refused(tmp_path):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    _recorded(real[1], given=None, args=None)

    unknown = "its export names no {} its client was given"
    both = f"{unknown.format('traces')}; {unknown.format('arguments')}"
    with pytest.raises(ValueError, match=f"r1 is not graded: {both}$"):
        grade(real, simulated)


def test_a_client_run_whose_export_lists_unnamed_requests_is_refused_naming_them(
    tmp_path,
):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    _recorded(simulated, **GRADED)
    report = grade(real, simulated)

    def unnamed(ids):
        export = json.loads((real[1] / CLIENT_RUN).read_text())
        rows = [{"x_request_id": i, "phase": "profiling", "trace": "t"} for i in ids]
        (real[1] / CLIENT_RUN).write_text(json.dumps(export | {"unnamed": rows}))

    # `report` read exports with no `unnamed`; one listing none grades the same.
    unnamed([])
    assert grade(real, simulated) == report
    unnamed(["x-1", "x-2"])
    why = "its export lists requests no step can be matched to: ['x-1', 'x-2']"
    with pytest.raises(ValueError, match=f"r1 is not graded: {re.escape(why)}$"):
        grade(real, simulated)


def _ran(monkeypatch, tmp_path, served, assumed=None):
    """Three real cells, and a simulated one whose run.json `compass run`
    wrote once its server, serving until stopped, served one client run per
    entry of `served` and was stopped during the last; with `assumed`, its
    engine read a `memory_readings` member stating that assumption."""
    sim = _serve(monkeypatch, tmp_path / "sim", None, served, stop="bound")
    run = json.loads((sim / "run.json").read_text()) | GRADED["run_file"]
    run |= {"step_roofline": False, "backend": GRADED["provenance"]}
    (sim / serve.RUN_FILE).write_text(json.dumps(run))
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    if assumed:
        engine = simulated / compass_run.LP_FILE.format("engine")
        hand_in = json.loads(engine.read_text()) | {"memory": {"assumed": assumed}}
        engine.write_text(json.dumps(hand_in))
    drive_run.write_grading(simulated, None)
    return real, simulated


def test_a_later_client_runs_refusal_does_not_refuse_the_first(monkeypatch, tmp_path):
    real, simulated = _ran(monkeypatch, tmp_path, [UNREFUSED, CLIENT_RUNS[1]])

    server = json.loads((simulated / compass_run.SUMMARY_FILE).read_text())
    reasons = dict(server["schedule"]["refusals"]["reasons"])
    assert "ended:frontend stopped by SIGTERM" in reasons and len(reasons) == 4
    grade(real, simulated)


def test_a_stop_during_the_first_client_run_refuses_it(monkeypatch, tmp_path):
    real, simulated = _ran(monkeypatch, tmp_path, [UNREFUSED])

    unknown = "its first client run has no summary, so its refusals are unknown"
    with pytest.raises(ValueError, match=f"sim is not graded: {unknown}$"):
        grade(real, simulated)


def test_a_client_run_that_sent_nothing_before_the_graded_one_refuses_it(
    monkeypatch, tmp_path
):
    empty = ([], {"steps": 0, "seconds": 0.0, "treatments": collections.Counter()}, [])
    real, simulated = _ran(monkeypatch, tmp_path, [empty, CLIENT_RUNS[1]])

    nothing = "its first client run priced no step, so it is not the graded one"
    with pytest.raises(ValueError, match=f"sim is not graded: {nothing}$"):
        grade(real, simulated)


def test_a_run_whose_engine_read_an_assumed_memory_reading_is_refused_naming_it(
    monkeypatch, tmp_path
):
    assumed = "copied from MI355X's memory_readings entry of this configuration"
    served = [UNREFUSED, CLIENT_RUNS[1]]
    real, simulated = _ran(monkeypatch, tmp_path, served, assumed)

    named = re.escape(f"its engines read assumed memory_readings members: {[assumed]}")
    with pytest.raises(ValueError, match=f"sim is not graded: {named}$"):
        grade(real, simulated)


CCTRACES_SIM = HOST | {"step_linear": {}, "client_runs": 1}


@pytest.mark.parametrize(
    "layout, run, sides",
    [
        ("whole", CCTRACES_SIM, [""]),
        ("export", CCTRACES_SIM, [""]),
        ("1p1d", CCTRACES_SIM | {"router_s": 0.0, "kv_write_req_s": 0.0}, ["-P", "-D"]),
    ],
    ids=["whole", "export", "1p1d"],
)
def test_a_cctraces_sim_cell_whose_engine_read_an_assumed_member_is_refused_naming_it(
    tmp_path, layout, run, sides
):
    """A cell `cctraces_sim.sh` wrote, with its client run's export or without,
    or through `pd_sim.sh`, whose run.json is the run file it ran."""
    cells = _four(tmp_path)
    *real, simulated = [_recorded(c) for c in cells] if layout == "export" else cells
    for cell in cells:
        (cell / "run.json").write_text(json.dumps(run))
    report = grade(real, simulated)

    def hand_ins(assumed):
        for side in sides:
            memory = {"assumed": assumed and f"{assumed}{side}"}
            engine = simulated / compass_run.LP_FILE.format(f"engine{side}")
            engine.write_text(json.dumps({"refusals": [], "memory": memory}))

    hand_ins(None)
    assert grade(real, simulated) == report
    hand_ins("copied from MI355X")
    named = re.escape(ASSUMED.format(sorted(f"copied from MI355X{s}" for s in sides)))
    with pytest.raises(ValueError, match=f"sim is not graded: {named}$"):
        grade(real, simulated)


def _aiter(tmp_path, monkeypatch):
    """An installed aiter of its own, its JIT directory `jit` returned holding
    one module the image built, and a home of its own, with no JIT directory
    named by the environment."""
    site = tmp_path / "site"
    (site / "aiter" / "jit" / "build" / "old" / "build").mkdir(parents=True)
    (site / "aiter" / "jit" / "build" / "old" / "build" / "old.so").write_bytes(b"")
    (site / "aiter" / "jit" / "old.so").write_bytes(b"")
    (site / "aiter" / "__init__.py").write_text("")
    old = time.time_ns() - 10**12
    for path in (site, *site.rglob("*")):
        os.utime(path, ns=(old, old))
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.delitem(sys.modules, "aiter", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in ("AITER_JIT_DIR", "AITER_ROOT_DIR"):
        monkeypatch.delenv(name, raising=False)
    return site / "aiter" / "jit"


def test_a_record_whose_server_built_an_aiter_module_is_refused_naming_it(tmp_path):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    builds = ["mha_varlen_fwd_bf16_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale"]
    _recorded(real[1], **{aiter_jit.BUILDS: builds})

    named = re.escape(f"aiter ran a JIT build of {builds} while its server served")
    with pytest.raises(ValueError, match=f"r1 is not graded: {named}, so a step"):
        grade(real, simulated)


def test_a_real_cctraces_sim_cell_whose_server_began_an_aiter_build_is_refused(
    tmp_path, monkeypatch
):
    """Its run.json as `cctraces_sim.sh` writes it, then given the entry as
    that script gives it, from the time its server began serving: a build
    the stop cut short, with no module written, counts."""
    jit = _aiter(tmp_path, monkeypatch)
    *real, simulated = _four(tmp_path / "cells")
    for cell in (*real, simulated):
        _recorded(cell)
        (cell / "run.json").write_text(json.dumps({"client_runs": 1} | HOST))
    report = grade(real, simulated)
    since = time.time_ns() - 10**9

    aiter_jit.main([str(real[0] / "run.json"), str(since)])
    assert json.loads((real[0] / "run.json").read_text()) == {"client_runs": 1} | HOST
    assert grade(real, simulated) == report

    (jit / "build" / "mha" / "blob.staging").mkdir(parents=True)
    (jit / "build" / "mha" / "blob.staging" / "kernel.cpp").write_text("")
    aiter_jit.main([str(real[2] / "run.json"), str(since)])
    named = re.escape("aiter ran a JIT build of ['mha'] while its server served")
    with pytest.raises(ValueError, match=f"r2 is not graded: {named}, so a step"):
        grade(real, simulated)


#: Requests rank 1 of a deployment's side holds, under rank 0's request ids.
RANK1 = {"ka": "kc", "kb": "kd"}


def _deployed(cell, late=None, **run):
    """`cell` as `deploy.sh` lays out a deployment: prefill/ and decode/ each
    `_recorded`'s steps, with `run`'s entries in its run.json, on DP rank 0
    and again on rank 1 as requests `kc` and `kd`, the ranks stepping
    together; on side `late` rank 1 runs its warmup's step after its first own
    one. The client's export, naming all four, is beside them;
    their requests are named as `_named_as_deployed` names them."""
    for side in SIDES:
        _recorded(_cell(cell / side, _schedule(), {7: "ka", 8: "kb"}), **run)
        steps = cell / side / "steps"
        lines = [json.loads(x) for x in (steps / "dp0.jsonl").read_text().splitlines()]
        rank1 = json.loads(json.dumps(lines))
        if side == late:
            rank1[:2] = rank1[1::-1]
        for line in rank1:
            line["dp_rank"] = 1
            line["named"] = [
                [r, RANK1.get(k, k), f"id-{RANK1.get(k, k)}"]
                for r, k, _ in line["named"]
            ]
        for rank, rank_lines in enumerate((lines, rank1)):
            for i, line in enumerate(rank_lines):
                line["step"] = i
            text = "".join(json.dumps(x) + "\n" for x in rank_lines)
            (steps / f"dp{rank}.jsonl").write_text(text)
        export = json.loads((cell / side / CLIENT_RUN).read_text())
        for k in RANK1.values():
            export["requests"][f"id-{k}"] = {"trace": f"trace-{k}", "sampling": {}}
        shutil.rmtree(cell / "artifacts", ignore_errors=True)
        (cell / side / "artifacts").rename(cell / "artifacts")
        (cell / CLIENT_RUN).write_text(json.dumps(export))
    (cell / "run.json").write_text(json.dumps({"kv_link": "inter_node"}))
    _named_as_deployed(cell)
    return cell


def _named_as_deployed(cell):
    """Name the requests of deployment `cell`, each side's record named as
    one server's, as a deployment's two records name them: prefill by its own
    response ids, `p-<id>`, which the export does not name; decode, its
    request ids 100 above prefill's, at its first step by its request id, and
    each joined in `transfers.jsonl` to its prefill DP rank and request id."""
    for path in sorted((cell / "prefill" / "steps").glob("dp*[0-9].jsonl")):
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        for line in lines:
            line["named"] = [[r, k, f"p-{i}"] for r, k, i in line["named"]]
        path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    sent = []
    for path in sorted((cell / "decode" / "steps").glob("dp*[0-9].jsonl")):
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        ids = {r: i for line in lines for r, _k, i in line["named"]}
        seen = set()
        for line in lines:
            first = [r for r, *_ in line["batch"] if r in ids and r not in seen]
            seen.update(first)
            line["named"] = [[r + 100, r + 100, ids[r]] for r in first]
            line["batch"] = [[r + 100, *rest] for r, *rest in line["batch"]]
        path.write_text("".join(json.dumps(x) + "\n" for x in lines))
        sent += [
            {"response_id": i, "remote_dp_rank": int(path.stem[2:]), "transfer_id": r}
            for r, i in ids.items()
        ]
    text = "".join(json.dumps(x) + "\n" for x in sent)
    (cell / "decode" / "steps" / TRANSFERS).write_text(text)


def test_a_deployment_is_graded_per_side_over_every_dp_rank(tmp_path):
    real = [_deployed(tmp_path / f"r{i}") for i in range(3)]
    simulated = _deployed(tmp_path / "sim", **GRADED)
    report = grade(real, simulated)

    assert set(report) == {
        *(f"{family} {side}" for family in ("counts", "distributions", "one run") for side in SIDES),
        "end to end",
    }  # fmt: skip
    for side in SIDES:
        counts = report[f"counts {side}"]
        # `test_counts_match_across_runs_whose_request_ids_differ`'s, once per rank.
        assert {k: g["simulated"] for k, g in counts.items()} == {
            "prefill steps": 6,
            "decode steps": 6,
            "prefill tokens": 906,
            "decode tokens": 10,
            "preemptions": 2,
            "prefill streaks": 4,
            "requests completed": 3.0,
        }
        assert {g["verdict"] for g in counts.values()} == {"exact"}
    assert report["end to end"]["ttft mean"]["verdict"] == "inside the real spread"


def test_a_deployment_side_whose_warmup_runs_on_past_its_first_own_step_on_any_rank_is_refused(
    tmp_path,
):
    real = [
        _deployed(tmp_path / f"r{i}", late="decode" if i == 1 else None)
        for i in range(3)
    ]
    simulated = _deployed(tmp_path / "sim", **GRADED)

    late = "1 from step 1, so its warmup has not ended when it starts"
    with pytest.raises(
        ValueError, match=f"r1/decode is not graded: steps from .*{late}$"
    ):
        grade(real, simulated)


def test_a_deployment_side_whose_server_built_an_aiter_module_is_refused_naming_it(
    tmp_path,
):
    real = [_deployed(tmp_path / f"r{i}") for i in range(3)]
    simulated = _deployed(tmp_path / "sim", **GRADED)
    builds = ["mha_varlen_fwd_bf16"]
    run = {"format_version": 2, aiter_jit.BUILDS: builds}
    (real[1] / "decode" / "run.json").write_text(json.dumps(run))

    named = re.escape(f"aiter ran a JIT build of {builds} while its server served")
    with pytest.raises(ValueError, match=f"r1/decode is not graded: {named}"):
        grade(real, simulated)


def test_deployments_and_single_servers_are_not_one_cell(tmp_path):
    real = [_deployed(tmp_path / f"r{i}") for i in range(2)]
    sim = _cell(tmp_path / "sim", _schedule(), {7: "ka", 8: "kb"})
    simulated = _recorded(sim, **GRADED)

    with pytest.raises(ValueError, match="the runs' servers wrote different sides"):
        grade(real, simulated)

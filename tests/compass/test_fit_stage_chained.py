# SPDX-License-Identifier: MIT
"""`compass fit steps --stage-form chained`: a later pipeline stage's vector is
stage 0's times the median ratio of its own device work, from the stage
before's device end plus its input's copy, to stage 0's, on the steps both
stages found their device idle at."""

import json

import pytest
from test_backend_step_linear import law

from atom.compass.backends.step_linear import StepLinearLaw
from atom.compass.fit.steps import _chained_stage, _stage_lines

MS = 1_000_000
HOP = (0.5e-3, 1e-6)  # 0.5 ms and 1 us a token


def line(step, launch_ms, busy_ms, gap_ms=5.0, tokens=1000, **extra):
    return {
        "step": step,
        "is_dummy_run": False,
        "batch": [[step, tokens, tokens]],
        "t_launch_event_ns": int(launch_ms * MS),
        "d_busy_ns": int(busy_ms * MS),
        "d_gap_ns": int(gap_ms * MS),
        **extra,
    }


def stages(own, n=9, draft_kv_ms=0.0):
    """Stage 0 and stage 1 lines: stage 0's step n runs `10 + n` ms from its
    launch at 100 n ms; stage 1 launches 1 ms after it and waits for stage 0's
    end and the 1.5 ms copy of its 1000 tokens, then runs `own` of stage 0's."""
    zero, one = {}, {}
    for s in range(n):
        t, b = 100.0 * s, 10.0 + s
        zero[s] = line(s, t, b)
        wait = t + b + 1.5 - (t + 1.0)
        one[s] = line(s, t + 1.0, wait + own * b + draft_kv_ms, d_draft_kv_ns=int(draft_kv_ms * MS))
    return zero, one


def test_a_chained_stage_is_its_own_work_from_the_stage_befores_end_and_hop():
    zero, one = stages(0.8)
    # a step whose stage-1 device was busy at its launch is left out
    one[3] = line(3, 301.0, 500.0, gap_ms=0.0)
    vector = law()["prefill"]
    scaled, figures = _chained_stage(1, [zero, zero, one], HOP, 1e-3, vector)
    assert figures["scale"] == pytest.approx(0.8, rel=1e-6)
    assert figures["steps"] == 8 and figures["form"] == "chained"
    assert figures["hop"] == list(HOP)
    for k, v in vector["coefficients"].items():
        assert scaled["coefficients"][k] == pytest.approx(0.8 * v, rel=1e-6)
    assert StepLinearLaw(dict(law(), stage_prefill={"1": scaled})).at_stage(1).prefill


def test_a_drafter_stages_draft_kv_pass_is_not_its_target_work():
    zero, one = stages(0.8, draft_kv_ms=2.0)
    assert _chained_stage(1, [zero, zero, one], HOP, 1e-3, law()["prefill"])[1][
        "scale"
    ] == pytest.approx(0.8, rel=1e-6)


def test_a_stage_launched_after_its_input_arrived_runs_its_whole_window():
    zero = {0: line(0, 0.0, 10.0)}
    one = {0: line(0, 50.0, 7.0)}  # long after stage 0's end and the copy
    assert _chained_stage(1, [zero, zero, one], HOP, 1e-3, law()["prefill"])[1][
        "scale"
    ] == pytest.approx(0.7)


def test_no_common_idle_step_is_refused():
    zero, one = stages(0.8)
    for s in one.values():
        s["d_gap_ns"] = 0
    with pytest.raises(ValueError, match="found the device idle at no common step"):
        _chained_stage(1, [zero, zero, one], HOP, 1e-3, law()["prefill"])


def test_stage_lines_keep_the_timed_lines_that_run_no_dummy(tmp_path):
    path = tmp_path / "dp0.jsonl"
    lines = [line(0, 0.0, 1.0), {**line(1, 1.0, 1.0), "is_dummy_run": True}, {"step": 2}]
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert list(_stage_lines(path)) == [0]

# SPDX-License-Identifier: MIT
"""`step_labels.launch_stalls` and the opt-in `STALL_ENV` cut: a prefill line
whose host launch window lasted `STALL_FACTOR` times its launch path's median
on its rank marks its step, and under the switch every rank's line at that
step is cut as `stalled`, keeping its host labels, where the paced cut misses
it (the device runs on past the stalled launch's end). Unset or ``0``, every
label and fit is as before.

No device.
"""

import json

import pytest
from test_backend_step_linear_fit import _engine_file

from atom.compass.backends import step_labels
from atom.compass.backends.step_linear_fit import fit, label_records

MS = 1_000_000


def _line(
    step, rank, launch, rows=((6000, 9000, False),), flags=("eager_body",), dummy=False
):
    """The fields `launch_stalls` reads, a launch window of `launch` ms."""
    return {
        "step": step,
        "dp_rank": rank,
        "is_dummy_run": dummy,
        "produces_output": True,
        "rows": [list(r) for r in rows],
        "rung": 1 if all(r[2] for r in rows) else None,
        "t_launch_ns": [step * 1000 * MS, step * 1000 * MS + int(launch * MS)],
        "launch": dict.fromkeys(flags, 1.0),
    }


def test_a_window_twice_its_paths_median_on_its_rank_marks_its_step():
    """Each rank and launch path is held to its own median: a tail-path line at
    twice the tail's median is a stall though under the body's; a decode line
    and a dummy run, however long, are none."""
    r0 = [_line(i, 0, 50) for i in range(20)]
    r0[7] = _line(7, 0, 100)  # 2.0 times: a stall
    r0[8] = _line(8, 0, 99)  # under twice: none
    r0.append(_line(30, 0, 500, rows=((1, 900, True),)))  # a decode
    r0.append(_line(31, 0, 500, dummy=True))
    r1 = [_line(i, 1, 30, flags=("eager_tail",)) for i in range(20)]
    r1[9] = _line(9, 1, 80, flags=("eager_tail",))

    assert step_labels.launch_stalls([r0, r1]) == {7, 9}
    assert step_labels.launch_stalls([r0, r1], factor=3.0) == frozenset()


WORK = (20 * MS, 10_000)
DRAFT_NS = 2 * MS


def _record_line(step, rank, t, tokens, launch):
    """A plain MTP record's prefill line (as test_step_labels_paced's) of DP
    rank `rank` at host time `t`: its device runs its work from `t`, longer
    than a normal launch, so no line is paced; `launch` ns its host launch."""
    target = WORK[0] + WORK[1] * tokens
    rows = [[tokens, tokens + 1000 + 7919 * step % 50000, False]]
    return {
        "step": step,
        "dp_rank": rank,
        "is_dummy_run": False,
        "produces_output": True,
        "batch": [[step, tokens, rows[0][1]]],
        "named": [],
        "rows": rows,
        "rung": None,
        "moe_rows": 2 * tokens,
        "moe_tokens": 2 * tokens,
        "t_enter_ns": t,
        "t_launch_ns": [t, t + launch],
        "t_recv_ns": [t + target + DRAFT_NS] * 2,
        "d_gap_ns": 0,
        "d_busy_ns": target + DRAFT_NS,
        "d_target_ns": target,
        "t_exit_ns": t + max(launch, target + DRAFT_NS),
        "launch": {"eager_body": 1.0},
    }


#: The step whose rank-1 host stalls: its launch ten times the others'.
STALL = 17


def _run(tmp_path):
    """Two DP ranks' records of 40 prefills, each launched in 40 ms but step
    `STALL`'s on rank 1, launched in 400 ms while its device ran on past it."""
    paths = []
    for rank in (0, 1):
        lines, t = [], 0
        for step in range(40):
            tokens = 6000 + 997 * step + 13 * rank
            launch = 400 * MS if (step, rank) == (STALL, 1) else 40 * MS
            lines.append(_record_line(step, rank, t, tokens, launch))
            t = lines[-1]["t_exit_ns"] + MS
        path = tmp_path / f"dp{rank}.jsonl"
        path.write_text("".join(json.dumps(x) + "\n" for x in lines))
        _engine_file(path)
        paths.append(path)
    return paths


@pytest.fixture
def record_label(monkeypatch):
    monkeypatch.setenv(step_labels.PREFILL_LABEL_ENV, "record")


@pytest.mark.parametrize("off", [None, "0"], ids=["unset", "0"])
def test_the_switch_cuts_the_stalled_step_on_every_rank_and_off_labels_as_before(
    tmp_path, monkeypatch, record_label, off
):
    paths = _run(tmp_path)
    if off is None:
        monkeypatch.delenv(step_labels.STALL_ENV, raising=False)
    else:
        monkeypatch.setenv(step_labels.STALL_ENV, off)
    before = label_records(paths, dp=2)[0]
    monkeypatch.setenv(step_labels.STALL_ENV, "1")
    after = label_records(paths, dp=2)[0]

    assert not any(s.cut for s in before)
    assert [(s.rank, s.step) for s in after] == [(s.rank, s.step) for s in before]
    for b, a in zip(before, after):
        if a.step == STALL:
            assert a.cut == step_labels.STALLED and a.device_s is None
            assert a.host == b.host and a.host["W_launch"] > 0
        else:
            assert a == b


def test_a_fit_counts_the_stalled_lines_and_without_the_switch_states_no_such_cut(
    tmp_path, monkeypatch, record_label
):
    """One rank's record: its own stall cuts its step; the fit counts it under
    `cut`, and the law fitted without the switch names no such cut and fits it."""
    path = _run(tmp_path)[1]
    monkeypatch.delenv(step_labels.STALL_ENV, raising=False)
    off = fit(label_records([path])[0], {"form": "base"})[0]
    monkeypatch.setenv(step_labels.STALL_ENV, "1")
    on = fit(label_records([path])[0], {"form": "base"})[0]

    assert step_labels.STALLED not in off["cut"] and on["cut"][step_labels.STALLED] == 1
    assert {k: v for k, v in on["cut"].items() if k != step_labels.STALLED} == off[
        "cut"
    ]
    assert on["prefill"]["fit"]["samples"] == off["prefill"]["fit"]["samples"] - 1


def test_a_switch_value_other_than_0_or_1_is_refused(
    tmp_path, monkeypatch, record_label
):
    monkeypatch.setenv(step_labels.STALL_ENV, "yes")

    with pytest.raises(ValueError, match=step_labels.STALL_ENV):
        label_records(_run(tmp_path), dp=2)

# SPDX-License-Identifier: MIT
"""`step_labels`: under `COMPASS_PREFILL_LABEL=record` a prefill whose device
work ends at its host launch's end ran at the launch's pace and is cut as
`paced`, keeping its host labels; one the device kept running past its launch
keeps its window's label, and the prefill vector fitted without the paced
spans prices the device's work, not the launch."""

import json

import pytest
from test_backend_step_linear_fit import _engine_file

from atom.compass.backends import step_labels
from atom.compass.backends.step_linear_fit import fit, label_records

MS = 1_000_000
#: The launch an eager output prefill takes whatever its tokens (#3720 cell B: 55-56 ms).
LAUNCH_NS = 55 * MS
#: The device's own work: an intercept and a slope per token, in ns.
WORK = (20 * MS, 10_000)
#: The drafter's share of a step's device time, constant, in ns.
DRAFT_NS = 2 * MS


def _work(tokens: int) -> int:
    return WORK[0] + WORK[1] * tokens


def _line(step, t, rows, target, launch, gap, mtp=True):
    """A line of an MTP record at host time `t`, where its device starts, idle
    for `gap` ns before it: its target runs `target` ns, its drafter
    `DRAFT_NS`, its launch from `t` for `launch` ns. Without `mtp`, a line of
    a record with no speculative decode: its window is the target's alone."""
    if not mtp:
        line = _line(step, t, rows, target - DRAFT_NS, launch, gap)
        del line["d_target_ns"]
        return line
    return {
        "step": step,
        "dp_rank": 0,
        "is_dummy_run": False,
        "produces_output": True,
        "batch": [[step, q, n] for q, n, _ in rows],
        "named": [],
        "rows": [list(r) for r in rows],
        "rung": None if not rows[0][2] else 1,
        "t_enter_ns": t,
        "t_launch_ns": [t, t + launch],
        "t_recv_ns": [t + target + DRAFT_NS] * 2,
        "d_gap_ns": gap,
        "d_busy_ns": target + DRAFT_NS,
        "d_target_ns": target,
        "t_exit_ns": t + max(launch, target + DRAFT_NS),
    }


def _plain_record(path, prefills, mtp=True):
    """A plain MTP record (without `mtp`, one with no speculative decode), no
    trace run's busy time: one decode step, then each of `prefills`, (tokens,
    paced): a paced one's device runs the launch's span, its work below it;
    another's work, longer than its launch. The device clock, each line's busy
    time and the idle before it, keeps the host's, so the host clock leads it
    by nothing."""
    lines = [_line(0, 0, [(1, 100, True)], 5 * MS, MS, 0, mtp)]
    for i, (tokens, paced) in enumerate(prefills, start=1):
        rows = [(tokens, tokens + 1000 + 7919 * i % 50000, False)]
        work = _work(tokens)
        target = LAUNCH_NS + MS if paced else work
        launch = LAUNCH_NS if paced else min(LAUNCH_NS, work - 30 * MS)
        last = lines[-1]
        t = last["t_exit_ns"] + MS
        idle = t - (last["t_enter_ns"] + last["d_busy_ns"])
        lines.append(_line(i, t, rows, target, launch, idle, mtp))
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    _engine_file(path)
    return path


#: Small prefills the launch paces, then ones the device runs past it.
PREFILLS = [(256 + 37 * i, True) for i in range(30)] + [
    (6000 + 997 * i, False) for i in range(30)
]


@pytest.fixture
def record_label(monkeypatch):
    monkeypatch.setenv(step_labels.PREFILL_LABEL_ENV, "record")


@pytest.mark.parametrize("mtp", [True, False], ids=["mtp", "no-spec"])
def test_a_prefill_ending_at_its_launch_end_is_cut_paced_with_its_host_labels(
    tmp_path, record_label, mtp
):
    """With no speculative decode (Kimi-K3.1 at #752's deployment: no line
    carries `d_target_ns`) a prefill's window is its target's alone: labelled
    from `d_busy_ns`, and cut as paced the same way."""
    record = _plain_record(tmp_path / "dp0.jsonl", PREFILLS, mtp)

    samples = {s.step: s for s in label_records([record])[0]}

    for i, (tokens, paced) in enumerate(PREFILLS, start=1):
        s = samples[i]
        if paced:
            assert s.cut == "paced" and s.device_s is None
            assert s.host["W_launch"] > 0 and s.host["W_pre"] == 0
        else:
            assert s.cut is None
            assert s.device_s == pytest.approx(_work(tokens) * 1e-9)
    assert samples[0].cut is None


@pytest.mark.parametrize("mtp", [True, False], ids=["mtp", "no-spec"])
def test_trace_labels_cut_no_prefill_as_paced(tmp_path, monkeypatch, mtp):
    monkeypatch.delenv(step_labels.PREFILL_LABEL_ENV, raising=False)
    record = _plain_record(tmp_path / "dp0.jsonl", PREFILLS, mtp)

    cuts = {s.cut for s in label_records([record])[0] if s.step}

    assert cuts == {"untraced"}


@pytest.mark.parametrize("mtp", [True, False], ids=["mtp", "no-spec"])
def test_the_prefill_vector_fits_the_device_s_work_not_the_launch(
    tmp_path, record_label, mtp
):
    """Fitted on the paced spans too, the intercept takes up the launch; with
    them cut, the vector prices the work the device ran, and the fit counts
    the paced steps under `cut`."""
    record = _plain_record(tmp_path / "dp0.jsonl", PREFILLS, mtp)
    samples = label_records([record])[0]

    law = fit(samples, {"form": "base"})[0]

    assert "prefill" not in law["unfitted"], law["unfitted"]
    assert law["cut"]["paced"] == sum(paced for _, paced in PREFILLS)
    got = law["prefill"]["coefficients"]
    assert got["step"] == pytest.approx(WORK[0] * 1e-9, rel=1e-3)
    assert got["tokens"] == pytest.approx(WORK[1] * 1e-9, rel=1e-3)

# SPDX-License-Identifier: MIT
"""`step_labels`: the opt-in `PREFILL_EXPOSED_ENV` cut. Under
`COMPASS_PREFILL_LABEL=record` and the switch at N, a windowed prefill line of
at most N prefill tokens of its own whose device started before its host
launch's end (launch-exposed, as a decode's) is cut as `exposed`, keeping its
host labels, where the paced cut misses it (its device ends a few ms past the
launch's end); a larger exposed line, its device running its own work past the
launch, and a line whose device started after its launch had ended keep their
windows. Unset or empty, every label and fit is as before.

No device.
"""

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
#: How far past its launch's end a launch-bound line's device ends: past `PACED_S`.
PAST_NS = 6 * MS


def _work(tokens: int) -> int:
    return WORK[0] + WORK[1] * tokens


def _line(step, t, start, rows, target, launch):
    """An MTP record's prefill line entered and launched at host time `t`, its
    device starting at `start` (the device clock keeps the host's: its own
    tokens' read returns at its target's end) and running its target
    `target` ns and its drafter `DRAFT_NS`."""
    end = start + target + DRAFT_NS
    return {
        "step": step,
        "dp_rank": 0,
        "is_dummy_run": False,
        "produces_output": True,
        "batch": [[step, q, n] for q, n, _ in rows],
        "named": [],
        "rows": [list(r) for r in rows],
        "rung": None,
        "t_enter_ns": t,
        "t_launch_ns": [t, t + launch],
        "t_read_ns": [start + target] * 2,
        "d_busy_ns": target + DRAFT_NS,
        "d_target_ns": target,
        "t_exit_ns": max(t + launch, end),
    }


#: (tokens, kind): `near`, a short line its launch bounds, its device starting
#: at the launch and ending `PAST_NS` past its end; `clean`, a short line whose
#: device starts after its launch's end (a busy device's next step), its work
#: its own; `long`, a line whose device starts at the launch and runs its own
#: work far past it.
PREFILLS = (
    [(300 + 53 * i, "near") for i in range(20)]
    + [(250 + 61 * i, "clean") for i in range(20)]
    + [(6000 + 997 * i, "long") for i in range(20)]
)
#: The switch's bound: past every short line, under every long one.
BOUND = 2048


def _record(path, prefills=PREFILLS):
    lines, device = [], 0
    t = MS
    for i, (tokens, kind) in enumerate(prefills):
        rows = [(tokens, tokens + 1000 + 7919 * i % 50000, False)]
        if kind == "near":
            start, target = t, LAUNCH_NS + PAST_NS
        elif kind == "clean":
            start, target = t + LAUNCH_NS + MS, _work(tokens)
        else:
            start, target = t, _work(tokens)
        line = _line(i, t, start, rows, target, LAUNCH_NS)
        line["d_gap_ns"] = start - device
        device = start + line["d_busy_ns"]
        lines.append(line)
        t = line["t_exit_ns"] + MS
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    _engine_file(path)
    return path


@pytest.fixture
def record_label(monkeypatch):
    monkeypatch.setenv(step_labels.PREFILL_LABEL_ENV, "record")
    monkeypatch.delenv(step_labels.PREFILL_EXPOSED_ENV, raising=False)


def test_the_record_holds_the_three_kinds_as_built(tmp_path, record_label):
    """The fixture's premise: no line is paced; the near and long lines are
    launch-exposed and the clean ones not; unswitched, none is cut."""
    record = _record(tmp_path / "dp0.jsonl")
    lines = step_labels._read(record)
    exposed, _ = step_labels._host_cut(lines)

    assert step_labels._paced(lines) == set()
    assert exposed == {i for i, (_, k) in enumerate(PREFILLS) if k != "clean"}
    assert not any(s.cut for s in label_records([record])[0])


@pytest.mark.parametrize("off", [None, ""], ids=["unset", "empty"])
def test_the_switch_cuts_the_short_exposed_lines_alone_and_off_labels_as_before(
    tmp_path, monkeypatch, record_label, off
):
    record = _record(tmp_path / "dp0.jsonl")
    if off is not None:
        monkeypatch.setenv(step_labels.PREFILL_EXPOSED_ENV, off)
    before = label_records([record])[0]
    monkeypatch.setenv(step_labels.PREFILL_EXPOSED_ENV, str(BOUND))
    after = label_records([record])[0]

    assert [s.step for s in after] == [s.step for s in before]
    for (tokens, kind), b, a in zip(PREFILLS, before, after):
        if kind == "near":
            assert a.cut == "exposed" and a.device_s is None
            assert a.host == b.host and a.host["W_launch"] > 0
        else:
            assert a == b
            assert a.cut is None
    assert before[0].device_s == pytest.approx((LAUNCH_NS + PAST_NS) * 1e-9)


def test_the_bound_counts_a_lines_own_prefill_tokens_inclusive(
    tmp_path, monkeypatch, record_label
):
    """A near line of exactly N tokens is cut and one of N + 1 kept."""
    prefills = [(BOUND, "near"), (BOUND + 1, "near"), *PREFILLS[20:]]
    record = _record(tmp_path / "dp0.jsonl", prefills)
    monkeypatch.setenv(step_labels.PREFILL_EXPOSED_ENV, str(BOUND))

    got = label_records([record])[0]

    assert got[0].cut == "exposed" and got[1].cut is None


def test_the_prefill_vector_fits_the_devices_work_only_with_the_switch(
    tmp_path, monkeypatch, record_label
):
    """Fitted with the launch-bound short lines, the intercept takes up the
    launch; with them cut, the vector prices the work the device ran, and the
    fit counts them under `exposed`; unswitched the law counts no such cut."""
    record = _record(tmp_path / "dp0.jsonl")
    off = fit(label_records([record])[0], {"form": "base"})[0]
    monkeypatch.setenv(step_labels.PREFILL_EXPOSED_ENV, str(BOUND))
    on = fit(label_records([record])[0], {"form": "base"})[0]

    assert off["cut"]["exposed"] == 0
    assert on["cut"]["exposed"] == sum(k == "near" for _, k in PREFILLS)
    assert {k: v for k, v in on["cut"].items() if k != "exposed"} == {
        k: v for k, v in off["cut"].items() if k != "exposed"
    }
    got = on["prefill"]["coefficients"]
    assert got["step"] == pytest.approx(WORK[0] * 1e-9, rel=1e-3)
    assert got["tokens"] == pytest.approx(WORK[1] * 1e-9, rel=1e-3)
    assert off["prefill"]["coefficients"]["step"] > 1.2 * WORK[0] * 1e-9


def test_trace_labels_are_unswitched(tmp_path, monkeypatch):
    """Without `COMPASS_PREFILL_LABEL=record` no prefill line is windowed: the
    switch cuts none, every one stays `untraced`."""
    monkeypatch.delenv(step_labels.PREFILL_LABEL_ENV, raising=False)
    monkeypatch.setenv(step_labels.PREFILL_EXPOSED_ENV, str(BOUND))
    record = _record(tmp_path / "dp0.jsonl")

    assert {s.cut for s in label_records([record])[0]} == {"untraced"}


@pytest.mark.parametrize("value", ["0", "-5", "2k", "2048.0", "yes"])
def test_a_value_other_than_a_positive_whole_number_is_refused(
    tmp_path, monkeypatch, record_label, value
):
    monkeypatch.setenv(step_labels.PREFILL_EXPOSED_ENV, value)

    with pytest.raises(ValueError, match=step_labels.PREFILL_EXPOSED_ENV):
        label_records([_record(tmp_path / "dp0.jsonl")])

# SPDX-License-Identifier: MIT
"""A launch's wait on earlier device work, taken out of its `W_launch` label
under ``COMPASS_FIT_LAUNCH_WAIT=1`` and kept by a run's recurrence.

A record whose lines state no `blocked_ns` times a launch window that holds
whatever its host spent in HIP's blocking calls. Where a kind's launch began
before the device finished the forward before it and outlasted that end
(`step_labels.launch_waits`), the host waited for it there: a host fit under
the switch takes that wait out of the label (`step_labels.wait_labels`) in the
kinds whose launches do so (`step_labels.wait_kinds`) and states them
(`step_labels.WAITS`); a run's `HostDeviceRecurrence` then starts such a
kind's launch no earlier than the device end of the forward before it. Unset,
no label moves, no host entry states the kinds and every launch starts at its
prepare end.

No device.
"""

import dataclasses
import json
from types import SimpleNamespace

import pytest

from atom.compass import run as compass_run
from atom.compass.backends import step_labels
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.fit import host as fit_host
from atom.compass.runner.overrides import HostDeviceRecurrence

MS = 1_000_000
WAITS = step_labels.WAITS


def _line(step, rows, output, launch, device, prev_end, blocked=False):
    """A real line: launch window `launch` (ms, host clock) and device span
    `device` (ms), the host clock and the device's at one lead, 0; the gap
    from the forward before it, ending at `prev_end` (ms), on all but the
    first."""
    line = {
        "step": step,
        "dp_rank": 0,
        "is_dummy_run": False,
        "produces_output": output,
        "rows": [list(r) for r in rows],
        "t_launch_ns": [launch[0] * MS, launch[1] * MS],
        "t_launch_event_ns": launch[0] * MS,
        "d_busy_ns": (device[1] - device[0]) * MS,
    }
    if prev_end is not None:
        line["d_gap_ns"] = (device[0] - prev_end) * MS
    if blocked:
        line["blocked_ns"] = {"hipStreamSynchronize": 1}
    return line


DECODE, PREFILL = ((1, 900, True),), ((500, 9000, False),)


def _lines():
    """A decode, a prefill whose launch began 1 ms in, 9 ms before the decode
    left the device, and ran 190 ms; a decode launched after the prefill left
    the device, one launched while that decode ran and done 0.2 ms later, a
    middle chunk that waited 4 ms, and one stating `blocked_ns`."""
    return [
        _line(0, DECODE, True, (0, 0.2), (0, 10), None),
        _line(1, PREFILL, True, (1, 191), (10, 195), 10),
        _line(2, DECODE, True, (200, 200.2), (200, 210), 195),
        _line(3, DECODE, True, (201, 201.2), (210, 220), 210),
        _line(4, PREFILL, False, (216, 400), (220, 401), 220),
        _line(5, PREFILL, False, (390, 600), (401, 602), 401, blocked=True),
    ]


def test_a_launch_that_outlasts_the_forward_before_it_waited_for_it():
    waits, began, outlasted = step_labels.launch_waits(_lines())
    assert waits == {1: ("prefill", 9 * MS), 4: ("middle", 4 * MS)}
    assert began == {"prefill": 1, "decode": 1, "middle": 1}
    assert outlasted == {"prefill": 1, "middle": 1}
    assert step_labels.wait_kinds(began, outlasted) == ["middle", "prefill"]
    # A decode launched while the forward before it ran ends long before it.
    assert step_labels.wait_kinds({"decode": 4}, {"decode": 1}) == []
    assert step_labels.wait_kinds({"decode": 4}, {"decode": 2}) == ["decode"]


@dataclasses.dataclass(frozen=True)
class _Sample:
    step: int
    rank: int
    view: BatchView
    host: dict
    provenance: dict


def _record(tmp_path):
    steps = tmp_path / "rec" / "steps"
    steps.mkdir(parents=True)
    (steps / "dp0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in _lines()))
    return tmp_path / "rec"


def _samples(record):
    run = step_labels.digest(record / "steps" / "dp0.jsonl")
    got = []
    for s in _lines():
        shapes = tuple(RequestShape(q, c, bool(d)) for q, c, d in s["rows"])
        view = BatchView(shapes, produces_output=s["produces_output"])
        window = (s["t_launch_ns"][1] - s["t_launch_ns"][0]) / 1e9
        host = {"E": 0.001, "W_launch": window}
        got.append(_Sample(s["step"], 0, view, host, {"run": run, "config": {}}))
    return got


def test_a_waiting_kind_s_launch_labels_lose_their_wait(tmp_path):
    record = _record(tmp_path)
    samples = _samples(record)
    out, report, kinds = step_labels.wait_labels(samples, [record])
    assert kinds == ["middle", "prefill"]
    launch = [s.host["W_launch"] for s in out]
    assert launch == pytest.approx([0.0002, 0.181, 0.0002, 0.0002, 0.180, 0.210])
    assert [s.host["E"] for s in out] == [0.001] * 6
    assert report["prefill"] == {
        "began": 1,
        "outlasted": 1,
        "moved": 1,
        "wait_s": 0.009,
    }
    assert report["decode"]["moved"] == 0
    # Another run's samples are not this record's.
    other = [
        dataclasses.replace(s, provenance={"run": "x", "config": {}}) for s in samples
    ]
    assert step_labels.wait_labels(other, [record])[0] == other


@pytest.mark.parametrize("value", [None, "0", "1"])
def test_a_host_fit_takes_the_waits_out_under_the_switch_only(
    tmp_path, monkeypatch, value
):
    record = _record(tmp_path)
    fitted = {"laws": {"prefill": {"W_launch": [0.05, 0, 0]}}}
    seen = []

    def fit(samples):
        seen.append([s.host["W_launch"] for s in samples])
        return dict(fitted), {"unfitted": {}}

    monkeypatch.setattr(fit_host, "fit_host", fit)
    monkeypatch.setattr(fit_host.publish, "runs", lambda samples: {})
    if value is None:
        monkeypatch.delenv(step_labels.WAIT_ENV, raising=False)
    else:
        monkeypatch.setenv(step_labels.WAIT_ENV, value)
    samples = _samples(record)
    host, report = fit_host.host_laws(samples, records=[record])
    windows = [s.host["W_launch"] for s in samples]
    if value == "1":
        assert host == {**fitted, WAITS: ["middle", "prefill"]}
        assert report[WAITS]["middle"]["moved"] == 1
        assert seen[0][1] == pytest.approx(windows[1] - 0.009)
    else:
        assert host == fitted and WAITS not in report
        assert seen[0] == windows


def test_the_switch_takes_0_or_1(monkeypatch):
    monkeypatch.setenv(step_labels.WAIT_ENV, "yes")
    with pytest.raises(ValueError, match="COMPASS_FIT_LAUNCH_WAIT"):
        step_labels.launch_wait_on()


def _batch(prefill, output, tokens):
    return SimpleNamespace(
        total_seqs_num_prefill=int(prefill),
        produces_output=lambda: output,
        total_tokens_num=tokens,
        total_seqs_num=1,
    )


LAWS = {
    k: {
        "E": [0, 0, 0],
        "W_pre": [0, 0, 0],
        "W_launch": [0.18, 0, 0],
        "W_post": [0, 0, 0],
    }
    for k in ("decode", "prefill", "middle")
}


def _ends(waits):
    """A middle chunk of 1 s on the device, then a prefill of 50 ms: each
    forward's device end."""
    clock = [0.0]
    rec = HostDeviceRecurrence(LAWS, (), lambda: clock[0], waits=waits)
    ends = []
    for batch, device in (
        (_batch(True, False, 8192), 1.0),
        (_batch(True, True, 500), 0.05),
    ):
        clock[0] += rec.charge(batch, device, 0.0)
        ends.append(rec.ends[-1])
    return ends


def test_a_waiting_kind_s_launch_starts_at_the_forward_before_it_s_device_end():
    # Unset: the prefill's 0.18 s launch runs beside the middle chunk's device.
    assert _ends(()) == pytest.approx([1.0, 1.05])
    # Waiting: it starts once the middle chunk has left the device.
    assert _ends(["prefill"]) == pytest.approx([1.0, 1.18])
    # A kind not named starts at its prepare end, as before.
    assert _ends(["decode"]) == pytest.approx([1.0, 1.05])


def test_a_run_file_s_host_entry_may_state_the_waiting_kinds(tmp_path):
    host = {"laws": LAWS, WAITS: ["prefill", "middle"]}
    compass_run._check_host(tmp_path / "run.json", host)
    with pytest.raises(ValueError, match="launch_waits"):
        compass_run._check_host(tmp_path / "run.json", {**host, WAITS: ["verify"]})
    with pytest.raises(ValueError, match="launch_waits"):
        compass_run._check_host(tmp_path / "run.json", {**host, WAITS: "prefill"})


# COMPASS_PREFILL_WAITS_PREVIOUS (a6a33bf02) beside this switch: that one takes the
# span from a middle chunk's staging-gate end to its launch start out of its
# W_pre, this one the span from the launch start to the previous forward's
# device end out of its W_launch, so together they take one wait once.


def _pair(prev_end_ms):
    """A decode whose device ends at `prev_end_ms`, then a middle chunk that
    enters at 1 ms, leaves its staging gate at 1.5 ms and launches over
    3-200 ms; both estimates of that end agree (the decode's device started
    at its launch, the lead is 0)."""
    prev = _line(0, DECODE, True, (0, 0.2), (0, prev_end_ms), None)
    start = max(prev_end_ms, 3)
    mid = _line(1, PREFILL, False, (3, 200), (start, start + 190), prev_end_ms)
    mid["t_enter_ns"], mid["t_gate_ns"] = 1 * MS, [1 * MS, int(1.5 * MS)]
    return prev, mid


@pytest.mark.parametrize("prev_end_ms", [10, 2])
def test_with_prefill_waits_previous_a_wait_is_taken_once(prev_end_ms):
    prev, mid = _pair(prev_end_ms)
    host = step_labels._segments(mid, False)
    pre = step_labels.take_prefill_wait(host, prev, mid, "middle")
    taken_pre = host["W_pre"] - pre["W_pre"]
    taken_launch = step_labels.launch_waits([prev, mid])[0].get(1, ("middle", 0))[1]
    # The two spans meet at the launch start: their sum is the one wait, from
    # the gate's end to the previous device end.
    assert taken_pre + taken_launch == (prev_end_ms - 1.5) * MS
    assert taken_launch == max(prev_end_ms - 3, 0) * MS


MIDDLE_PRE = {**LAWS, "middle": {**LAWS["middle"], "W_pre": [0.01, 0, 0]}}


@pytest.mark.parametrize(
    "prefill_waits, launch_waits, end",
    [
        # The first chunk's device runs 0.01-1.01 s, after its 10 ms W_pre.
        ("0", (), 1.06),  # the second's launch runs beside that device
        ("0", ["middle"], 1.19),  # its launch waits for 1.01 s, then 0.18 s
        ("1", (), 1.20),  # its W_pre waits, then 10 ms W_pre and the launch
        ("1", ["middle"], 1.20),  # the same: the one wait bounds both starts
    ],
)
def test_with_prefill_waits_previous_a_run_waits_once(
    monkeypatch, prefill_waits, launch_waits, end
):
    monkeypatch.setenv(step_labels.PREFILL_WAIT_ENV, prefill_waits)
    clock = [0.0]
    rec = HostDeviceRecurrence(MIDDLE_PRE, (), lambda: clock[0], waits=launch_waits)
    for device in (1.0, 0.05):
        clock[0] += rec.charge(_batch(True, False, 8192), device, 0.0)
    assert rec.ends[-1] == pytest.approx(end)

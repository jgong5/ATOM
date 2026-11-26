# SPDX-License-Identifier: MIT
"""A middle prefill chunk waits for the previous forward's device work (opt-in COMPASS_MIDDLE_WAITS_PREVIOUS=1).

ATOM's host blocks, between a middle chunk's staging gate and its launch, until
the previous forward's device work is done: on Qwen3.8-Flash-Next's records a
middle chunk's `W_pre` less that wait was 0.4-0.8 ms while `W_pre` ran from
0.8 ms to 0.55 s. Under the switch the labels take the wait out of `W_pre` and
the simulated host waits for the previous forward's device end instead.
"""

import math

from atom.compass.backends import step_labels
from atom.compass.backends.step_labels import middle_wait_ns
from atom.compass.runner.overrides import HostDeviceRecurrence

MS = 1_000_000


def _line(enter, gate_end, launch, busy):
    return {
        "t_enter_ns": enter,
        "t_gate_ns": (enter, gate_end),
        "t_launch_ns": (launch, launch + 70 * MS),
        "d_busy_ns": busy,
    }


def test_the_wait_is_the_previous_device_end_past_the_gate():
    prev = _line(0, 0, 10 * MS, 600 * MS)  # device until 610 ms
    s = _line(80 * MS, 80 * MS, 611 * MS, 600 * MS)  # launched once it was done
    assert middle_wait_ns(prev, s) == 530 * MS
    early = _line(700 * MS, 700 * MS, 701 * MS, 50 * MS)  # previous work long done
    assert middle_wait_ns(prev, early) == 0
    assert middle_wait_ns(None, s) == 0


def test_the_switch_reads_its_variable(monkeypatch):
    monkeypatch.delenv(step_labels.MIDDLE_WAIT_ENV, raising=False)
    assert not step_labels.middle_waits()
    monkeypatch.setenv(step_labels.MIDDLE_WAIT_ENV, "1")
    assert step_labels.middle_waits()


def _recurrence(now):
    laws = {
        kind: {s: [0.001, 0.0, 0.0] for s in ("E", "W_pre", "W_launch", "W_post")}
        for kind in ("prefill", "middle", "decode")
    }
    r = object.__new__(HostDeviceRecurrence)
    r.laws, r.columns, r.paths, r.skip, r.mtp = laws, {}, {}, frozenset(), False
    r.ends = __import__("collections").deque([-math.inf, 0.5], maxlen=2)
    r.output_end = -math.inf
    return r


def test_a_middle_chunk_starts_its_prepare_after_the_previous_device_end(monkeypatch):
    r = _recurrence(0.0)
    monkeypatch.delenv(step_labels.MIDDLE_WAIT_ENV, raising=False)
    off, _ = r.prepare_end(0.0, "middle", (1, 0, 0))
    monkeypatch.setenv(step_labels.MIDDLE_WAIT_ENV, "1")
    on, _ = r.prepare_end(0.0, "middle", (1, 0, 0))
    decode, _ = r.prepare_end(0.0, "decode", (1, 0, 0))
    assert math.isclose(off, 0.002)  # E then W_pre, the device two back long done
    assert math.isclose(on, 0.501)  # the previous device end, then W_pre
    assert math.isclose(decode, 0.002)  # other kinds keep their own waits

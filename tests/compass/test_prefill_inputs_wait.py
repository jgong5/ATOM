# SPDX-License-Identifier: MIT
"""An MTP output prefill waits for the previous forward's device work (opt-in
COMPASS_PREFILL_INPUTS_WAIT_PREVIOUS=1).

ATOM's host, past an output prefill's MTP status wait, blocks until the previous
forward's device work is done before it launches: behind a middle chunk, which
leaves no MTP status to wait for, that wait lands in `W_inputs`. On
GLM-5.3-Flash's step B record `W_inputs` less that wait was 1.6-5.3 ms while
`W_inputs` ran from 12 ms to 214 ms (medians). Under the switch the labels take
the wait out of `W_inputs` and the simulated host waits for the previous
forward's device end instead.
"""

import collections
import math

from atom.compass.backends import step_labels
from atom.compass.backends.step_labels import inputs_wait_ns
from atom.compass.runner.overrides import HostDeviceRecurrence

MS = 1_000_000


def _line(enter, mtp_end, launch, busy):
    return {
        "t_enter_ns": enter,
        "t_gate_ns": (enter, enter),
        "t_mtp_ns": (enter, mtp_end),
        "t_launch_ns": (launch, launch + 70 * MS),
        "d_busy_ns": busy,
    }


def test_the_wait_is_the_previous_device_end_past_the_mtp_status_wait():
    prev = _line(0, 0, 10 * MS, 300 * MS)  # device until 310 ms
    s = _line(80 * MS, 81 * MS, 313 * MS, 200 * MS)  # launched 3 ms after it
    assert inputs_wait_ns(prev, s) == 229 * MS
    held = _line(80 * MS, 311 * MS, 314 * MS, 200 * MS)  # its MTP status wait held it
    assert inputs_wait_ns(prev, held) == 0
    assert inputs_wait_ns(None, s) == 0
    unstamped = {k: v for k, v in s.items() if k != "t_mtp_ns"}
    assert inputs_wait_ns(prev, unstamped) == 0


def test_the_switch_reads_its_variable(monkeypatch):
    monkeypatch.delenv(step_labels.INPUTS_WAIT_ENV, raising=False)
    assert not step_labels.inputs_waits()
    monkeypatch.setenv(step_labels.INPUTS_WAIT_ENV, "1")
    assert step_labels.inputs_waits()


def _recurrence(mtp):
    host = ("E", "W_pre", "W_launch", "W_post")
    laws = {kind: {s: [0.001, 0.0, 0.0] for s in (*host, "W_inputs")} for kind in ("prefill", "decode")}
    laws["middle"] = {s: [0.001, 0.0, 0.0] for s in host}  # a middle chunk runs no MTP inputs
    r = object.__new__(HostDeviceRecurrence)
    r.laws, r.columns, r.paths, r.skip, r.mtp = laws, {}, {}, frozenset(), mtp
    r.ends = collections.deque([-math.inf, 0.5], maxlen=2)
    r.output_end = -math.inf
    return r


def test_an_output_prefill_launches_after_the_previous_device_end(monkeypatch):
    r = _recurrence(True)
    monkeypatch.delenv(step_labels.INPUTS_WAIT_ENV, raising=False)
    off, _ = r.prepare_end(0.0, "prefill", (1, 0, 0))
    monkeypatch.setenv(step_labels.INPUTS_WAIT_ENV, "1")
    on, _ = r.prepare_end(0.0, "prefill", (1, 0, 0))
    decode, _ = r.prepare_end(0.0, "decode", (1, 0, 0))
    middle, _ = r.prepare_end(0.0, "middle", (1, 0, 0))
    plain, _ = _recurrence(False).prepare_end(0.0, "prefill", (1, 0, 0))
    assert math.isclose(off, 0.003)  # E, W_pre, W_inputs; the device two back long done
    assert math.isclose(on, 0.501)  # the previous device end, then W_inputs
    assert math.isclose(decode, 0.003)  # other kinds keep their own waits
    assert math.isclose(middle, 0.002)  # a middle chunk has no W_inputs
    assert math.isclose(plain, 0.003)  # no MTP: no W_inputs wait to move

# SPDX-License-Identifier: MIT
"""COMPASS_DP_WAKE_S (#734 Cell B): while its group was idle at its last DP
state sync, a DP member back from its idle point waits the wake's seconds
there (next_event, the rounds every member asks alike) before its input
pull, and its next idle point waits as before; a busy group's member pulls
at once; the waits' clocks are handed in. The variable is unset by default
and refuses a negative or non-numeric value. No device."""

import pytest

from atom.compass import run as compass_run
from atom.model_engine.engine_core import DPEngineCoreProc
from atom.utils import clock


class _Clock:
    """An LP clock whose next_event grants an arrival at `arrivals` first."""

    def __init__(self, arrivals=()):
        self.now, self.in_run, self.calls, self.idle_calls = 10.0, True, 0, 0
        self.arrivals, self.asked = list(arrivals), []

    def next_event(self, t):
        self.calls += 1
        self.asked.append(t)
        early = [a for a in self.arrivals if self.now < a < t]
        self.now = min(early) if early else t
        return self.now


def _member(monkeypatch, rt, group_busy):
    """A member under `_wake_before_pull(0.002)`, its sync answering whether
    the group is busy from `group_busy` in turn, its pull a no-op."""
    busy = iter(group_busy)
    pulls = []

    def sync(self, local, shutdown=False, held=False):
        return (next(busy), shutdown, held)

    def pull(self):
        pulls.append(rt.now)
        return False

    monkeypatch.setattr(DPEngineCoreProc, "_sync_dp_state", sync)
    monkeypatch.setattr(DPEngineCoreProc, "pull_and_process_input_queue", pull)
    monkeypatch.setattr(clock, "installed", lambda: rt)
    compass_run._wake_before_pull(0.002)
    return DPEngineCoreProc.__new__(DPEngineCoreProc), pulls


def test_an_idle_group_s_member_waits_the_wake_before_its_pull(monkeypatch):
    # An arrival (a lookup's reply) inside the wait: it is taken, and the
    # wait goes on to its end.
    rt = _Clock(arrivals=[10.0005])
    member, pulls = _member(monkeypatch, rt, [False, True, False])
    member.pull_and_process_input_queue()  # before any sync: idle
    assert rt.asked == [pytest.approx(10.002), pytest.approx(10.002)]
    assert pulls == [pytest.approx(10.002)] and rt.idle_calls == rt.calls
    member._sync_dp_state(False)  # the group is idle: the next pull waits
    member.pull_and_process_input_queue()
    assert pulls[-1] == pytest.approx(10.004)
    member._sync_dp_state(True)  # busy: the next pull does not wait
    member.pull_and_process_input_queue()
    assert pulls[-1] == pytest.approx(10.004) and len(rt.asked) == 3
    assert rt.dp_wakes == [pytest.approx(10.002), pytest.approx(10.004)]
    member._sync_dp_state(False)
    member.pull_and_process_input_queue()
    assert pulls[-1] == pytest.approx(10.006)


def test_the_variable_is_off_by_default_and_refuses_bad_values(monkeypatch):
    monkeypatch.delenv(compass_run.DP_WAKE_ENV, raising=False)
    assert compass_run.dp_wake_s() == 0.0
    monkeypatch.setenv(compass_run.DP_WAKE_ENV, "0.00134")
    assert compass_run.dp_wake_s() == pytest.approx(0.00134)
    for bad in ("-1", "x", "inf"):
        monkeypatch.setenv(compass_run.DP_WAKE_ENV, bad)
        with pytest.raises(ValueError, match="not a non-negative number"):
            compass_run.dp_wake_s()

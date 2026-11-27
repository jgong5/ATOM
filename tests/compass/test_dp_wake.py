# SPDX-License-Identifier: MIT
"""COMPASS_DP_WAKE_S (#734 Cell B): a DP member whose group was idle at its
last DP state sync takes the wake's seconds before the sync that first
carries its work, once a wake, and hands in their clocks; a member with
none, or a group already busy, takes none; the variable is unset by default
and refuses a negative or non-numeric value. No device."""

import pytest

from atom.compass import run as compass_run
from atom.model_engine.engine_core import DPEngineCoreProc
from atom.utils import clock


class _Clock:
    def __init__(self):
        self.now, self.in_run, self.asked = 10.0, True, []

    def advance_to(self, t):
        self.asked.append(t)
        self.now = t


def _member(monkeypatch, peers_busy=()):
    """A member under `_wake_before_sync(0.002)`, its sync answering the
    group's state as MAX over its own and each of `peers_busy` in turn."""
    busy = iter(peers_busy)

    def sync(self, local, shutdown=False, held=False):
        return (local or next(busy, False), shutdown, held)

    monkeypatch.setattr(DPEngineCoreProc, "_sync_dp_state", sync)
    rt = _Clock()
    monkeypatch.setattr(clock, "installed", lambda: rt)
    compass_run._wake_before_sync(0.002)
    return DPEngineCoreProc.__new__(DPEngineCoreProc), rt


def test_a_member_that_wakes_its_idle_group_takes_the_wake_once(monkeypatch):
    member, rt = _member(monkeypatch)
    assert member._sync_dp_state(False) == (False, False, False)
    assert rt.asked == []
    # Its first sync with work after an idle one: the wake, then the sync.
    assert member._sync_dp_state(True)[0] is True
    assert rt.asked == [pytest.approx(10.002)]
    # The group is busy now: no wake until it idles again.
    member._sync_dp_state(True)
    assert len(rt.asked) == 1
    member._sync_dp_state(False)
    member._sync_dp_state(True)
    assert rt.asked[-1] == pytest.approx(10.004) and rt.dp_wakes == rt.asked


def test_a_peer_woken_by_another_member_takes_none(monkeypatch):
    member, rt = _member(monkeypatch, peers_busy=(True, True))
    member._sync_dp_state(False)  # a peer holds work: the group is busy
    member._sync_dp_state(True)
    assert rt.asked == []


def test_the_variable_is_off_by_default_and_refuses_bad_values(monkeypatch):
    monkeypatch.delenv(compass_run.DP_WAKE_ENV, raising=False)
    assert compass_run.dp_wake_s() == 0.0
    monkeypatch.setenv(compass_run.DP_WAKE_ENV, "0.00134")
    assert compass_run.dp_wake_s() == pytest.approx(0.00134)
    for bad in ("-1", "x", "inf"):
        monkeypatch.setenv(compass_run.DP_WAKE_ENV, bad)
        with pytest.raises(ValueError, match="not a non-negative number"):
            compass_run.dp_wake_s()

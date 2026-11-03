# SPDX-License-Identifier: MIT
"""Under MTP, a decode's prepare queues device work ahead of its launch. The
gate feeds it as the record's `d_gap_ns`: a busy launch's device starts that
long after the last forward's device end, and an idle one's at its prepare
end, which covers the idle time the gap also holds."""

import pytest
from test_parity_recurrence import MS, line

from atom.compass.parity import recurrence

#: Milliseconds of device work each decode's prepare queues ahead of its launch.
QUEUED = 0.25


def _decodes(idle=(), n=8):
    """`n` MTP decodes of one request written by the wait rules: 1 ms per host
    segment, 10 ms of target and then 0.5 of draft. Each status wait returns at
    the previous target's end, so each launch finds the draft still running:
    the target starts `QUEUED` after it ends. A forward in `idle` enters 20 ms
    late and finds the device idle: the work ran during its prepare, and the
    target starts at the launch."""
    steps, exit_, end, done = [], -1.0, None, None
    for i in range(n):
        enter = exit_ + 1 + 20 * (i in idle)
        m0 = enter + 1
        m1 = m0 if done is None else max(m0, done)
        start = m1 if end is None else max(end + QUEUED, m1)
        launch = (m1, m1 + 1)
        s = line(i, [[2, 150 + i, True]], True, enter, (enter,) * 2, launch, None, 0)
        s["t_mtp_ns"] = [m0 * MS, m1 * MS]
        s["t_recv_ns"] = [(m1 + 1) * MS] * 2
        s["t_draft_recv_ns"] = [(m1 + 2) * MS] * 2
        exit_ = m1 + 2
        s["t_exit_ns"] = exit_ * MS
        s["d_busy_ns"], s["d_target_ns"] = 10.5 * MS, 10 * MS
        if end is not None:
            s["d_gap_ns"] = round((start - end) * MS)
        done, end = start + 10, start + 10.5
        steps.append(s)
    return steps


def test_a_busy_mtp_launch_starts_its_device_after_the_work_its_prepare_queued():
    """Uncharged, each decode period runs short by `QUEUED` of 10.75 ms."""
    report = recurrence.check(_decodes())
    assert report["kinds"]["decode"][1:4] == pytest.approx((0, 0, 0), abs=1e-9)
    assert report["pass"] and report["total"] == pytest.approx(0, abs=1e-9)


def test_an_idle_mtp_launch_is_not_charged_the_idle_time_its_gap_holds():
    """Step 4's gap of 12.75 ms is mostly the device idle while its host
    entered late; its target starts at its launch."""
    steps = _decodes(idle={4})
    assert steps[4]["d_gap_ns"] == 12.75 * MS
    report = recurrence.check(steps)
    assert report["kinds"]["decode"][1:4] == pytest.approx((0, 0, 0), abs=1e-9)
    assert report["pass"] and report["total"] == pytest.approx(0, abs=1e-9)

# SPDX-License-Identifier: MIT
"""The host's clock drifts from the device's over a record, and a host can
stall between recording a launch's event and stamping the launch: the lead
that places a device event on the host clock follows the drift and leaves
that launch out, so the MTP status copy's label stays the copy's latency and
a host-paced draft is still found."""

import pytest
from test_parity_recurrence import MS, line

from atom.compass.backends.step_labels import ANCHORS, host_paced_drafts, mtp_copy

#: Milliseconds from a target's end to its status wait's return.
COPY = 0.25
#: Nanoseconds the host's clock gains on the device's per forward.
DRIFT = 2_000
#: Milliseconds a stalled host takes from recording a launch's event to its stamp.
STALL = 2.0
#: Forwards that enter 20 ms late and find the device idle.
LATE = {50, 100, 150}


def _decodes(n=200, stall=None, posts=None):
    """`n` MTP decodes of one request written by the wait rules, in ms on the
    device's clock: each enters 1 after the last exit and its status wait
    returns `COPY` after the last target's end, or at once in a `LATE`
    forward. It records its launch's event there and stamps the launch then,
    `STALL` later at step `stall`; the device runs 10 of target once both the
    last forward and that launch are in, then 0.5 of draft. The host enters its
    draft copy wait 2 after the launch, `posts[i]` more, and exits there. Every
    host stamp of forward `i` reads `i * DRIFT` ns more."""
    steps, exit_, end, done = [], -1.0, None, None
    for i in range(n):
        enter = exit_ + 1 + 20 * (i in LATE)
        m0 = enter + 1
        m1 = m0 if done is None else max(m0, done + COPY)
        l0 = m1 + STALL * (i == stall)
        start = m1 if end is None else max(end, m1)
        target = max(start, l0) + 10
        exit_ = l0 + 2 + (posts or {}).get(i, 0)
        rows, gate = [[2, 150 + i, True]], (enter, enter)
        s = line(i, rows, True, enter, gate, (l0, l0 + 1), None, exit_)
        s["t_mtp_ns"] = [m0 * MS, m1 * MS]
        s["t_recv_ns"] = [(l0 + 1) * MS] * 2
        s["t_draft_recv_ns"] = [exit_ * MS] * 2
        s["d_busy_ns"] = round((target + 0.5 - start) * MS)
        s["d_target_ns"] = round((target - start) * MS)
        if end is not None:
            s["d_gap_ns"] = round((start - end) * MS)
        done, end = target, target + 0.5
        drift = i * DRIFT
        for key in [k for k in s if k.startswith("t_")]:
            v = s[key]
            s[key] = [t + drift for t in v] if isinstance(v, list) else v + drift
        steps.append(s)
    return steps


@pytest.mark.parametrize("stall", [None, 120])
def test_the_status_copy_label_follows_the_drift_and_leaves_a_stalled_launch_out(stall):
    """The drift reaches 0.4 ms, past the copy's 0.25, and the stall 2 ms. A
    lead read off the whole record places the early events late, or every
    event late by the stall; carried along the status waits it is off by no
    more than the drift over `ANCHORS` forwards."""
    copy = mtp_copy(_decodes(stall=stall))
    waited = {n: c for n, (c, began) in copy.items() if began}
    assert waited.keys() == set(range(1, 200)) - LATE
    assert waited == {n: pytest.approx(COPY * MS, abs=ANCHORS * DRIFT) for n in waited}


def test_a_host_paced_draft_is_found_by_the_lead_at_its_step():
    """Step 60's host enters its draft copy wait 0.75 ms after its device
    reached the draft, step 70's 0.25 ms before. A lead read off the whole
    record, the stalled launch's at step 120, places step 60's event 1.87 ms
    late and hides it."""
    steps = _decodes(stall=120, posts={60: 9, 70: 8})
    assert host_paced_drafts(steps) == {60}

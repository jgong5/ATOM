# SPDX-License-Identifier: MIT
"""Rounds granted ahead stop at an offered arrival, and so does the ceiling
of each held member reply: a member steps on its own through the rounds
granted ahead of it and no further. (With the Mooncake Store an engine's
request channels are offering, so a request arriving below the ceiling stops
them.)"""

import pytest

from atom.compass.clock import REPORT, TAR, ClockAbort, ClockAuthority, TimelineLog

from .test_member_join import ENGINE, FRONTEND, RANKS, REQ0, REQ1, _dp_table, _own


def _stopped_ahead():
    """dp0 asks rounds 2.0, 2.5 and 3.0, dp1 returns at 2.0 owing the rest,
    and a request at 2.7 on dp1's offering channel waits: 2.5 is granted
    ahead, 3.0 is not."""
    table = _dp_table(("engine", "engine"))
    table.mark_offering(REQ1)
    ca = ClockAuthority(
        table,
        timeline=TimelineLog(),
        members={ENGINE: {rank: _own(rank) for rank in RANKS}},
        ceilings=True,
    )
    log = [(REQ0, 0, 1.5), (REQ0, 1, 2.9), (REQ1, 0, 2.7)]
    assert ca.on_request(FRONTEND, TAR, 3.5, log) == []
    replies = ca.on_request(ENGINE, TAR, 2.0, [], member="dp0", rounds=[2.5, 3.0])
    replies += ca.on_request(ENGINE, TAR, 2.0, [], member="dp1")
    return replies, ca


def test_a_held_reply_ceiling_stops_at_the_first_round_not_granted_ahead():
    replies, ca = _stopped_ahead()
    (dp1,) = [r for r in replies if r[0] == (ENGINE, "dp1")]
    assert dp1[1] == 2.0 and dp1[2].ceiling == 3.0
    assert ca._owing[ENGINE]["dp1"][1] == [2.5]


def test_the_member_steps_the_round_granted_ahead_and_asks_the_next():
    _, ca = _stopped_ahead()
    ca.on_request(ENGINE, TAR, 3.0, [], member="dp1", steps=[(TAR, 2.5, 2.5, 0)])
    assert not ca._owing[ENGINE].get("dp1", [0, []])[1]


def test_a_step_past_the_rounds_granted_ahead_aborts_as_before():
    _, ca = _stopped_ahead()
    steps = [(TAR, 2.5, 2.5, 0), (TAR, 3.0, 3.0, 0)]
    with pytest.raises(ClockAbort, match=r"dp1 stepped on its own to TAR\(3.0\)"):
        ca.on_request(ENGINE, REPORT, 3.0, [], member="dp1", steps=steps)

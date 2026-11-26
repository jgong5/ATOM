# SPDX-License-Identifier: MIT
"""A local step below a grant's ceiling releases the messages sent ahead that
it reaches, each at its arrival, and removes them from the sent-ahead list in
place; a step that reaches none leaves that list as it was."""

from .test_lp_runtime import ENGINE, INF, REQ, _receiver, _runtime, _under

AHEAD = {REQ: [(0, 10.1), (1, 10.3), (2, 10.3), (3, 10.6)]}


def _granted_ahead(n, *replies):
    """An engine at 10.0 whose grant sent `AHEAD` ahead below its ceiling 10.8,
    with a handler that takes `n` of them, recording the clock it sees."""
    rt, call = _runtime(ENGINE, (10.0, _under(10.8, AHEAD)), *replies)
    seen = []
    handler = _receiver(rt, REQ, n, seen)
    call(rt.next_event, INF)
    return rt, call, seen, handler


def test_a_step_that_reaches_none_releases_nothing_and_keeps_the_list():
    rt, call, seen, handler = _granted_ahead(0)
    kept, entries = rt.ahead, list(rt.ahead)
    call(rt.advance_to, 10.05)
    assert seen == [] and rt.now == 10.05
    assert rt.released[REQ] == set()
    assert rt.ahead is kept and kept == entries
    assert rt.steps == [("TAR", 10.05, 10.05, 0)]
    handler.join(5)


def test_steps_that_reach_some_release_them_at_their_arrivals_and_keep_the_rest():
    rt, call, seen, handler = _granted_ahead(3)
    kept = rt.ahead
    call(rt.advance_to, 10.1)  # exactly the first arrival
    assert seen == [(REQ, 0, 10.1)] and rt.now == 10.1
    call(rt.advance_to, 10.3)  # exactly the next two
    assert seen[1:] == [(REQ, 1, 10.3), (REQ, 2, 10.3)] and rt.now == 10.3
    assert rt.ahead is kept and kept == [(10.6, REQ, 3)]
    assert [s[2] for s in rt.steps] == [10.1, 10.3]
    handler.join(5)
    assert not handler.is_alive()


def test_a_step_that_reaches_all_releases_each_once_and_empties_the_list():
    rt, call, seen, handler = _granted_ahead(4, (INF, {}))
    kept = rt.ahead
    call(rt.advance_to, 10.7)
    assert seen == [(REQ, 0, 10.1), (REQ, 1, 10.3), (REQ, 2, 10.3), (REQ, 3, 10.6)]
    assert rt.now == 10.7 and rt.ahead is kept and kept == []
    handler.join(5)
    assert not handler.is_alive()
    # Past the list, the next call goes to the authority and carries the step.
    assert call(rt.next_event, INF) == INF
    assert rt.conn.sent[-1] == ("NER", INF, [], INF, [("TAR", 10.7, 10.7, 0)], True)

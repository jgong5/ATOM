"""A closed loop that quietly serialises a session is not a closed loop.

Three ways this mode can look right and be wrong, and all of them are silent.

It can send a session's requests one at a time -- every request still
completes, the artifact still says `clients: 16`, and the measured concurrency
is a fraction of what the recording had, because 43.5% of the corpus's requests
overlap another request of their own session. Or it can send them all at once,
which invents concurrency the recording never had: only 8.4% of sub-agent
window pairs actually overlap.

Or it can drop the think time. Between a turn coming back and the next one
going out sits a user reading and typing, and on this corpus that gap is most
of the session -- 32 sessions whose requests total a few hours of engine work
span 98,944 seconds end to end. A run that drops it still finishes, still
reports every request, and has replaced an agentic workload with a hammer
benchmark.

So the tests below pin the *shape* of what gets issued, not just that it
finishes: which requests wait for which, how long they then wait, and which
slot runs them.
"""
import importlib.util
import threading
import time
from pathlib import Path

import pytest


def _module():
    spec = importlib.util.spec_from_file_location(
        "replay_closed_mod",
        Path(__file__).resolve().parents[2] / "scripts/compass/replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(session, arrival_s, api_time_s, tokens=640):
    return {"session": session, "arrival_s": float(arrival_s),
            "api_time_s": float(api_time_s), "input_tokens": tokens,
            "output_tokens": 16}


def _ok(i):
    return {"index": i, "ok": True, "response": {}}


class TestTheRecordedOverlapIsWhatIsReproduced:
    def test_requests_that_did_not_overlap_are_sequential(self):
        mod = _module()
        rows = [_row(0, 0.0, 1.0), _row(0, 1.0, 1.0), _row(0, 2.0, 1.0)]
        deps, _ = mod._session_plan(rows)
        assert deps == [[], [0], [0, 1]]

    def test_requests_that_did_overlap_are_concurrent(self):
        mod = _module()
        # A turn at t=0 running 10s, with two sub-agents fired inside it.
        rows = [_row(0, 0.0, 10.0), _row(0, 1.0, 2.0), _row(0, 1.5, 2.0)]
        deps, _ = mod._session_plan(rows)
        assert deps[1] == [] and deps[2] == []

    def test_a_sub_agent_that_finished_first_still_blocks_the_next_turn(self):
        mod = _module()
        rows = [_row(0, 0.0, 1.0), _row(0, 0.2, 0.3), _row(0, 2.0, 1.0)]
        # Row 2 starts after both, so it waits for both -- including the
        # sub-agent, which is the whole point of reading the windows rather
        # than the nesting.
        assert mod._session_plan(rows)[0][2] == [0, 1]

    def test_a_row_with_no_recorded_duration_is_treated_as_instantaneous(self):
        mod = _module()
        # api_time_s absent -> zero-length window, ending where it starts. The
        # conservative reading: it blocks what starts after it and overlaps
        # nothing. The alternative -- an unbounded window -- would make one
        # missing field serialise the rest of the session.
        rows = [{"session": 0, "arrival_s": 0.0}, _row(0, 0.0, 1.0),
                _row(0, -1.0, 3.0)]
        deps, _ = mod._session_plan(rows)
        assert deps[1] == [0]      # starts at its end, so waits for it
        assert deps[0] == []       # the row still running at t=0 does not block it


class TestTheThinkTimeIsKept:
    def test_the_recorded_gap_between_turns_survives(self):
        mod = _module()
        # Turn one runs 0->1, the user reads for 4s, turn two goes out at 5.
        rows = [_row(0, 0.0, 1.0), _row(0, 5.0, 1.0)]
        deps, think = mod._session_plan(rows)
        assert deps[1] == [0]
        assert think == [0.0, 4.0]

    def test_a_gap_is_measured_from_the_last_predecessor_not_the_first(self):
        mod = _module()
        # A turn and a sub-agent, the sub-agent finishing last. The user could
        # not have read the answer before the slower branch returned.
        rows = [_row(0, 0.0, 1.0), _row(0, 0.2, 3.0), _row(0, 5.0, 1.0)]
        deps, think = mod._session_plan(rows)
        assert deps[2] == [0, 1]
        assert think[2] == pytest.approx(1.8)   # 5.0 - 3.2, not 5.0 - 1.0

    def test_a_branch_already_open_measures_from_when_the_session_opened(self):
        mod = _module()
        # Both rows wait for nothing, so there is no predecessor to measure
        # from. The session opening is the only shared origin they have.
        rows = [_row(0, 10.0, 5.0), _row(0, 12.0, 1.0)]
        deps, think = mod._session_plan(rows)
        assert deps == [[], []]
        assert think == [0.0, 2.0]

    def test_a_row_that_overlapped_its_predecessor_does_not_wait_for_it(self):
        mod = _module()
        # It went out a millisecond before the previous turn came back, so it
        # never waited on it and never thought about it. Treating the overlap
        # as a dependency would insert think time that was never spent.
        rows = [_row(0, 0.0, 1.0), _row(0, 0.999, 1.0)]
        deps, think = mod._session_plan(rows)
        assert deps[1] == []
        assert think == [0.0, 0.999]   # measured from the session opening


class TestSessionsAreNotSplitAcrossClients:
    def test_rows_group_by_session_in_first_arrival_order(self):
        mod = _module()
        workload = [_row(7, 5.0, 1.0), _row(3, 1.0, 1.0), _row(7, 6.0, 1.0),
                    _row(3, 2.0, 1.0)]
        assert mod._sessions(workload) == [[1, 3], [0, 2]]

    def test_every_request_of_a_session_runs_on_one_client(self):
        mod = _module()
        workload = [_row(s, i, 1.0) for s in range(4) for i in range(3)]
        groups, assignment, _, deps, think, selected = mod._dag(workload, 2, 0)
        results = mod._closed_loop(workload, groups, assignment, deps, think, _ok)
        assert len(results) == len(workload)

        # Each client's work is whole sessions, no session is dealt twice, and
        # between them they cover the trace.
        by_client = {}
        for c, slot in enumerate(assignment):
            for g in slot:
                sessions = {workload[i]["session"] for i in groups[g]}
                assert len(sessions) == 1
                by_client.setdefault(sessions.pop(), []).append(c)
        assert sorted(by_client) == [0, 1, 2, 3]
        assert all(len(v) == 1 for v in by_client.values())
        assert selected == list(range(len(workload)))


class TestTheDealIsTheSameOnBothSides:
    """The two sides run the same pool; if they deal it differently they are
    two experiments, and the paired report cannot tell that from model error."""

    def test_the_longest_session_is_dealt_first(self):
        mod = _module()
        # One long session and three short ones over two slots. Round-robin
        # would put the long one with a short one and leave the other slot
        # idle for most of the run; longest-first does not.
        workload = ([_row(0, 0.0, 10.0)]
                    + [_row(s, s, 1.0) for s in range(1, 4)])
        groups = mod._sessions(workload)
        assignment, spans = mod._balanced_deal(groups, workload, 2)
        assert spans == [10.0, 1.0, 1.0, 1.0]
        assert assignment == [[0], [1, 2, 3]]

    def test_the_deal_does_not_depend_on_anything_but_the_trace(self):
        mod = _module()
        workload = [_row(s, s, 1.0 + s) for s in range(9)]
        first = mod._dag(workload, 3, 0)[1]
        assert first == mod._dag(workload, 3, 0)[1]

    def test_assignment_does_not_depend_on_how_fast_requests_return(self):
        mod = _module()
        workload = [_row(s, s, 1.0) for s in range(9)]

        def run(delay_of):
            def send(i):
                time.sleep(delay_of(i))
                return _ok(i)
            groups, assignment, _, deps, think = mod._dag(workload, 3, 0)[:5]
            got = mod._closed_loop(workload, groups, assignment, deps, think,
                                   send)
            return assignment, sorted(r["index"] for r in got)

        # Slow sessions on one side, fast on the other. A shared work queue
        # would deal different sessions to different clients; dealing up front
        # must not.
        assert run(lambda i: 0.0) == run(lambda i: 0.02 if i % 3 else 0.0)

    def test_sessions_per_client_bounds_the_work_evenly(self):
        mod = _module()
        workload = [_row(s, s, 1.0) for s in range(20)]
        assignment = mod._dag(workload, 4, 2)[1]
        assert [len(a) for a in assignment] == [2, 2, 2, 2]

    def test_the_digest_separates_two_different_deals(self):
        mod = _module()
        workload = [_row(s, s, 1.0 + s) for s in range(6)]

        def digest(clients):
            _, assignment, _, deps, think, selected = mod._dag(
                workload, clients, 0)
            return mod._dag_digest(assignment, deps, think, selected)

        assert digest(2) == digest(2)
        assert digest(2) != digest(3)


class TestASlotTakesTheNextSessionWhenItIsDone:
    def test_the_next_sessions_opening_requests_wait_for_the_whole_previous_one(self):
        mod = _module()
        workload = [_row(0, 0.0, 1.0), _row(0, 2.0, 1.0), _row(1, 10.0, 1.0)]
        _, assignment, _, deps, think, _ = mod._dag(workload, 1, 0)
        assert assignment == [[0, 1]]
        # Not just the last row of session 0 -- all of them. A sub-agent that
        # started late can still be running when the main turn has returned,
        # and the slot is not free until it is not.
        assert deps[2] == [0, 1]
        # And no think time on that edge: the gap between two sessions in the
        # trace is a gap between two different users.
        assert think[2] == 0.0

    def test_the_first_session_of_a_slot_waits_for_nothing(self):
        mod = _module()
        workload = [_row(0, 0.0, 1.0), _row(1, 10.0, 1.0)]
        _, _, _, deps, think, _ = mod._dag(workload, 2, 0)
        assert deps[0] == [] and deps[1] == []
        assert think[0] == 0.0 and think[1] == 0.0

    def test_the_between_session_gap_is_not_slept(self):
        mod = _module()
        # Two sessions 100s apart in the recording, one slot. The recorded gap
        # belongs to a different user, so a slot must pick the next session up
        # immediately -- otherwise a sweep over client counts measures the
        # trace's own idle time instead of the engine's throughput.
        workload = [_row(0, 0.0, 0.0), _row(1, 100.0, 0.0)]
        groups, assignment, _, deps, think, _ = mod._dag(workload, 1, 0)
        began = time.monotonic()
        mod._closed_loop(workload, groups, assignment, deps, think, _ok)
        assert time.monotonic() - began < 1.0


class TestTheThinkTimeIsReallySlept:
    def test_the_paced_executor_waits_the_recorded_gap(self):
        mod = _module()
        workload = [_row(0, 0.0, 0.0), _row(0, 0.05, 0.0)]
        groups, assignment, _, deps, think, _ = mod._dag(workload, 1, 0)
        assert think[1] == 0.05
        began = time.monotonic()
        mod._closed_loop(workload, groups, assignment, deps, think, _ok)
        # An executor that declared the gap and then did not sleep it would
        # come back instantly, and the real side would be a burst.
        assert time.monotonic() - began >= 0.045


class TestConcurrencyIsBoundedByTheClients:
    def test_one_client_never_has_two_sessions_open(self):
        mod = _module()
        # Three sessions, each a single long request. With one client they
        # must not overlap; a bug that spawns per session rather than per
        # client would let them.
        workload = [_row(s, 0.0, 1.0) for s in range(3)]
        live = 0
        peak = 0
        lock = threading.Lock()

        def send(i):
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(0.02)
            with lock:
                live -= 1
            return _ok(i)

        groups, assignment, _, deps, think, _ = mod._dag(workload, 1, 0)
        mod._closed_loop(workload, groups, assignment, deps, think, send)
        assert peak == 1

    def test_in_session_overlap_really_happens_at_the_socket(self):
        mod = _module()
        # One turn overlapped by two sub-agents, fired a millisecond apart. If
        # the loop serialises the session, peak is 1 and the run measures a
        # workload the corpus does not contain.
        workload = [_row(0, 0.0, 10.0), _row(0, 0.001, 2.0),
                    _row(0, 0.002, 2.0)]
        live = 0
        peak = 0
        lock = threading.Lock()

        def send(i):
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(0.05)
            with lock:
                live -= 1
            return _ok(i)

        groups, assignment, _, deps, think, _ = mod._dag(workload, 1, 0)
        mod._closed_loop(workload, groups, assignment, deps, think, send)
        assert peak == 3

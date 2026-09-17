"""A closed loop that quietly serialises a session is not a closed loop.

Two ways this mode can look right and be wrong, and both are silent. It can
send a session's requests one at a time -- every request still completes, the
artifact still says `clients: 16`, and the measured concurrency is a fraction
of what the recording had, because 43.5% of the corpus's requests overlap
another request of their own session. Or it can send them all at once, which
invents concurrency the recording never had: only 8.4% of sub-agent window
pairs actually overlap.

So the tests below pin the *shape* of what gets issued, not just that it
finishes.
"""
import importlib.util
import threading
import time
from pathlib import Path


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


class TestTheRecordedOverlapIsWhatIsReproduced:
    def test_requests_that_did_not_overlap_are_sequential(self):
        mod = _module()
        rows = [_row(0, 0.0, 1.0), _row(0, 1.0, 1.0), _row(0, 2.0, 1.0)]
        assert mod._dependencies(rows) == [[], [0], [0, 1]]

    def test_requests_that_did_overlap_are_concurrent(self):
        mod = _module()
        # A turn at t=0 running 10s, with two sub-agents fired inside it.
        rows = [_row(0, 0.0, 10.0), _row(0, 1.0, 2.0), _row(0, 1.5, 2.0)]
        deps = mod._dependencies(rows)
        assert deps[1] == [] and deps[2] == []

    def test_a_sub_agent_that_finished_first_still_blocks_the_next_turn(self):
        mod = _module()
        rows = [_row(0, 0.0, 1.0), _row(0, 0.2, 0.3), _row(0, 2.0, 1.0)]
        # Row 2 starts after both, so it waits for both -- including the
        # sub-agent, which is the whole point of reading the windows rather
        # than the nesting.
        assert mod._dependencies(rows)[2] == [0, 1]

    def test_a_row_with_no_recorded_duration_is_treated_as_instantaneous(self):
        mod = _module()
        # api_time_s absent -> zero-length window, ending where it starts. The
        # conservative reading: it blocks what starts after it and overlaps
        # nothing. The alternative -- an unbounded window -- would make one
        # missing field serialise the rest of the session.
        rows = [{"session": 0, "arrival_s": 0.0}, _row(0, 0.0, 1.0),
                _row(0, -1.0, 3.0)]
        deps = mod._dependencies(rows)
        assert deps[1] == [0]      # starts at its end, so waits for it
        assert deps[0] == []       # the row still running at t=0 does not block it


class TestSessionsAreNotSplitAcrossClients:
    def test_rows_group_by_session_in_first_arrival_order(self):
        mod = _module()
        workload = [_row(7, 5.0, 1.0), _row(3, 1.0, 1.0), _row(7, 6.0, 1.0),
                    _row(3, 2.0, 1.0)]
        assert mod._sessions(workload) == [[1, 3], [0, 2]]

    def test_every_request_of_a_session_runs_on_one_client(self):
        mod = _module()
        workload = [_row(s, i, 1.0) for s in range(4) for i in range(3)]
        results, assignment = mod._closed_loop(
            workload, 2, 0, lambda i: {"index": i, "ok": True, "response": {}})
        assert len(results) == len(workload)

        # Each client's work is whole sessions, no session is dealt twice, and
        # between them they cover the trace.
        by_client = {}
        for c, groups in enumerate(assignment):
            for idxs in groups:
                sessions = {workload[i]["session"] for i in idxs}
                assert len(sessions) == 1
                by_client.setdefault(sessions.pop(), []).append(c)
        assert sorted(by_client) == [0, 1, 2, 3]
        assert all(len(v) == 1 for v in by_client.values())
        covered = sorted(i for a in assignment for g in a for i in g)
        assert covered == list(range(len(workload)))


class TestBothSidesRunTheSameSessionsInTheSameOrder:
    def test_assignment_does_not_depend_on_how_fast_requests_return(self):
        mod = _module()
        workload = [_row(s, 0.0, 1.0) for s in range(9)]

        def run(delay_of):
            def send(i):
                time.sleep(delay_of(i))
                return {"index": i, "ok": True, "response": {}}
            return mod._closed_loop(workload, 3, 0, send)[1]

        # Slow sessions on one side, fast on the other. A shared work queue
        # would deal different sessions to different clients; dealing up front
        # must not.
        assert run(lambda i: 0.0) == run(lambda i: 0.02 if i % 3 else 0.0)

    def test_sessions_per_client_bounds_the_work_evenly(self):
        mod = _module()
        workload = [_row(s, 0.0, 1.0) for s in range(20)]
        _, assignment = mod._closed_loop(
            workload, 4, 2, lambda i: {"index": i, "ok": True, "response": {}})
        assert [len(a) for a in assignment] == [2, 2, 2, 2]


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
            return {"index": i, "ok": True, "response": {}}

        mod._closed_loop(workload, 1, 0, send)
        assert peak == 1

    def test_in_session_overlap_really_happens_at_the_socket(self):
        mod = _module()
        # One turn overlapped by two sub-agents. If the loop serialises the
        # session, peak is 1 and the run measures a workload the corpus does
        # not contain.
        workload = [_row(0, 0.0, 10.0), _row(0, 1.0, 2.0), _row(0, 1.5, 2.0)]
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
            return {"index": i, "ok": True, "response": {}}

        mod._closed_loop(workload, 1, 0, send)
        assert peak == 3

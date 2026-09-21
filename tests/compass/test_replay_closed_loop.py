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
import random
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


def _stream_row(session, stream, arrival_s, api_time_s, tokens=640):
    row = _row(session, arrival_s, api_time_s, tokens)
    row["stream"] = stream
    return row


def _ok(e):
    return {"index": e["eid"], "row": e["row"], "ok": True, "response": {}}


def _run(mod, workload, *, clients, duration=0.0, instances=0, send=None,
         startup=False, seed=0):
    """A recycling run with no server behind it. Returns `(plan, results)`."""
    out: dict = {}
    groups = mod._sessions(workload)
    plan, _, _ = mod._recycle(
        workload, groups, clients=clients, duration=duration,
        instances=instances or (0 if duration else 1),
        sampler=mod._Sampler(len(groups), seed, "sequential"),
        rng=random.Random(seed), guard=mod._IdleGuard(0.0),
        send=send or _ok, out=out, startup=startup)
    return plan, out


def _peak(mod, workload, *, clients, instances):
    """Highest number of requests on the wire at once."""
    live = peak = 0
    lock = threading.Lock()

    def send(e):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.05)
        with lock:
            live -= 1
        return _ok(e)

    _run(mod, workload, clients=clients, instances=instances, send=send)
    return peak


class TestTheRecordedOverlapIsWhatIsReproduced:
    def test_requests_that_did_not_overlap_are_sequential(self):
        mod = _module()
        rows = [_row(0, 0.0, 1.0), _row(0, 1.0, 1.0), _row(0, 2.0, 1.0)]
        deps, _ = mod._session_plan(rows)
        # Row 2 waits on row 1 only: row 1 already waits on row 0, so the
        # second edge states nothing the first does not. The release instant is
        # `max(ends)` either way -- pruning changes the declared payload and
        # not the schedule.
        assert deps == [[], [0], [1]]

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
        # api_time_s absent -> zero-length window, ending where it starts, so
        # one missing field cannot serialise the rest of the session by
        # implying an unbounded window.
        #
        # It does not block a row starting at that same instant. Golden wants
        # the predecessor to have started strictly before, not merely to have
        # finished by then, and a zero-width interval at t never satisfies that
        # for a row at t.
        rows = [{"session": 0, "arrival_s": 0.0}, _row(0, 0.0, 1.0),
                _row(0, -1.0, 3.0)]
        deps, _ = mod._session_plan(rows)
        assert deps[1] == []
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

    def test_every_request_of_an_instance_comes_from_one_session(self):
        mod = _module()
        workload = [_row(s, i, 1.0) for s in range(4) for i in range(3)]
        plan, out = _run(mod, workload, clients=2, instances=2)
        assert len(out) == len(plan)
        for key in {(e["lane"], e["instance"]) for e in plan}:
            rows = [e for e in plan if (e["lane"], e["instance"]) == key]
            assert len({workload[e["row"]]["session"] for e in rows}) == 1


class TestALaneRecyclesUntilTheClockSaysStop:
    """The pool's size must not decide the run's length.

    Dealing a fixed slice of sessions to slots made a small trace produce a
    short run with idle lanes -- c16 on a 16-session trace realised 9.1 lanes
    of 16, which was then reported as a fact about the workload rather than as
    an artifact of the deal. Bounded by the clock, a small pool is simply
    replayed more times.
    """

    def test_one_session_is_replayed_many_times(self):
        mod = _module()
        workload = [_row(0, 0.0, 0.0), _row(0, 0.0, 0.0)]
        plan, _ = _run(mod, workload, clients=1, duration=0.3)
        assert len({e["instance"] for e in plan}) > 1

    def test_the_pool_does_not_bound_the_run(self):
        mod = _module()
        workload = [_row(s, 0.0, 0.0) for s in range(2)]
        short, _ = _run(mod, workload, clients=1, duration=0.15)
        long, _ = _run(mod, workload, clients=1, duration=0.6)
        assert len(long) > len(short)

    def test_every_lane_keeps_working_for_the_whole_window(self):
        mod = _module()
        # Four lanes, one session in the pool. A fixed deal would leave three
        # of them with nothing.
        workload = [_row(0, 0.0, 0.0)]
        plan, _ = _run(mod, workload, clients=4, duration=0.3)
        assert {e["lane"] for e in plan} == {0, 1, 2, 3}
        assert all(len({e["instance"] for e in plan if e["lane"] == c}) > 1
                   for c in range(4))

    def test_the_sampler_is_seeded(self):
        mod = _module()
        workload = [_row(s, 0.0, 0.0) for s in range(6)]

        def drawn():
            sampler = mod._Sampler(6, 7, "shuffle")
            return [sampler.draw() for _ in range(12)]

        assert drawn() == drawn()
        assert sorted(drawn()[:6]) == list(range(6))


class TestStartupSamplingAndWarmup:
    """A lane's first session is already in progress, and its prefix is warm.

    Starting every session at turn 0 measures a server whose every session is
    cold -- maximum prefill, minimum reuse -- which is not the steady state the
    throughput number is supposed to describe.
    """

    def test_a_session_joined_midway_skips_the_turns_before_it(self):
        mod = _module()
        rows = [_row(0, float(i), 0.1) for i in range(10)]
        warm, profiled = mod._tstar_split(rows, 0.5)
        assert profiled == [5, 6, 7, 8, 9]
        assert warm == 4

    def test_the_turn_before_it_is_sent_unmeasured(self):
        mod = _module()
        workload = [_row(0, float(i), 0.0) for i in range(10)]
        plan = mod._instance(workload, list(range(10)), eid0=0, lane=0,
                             instance=0, session=0, marker="abc", ratio=0.5)
        warm = [e for e in plan if e["phase"] == "warmup"]
        assert len(warm) == 1 and warm[0]["row"] == 4
        assert [e["row"] for e in plan if e["phase"] == "profile"] == [5, 6, 7, 8, 9]

    def test_the_first_profiled_turn_starts_at_the_boundary(self):
        mod = _module()
        # A 30s gap between the warmup turn and the first profiled one. Sleeping
        # it would put the lane's t* 30s past every other lane's, and the
        # boundary would stop being a boundary.
        workload = [_row(0, 0.0, 1.0), _row(0, 30.0, 1.0)]
        plan = mod._instance(workload, [0, 1], eid0=0, lane=0, instance=0,
                             session=0, marker="abc", ratio=1.0)
        first = [e for e in plan if e["phase"] == "profile"][0]
        assert first["think_s"] == 0.0
        assert first["think_recorded_s"] == 29.0

    def test_a_recycled_session_replays_from_turn_zero(self):
        mod = _module()
        workload = [_row(0, float(i), 0.0) for i in range(6)]
        plan = mod._instance(workload, list(range(6)), eid0=0, lane=0,
                             instance=3, session=0, marker="abc", ratio=0.0)
        assert not [e for e in plan if e["phase"] == "warmup"]
        assert [e["row"] for e in plan] == [0, 1, 2, 3, 4, 5]

    def test_a_failed_root_warmup_refuses_the_run(self):
        mod = _module()
        workload = [_row(0, float(i), 0.0) for i in range(6)]

        def send(e):
            if e["phase"] == "warmup":
                return {"index": e["eid"], "row": e["row"], "ok": False,
                        "error": "TimeoutError: timed out"}
            return _ok(e)

        with pytest.raises(SystemExit) as exc:
            mod._recycle(workload, mod._sessions(workload), clients=1,
                         duration=0.0, instances=1,
                         sampler=mod._Sampler(1, 0, "sequential"),
                         rng=random.Random(0), guard=mod._IdleGuard(0.0),
                         send=send, out={}, startup=True)
        assert "warmup" in str(exc.value)


class TestTheCacheBustMarker:
    """A recycled session must not find the previous instance's blocks warm.

    Without a marker the second instance of a trace prefills almost nothing, so
    the run reports reuse the recording never had and a throughput figure to
    match.
    """

    def test_two_instances_do_not_share_a_leading_block(self):
        from atom.compass.workload import MARKER_TOKENS, prompt_of_hash_ids

        one = prompt_of_hash_ids([1, 2], 128, marker="00000000000a").split()
        two = prompt_of_hash_ids([1, 2], 128, marker="00000000000b").split()
        assert one[:MARKER_TOKENS] != two[:MARKER_TOKENS]

    def test_the_marker_is_one_native_block(self):
        from atom.compass.workload import MARKER_TOKENS
        # 64-token source blocks stay aligned to the engine's 16-token blocks
        # only if the shift is a multiple of 16.
        assert MARKER_TOKENS == 16 and 64 % MARKER_TOKENS == 0

    def test_the_marker_does_not_change_the_length(self):
        from atom.compass.workload import prompt_of_hash_ids

        for tokens in (64, 128, 1024):
            ids = list(range(1, tokens // 64 + 1))
            plain = prompt_of_hash_ids(ids, tokens)
            marked = prompt_of_hash_ids(ids, tokens, marker="0123456789ab")
            assert len(plain.split()) == len(marked.split()) == tokens

    def test_turns_of_one_instance_still_share(self):
        from atom.compass.workload import prompt_of_hash_ids

        first = prompt_of_hash_ids([1, 2], 128, marker="0123456789ab").split()
        second = prompt_of_hash_ids([1, 2, 3], 192, marker="0123456789ab").split()
        assert second[:128] == first

    def test_every_turn_of_an_instance_carries_the_same_marker(self):
        mod = _module()
        workload = [_row(0, float(i), 0.0) for i in range(4)]
        plan, _ = _run(mod, workload, clients=1, instances=2)
        for key in {(e["lane"], e["instance"]) for e in plan}:
            markers = {e["marker"] for e in plan
                       if (e["lane"], e["instance"]) == key}
            assert len(markers) == 1
        assert len({e["marker"] for e in plan}) == len(
            {(e["lane"], e["instance"]) for e in plan})


class TestThePerStreamSpine:
    """Within a chain, turn k+1 is a reply to turn k.

    The earlier rule read only the recorded windows, session-wide. On a
    recording where two consecutive root turns overlap by a hair -- different
    clocks, or a retry -- it replayed a conversation as a fan-out, putting two
    turns of the same chain in the server at once. The token counts are
    unchanged, so nothing in the aggregate shows it.
    """

    def test_a_chain_is_sequential_even_where_the_recording_overlaps(self):
        mod = _module()
        rows = [_stream_row(0, 0, 0.0, 1.0), _stream_row(0, 0, 0.999, 1.0)]
        deps, think = mod._session_plan(rows)
        assert deps[1] == [0]
        assert think[1] == 0.0      # it never actually waited, so it does not

    def test_two_chains_that_overlap_are_concurrent(self):
        mod = _module()
        rows = [_stream_row(0, 0, 0.0, 10.0), _stream_row(0, 1, 1.0, 2.0),
                _stream_row(0, 2, 1.5, 2.0)]
        deps, _ = mod._session_plan(rows)
        assert deps[1] == [] and deps[2] == []

    def test_a_chain_waits_for_another_chains_completed_request(self):
        mod = _module()
        rows = [_stream_row(0, 0, 0.0, 1.0), _stream_row(0, 1, 0.2, 0.3),
                _stream_row(0, 0, 2.0, 1.0)]
        deps, think = mod._session_plan(rows)
        assert deps[2] == [0, 1]
        assert think[2] == pytest.approx(1.0)

    def test_only_the_latest_of_a_chain_is_kept(self):
        mod = _module()
        # Three finished turns of one other chain. Waiting on all three says
        # the same thing as waiting on the last, and the declared payload
        # carries the edges.
        rows = [_stream_row(0, 1, 0.0, 0.1), _stream_row(0, 1, 0.5, 0.1),
                _stream_row(0, 1, 1.0, 0.1), _stream_row(0, 0, 5.0, 1.0)]
        deps, _ = mod._session_plan(rows)
        assert deps[3] == [2]

    def test_a_zero_width_predecessor_at_the_same_instant_is_not_one(self):
        mod = _module()
        # Golden requires the predecessor to have started strictly before, not
        # merely to have finished by then.
        rows = [_stream_row(0, 1, 5.0, 0.0), _stream_row(0, 0, 5.0, 1.0)]
        deps, _ = mod._session_plan(rows)
        assert deps[1] == []

    def test_a_trace_without_chain_ids_keeps_the_old_rule(self):
        mod = _module()
        rows = [_row(0, 0.0, 1.0), _row(0, 0.2, 0.3), _row(0, 2.0, 1.0)]
        assert mod._session_plan(rows)[0][2] == [0, 1]


class TestTheIdleGuard:
    """Dead air is skipped; a gap with work behind it is not.

    This corpus's think times are most of its wall clock -- 32 sessions with a
    few hours of engine work span 98,944 seconds -- so a faithful replay spends
    nearly all of a fixed window asleep.
    """

    def test_dead_air_is_shifted_forward(self):
        mod = _module()
        guard = mod._IdleGuard(0.2, period=0.02).start()
        began = time.monotonic()
        guard.sleep(5.0)
        elapsed = time.monotonic() - began
        guard.stop()
        assert elapsed < 1.5
        assert guard.shifts >= 1
        assert guard.shifted_s > 4.0

    def test_a_gap_with_a_request_in_flight_is_not_touched(self):
        mod = _module()
        guard = mod._IdleGuard(0.05, period=0.02).start()
        threading.Thread(target=guard.sleep, args=(5.0,), daemon=True).start()
        with guard.sending():
            time.sleep(0.25)
            shifts = guard.shifts
        guard.stop()
        assert shifts == 0

    def test_one_shift_moves_every_pending_timer_by_the_same_amount(self):
        mod = _module()
        # Shifting each timer to the cap independently would close the gaps
        # between lanes, which is the arrival process between sessions.
        guard = mod._IdleGuard(0.2, period=0.02)
        for seconds in (5.0, 6.0, 9.0):
            threading.Thread(target=guard.sleep, args=(seconds,),
                             daemon=True).start()
        time.sleep(0.1)
        with guard.cv:
            before = sorted(guard._deadlines.values())
        guard.start()
        time.sleep(0.15)
        with guard.cv:
            after = sorted(guard._deadlines.values())
        guard.stop()
        assert len(before) == len(after) == 3
        moved = [b - a for a, b in zip(after, before)]
        assert max(moved) - min(moved) < 1e-6
        assert min(moved) > 4.0

    def test_a_cap_of_zero_leaves_the_clock_alone(self):
        mod = _module()
        guard = mod._IdleGuard(0.0).start()
        began = time.monotonic()
        guard.sleep(0.3)
        elapsed = time.monotonic() - began
        guard.stop()
        assert elapsed >= 0.28
        assert guard.shifts == 0


class TestTheScheduleIsWhatPairsTheTwoSides:
    def test_the_digest_separates_two_schedules(self):
        mod = _module()
        workload = [_row(s, float(s), 1.0) for s in range(6)]
        one, _ = _run(mod, workload, clients=2, instances=1)
        two, _ = _run(mod, workload, clients=3, instances=1)
        assert mod._plan_digest(one) == mod._plan_digest(one)
        assert mod._plan_digest(one) != mod._plan_digest(two)

    def test_a_recorded_schedule_replays_the_same_executions(self):
        mod = _module()
        workload = [_row(s, float(s), 0.0) for s in range(4)]
        plan, first = _run(mod, workload, clients=2, instances=2)
        again: dict = {}
        mod._replay_schedule(plan, _ok, again, mod._IdleGuard(0.0))
        assert sorted(again) == sorted(first)
        assert [again[k]["row"] for k in sorted(again)] == \
               [first[k]["row"] for k in sorted(first)]

    def test_an_instance_waits_for_the_previous_one_on_its_lane(self):
        mod = _module()
        workload = [_row(0, 0.0, 1.0), _row(0, 2.0, 1.0)]
        plan, _ = _run(mod, workload, clients=1, instances=2)
        second = [e for e in plan if e["instance"] == 1]
        first = [e for e in plan if e["instance"] == 0]
        # Not just the last row of the previous session -- all of its open
        # ends. A sub-agent that started late can still be running when the
        # main turn has returned, and the lane is not free until it is not.
        assert second[0]["deps"] == mod._terminals(first)
        # And no think time on that edge: the gap between two sessions in the
        # trace is a gap between two different users.
        assert second[0]["think_s"] == 0.0


class TestTheThinkTimeIsReallySlept:
    def test_the_paced_executor_waits_and_records_what_it_waited(self):
        mod = _module()
        workload = [_row(0, 0.0, 0.0), _row(0, 0.05, 0.0)]
        began = time.monotonic()
        plan, _ = _run(mod, workload, clients=1, instances=1)
        # An executor that declared the gap and then did not sleep it would
        # come back instantly, and the real side would be a burst.
        assert time.monotonic() - began >= 0.045
        # think_s is the gap this lane ACTUALLY waited (replay.py:664), not
        # the one the corpus asked for, so a real 50 ms sleep records as
        # 50.08 ms and exact equality cannot hold. The idle guard can also
        # shorten a wait, so the tolerance is two-sided. The first edge is
        # still exactly zero: nothing is slept there at all.
        think = [e["think_s"] for e in plan]
        assert think[0] == 0.0
        assert think[1] == pytest.approx(0.05, abs=5e-3)

    def test_the_between_session_gap_is_not_slept(self):
        mod = _module()
        workload = [_row(0, 0.0, 0.0), _row(1, 100.0, 0.0)]
        began = time.monotonic()
        _run(mod, workload, clients=1, instances=2)
        assert time.monotonic() - began < 1.0


class TestConcurrencyIsBoundedByTheLanes:
    def test_one_lane_never_has_two_sessions_open(self):
        mod = _module()
        workload = [_row(s, 0.0, 1.0) for s in range(3)]
        peak = _peak(mod, workload, clients=1, instances=3)
        assert peak == 1

    def test_in_session_overlap_really_happens_at_the_socket(self):
        mod = _module()
        # One turn overlapped by two sub-agents, fired a millisecond apart. If
        # the loop serialises the session, peak is 1 and the run measures a
        # workload the corpus does not contain.
        workload = [_stream_row(0, 0, 0.0, 10.0), _stream_row(0, 1, 0.001, 2.0),
                    _stream_row(0, 2, 0.002, 2.0)]
        assert _peak(mod, workload, clients=1, instances=1) == 3

    def test_sub_agents_do_not_take_a_lane_of_their_own(self):
        mod = _module()
        # Two lanes, each holding a session that fans out three ways. In-flight
        # requests reach six, which is the workload and not an error: golden
        # counts lanes, not requests.
        workload = [_stream_row(s, k, 0.0, 5.0)
                    for s in range(2) for k in range(3)]
        assert _peak(mod, workload, clients=2, instances=1) == 6


class TestTheWindowClosesOnALaneMidSession:
    """A rung must end at its deadline, not at the end of a recorded session.

    The lane loop used to consult the clock only *between* instances, so a
    lane that had started a session ran it to the end whatever
    `--benchmark-duration` said. On the corpus that is not a rounding error:
    the recorded per-session span is 8,468 s at the median and 351,982 s at
    the longest, so a 1,800 s rung would have run for one recorded session.
    It showed up as a dry run sitting at 0.2% CPU for ten minutes -- asleep in
    think time, not wedged.

    The fix may not be to shorten the think time. The scenario the corpus ships
    with sets `forbid_ignore_trace_delays` and `forbid_inter_turn_delay_cap`,
    and the only compression it allows is the system-wide idle gap. So the turn
    behind an unfinished think time is simply not sent, and the run ends with
    its lanes mid-session -- which is what the same scenario's
    `minimum_profile_metric_coverage_ratio` of 0.95 is there to tolerate.
    """

    @staticmethod
    def _slow_session(turns, gap):
        """One session whose turns are `gap` seconds apart."""
        return [_row(0, i * gap, 0.0) for i in range(turns)]

    def _recycled(self, mod, workload, *, clients=1, duration=0.4, send=None):
        out, guard = {}, mod._IdleGuard(0.0)
        groups = mod._sessions(workload)
        began = time.monotonic()
        plan, _, ended = mod._recycle(
            workload, groups, clients=clients, duration=duration, instances=0,
            sampler=mod._Sampler(len(groups), 0, "sequential"),
            rng=random.Random(0), guard=guard, send=send or _ok, out=out,
            startup=False)
        return plan, out, guard, ended - began

    def test_a_long_session_does_not_outlast_the_window(self):
        mod = _module()
        # Ten turns a second apart: the session is ten seconds long and the
        # window is four tenths of one.
        plan, _, _, wall = self._recycled(
            mod, self._slow_session(10, 1.0), duration=0.4)
        assert wall < 3.0, f"the rung ran {wall:.1f}s past a 0.4s window"
        assert len(plan) < 10

    def test_the_turns_it_did_not_reach_are_not_in_the_plan(self):
        mod = _module()
        plan, out, guard, _ = self._recycled(
            mod, self._slow_session(10, 1.0), duration=0.4)
        assert guard.cut > 0
        # The plan is the schedule as executed: every entry left in it was
        # sent, and none of them waits on one that was not. A dangling edge
        # here is what would make the modelled side replay a request the real
        # side never issued.
        kept = {e["eid"] for e in plan}
        assert all(d in kept for e in plan for d in e["deps"])
        assert kept <= set(out)

    def test_nothing_goes_on_the_wire_after_the_deadline(self):
        mod = _module()
        sent = []
        lock = threading.Lock()

        def send(e):
            with lock:
                sent.append(time.monotonic())
            return _ok(e)

        began = time.monotonic()
        self._recycled(mod, self._slow_session(20, 0.2), duration=0.5,
                       send=send)
        assert sent, "nothing was sent at all"
        assert max(sent) - began < 0.5 + 0.3

    def test_an_unfinished_think_time_is_abandoned_not_shortened(self):
        mod = _module()
        # A turn 0.3s behind its predecessor, inside a window with room for
        # it. Cutting the sleep short to fit would be the inter-turn cap the
        # scenario forbids; the gap must still be paid in full.
        gaps = []
        last = [None]
        lock = threading.Lock()

        def send(e):
            with lock:
                now = time.monotonic()
                if last[0] is not None:
                    gaps.append(now - last[0])
                last[0] = now
            return _ok(e)

        self._recycled(mod, self._slow_session(3, 0.3), duration=5.0,
                       send=send)
        # A lane recycles several times in five seconds, and the gap across an
        # instance boundary is zero by design -- it belongs to two different
        # users. So every gap is either that boundary or a recorded think time
        # paid in full; a value in between is a shortened sleep.
        assert gaps
        assert not [g for g in gaps if 0.02 < g < 0.25]
        assert max(gaps) >= 0.25

    def test_an_unbounded_run_is_not_cut(self):
        mod = _module()
        # No duration and no deadline: the modelled side replays a recorded
        # schedule in full, and a guard left armed from the real side would
        # silently truncate it.
        plan, _, guard, _ = self._recycled(
            mod, self._slow_session(4, 0.01), duration=0.0)
        assert guard.expires_at is None
        assert guard.cut == 0
        assert len(plan) == 4

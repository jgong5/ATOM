# SPDX-License-Identifier: MIT
"""What a run of the clock says about itself.

Three records and the rules that go with them. The timeline is off unless a run
asks for it and has to cost nothing when off, which is measured here rather than
asserted. The dump has to be honest about the one thing it cannot tell -- a run
that finished and a run that is stuck reach the same state, and nothing in the
protocol separates them. And the summary has to come out the same twice for one
configuration, which is only possible if the numbers that move with the order
requests arrived in are kept apart from the ones that do not.
"""

import ast
import itertools
import json
import math
import pathlib
import re
import time

import pytest

from atom.compass.clock import (
    SPEED_TARGET_RATIO,
    ClockAuthority,
    ClockDeadlock,
    DetectorState,
    DriverDiscipline,
    Grant,
    LinkClass,
    LookaheadMatrix,
    LpId,
    LpRegistry,
    LpState,
    LpStatus,
    RefusalTally,
    RunSummary,
    StallKind,
    TimelineLog,
    deadlock_dump,
    stall_kind,
)

CLOCK_PACKAGE = (
    pathlib.Path(__file__).resolve().parents[2] / "atom" / "compass" / "clock"
)

# A four-participant configuration with unequal floors and unequal work, which
# is what makes the arrival-order question interesting: with equal floors and
# equal horizons every request order resolves identically and the property under
# test is not exercised at all.
#
# Every participant's last event is at the same time, and that is not
# decoration. The protocol has no way for a participant to say it has finished:
# one that has drained its work can only keep asking, and while any peer is
# still executing it keeps being granted a floor's worth of time and walks
# forward for ever. So a bounded run is one where the work runs out everywhere
# at once, and the harness below has to impose that because nothing in the
# clock can.
NAMES = ("decode", "engine", "prefill", "traffic-source")
FLOORS = (0.5, 1.0, 2.0)
LAST_EVENT = 12.0
WORK = (
    (1.0, 6.0, LAST_EVENT),
    (2.0, 9.0, LAST_EVENT),
    (3.0, 7.0, LAST_EVENT),
    (4.0, 11.0, LAST_EVENT),
)


def _topology(names=NAMES, floors=FLOORS):
    registry = LpRegistry()
    ids = [registry.register(LpId(name)) for name in names]
    matrix = LookaheadMatrix(registry)
    for source_index, source in enumerate(ids):
        for target_index, target in enumerate(ids):
            if source is target:
                continue
            floor = floors[(source_index + target_index) % len(floors)]
            matrix.declare(source, target, LinkClass.PREFILL_TO_DECODE, floor)
    return registry, matrix, ids


def _drive(order, timeline=None, names=NAMES, floors=FLOORS, work=WORK, rounds=4000):
    """Run one fixed workload to exhaustion, asking in `order`.

    A participant asks only while it is executing and only while it has work
    left, so one that has been refused stays parked until a peer moves rather
    than stepping its clock forward a floor at a time. That is the discipline
    the grant traffic can afford, and it is also what lets this finish.
    """
    registry, matrix, ids = _topology(names, floors)
    authority = ClockAuthority(registry, matrix, timeline=timeline)
    todo = {lp_id: list(times) for lp_id, times in zip(ids, work)}
    for _ in range(rounds):
        moved = False
        for index in order:
            lp_id = ids[index]
            if authority.held_grant(lp_id) is not None:
                grant = authority.take_up_grant(lp_id)
                moved = True
                if todo[lp_id] and todo[lp_id][0] <= grant.advance_to:
                    todo[lp_id].pop(0)
                    if todo[lp_id]:
                        target = ids[(index + 1) % len(ids)]
                        floor = matrix.lookahead(lp_id, target)
                        authority.schedule_event(
                            lp_id, target, authority.now(lp_id) + floor
                        )
            if authority.state(lp_id).status is not LpStatus.RUNNING:
                continue
            if not todo[lp_id]:
                continue
            if authority.request_advance(lp_id, todo[lp_id][0]):
                moved = True
        if not moved:
            return authority
    raise AssertionError("the workload did not finish; it is not a bounded run")


def _pairwise_matrix(floor=0.0, names=("engine", "traffic-source")):
    registry, matrix, ids = _topology(names, (floor,))
    return registry, matrix, ids


# --- the timeline ------------------------------------------------------------


def test_the_timeline_is_off_unless_a_run_asks_for_it():
    registry, matrix, _ids = _pairwise_matrix(1.0e-3)
    assert ClockAuthority(registry, matrix).timeline is None


def test_a_run_with_the_log_off_issues_exactly_the_grants_it_issues_with_it_on():
    # The log observes; it must not participate. Same workload, same order, one
    # arm logging and one not, and the grant ledgers compared element by element.
    log = TimelineLog()
    quiet = _drive((0, 1, 2, 3))
    loud = _drive((0, 1, 2, 3), timeline=log)
    assert quiet.states() == loud.states()
    assert [quiet.grants_issued(lp) for lp in quiet.registry.ids()] == [
        loud.grants_issued(lp) for lp in loud.registry.ids()
    ]
    assert len(log) == loud.grants_issued()


def test_nothing_is_written_or_called_when_the_log_is_off():
    def sink(_line):
        raise AssertionError("a sink was called for a run that asked for no log")

    TimelineLog(sink)  # built but never handed to the clock
    authority = _drive((0, 1, 2, 3))
    assert authority.timeline is None


def _per_grant_seconds(timeline, rounds=4000):
    """Wall seconds per grant on the tightest loop that issues one.

    Two participants asking in turn, so almost all of what is timed is the
    clock's own path rather than a driver's bookkeeping.
    """
    registry, matrix, ids = _topology(("engine", "traffic-source"), (1.0,))
    authority = ClockAuthority(registry, matrix, timeline=timeline)
    first, second = ids
    started = time.perf_counter()
    for _ in range(rounds):
        authority.request_advance(first, math.inf)
        authority.take_up_grant(first)
        authority.request_advance(second, math.inf)
        authority.take_up_grant(second)
    return (time.perf_counter() - started) / (2 * rounds)


def test_the_log_costs_less_when_off_than_when_on():
    # The claim is that the default costs nothing, and the honest way to state
    # it here is as a comparison: with the log off a grant does one
    # `is not None` test, and with it on it formats five fields and allocates a
    # record. Best of three, because a shared host adds time and never removes
    # it. The absolute figure belongs with the run that measured it, not in an
    # assertion that would read as a claim about this machine.
    off = min(_per_grant_seconds(None) for _ in range(3))
    on = min(_per_grant_seconds(TimelineLog()) for _ in range(3))
    assert off * 1.15 < on, f"off {off * 1e6:.3f} us/grant, on {on * 1e6:.3f} us/grant"


def test_one_record_per_granted_advance_in_the_order_the_clock_issued_them():
    log = TimelineLog()
    authority = _drive((3, 2, 1, 0), timeline=log)
    assert len(log) == authority.grants_issued()
    per_participant = {}
    for entry in log.records():
        per_participant.setdefault(entry.lp_id, []).append(entry)
    for lp_id, entries in per_participant.items():
        assert len(entries) == authority.grants_issued(lp_id)
        # Append-only and continuous: each record starts where the last ended.
        for earlier, later in itertools.pairwise(entries):
            assert later.virtual_time_from == earlier.virtual_time_to


def test_a_record_is_the_five_declared_columns_and_names_the_participant_verbatim():
    log = TimelineLog()
    authority = _drive((0, 1, 2, 3), timeline=log)
    names = [str(lp_id) for lp_id in authority.registry.ids()]
    for entry, line in zip(log.records(), log.lines()):
        columns = line.split(" ", 4)
        assert len(columns) == 5
        assert columns[0] == str(entry.lp_id)
        assert columns[0] in names
        assert float(columns[1]) == entry.virtual_time_from
        assert float(columns[2]) == entry.virtual_time_to
        assert columns[3] == entry.event
        assert columns[4] == entry.detail


def test_an_advance_says_what_ended_it():
    log = TimelineLog()
    _drive((0, 1, 2, 3), timeline=log)
    events = sorted({entry.event for entry in log.records()})
    assert set(events) <= {"bound", "horizon", "release", "tie"}
    assert "horizon" in events and "bound" in events
    for entry in log.records():
        if entry.event == "release":
            assert entry.virtual_time_from == entry.virtual_time_to
        else:
            assert entry.virtual_time_to > entry.virtual_time_from


def test_an_advance_both_could_have_ended_is_neither_of_them():
    # `bound` is read as the count of advances a peer cut short, and that
    # reading is the only measured check on a declared discipline. Where the
    # participant's own event lands on the same instant as the bound, nothing
    # was cut short, and on a symmetric arrangement that is a large share of
    # the run rather than a curiosity.
    log = TimelineLog()
    engine = LpId("engine")
    peer = LpId("traffic-source")
    assert log.record(Grant(engine, 0.0, 5.0, 5.0, peer), 5.0).event == "tie"
    assert log.record(Grant(engine, 0.0, 5.0, 5.0, peer), 9.0).event == "bound"
    assert log.record(Grant(engine, 0.0, 5.0, 9.0, peer), 5.0).event == "horizon"
    assert log.ended_at_peer_bound() == 1


def test_a_tie_is_common_enough_on_a_symmetric_arrangement_to_matter():
    # Equal floors everywhere, which is the shape a ring of pipeline stages
    # has. Counted here so the classification is not defended by argument.
    log = TimelineLog()
    _drive((0, 1, 2, 3), timeline=log, floors=(1.0,))
    ties = sum(1 for entry in log.records() if entry.event == "tie")
    assert ties > 0
    assert ties > len(log) // 5
    assert log.ended_at_peer_bound() == sum(
        1 for entry in log.records() if entry.event == "bound"
    )


def test_a_release_is_recorded_as_a_grant_of_no_span():
    log = TimelineLog()
    engine = LpId("engine")
    entry = log.record(Grant(engine, 4.0, 4.0, 4.0, LpId("traffic-source")), math.inf)
    assert entry.event == "release"
    assert entry.virtual_time_from == entry.virtual_time_to == 4.0


def test_a_sink_sees_every_line_as_it_is_written():
    seen = []
    log = TimelineLog(seen.append)
    _drive((0, 1, 2, 3), timeline=log)
    assert seen == list(log.lines())


def test_a_sink_that_is_not_callable_is_refused_where_it_is_handed_over():
    with pytest.raises(TypeError, match="sink must be callable"):
        TimelineLog("timeline.log")


def test_a_streaming_log_can_be_asked_not_to_keep_what_it_streamed():
    # A sink alone does not relieve the memory: every record is retained as
    # well, so a caller streaming to a file still holds the whole run. Where
    # each grant covers one microsecond floor that is millions of live records
    # per simulated second, and the object graph arrives before the text does.
    seen = []
    streaming = TimelineLog(seen.append, retain=False)
    quiet = TimelineLog()
    _drive((0, 1, 2, 3), timeline=streaming)
    _drive((0, 1, 2, 3), timeline=quiet)
    assert len(streaming) == len(quiet)
    assert streaming.ended_at_peer_bound() == quiet.ended_at_peer_bound()
    assert seen == list(quiet.lines())
    with pytest.raises(ValueError, match="asked not to retain"):
        streaming.records()


def test_a_log_that_would_write_nothing_anywhere_is_refused():
    with pytest.raises(ValueError, match="would write nothing"):
        TimelineLog(retain=False)


def test_the_share_of_advances_a_peer_cut_short_is_available_as_a_number():
    log = TimelineLog()
    _drive((0, 1, 2, 3), timeline=log)
    counted = sum(1 for entry in log.records() if entry.event == "bound")
    assert log.ended_at_peer_bound() == counted


# --- the dump ----------------------------------------------------------------


def _rows(dump):
    """The dump's indented lookahead row for each participant, keyed by name."""
    rows = {}
    current = None
    for line in dump.splitlines():
        if line.startswith("    "):
            if current is not None:
                rows[current].append(line)
        elif line and " " in line and not line.startswith("the clock"):
            current = line.split(" ", 1)[0]
            rows[current] = []
    return rows


def _stalled():
    """Three participants, zero floors, nobody with anything left to do."""
    registry, matrix, ids = _topology(("decode", "engine", "traffic-source"), (0.0,))
    authority = ClockAuthority(registry, matrix)
    for lp_id in ids[:-1]:
        authority.request_advance(lp_id, math.inf)
    return authority, ids


def test_a_stall_is_classified_and_a_clock_that_can_still_move_is_not_one():
    registry, matrix, _ids = _topology()
    authority = ClockAuthority(registry, matrix)
    assert stall_kind(authority) is StallKind.NOT_STALLED
    assert "not a stall" in deadlock_dump(authority)


def test_the_dump_names_every_participants_clock_state_and_bounding_row():
    authority, ids = _stalled()
    with pytest.raises(ClockDeadlock) as raised:
        authority.request_advance(ids[-1], math.inf)
    dump = raised.value.table
    for lp_id in ids:
        assert f"{lp_id} blocked-on-message clock 0.0s" in dump
        assert f"{lp_id} blocked-on-message" in dump
        for peer in ids:
            if peer is not lp_id:
                assert f"    from {peer} floor 0.0s" in dump
    assert str(raised.value.table) in str(raised.value)


def test_the_dump_marks_the_one_term_in_the_row_that_binds():
    # A participant still executing holds everyone else at its own clock, so the
    # binding term is nameable and the dump names it rather than leaving a
    # reader to redo the minimum by hand.
    registry, matrix, ids = _topology(("decode", "engine", "traffic-source"), (0.5,))
    authority = ClockAuthority(registry, matrix)
    authority.request_advance(ids[0], math.inf)
    dump = deadlock_dump(authority)
    # One marked term per participant: the row is a minimum, and the dump says
    # which element of it the minimum came from rather than leaving a reader to
    # redo the arithmetic.
    assert dump.count("<- binds") == len(ids)
    rows = _rows(dump)
    binding = [line for line in rows[str(ids[0])] if "<- binds" in line]
    assert len(binding) == 1
    assert binding[0].startswith(
        f"    from {ids[1]} floor 0.5s earliest 0.0s gives 0.5s"
    )


def test_a_stall_with_no_event_anywhere_cannot_tell_a_finished_run_from_a_deadlock():
    authority, ids = _stalled()
    with pytest.raises(ClockDeadlock) as raised:
        authority.request_advance(ids[-1], math.inf)
    dump = raised.value.table
    assert str(StallKind.NO_FUTURE_EVENT) in dump
    assert "A run that has finished its work reaches exactly this state" in dump
    assert "no way for a participant to say it has finished" in dump
    # And it says which of the two readings the state leans towards, as a
    # reading of the same evidence rather than as a second source.
    assert "3 of 3 participants were never granted any time" in dump
    assert "narrows the question rather than answering it" in dump


def test_a_finished_looking_stall_says_so_without_claiming_the_run_finished():
    # The other reading of the same shape: everybody moved, everybody drained,
    # nobody knows of anything more. It is what the end of a clean run looks
    # like, and it is also what two participants waiting on each other look
    # like, so the dump reports the evidence and stops short of the verdict.
    registry, matrix, ids = _topology(("engine", "traffic-source"), (0.0,))
    authority = ClockAuthority(registry, matrix)
    authority.request_advance(ids[0], 5.0)
    authority.request_advance(ids[1], 5.0)
    for lp_id in ids:
        authority.take_up_grant(lp_id)
    authority.request_advance(ids[0], math.inf)
    with pytest.raises(ClockDeadlock) as raised:
        authority.request_advance(ids[1], math.inf)
    dump = raised.value.table
    assert "every participant was granted time" in dump
    assert "the furthest clock reached 5.0s" in dump
    assert "a run stuck part way through would look the same" in dump


class _UnreachableEventAuthority(ClockAuthority):
    """A stall the rule is not supposed to be able to produce.

    Overrides what every participant declares and nothing else, so the dump
    under test reads the same accessors it reads in a real abort. The rule
    itself makes this state unreachable; the dump still has to say the right
    thing about it, because if it ever appears it is the evidence that the rule
    was wrong.
    """

    def states(self):
        return tuple(
            LpState(state.lp_id, state.now, 12.0, LpStatus.BLOCKED_ON_MESSAGE)
            for state in super().states()
        )


def test_a_stall_holding_an_event_nobody_can_reach_is_not_a_finished_run():
    registry, matrix, _ids = _topology()
    authority = _UnreachableEventAuthority(registry, matrix)
    assert stall_kind(authority) is StallKind.EVENT_UNREACHABLE
    dump = deadlock_dump(authority)
    assert "This is not a finished run" in dump
    assert "no way for a participant to say it has finished" not in dump


# --- the summary -------------------------------------------------------------


def _summary(authority, wall_seconds=0.25, **kwargs):
    return RunSummary.of(
        authority,
        wall_seconds,
        DriverDiscipline.PARK_WHEN_REFUSED,
        **kwargs,
    )


def _orders():
    orders = [()]
    for _ in range(4):
        orders = [order + (index,) for order in orders for index in range(4)]
        orders = [order for order in orders if len(set(order)) == len(order)]
    return orders


def test_the_schedule_half_of_the_summary_is_identical_under_every_arrival_order():
    # The property the whole design rests on: the sequence of (participant,
    # virtual time, event) may not vary with wall-clock interleaving. A summary
    # field that varies with arrival order is therefore not reproducible, and
    # this is the half that must be.
    records = [
        json.dumps(_summary(_drive(order)).schedule_record()) for order in _orders()
    ]
    assert len(_orders()) == 24
    assert len(set(records)) == 1, f"{len(set(records))} distinct schedule records"


def test_the_cost_half_moves_with_arrival_order_and_says_which_numbers_do():
    # The other side of the split, and the reason it exists: on this
    # configuration the 24 request orders give one schedule and several grant
    # counts. A reader handed the count alone would take it for a property of
    # the configuration, so it is in the half whose first field says what it
    # moves with.
    counts = {_summary(_drive(order)).grants_issued for order in _orders()}
    record = _summary(_drive((0, 1, 2, 3))).cost_record()
    assert record["varies_with"] == ["arrival order", "driver discipline", "host"]
    assert record["grants_issued_total"] in counts
    assert len(counts) > 1, "arrival order left the grant count alone here"


def test_a_summary_cannot_be_written_without_naming_how_the_run_asked_for_time():
    authority = _drive((0, 1, 2, 3))
    with pytest.raises(TypeError, match="means nothing without it"):
        RunSummary.of(authority, 0.25, "park-when-refused")
    with pytest.raises(TypeError):
        RunSummary.of(authority, 0.25)
    named = RunSummary.of(authority, 0.25, DriverDiscipline.RE_ASK_AFTER_TAKE_UP)
    assert named.cost_record()["driver_discipline"] == "re-ask-after-take-up"


def test_the_grant_count_carries_the_fingerprint_of_the_discipline_when_the_log_is_on():
    log = TimelineLog()
    authority = _drive((0, 1, 2, 3), timeline=log)
    with_log = _summary(authority).cost_record()
    assert with_log["grants_recorded"] == authority.grants_issued()
    assert with_log["grants_ended_at_peer_bound"] == log.ended_at_peer_bound()
    without_log = _summary(_drive((0, 1, 2, 3))).cost_record()
    assert without_log["grants_recorded"] is None
    assert without_log["grants_ended_at_peer_bound"] is None


def test_an_empty_record_count_means_the_log_was_off_and_can_mean_nothing_else():
    # The summary reads the log off the clock rather than taking a second copy
    # from the caller. Otherwise an empty count means either "the log was off"
    # or "whoever built the summary forgot to pass it", and those are opposite
    # statements about the run -- on a clock that logged every grant.
    log = TimelineLog()
    logging_clock = _drive((0, 1, 2, 3), timeline=log)
    quiet_clock = _drive((0, 1, 2, 3))
    assert logging_clock.timeline is log
    assert quiet_clock.timeline is None
    assert _summary(logging_clock).grants_recorded == len(log)
    assert _summary(quiet_clock).grants_recorded is None
    # And there is no way to describe this run with somebody else's log.
    with pytest.raises(TypeError):
        RunSummary.of(
            quiet_clock,
            0.25,
            DriverDiscipline.PARK_WHEN_REFUSED,
            timeline=TimelineLog(),
        )


def test_grants_are_reported_per_participant_and_in_total():
    authority = _drive((0, 1, 2, 3))
    summary = _summary(authority)
    assert [name for name, _ in summary.grants] == list(NAMES)
    assert summary.grants_issued == authority.grants_issued()
    for name, count in summary.grants:
        assert count == authority.grants_issued(LpId(name))


def test_the_speed_result_is_simulated_seconds_over_wall_seconds():
    authority = _drive((0, 1, 2, 3))
    simulated = max(authority.now(lp) for lp in authority.registry.ids())
    fast = _summary(authority, wall_seconds=simulated / 10.0)
    slow = _summary(authority, wall_seconds=simulated / 2.0)
    assert fast.simulated_seconds == simulated - authority.start_time
    assert fast.speed_ratio == pytest.approx(10.0)
    assert fast.meets_speed_target
    assert fast.speed_refused is None
    assert slow.speed_ratio == pytest.approx(2.0)
    assert not slow.meets_speed_target
    assert SPEED_TARGET_RATIO == 5.0


def test_a_run_that_spent_no_wall_time_has_no_speed_result_rather_than_an_unlimited_one():
    # This is the field the acceptance gate reads, so an unlimited ratio would
    # be a guess that reads as a pass. It is also the field that would put a
    # value in the record that no reader outside Python can carry.
    authority = _drive((0, 1, 2, 3))
    summary = _summary(authority, wall_seconds=0.0)
    assert summary.speed_ratio is None
    assert summary.meets_speed_target is None
    assert summary.speed_refused == (
        "no wall time was measured, so this run has no speed result"
    )


def test_a_duration_that_is_not_a_finite_number_is_refused_where_it_enters():
    authority = _drive((0, 1, 2, 3))
    for bad in (math.inf, float("nan"), -1.0):
        with pytest.raises(ValueError, match="finite number of seconds"):
            _summary(authority, wall_seconds=bad)
        with pytest.raises(ValueError, match="finite number of seconds"):
            _summary(authority, lazy_trace_wall_seconds=bad)


def test_lazy_traces_are_carried_as_a_count_and_the_wall_seconds_they_took():
    authority = _drive((0, 1, 2, 3))
    record = _summary(
        authority, lazy_traces=7, lazy_trace_wall_seconds=1.5
    ).cost_record()
    assert record["lazy_traces"] == 7
    assert record["lazy_trace_wall_seconds"] == 1.5


def test_the_detectors_are_split_by_whether_their_reading_is_reproducible():
    authority = _drive((0, 1, 2, 3))
    detectors = DetectorState(stragglers=0, watchdog_warnings=3, clock_lint="passed")
    summary = _summary(authority, detectors=detectors)
    assert summary.schedule_record()["stragglers"] == 0
    assert summary.schedule_record()["clock_lint"] == "passed"
    # A watchdog fires on a wall-clock threshold, so its count is a property of
    # the host and belongs with the other numbers that are.
    assert summary.cost_record()["watchdog_warnings"] == 3


def test_refusals_are_counted_by_step_by_predicted_second_and_by_distinct_reason():
    tally = RefusalTally.of(
        [
            "no price for this shape",
            "outside the measured hull",
            "no price for this shape",
        ],
        steps=50,
        refused_predicted_seconds=1.5,
        predicted_seconds=30.0,
    )
    record = tally.record()
    assert record["count"] == 3
    assert record["fraction_of_steps"] == pytest.approx(3 / 50)
    assert record["fraction_of_predicted_seconds"] == pytest.approx(0.05)
    assert record["reasons"] == [
        ["no price for this shape", 2],
        ["outside the measured hull", 1],
    ]


def test_refusal_reasons_come_out_in_one_order_whatever_order_they_arrived_in():
    forwards = RefusalTally.of(["b", "a", "c", "a"], steps=4).record()["reasons"]
    backwards = RefusalTally.of(["a", "c", "a", "b"], steps=4).record()["reasons"]
    assert forwards == backwards == [["a", 2], ["b", 1], ["c", 1]]


def test_an_empty_run_divides_by_nothing():
    tally = RefusalTally()
    assert tally.fraction_of_steps == 0.0
    assert tally.fraction_of_predicted_seconds == 0.0
    registry, matrix, _ids = _topology()
    summary = _summary(ClockAuthority(registry, matrix), wall_seconds=0.0)
    assert summary.simulated_seconds == 0.0
    assert summary.speed_ratio is None
    assert json.dumps(summary.as_record(), allow_nan=False)


def test_the_summary_is_a_value_that_can_be_read_without_the_machine_that_made_it():
    log = TimelineLog()
    authority = _drive((0, 1, 2, 3), timeline=log)
    record = _summary(authority, refusals=RefusalTally.of(["x"], 9)).as_record()
    assert sorted(record) == ["cost", "schedule"]
    # `allow_nan=False` is the point: Python's encoder emits `Infinity` and
    # `NaN` by default, which are extensions no strict reader has to accept,
    # and a record that only round-trips through the library that wrote it is
    # not a record that can be read without the machine that produced it.
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert len(log) == authority.grants_issued()


# --- what the records may not contain ----------------------------------------

CITATION = re.compile(
    r"\bD\d+(\.\d+)?\b|\bT\d+\b|\bW\d+(\.\d+)?\b|\bP0\.\d+\b|\bprinciple \d+\b"
    r"|\bGate \d+\b",
    re.IGNORECASE,
)


def test_nothing_the_clock_emits_cites_a_design_document():
    # The records here are the run's own evidence, so a document number inside
    # one is the same mistake as one in a comment and harder to find later: it
    # travels with the artifact into whatever reads it.
    log = TimelineLog()
    authority = _drive((0, 1, 2, 3), timeline=log)
    emitted = list(log.lines())
    emitted.append(deadlock_dump(authority))
    emitted.append(json.dumps(_summary(authority).as_record()))
    emitted += [str(kind) for kind in StallKind]
    emitted += [str(discipline) for discipline in DriverDiscipline]
    stalled, ids = _stalled()
    with pytest.raises(ClockDeadlock) as raised:
        stalled.request_advance(ids[-1], math.inf)
    emitted.append(str(raised.value))
    offenders = [text for text in emitted if CITATION.search(text)]
    assert not offenders, f"emitted data cites a design document: {offenders[:3]}"


def test_the_observability_module_reaches_no_clock_and_no_socket():
    # The package-wide import allowlist covers this module too. Stated again
    # from the other end, because the one thing a summary would plausibly want
    # is the wall time it reports, and taking it as an argument instead is what
    # keeps the record auditable away from the host that produced it.
    source = (CLOCK_PACKAGE / "observability.py").read_text()
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            imported.append((node.module or "").split(".")[0])
    assert sorted(dict.fromkeys(imported)) == ["dataclasses", "enum", "math"]
    assert "wall_seconds: float" in source


def _resolve_function():
    tree = ast.parse((CLOCK_PACKAGE / "authority.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve":
            return node
    raise AssertionError("the grant path is no longer a function named _resolve")


def test_no_work_the_log_needs_happens_outside_the_guard():
    # This is a shape check, not a cost check, and the difference is measured
    # rather than assumed: a version that builds one tuple of horizons *before*
    # the guard still has exactly one `if`, still tests `is not None` and still
    # has no `else`, and costs a real fraction of a microsecond on every grant
    # of a run that asked for no log. What separates the two is not the guard's
    # shape but what stands outside it, so that is what is asserted. The
    # measurement recorded with this task is the evidence about cost; this
    # cannot be that, and should not be read as it.
    resolve = _resolve_function()
    guards = [
        statement
        for statement in resolve.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "self._timeline is not None"
    ]
    assert len(guards) == 1, "the grant path no longer has exactly one log guard"
    guard = guards[0]
    assert not guard.orelse
    grant_loops = [
        statement
        for statement in resolve.body
        if isinstance(statement, ast.For) and "issued.append" in ast.unparse(statement)
    ]
    assert len(grant_loops) == 1
    for statement in resolve.body:
        if statement is guard or statement is grant_loops[0]:
            continue
        text = ast.unparse(statement)
        assert "_timeline" not in text, f"the log is touched outside its guard: {text}"
        assert "self._next" not in text, f"a horizon is read for nobody: {text}"
    # And nothing outside the guard walks the grants a second time, which is
    # the shape the measured mutant took.
    for statement in resolve.body:
        if statement is guard:
            continue
        for node in ast.walk(statement):
            if isinstance(node, (ast.For, ast.comprehension)):
                assert "issued" not in ast.unparse(node.iter), (
                    "the grants are walked again outside the guard: "
                    f"{ast.unparse(node.iter)}"
                )

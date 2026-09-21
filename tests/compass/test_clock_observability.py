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
    assert set(events) <= {"bound", "horizon", "release"}
    assert "horizon" in events and "bound" in events
    for entry in log.records():
        if entry.event == "release":
            assert entry.virtual_time_from == entry.virtual_time_to
        else:
            assert entry.virtual_time_to > entry.virtual_time_from


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
        assert f"{lp_id} blocked-on-message clock 0s" in dump
        assert f"{lp_id} blocked-on-message" in dump
        for peer in ids:
            if peer is not lp_id:
                assert f"    from {peer} floor 0s" in dump
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
    assert binding[0].startswith(f"    from {ids[1]} floor 0.5s earliest 0s gives 0.5s")


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
    assert "the furthest clock reached 5s" in dump
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


def _summary(authority, wall_seconds=0.25, timeline=None, **kwargs):
    return RunSummary.of(
        authority,
        wall_seconds,
        DriverDiscipline.PARK_WHEN_REFUSED,
        timeline=timeline,
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
    with_log = _summary(authority, timeline=log).cost_record()
    assert with_log["grants_recorded"] == authority.grants_issued()
    assert with_log["grants_ended_at_peer_bound"] == log.ended_at_peer_bound()
    without_log = _summary(authority).cost_record()
    assert without_log["grants_recorded"] is None
    assert without_log["grants_ended_at_peer_bound"] is None


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
    assert slow.speed_ratio == pytest.approx(2.0)
    assert not slow.meets_speed_target
    assert SPEED_TARGET_RATIO == 5.0


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
    assert summary.speed_ratio == math.inf
    assert summary.meets_speed_target


def test_the_summary_is_a_value_that_can_be_read_without_the_machine_that_made_it():
    log = TimelineLog()
    authority = _drive((0, 1, 2, 3), timeline=log)
    record = _summary(
        authority, timeline=log, refusals=RefusalTally.of(["x"], 9)
    ).as_record()
    assert sorted(record) == ["cost", "schedule"]
    assert json.loads(json.dumps(record)) == record


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
    emitted.append(json.dumps(_summary(authority, timeline=log).as_record()))
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


def test_the_timeline_hook_is_one_test_when_the_log_is_off():
    # What "costs nothing when off" is allowed to mean, checked structurally so
    # it cannot drift into a loop that runs whether or not anyone asked for a
    # log. The measurement above says what it costs; this says what it is.
    tree = ast.parse((CLOCK_PACKAGE / "authority.py").read_text())
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "self._timeline" in ast.unparse(node.test)
    ]
    assert len(guards) == 1
    assert ast.unparse(guards[0].test) == "self._timeline is not None"
    assert not guards[0].orelse

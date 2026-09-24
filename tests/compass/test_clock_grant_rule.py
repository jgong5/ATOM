# SPDX-License-Identifier: MIT
"""`atom.compass.clock.authority`: who may move their clock, and how far.

What each group of tests here is defending:

* **The rule reads a clock, not a horizon.** The bound a participant places on
  its peers comes from where its clock stands, never from where it says its
  next event is. The two differ on a topology that is one page of set-up, and
  `TestTheRuleReadsAClockAndNotAHorizon` runs both of them on it: the rejected
  one is implemented here, against the real state machine, so the only thing
  that differs between the two measurements is the quantity the bound is taken
  over.
* **A backdated event stops the run.** Not a warning and not a flag. The failure
  it catches does not crash anything -- the run finishes and reports a plausible
  latency -- which is the whole reason it has to be loud. It is a raise and not
  an `assert`, because `python -O` deletes an `assert`, and a check specified as
  always on cannot be one.
* **A stall stops the run too.** There is no timeout anywhere in this package,
  deliberately: a timeout that releases a run which never became valid produces
  a complete-looking result with no failures reported, and that has already cost
  this project a day of conclusions.
* **Two participants eligible at the same simulated time are served in name
  order.** Never in the order they asked. The order they ask in depends on
  process start-up and the host's scheduler, so a run that grants in arrival
  order is a different run every time.
* **Both degenerate shapes work.** One participant is a local clock. Every floor
  at zero is a single event loop spread over several processes -- correct,
  serialized, and not an error.
"""

import ast
import math
from pathlib import Path

import pytest

from atom.compass.clock import (
    BackdatedEvent,
    ClockAbort,
    ClockAuthority,
    ClockDeadlock,
    Grant,
    LinkClass,
    LookaheadMatrix,
    LpId,
    LpRegistry,
    LpState,
    LpStatus,
)

CLOCK_PACKAGE = Path(__file__).resolve().parents[2] / "atom" / "compass" / "clock"


def _clock(names, floor=0.0, link_class=LinkClass.TRAFFIC_TO_ENGINE, cls=None):
    """A clock over `names`, every ordered pair declared at the same floor."""
    registry = LpRegistry()
    ids = [registry.register(LpId(name)) for name in names]
    matrix = LookaheadMatrix(registry)
    for source in ids:
        for target in ids:
            if source != target:
                matrix.declare(source, target, link_class, floor)
    return (cls or ClockAuthority)(registry, matrix), ids


def _ledger(grants):
    """A grant sequence as plain values, for comparing two runs."""
    return [(str(g.lp_id), g.advance_from, g.advance_to) for g in grants]


# --- the arithmetic of the bound ---------------------------------------------


def test_the_bound_is_a_peers_clock_plus_the_floor_into_this_participant():
    clock, (engine, traffic) = _clock(["engine", "traffic-source"], floor=9.0e-3)
    # Both are executing at zero, so each bounds the other at one floor out.
    assert clock.grant_bound(engine) == pytest.approx(9.0e-3)
    assert clock.grant_bound(traffic) == pytest.approx(9.0e-3)


def test_the_bound_takes_the_tightest_peer_and_names_it():
    registry = LpRegistry()
    engine = registry.register(LpId("engine"))
    stage = registry.register(LpId("pipeline-stage-1"))
    traffic = registry.register(LpId("traffic-source"))
    matrix = LookaheadMatrix(registry)
    matrix.declare(traffic, engine, LinkClass.TRAFFIC_TO_ENGINE, 9.0e-3)
    matrix.declare(stage, engine, LinkClass.PIPELINE_STAGE_TO_STAGE, 1.0e-6)
    for source, target in matrix.undeclared():
        matrix.declare(source, target, LinkClass.PREFILL_TO_DECODE, 1.0e-3)
    clock = ClockAuthority(registry, matrix)
    grants = clock.request_advance(engine, 10.0)
    assert [g.lp_id for g in grants] == [engine]
    assert grants[0].advance_to == pytest.approx(1.0e-6)
    assert grants[0].bound_from == stage
    assert clock.now(traffic) == 0.0


def test_a_participant_asking_for_less_than_the_bound_gets_what_it_asked_for():
    clock, (engine, _traffic) = _clock(["engine", "traffic-source"], floor=9.0e-3)
    grants = clock.request_advance(engine, 1.0e-3)
    assert grants[0].advance_to == pytest.approx(1.0e-3)
    assert grants[0].bound == pytest.approx(9.0e-3)


def test_a_grant_records_where_it_came_from_and_how_much_time_it_covers():
    clock, (only,) = _clock(["engine"])
    (grant,) = clock.request_advance(only, 4.0)
    assert isinstance(grant, Grant)
    assert (grant.advance_from, grant.advance_to, grant.seconds) == (0.0, 4.0, 4.0)


# --- the rule reads a clock, not a horizon -----------------------------------


class _HorizonRuleAuthority(ClockAuthority):
    """The rejected rule, implemented here so both can be measured.

    It bounds a participant by where its peers say their next events are instead
    of by where their clocks stand. Everything else is the real thing -- the same
    state machine, the same two safety checks, the same stall check -- so the
    only difference between the two measurements below is the quantity the bound
    is a minimum over.
    """

    def earliest_emission_times(self):
        return {state.lp_id: state.next_event for state in self.states()}


#: The topology both rules are measured on. Two participants, no delay in
#: either direction, both clocks at zero. The engine knows of an event of its
#: own at 20 s. The traffic source is executing at 0 s and knows of no event of
#: its own -- it is part-way through the work that produces one.
ENGINE_KNOWN_EVENT_SECONDS = 20.0


class TestTheRuleReadsAClockAndNotAHorizon:
    def test_the_horizon_rule_backdates_an_event_by_twenty_seconds(self):
        clock, (engine, traffic) = _clock(
            ["engine", "traffic-source"], cls=_HorizonRuleAuthority
        )
        (grant,) = clock.request_advance(engine, ENGINE_KNOWN_EVENT_SECONDS)
        # Nothing bounds the engine, because the traffic source's horizon is
        # empty -- so it is released all the way to its own next event while the
        # traffic source is still standing at zero.
        assert grant.advance_to == ENGINE_KNOWN_EVENT_SECONDS
        assert clock.now(engine) == ENGINE_KNOWN_EVENT_SECONDS
        assert clock.now(traffic) == 0.0
        with pytest.raises(BackdatedEvent) as abort:
            clock.schedule_event(traffic, engine, 0.0)
        assert "20s into its past" in str(abort.value)

    def test_the_clock_rule_holds_the_engine_at_zero_and_accepts_the_event(self):
        clock, (engine, traffic) = _clock(["engine", "traffic-source"])
        assert clock.grant_bound(engine) == 0.0
        assert clock.request_advance(engine, ENGINE_KNOWN_EVENT_SECONDS) == ()
        assert clock.now(engine) == 0.0
        released = clock.schedule_event(traffic, engine, 0.0)
        assert [g.lp_id for g in released] == [engine]
        assert clock.now(engine) == 0.0
        assert clock.state(engine).next_event == 0.0

    def test_the_two_rules_differ_only_in_the_bound_they_compute(self):
        # Stated as one assertion so the result is a comparison and not two
        # separate observations: same topology, same event, 20 s apart.
        horizon, (engine, _traffic) = _clock(
            ["engine", "traffic-source"], cls=_HorizonRuleAuthority
        )
        clocks, (engine2, _traffic2) = _clock(["engine", "traffic-source"])
        assert horizon.grant_bound(engine) == math.inf
        assert clocks.grant_bound(engine2) == 0.0

    def test_the_difference_survives_a_floor_that_is_not_zero(self):
        # The argument is not about zero floors. At 1 ms the clock rule releases
        # the engine by exactly one floor and the horizon rule still releases it
        # by twenty seconds.
        horizon, (engine, traffic) = _clock(
            ["engine", "traffic-source"], floor=1.0e-3, cls=_HorizonRuleAuthority
        )
        clocks, (engine2, traffic2) = _clock(["engine", "traffic-source"], floor=1.0e-3)
        (bad,) = horizon.request_advance(engine, ENGINE_KNOWN_EVENT_SECONDS)
        (good,) = clocks.request_advance(engine2, ENGINE_KNOWN_EVENT_SECONDS)
        assert bad.advance_to == ENGINE_KNOWN_EVENT_SECONDS
        assert good.advance_to == pytest.approx(1.0e-3)
        with pytest.raises(BackdatedEvent):
            horizon.schedule_event(traffic, engine, 1.0e-3)
        clocks.schedule_event(traffic2, engine2, 1.0e-3)


# --- the peer set comes from the registry, not from the sized pairs ---------


class _SizedPairsAsPeersAuthority(ClockAuthority):
    """The peer set taken from the pairs that happen to have been sized.

    The same class of silent error as the horizon rule, and the reason it is
    worth measuring separately: a term missing from a minimum does not behave
    like a term worth zero. A zero lowers the minimum; an absence raises it, so
    the peer it belonged to stops constraining anything -- and the failure
    surfaces one event later, in a different participant from the one that was
    mis-bounded.

    Both overrides are needed, and that is the point. The set-up check would
    refuse this matrix outright, so measuring what the check prevents means
    skipping it first.
    """

    def require_sized_peers(self):
        return

    def peers(self, lp_id):
        absent = {pair: None for pair in self.lookahead.undeclared()}
        return tuple(
            member
            for member in self.registry.ids()
            if member != lp_id and (member, lp_id) not in absent
        )


def _disaggregated(with_the_traffic_leg):
    """Three participants, and one leg that is easy to forget to size.

    A traffic source feeds a prefill role and, directly, a decode role. The
    role-to-role floors are long; the traffic source reaches decode with no
    delay at all, which makes it the tightest peer decode has and the one whose
    omission matters most.
    """
    registry = LpRegistry()
    decode = registry.register(LpId("decode"))
    prefill = registry.register(LpId("prefill"))
    traffic = registry.register(LpId("traffic-source"))
    matrix = LookaheadMatrix(registry)
    for source, target in ((decode, prefill), (prefill, decode)):
        matrix.declare(source, target, LinkClass.PREFILL_TO_DECODE, 10.0)
    for source, target in (
        (decode, traffic),
        (prefill, traffic),
        (traffic, prefill),
    ):
        matrix.declare(source, target, LinkClass.TRAFFIC_TO_ENGINE, 10.0)
    if with_the_traffic_leg:
        matrix.declare(traffic, decode, LinkClass.TRAFFIC_TO_ENGINE, 0.0)
    return registry, matrix, decode, prefill, traffic


class TestThePeerSetComesFromTheRegistry:
    def test_an_omitted_leg_lifts_the_bound_instead_of_tightening_it(self):
        # The two walks, measured side by side on one state. Over the pairs that
        # were sized, the traffic source is not a peer of decode at all, so the
        # tightest constraint decode has simply is not counted -- and the
        # difference runs the wrong way: 10 s of licence rather than none.
        _, incomplete, decode, _prefill, _traffic = _disaggregated(False)
        registry, complete, decode, prefill, traffic = _disaggregated(True)
        standing = {decode: 0.0, prefill: 0.0, traffic: 0.0}
        absent = {pair: None for pair in incomplete.undeclared()}
        over_sized_pairs = min(
            standing[peer] + incomplete.lookahead(peer, decode)
            for peer in registry
            if peer != decode and (peer, decode) not in absent
        )
        over_peers = min(
            standing[peer] + complete.lookahead(peer, decode)
            for peer in registry
            if peer != decode
        )
        assert (over_sized_pairs, over_peers) == (10.0, 0.0)

    def test_the_row_itself_refuses_rather_than_coming_back_short(self):
        # The same claim one level down: asking for the row a minimum would be
        # taken over does not quietly hand back a row with a peer missing.
        _, incomplete, decode, _prefill, _traffic = _disaggregated(False)
        with pytest.raises(KeyError, match="traffic-source -> decode"):
            incomplete.inbound(decode)

    def test_a_peer_set_from_sized_pairs_grants_decode_past_the_traffic_source(self):
        registry, matrix, decode, prefill, traffic = _disaggregated(False)
        clock = _SizedPairsAsPeersAuthority(registry, matrix)
        (grant,) = clock.request_advance(decode, 20.0)
        assert (grant.advance_to, grant.bound_from) == (10.0, prefill)
        assert clock.now(traffic) == 0.0
        # The leg gets sized later, as legs do. The first event carried on it
        # is ten seconds into decode's past, and decode has already decided
        # what it does at every one of those ten seconds.
        matrix.declare(traffic, decode, LinkClass.TRAFFIC_TO_ENGINE, 0.0)
        with pytest.raises(BackdatedEvent) as abort:
            clock.schedule_event(traffic, decode, 0.0)
        assert "10s into its past" in str(abort.value)

    def test_the_clock_refuses_to_start_on_a_matrix_with_that_leg_missing(self):
        # Caught while the run is being configured, and naming the pair, rather
        # than one step later as a number computed from too few terms.
        registry, matrix, _decode, _prefill, _traffic = _disaggregated(False)
        with pytest.raises(KeyError, match="traffic-source -> decode"):
            ClockAuthority(registry, matrix)

    def test_with_the_leg_declared_decode_is_held_and_the_event_is_accepted(self):
        registry, matrix, decode, _prefill, traffic = _disaggregated(True)
        clock = ClockAuthority(registry, matrix)
        assert clock.grant_bound(decode) == 0.0
        assert clock.request_advance(decode, 20.0) == ()
        assert clock.now(decode) == 0.0
        released = clock.schedule_event(traffic, decode, 0.0)
        assert [g.lp_id for g in released] == [decode]


# --- safety: a backdated event stops the run ---------------------------------


def test_an_event_earlier_than_the_senders_own_floor_aborts():
    clock, (engine, traffic) = _clock(["engine", "traffic-source"], floor=9.0e-3)
    with pytest.raises(BackdatedEvent) as abort:
        clock.schedule_event(traffic, engine, 1.0e-3)
    assert "0.009s floor on traffic-source -> engine" in str(abort.value)


def test_the_recipients_clock_check_cannot_fire_under_this_rule():
    # And that is the claim, not an omission. A participant is never granted
    # past a peer's clock plus the floor between them, so an event that clears
    # the sender's floor has already cleared the recipient's clock. The check is
    # kept because it is the only one that catches a bound computed from the
    # wrong quantity, and both measurements of that are above.
    clock, (engine, traffic) = _clock(["engine", "traffic-source"], floor=1.0)
    clock.request_advance(engine, 5.0)
    clock.take_up_grant(engine)
    assert clock.now(engine) == clock.now(traffic) + 1.0
    clock.schedule_event(traffic, engine, 1.0)


def test_the_abort_carries_every_participants_clock_and_status():
    clock, (engine, traffic) = _clock(["engine", "traffic-source"], floor=9.0e-3)
    clock.request_advance(engine, 5.0)
    with pytest.raises(BackdatedEvent) as abort:
        clock.schedule_event(traffic, engine, 0.0)
    table = abort.value.table
    assert "engine" in table and "traffic-source" in table
    assert str(LpStatus.GRANTED) in table
    assert table in str(abort.value)


def test_the_safety_check_is_not_an_assert_statement():
    # `python -O` removes an `assert`. A check that is specified as always on,
    # not behind a flag, cannot be one -- so the module contains none at all.
    tree = ast.parse((CLOCK_PACKAGE / "authority.py").read_text())
    lines = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    assert not lines, f"authority.py asserts at {lines}; raise instead"


def test_a_waiting_participant_cannot_produce_an_event():
    # The rule looks past a waiting participant to the event it is waiting for.
    # That is only sound because a waiting participant emits nothing.
    clock, (engine, traffic) = _clock(["engine", "traffic-source"])
    clock.request_advance(engine, 5.0)
    assert clock.state(engine).status is LpStatus.BLOCKED_ON_MESSAGE
    with pytest.raises(ClockAbort) as abort:
        clock.schedule_event(engine, traffic, 9.0)
    assert "waiting for time to be granted" in str(abort.value)


def test_a_participant_cannot_declare_an_event_behind_its_own_clock():
    clock, (engine, _traffic) = _clock(["engine", "traffic-source"], floor=1.0)
    clock.request_advance(engine, 5.0)
    clock.take_up_grant(engine)
    with pytest.raises(BackdatedEvent):
        clock.request_advance(engine, 0.5)


# --- a stall stops the run ---------------------------------------------------


def test_everyone_waiting_with_no_known_event_aborts_loudly():
    clock, (engine, traffic) = _clock(["engine", "traffic-source"])
    assert clock.request_advance(engine) == ()
    with pytest.raises(ClockDeadlock) as abort:
        clock.request_advance(traffic)
    assert "none knows of a future event" in str(abort.value)
    assert "engine" in abort.value.table and "traffic-source" in abort.value.table


def test_a_floor_above_zero_delays_the_stall_by_one_step_and_no_more():
    # A participant with an empty horizon is still released by one floor: it is
    # stepping over idle time and finding nothing there. What matters is that it
    # is one step and not an indefinite crawl -- the stall is reached and
    # reported rather than approached forever.
    clock, (engine, traffic) = _clock(["engine", "traffic-source"], floor=9.0e-3)
    (stepped,) = clock.request_advance(engine)
    assert stepped.advance_to == pytest.approx(9.0e-3)
    clock.take_up_grant(engine)
    assert clock.request_advance(engine) == ()
    with pytest.raises(ClockDeadlock):
        clock.request_advance(traffic)


def test_one_participant_with_nothing_left_to_do_is_the_same_stall():
    clock, (only,) = _clock(["engine"])
    with pytest.raises(ClockDeadlock):
        clock.request_advance(only)


def test_a_known_event_anywhere_is_enough_to_release_everyone():
    # The converse of the stall: whenever any participant knows of an event, at
    # least one grant is issued. That is what makes the stall check exact rather
    # than a guess that something has gone wrong.
    clock, ids = _clock(["alpha", "beta", "gamma"])
    assert clock.request_advance(ids[0]) == ()
    assert clock.request_advance(ids[1]) == ()
    grants = clock.request_advance(ids[2], 3.0)
    assert [g.advance_to for g in grants] == [3.0, 3.0, 3.0]


# --- no grant steps over an event the clock has already accepted -------------


def _legal_runs(seed_count, floor_choices, size_choices):
    """Drive the clock through legal operations and watch every grant.

    The invariant is the one the whole module exists for, stated over the
    clock's own record rather than over any one call: at the moment a grant is
    issued, the participant receiving it must not be carried past an event that
    has already been accepted on it and that it has not yet been released to
    reach. A run that steps over one produces no error -- the participant simply
    advances, and the event arrives in its past later, in code that has no way
    to know -- so nothing here can rely on an abort being raised.

    Driven by a fixed pseudo-random sequence rather than a library, so the runs
    are the same on every machine and in every process.
    """
    stepped_over = []
    state = 12345
    for run in range(seed_count):
        size = size_choices[run % len(size_choices)]
        floor = floor_choices[run % len(floor_choices)]
        clock, ids = _clock([f"lp-{index}" for index in range(size)], floor=floor)
        pending = {lp_id: [] for lp_id in ids}
        for _ in range(24):
            state = (state * 1103515245 + 12345) % (1 << 31)
            choice = state % 3
            actor = ids[(state >> 8) % size]
            if choice == 0 and clock.held_grant(actor) is not None:
                clock.take_up_grant(actor)
                continue
            if choice == 1 and clock.state(actor).status.may_produce_events:
                target = ids[(state >> 16) % size]
                if target == actor:
                    continue
                floor_out = clock.lookahead.lookahead(actor, target)
                when = clock.now(actor) + floor_out + ((state >> 4) % 5) * 0.25
                clock.schedule_event(actor, target, when)
                pending[target].append(when)
                pending[target] = [
                    ts for ts in pending[target] if ts > clock.now(target)
                ]
                continue
            if clock.held_grant(actor) is not None:
                continue
            horizon = clock.now(actor) + 1.0 + ((state >> 12) % 8)
            try:
                grants = clock.request_advance(actor, horizon)
            except ClockDeadlock:
                break
            for grant in grants:
                held = pending[grant.lp_id]
                if held and grant.advance_to > min(held):
                    stepped_over.append(
                        (run, str(grant.lp_id), min(held), grant.advance_to)
                    )
                pending[grant.lp_id] = [ts for ts in held if ts > grant.advance_to]
    return stepped_over


def test_no_grant_steps_over_an_accepted_event():
    # 400 legal runs across two to five participants and three floors. The
    # single case this generalises was a grant of 0 s -> 10 s over an event
    # accepted at 0 s, with nothing raised: the clock had replaced its record of
    # that event with the participant's own declared horizon.
    stepped_over = _legal_runs(400, (0.0, 1.0e-3, 1.0), (2, 3, 4, 5))
    assert stepped_over == []


def test_a_declared_horizon_does_not_erase_an_accepted_event():
    # The single case, kept beside the fuzzer because it names the mechanism.
    clock, (engine, traffic) = _clock(["engine", "traffic-source"])
    clock.schedule_event(traffic, engine, 0.0)
    assert clock.state(engine).next_event == 0.0
    clock.request_advance(engine, 10.0)
    assert clock.state(engine).next_event == 0.0
    clock.request_advance(traffic, math.inf)
    assert clock.now(engine) == 0.0


def test_taking_up_a_grant_drops_only_what_the_grant_reached():
    # Both halves of the rule, on one clock. The event at 5 s outlives a grant
    # that only reached 2 s; the horizon the grant did reach does not, because a
    # horizon pinned behind the clock freezes the run on grants of no span.
    clock, (engine, traffic) = _clock(["engine", "traffic-source"], floor=3.0)
    clock.schedule_event(traffic, engine, 5.0)
    clock.request_advance(engine, math.inf)
    grant = clock.take_up_grant(engine)
    assert grant.advance_to == 3.0
    assert clock.state(engine).next_event == 5.0
    clock.request_advance(traffic, 4.0)
    clock.take_up_grant(traffic)
    clock.request_advance(engine, math.inf)
    clock.take_up_grant(engine)
    assert clock.now(engine) == 5.0
    assert clock.state(engine).next_event == math.inf


def test_a_second_event_in_flight_outlives_reaching_the_first():
    # The fuzzer's finding, and the reason the accepted side is a list rather
    # than one slot: reaching the event at 1.5 s must not forget the one at
    # 3 s, which the participant has not been told about either.
    clock, (engine, traffic) = _clock(["engine", "traffic-source"], floor=1.5)
    clock.schedule_event(traffic, engine, 3.0)
    clock.schedule_event(traffic, engine, 1.5)
    assert clock.state(engine).next_event == 1.5
    clock.request_advance(engine, math.inf)
    clock.take_up_grant(engine)
    assert clock.now(engine) == 1.5
    assert clock.state(engine).next_event == 3.0


def test_a_run_that_reaches_its_horizon_keeps_moving():
    # The failure mode of folding the horizon in without ever dropping it: the
    # record stays pinned at a timestamp already passed and every later grant
    # has no span. Measured at 0.5 s, forever, before the drop was added.
    clock, (only,) = _clock(["engine"])
    for step in range(1, 6):
        clock.request_advance(only, step * 0.5)
        clock.take_up_grant(only)
    assert clock.now(only) == 2.5


# --- determinism -------------------------------------------------------------


def test_grants_come_out_in_participant_order_not_arrival_order():
    orders = ((0, 1, 2), (2, 1, 0), (1, 2, 0), (2, 0, 1))
    ledgers = []
    for order in orders:
        clock, ids = _clock(["alpha", "beta", "gamma"])
        issued = []
        for index in order:
            issued += list(clock.request_advance(ids[index], 5.0))
        ledgers.append(_ledger(issued))
    assert ledgers[0] == [
        ("alpha", 0.0, 5.0),
        ("beta", 0.0, 5.0),
        ("gamma", 0.0, 5.0),
    ]
    assert ledgers.count(ledgers[0]) == len(orders)


def test_a_tie_on_the_bound_keeps_the_first_peer_in_the_total_order():
    clock, ids = _clock(["alpha", "beta", "gamma"])
    grants = clock.request_advance(ids[2], 5.0)
    assert grants == ()
    assert clock.grant_bound(ids[2]) == 0.0
    clock, ids = _clock(["alpha", "beta", "gamma"], floor=1.0e-3)
    (grant,) = clock.request_advance(ids[2], 5.0)
    assert grant.bound_from == ids[0]


# --- one participant is a local clock ----------------------------------------


def test_one_participant_advances_straight_to_its_own_next_event():
    clock, (only,) = _clock(["engine"])
    assert clock.grant_bound(only) == math.inf
    for horizon in (4.0, 9.0, 9.5):
        (grant,) = clock.request_advance(only, horizon)
        assert (grant.advance_to, grant.bound, grant.bound_from) == (
            horizon,
            math.inf,
            None,
        )
        clock.take_up_grant(only)
    assert clock.now(only) == 9.5
    assert clock.grants_issued(only) == 3
    assert clock.grants_issued() == 3


# --- every floor at zero is a single event loop ------------------------------


def test_every_floor_at_zero_is_correct_serialized_and_not_an_error():
    clock, ids = _clock(["alpha", "beta", "gamma"], floor=0.0)
    horizons = {"alpha": 5.0, "beta": 1.0, "gamma": 7.0}
    issued = []
    for lp_id in ids:
        issued += list(clock.request_advance(lp_id, horizons[str(lp_id)]))
    # Nobody passes the earliest event anywhere in the run, so exactly one
    # event's worth of time is released at a time.
    earliest = min(horizons.values())
    assert [(str(g.lp_id), g.advance_to) for g in issued] == [
        ("alpha", earliest),
        ("beta", earliest),
        ("gamma", earliest),
    ]
    assert all(clock.now(lp_id) == earliest for lp_id in ids)


def test_a_floor_above_zero_buys_overlap_and_nothing_else():
    # At a zero floor nobody moves until the last participant has declared its
    # horizon: one event's worth of time is released at a time. At 1 ms each is
    # released as soon as it asks, by one floor, without waiting for the others.
    # That is the whole of what a floor buys, and it is speed, not correctness.
    serialized, ids = _clock(["alpha", "beta", "gamma"], floor=0.0)
    overlapped, others = _clock(["alpha", "beta", "gamma"], floor=1.0e-3)
    horizons = (5.0, 1.0, 7.0)
    held = [serialized.request_advance(ids[i], horizons[i]) for i in (0, 1)]
    moved = [overlapped.request_advance(others[i], horizons[i]) for i in (0, 1)]
    assert held == [(), ()]
    assert [len(grants) for grants in moved] == [1, 1]
    assert [grants[0].advance_to for grants in moved] == pytest.approx([1.0e-3, 1.0e-3])


# --- the state machine -------------------------------------------------------


def test_a_participant_walks_running_waiting_granted_running():
    clock, (engine, traffic) = _clock(["engine", "traffic-source"], floor=1.0)
    assert clock.state(engine).status is LpStatus.RUNNING
    clock.request_advance(traffic, 30.0)
    assert clock.state(traffic).status is LpStatus.GRANTED
    clock.take_up_grant(traffic)
    assert clock.state(traffic).status is LpStatus.RUNNING
    clock.request_advance(engine, 40.0)
    clock.take_up_grant(engine)
    assert clock.request_advance(engine, 40.0) == ()
    assert clock.state(engine).status is LpStatus.BLOCKED_ON_MESSAGE


def test_the_state_of_every_participant_reads_in_the_total_order():
    clock, _ids = _clock(["alpha", "beta", "gamma"])
    states = clock.states()
    assert [str(state.lp_id) for state in states] == ["alpha", "beta", "gamma"]
    assert all(isinstance(state, LpState) for state in states)
    assert all(state.next_event == math.inf for state in states)


def test_asking_again_while_holding_a_grant_is_refused():
    clock, (only,) = _clock(["engine"])
    clock.request_advance(only, 4.0)
    with pytest.raises(ValueError, match="has not taken up"):
        clock.request_advance(only, 8.0)


def test_taking_up_a_grant_that_was_not_issued_is_refused():
    clock, (only,) = _clock(["engine"])
    with pytest.raises(ValueError, match="holds no grant"):
        clock.take_up_grant(only)


def test_an_event_on_yourself_is_refused():
    clock, (engine, _traffic) = _clock(["engine", "traffic-source"])
    with pytest.raises(ValueError, match="cannot schedule an event on itself"):
        clock.schedule_event(engine, engine, 1.0)


# --- what the clock refuses to be built from ---------------------------------


def test_a_pair_with_no_declared_floor_is_refused_at_construction():
    registry = LpRegistry()
    engine = registry.register(LpId("engine"))
    traffic = registry.register(LpId("traffic-source"))
    matrix = LookaheadMatrix(registry)
    matrix.declare(traffic, engine, LinkClass.TRAFFIC_TO_ENGINE, 9.0e-3)
    with pytest.raises(KeyError, match="engine -> traffic-source"):
        ClockAuthority(registry, matrix)


def test_a_clock_with_no_participants_is_refused():
    registry = LpRegistry()
    with pytest.raises(ValueError, match="at least one participant"):
        ClockAuthority(registry, LookaheadMatrix(registry))


def test_a_participant_registered_after_the_clock_was_built_is_refused():
    registry = LpRegistry()
    registry.register(LpId("engine"))
    clock = ClockAuthority(registry, LookaheadMatrix(registry))
    late = registry.register(LpId("decode"))
    with pytest.raises(KeyError, match="not a participant"):
        clock.request_advance(late, 1.0)


def test_a_start_time_that_is_not_a_number_of_seconds_is_refused():
    registry = LpRegistry()
    registry.register(LpId("engine"))
    with pytest.raises(ValueError, match="finite"):
        ClockAuthority(registry, LookaheadMatrix(registry), start_time=math.inf)


def test_the_clock_can_start_somewhere_other_than_zero():
    registry = LpRegistry()
    engine = registry.register(LpId("engine"))
    clock = ClockAuthority(registry, LookaheadMatrix(registry), start_time=100.0)
    (grant,) = clock.request_advance(engine, 101.0)
    assert (grant.advance_from, grant.advance_to) == (100.0, 101.0)

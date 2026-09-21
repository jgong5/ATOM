# SPDX-License-Identifier: MIT
"""ATOM's deployments driven against the clock, with no ATOM and no GPU.

What each group here is defending:

* **The collapse.** A tensor- or data-parallel group is one participant however
  wide it is, and only three things make another one: a prefill/decode role
  boundary, a pipeline stage, and an independent replica behind the router. The
  six deployments are asserted by participant count, because that count is the
  whole of the claim -- 512 GPUs is 65 participants, and 4 GPUs is 2.
* **Every deployment runs.** Each one is driven to the end of a trace shaped
  like the measured one, and no safety check in the clock or in the harness
  fires on any of them.
* **No grant steps over an undelivered message.** The clock cannot check this
  about itself; it knows what it accepted, and only the driver knows what the
  participant was handed. Every run here drives a participant holding two events
  at **two distinct timestamps** and grants it the earlier one while the later
  one stands, because two events at one timestamp are one horizon record and a
  clock that kept a single record would carry them correctly. A clock kept to
  one record is run against a deployment below and fails it.
* **The driver decides the cost.** Three disciplines, one deployment, same
  trace, same schedule: the two that are not the contract pay over five hundred
  times as many grants, and on a deployment whose tightest floor is a
  microsecond neither of them finishes at all.
* **A finished run is a deadlock.** Every clean run here ends by raising
  `ClockDeadlock` with every participant reporting it had finished, because
  there is no call that says so. The test asserts that pairing rather than
  working around it.

The trace is shortened. The full-length one -- 106 prefill and 4,346 decode
steps, the shape of a measured prior run -- is measured separately and reported
in the pull request; these runs keep that shape and cut the length, so this
stays a tier rather than becoming a job. The 65-participant run is cut hardest
because the rule's own arithmetic is quadratic in the participant count and
costs about 1.3 ms per grant there.
"""

import collections
import dataclasses

import pytest

from atom.compass.clock import ClockAuthority, ClockDeadlock

from . import participants as behaviours
from .deployments import DEPLOYMENTS, Deployment, Replica, Role, clock_parts
from .harness import Discipline, SteppedOverEvent, SyntheticRun
from .participants import DESIGN_WORKLOAD, Message, scaled

#: Short enough that the whole file is a few seconds, long enough that every
#: deployment serves more than one request and holds more than one event.
BRIEF = scaled(DESIGN_WORKLOAD, requests=16, decode_steps=2)

#: The 65-participant deployment gets a shorter one still; see the module note.
BRIEFEST = scaled(DESIGN_WORKLOAD, requests=4, decode_steps=1)

#: Deployment, expected participants, expected GPUs.
EXPECTED = (
    ("tp4-one-server", 2, 4),
    ("tp4-prefill-tp4-decode", 3, 8),
    ("tp8-role-disaggregated", 3, 16),
    ("tp8-pp4", 5, 32),
    ("eight-replicas-each-tp8", 17, 128),
    ("eight-replicas-each-tp8-pp4", 65, 512),
)


def workload_for(deployment):
    return BRIEFEST if len(deployment.participants) > 17 else BRIEF


def expected_steps(deployment, workload):
    """Every step of every request, counted once per stage that charges it."""
    total = 0
    for role, steps in (
        (Role.ENGINE, workload.prefill_steps + workload.decode_steps),
        (Role.PREFILL, workload.prefill_steps),
        (Role.DECODE, workload.decode_steps),
    ):
        replicas = deployment.of_role(role)
        if replicas:
            total += workload.requests * steps * replicas[0].pp
    return total


def run(deployment, **kwargs):
    return SyntheticRun(deployment, workload_for(deployment), **kwargs).run()


@pytest.fixture(scope="module", params=DEPLOYMENTS, ids=lambda d: d.name)
def report(request):
    """Every deployment, driven once under the contract, shared by the tests."""
    return run(request.param)


class TestTheCollapse:
    """What makes a participant, and what only makes GPUs."""

    @pytest.mark.parametrize("name,participants,gpus", EXPECTED)
    def test_each_deployment_has_the_participants_the_collapse_gives_it(
        self, name, participants, gpus
    ):
        deployment = next(d for d in DEPLOYMENTS if d.name == name)
        assert len(deployment.participants) == participants
        assert deployment.gpus == gpus

    def test_widening_tensor_and_data_parallelism_adds_no_participant(self):
        """Four ways wide and eight ways wide are the same one participant."""
        narrow = Deployment("narrow", (Replica(Role.ENGINE, 0, 1, 1),))
        wide = Deployment("wide", (Replica(Role.ENGINE, 0, 8, 1),))
        assert len(narrow.participants) == len(wide.participants) == 2
        assert (narrow.gpus, wide.gpus) == (1, 8)

    def test_a_pipeline_is_the_one_group_the_collapse_does_not_cover(self):
        """Every stage is its own participant; the width above it is not."""
        staged = Deployment("staged", (Replica(Role.ENGINE, 0, 8, 4),))
        assert len(staged.participants) == 5
        assert staged.gpus == 32

    def test_every_ordered_pair_of_participants_has_a_declared_floor(self):
        """The matrix is complete, so no peer drops out of a minimum."""
        for deployment in DEPLOYMENTS:
            _registry, matrix = clock_parts(deployment)
            assert matrix.undeclared() == ()


class TestEveryDeploymentRuns:
    """The six arrangements, driven end to end."""

    def test_the_run_ends_with_every_participant_finished(self, report):
        assert report.unfinished == ()

    def test_every_step_of_the_trace_was_charged(self, report):
        deployment = next(d for d in DEPLOYMENTS if d.name == report.deployment)
        assert report.steps == expected_steps(deployment, workload_for(deployment))

    def test_the_clock_and_the_driver_agree_the_run_is_over(self, report):
        """A clean run ends on the deadlock, and that is the only way to know.

        Nothing in the protocol says a participant has finished, so the state
        that ends a healthy run is the state that reports a hung one. What
        separates them is the participants' own answer, which the clock cannot
        be asked for -- so a stop with no unfinished participants is the end of
        a run and a stop with some is a real deadlock.
        """
        assert "none knows of a future event" in report.stopped_by
        assert report.unfinished == ()

    def test_no_grant_moved_a_participant_past_an_undelivered_message(self, report):
        """Checked on every grant as the run was driven; a failure raises there.

        The assertion below is that grants were issued at all, because a run
        that issued none would satisfy the check vacuously.
        """
        assert report.grants > 0

    def test_participants_held_two_events_at_two_different_times(self, report):
        """Counting messages does not reach the case the second record is for.

        Two messages at one timestamp are one horizon record, so a run that
        only ever produced those would pass a count of events in flight while
        exercising nothing a single-record horizon gets wrong. What has to
        happen is a participant holding two events at two times *and* being
        granted the earlier one while the later one still stands -- both
        counted here, and the mutant below fails on exactly this.
        """
        assert report.events_in_flight >= 2
        assert report.event_times_in_flight >= 2
        assert report.grants_that_kept_a_later_event > 0

    def test_the_protocol_cost_stays_near_one_grant_per_step_per_participant(
        self, report
    ):
        """Where a lost driver discipline shows up before it shows up anywhere else.

        One grant per participant per event is the floor -- a participant that
        does something has to be granted the time to do it in. Measured on this
        shortened trace it is 1.7 to 3.6, and on the full-length trace 1.3 to
        2.1; the two differ because a shorter trace pays the same idle rounds
        over fewer steps, and the band below is set wide enough to hold both.
        The extra over 1.0 is the grant an idle participant has to take up
        before it is allowed to ask again. A driver that stopped serving the
        furthest-behind participant last would not be slightly outside this
        band; on the pipelined deployments it would be outside it by three
        orders of magnitude.
        """
        per_step = report.grants / (report.steps * report.participants)
        assert 1.0 < per_step < 8.0


class TestEveryInventoryCategoryIsDriven:
    """A behaviour that nothing exercises is not modelling anything."""

    def test_steps_are_charged_blocks_are_real_and_the_drain_keeps_a_cadence(
        self, monkeypatch
    ):
        """One run has to reach all three of the shapes the inventory names.

        The count that matters is the **blocking** read, not the call that
        asks for none. `receive` is called on every grant and most of those
        calls ask for zero messages, so counting calls says nothing; counting
        `inbox.get(timeout=...)` counts the real parks, and that number is
        pinned to the messages rather than merely compared with them -- every
        message is read exactly once, so anything other than equality means a
        message was delivered without a park or a park happened without one.

        The drain is pinned the same way. `poll` is reached only through
        `_drain`, so equality is the statement -- a drain tick that stopped
        polling would break it, and the inequality it replaces could not.
        """
        seen = {"poll": 0, "_drain": 0}
        blocking_reads = collections.Counter()

        def counting(owner, name):
            original = getattr(owner, name)

            def wrapped(self, *args, _original=original, _name=name):
                seen[_name] += 1
                return _original(self, *args)

            monkeypatch.setattr(owner, name, wrapped)

        def counted_receive(self, count):
            if count:
                blocking_reads[str(self.lp_id)] += count
            return [
                self.inbox.get(timeout=behaviours.REAL_BLOCK_SECONDS)
                for _ in range(count)
            ]

        monkeypatch.setattr(behaviours._Participant, "receive", counted_receive)
        counting(behaviours._Participant, "poll")
        counting(behaviours.EngineStage, "_drain")
        deployment = DEPLOYMENTS[1]
        report = SyntheticRun(deployment, BRIEF).run()
        assert report.steps > 0
        assert sum(blocking_reads.values()) == report.messages
        assert set(blocking_reads) == {str(lp_id) for lp_id in deployment.participants}
        assert seen["_drain"] > 0
        assert seen["poll"] == seen["_drain"]

    def test_an_idle_stretch_is_crossed_in_one_grant(self, report):
        """The idle step loop jumps rather than spinning, which is the point.

        Arrivals in this trace are seconds apart and the tightest floor is a
        microsecond, so a run that crawled through the gaps would show a
        longest grant of about a floor. Every deployment here shows one at
        least six orders of magnitude larger than that.
        """
        assert report.longest_grant_seconds > 1.0


class TestTheDriverDecidesTheCost:
    """Same clock, same trace, same schedule, three disciplines."""

    DEPLOYMENT = DEPLOYMENTS[3]
    TRACE = scaled(DESIGN_WORKLOAD, requests=4, decode_steps=2)
    CAP = 200_000

    def _run(self, discipline):
        return SyntheticRun(self.DEPLOYMENT, self.TRACE, discipline, self.CAP).run()

    def test_the_contract_finishes_and_the_other_two_do_not(self):
        """A microsecond floor is where the difference stops being an opinion."""
        contract = self._run(Discipline.PARK_LAGGARD_LAST)
        assert contract.unfinished == ()
        for discipline in (
            Discipline.PARK_IN_NAME_ORDER,
            Discipline.TAKE_UP_AND_REASK,
        ):
            crawling = self._run(discipline)
            assert crawling.unfinished != ()
            assert crawling.grants >= self.CAP
            assert crawling.grants > 20 * contract.grants
            assert crawling.modelled_seconds < contract.modelled_seconds

    def test_serving_the_laggard_last_is_what_reaches_the_parked_state(self):
        """It is the ordering, not the asking, that the tight floor punishes.

        Both disciplines here ask until they are refused. The only difference
        is which participant is offered its turn last, and on a two-role
        deployment with millisecond floors that alone costs half as many grants
        again while charging exactly the same steps. Both are capped, because
        the losing one does not always stop on its own: on a longer trace than
        this one, served in name order, a three-participant run that has charged
        every step of its work keeps being granted time -- 400,000 grants and
        1,271 s of modelled time past the last of its 168 steps, about 3 ms a
        grant -- rather than reaching the state that ends a run.
        """
        pair = DEPLOYMENTS[1]
        trace = scaled(DESIGN_WORKLOAD, requests=4, decode_steps=4)
        last = SyntheticRun(pair, trace, Discipline.PARK_LAGGARD_LAST, self.CAP).run()
        named = SyntheticRun(pair, trace, Discipline.PARK_IN_NAME_ORDER, self.CAP).run()
        assert last.steps == named.steps
        assert last.grants < named.grants
        assert last.unfinished == named.unfinished == ()


class _OneRecordHorizon(ClockAuthority):
    """A clock that keeps only the earliest event accepted for a participant.

    The shape a horizon has when its accepted side is one slot instead of a
    list. It is correct for a participant holding one event, and correct for a
    participant holding several at one timestamp, because the horizon is the
    minimum either way. It loses the second of two events at two timestamps:
    reaching the first empties the slot, and nothing then holds the participant
    at the second.
    """

    def schedule_event(self, source, target, timestamp):
        grants = super().schedule_event(source, target, timestamp)
        accepted = self._accepted[target]
        if len(accepted) > 1:
            self._accepted[target] = [min(accepted)]
            self._restate_horizon(target)
        return grants


class TestTheChecksFire:
    """A harness whose checks cannot fail is not checking anything."""

    def test_a_clock_that_keeps_one_horizon_record_is_caught(self):
        """The defect the second record exists for, run against this harness.

        Not a broken driver: the driver is untouched and the clock is the thing
        narrowed. It is caught because the traffic source offers a pair of
        requests a tokenisation apart, so the engine they go to holds two
        events at two timestamps and is granted the first while the second
        stands -- the one shape in which the record that was dropped was the
        one carrying an event.
        """
        run_under_test = SyntheticRun(
            DEPLOYMENTS[1], scaled(DESIGN_WORKLOAD, 4, 2), clock=_OneRecordHorizon
        )
        with pytest.raises((SteppedOverEvent, ClockDeadlock)) as raised:
            run_under_test.run()
        assert raised.type is SteppedOverEvent

    def test_the_one_record_clock_passes_the_run_that_never_holds_two_times(self):
        """The mutant is discriminating, which is the other half of the claim.

        A check that failed whatever it was pointed at would prove nothing
        about the shape it is named for. Stretch the tokenisation slice past
        twice the admission delay and the engine is released to the first
        request before the second is offered, so the run holds one event at a
        time -- and the narrowed clock, which is wrong, completes it with
        nothing raised. That is the state the six deployments were in before
        the slice was charged, and it is why a grant count taken from them was
        no evidence about the second record.
        """
        trace = scaled(DESIGN_WORKLOAD, 4, 2)
        one_at_a_time = dataclasses.replace(trace, tokenise_seconds=44.0e-3)
        report = SyntheticRun(DEPLOYMENTS[1], one_at_a_time).run()
        assert report.event_times_in_flight == 1
        assert report.grants_that_kept_a_later_event == 0
        narrowed = SyntheticRun(
            DEPLOYMENTS[1], one_at_a_time, clock=_OneRecordHorizon
        ).run()
        assert narrowed.unfinished == ()
        assert narrowed.grants == report.grants

    def test_a_message_held_back_is_reported_as_a_step_over(self):
        """Break the delivery, not the clock, and the harness must notice."""
        deployment = DEPLOYMENTS[1]
        run_under_test = SyntheticRun(deployment, scaled(DESIGN_WORKLOAD, 4, 2))
        run_under_test._hand_over = lambda lp_id, grant: 0
        with pytest.raises((SteppedOverEvent, ClockDeadlock)) as raised:
            run_under_test.run()
        assert raised.type is SteppedOverEvent

    def test_a_message_never_sent_leaves_its_participant_unfinished(self):
        """The same stop, told apart by who says they finished.

        A run that ends cleanly and a run that hangs raise one exception with
        one table. Dropping every response leaves the traffic source waiting
        for something nobody will send, and the only thing that distinguishes
        that stop from the healthy one above is the participants' own answer.
        """
        run_under_test = SyntheticRun(DEPLOYMENTS[1], scaled(DESIGN_WORKLOAD, 4, 2))
        delivered = run_under_test.send

        def drop_responses(source, target, when, payload):
            if payload[0] is not Message.RESPONSE:
                delivered(source, target, when, payload)

        run_under_test.send = drop_responses
        report = run_under_test.run()
        assert "none knows of a future event" in report.stopped_by
        assert report.unfinished == ("traffic-source",)

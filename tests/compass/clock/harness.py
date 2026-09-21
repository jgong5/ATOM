# SPDX-License-Identifier: MIT
"""Drives synthetic participants against the clock and reports what it cost.

The driver is not a detail of this file. **Whether a run steps over idle time or
crawls through it is decided here, by three to five orders of magnitude**, and
the rule is the same in both cases. A participant that is executing, or holding
a grant it has not taken up, bounds every peer at its own clock plus one floor.
So the clock can only look past everybody to the next event anywhere at a moment
when *every* participant is parked, and everything below is about reaching that
moment.

Two things are needed, and the second was found by measurement here rather than
being known beforehand.

**Ask until refused, then stay parked.** A participant asks, takes up whatever it
is given, and asks again until the clock refuses it; a refused participant does
not ask again until a peer's state changes. A driver that instead takes up one
grant and goes round its own loop keeps somebody executing at all times, and the
run creeps forward one floor at a time -- finer, not coarser, the tighter the
floor.

**Serve the participant whose clock is furthest behind last.** The
furthest-behind participant is the one holding everybody else's bound, and while
it stands still the others can catch up to it and park. Serve it first instead
and it advances one floor, which puts every peer one floor behind it, and they
are all granted that floor, and one of them is then the laggard. Measured on the
four-request trace: the two-role deployment costs 1,723 grants served in name
order and 729 served laggard-last, and the pipelined one does not finish at all
in name order -- 2,000,000 grants bought 0.454 s of modelled time and 11 of its
660 steps, because its tightest floor is a microsecond. Laggard-last runs the
same trace in 7,997 grants.

The driver also holds the check the clock cannot make for itself. The clock
knows the events it has accepted for a participant; only the harness knows which
of them it has actually handed over. **No grant may move a participant past the
timestamp of a message it has not been given**, and that is asserted on every
grant of every run here. It is the shape a single-event driver cannot see:
reaching the first of two events in flight must not forget the second.

**There is no way to say a participant has finished.** A run that ends with
everybody parked and nothing left to do is exactly the deadlock condition, so a
clean run and a hung one raise the same exception carrying the same table. The
driver therefore treats that exception as the end of the run and then asks the
participants themselves whether they had in fact finished, because the clock
cannot be asked. A stop that reports `unfinished` names is a real deadlock; one
that reports none is a run that ended.
"""

import time
from dataclasses import dataclass
from enum import Enum

from atom.compass.clock import ClockAuthority, ClockDeadlock, LpStatus

from .deployments import DEPLOYMENTS, clock_parts
from .participants import DESIGN_WORKLOAD, EngineStage, build, scaled


class Discipline(Enum):
    """How the driver decides who asks the clock for time, and when."""

    #: Ask until refused, and serve the furthest-behind participant last. The
    #: contract: it is the only one of the three that reaches the all-parked
    #: state, which is the only state in which the clock skips idle time.
    PARK_LAGGARD_LAST = "park-laggard-last"

    #: Ask until refused, but serve in name order. Correct, and it pays one
    #: floor per pass for every participant that is behind.
    PARK_IN_NAME_ORDER = "park-in-name-order"

    #: Take up one grant and go round the driver's own loop again. Always leaves
    #: somebody executing, so nothing is ever granted past one floor.
    TAKE_UP_AND_REASK = "take-up-and-re-ask"


class SteppedOverEvent(AssertionError):
    """A participant was moved past a message it had not been handed."""


class GrantsExhausted(Exception):
    """A run hit its grant cap. Only a crawling discipline ever does."""


@dataclass(frozen=True)
class RunReport:
    """What one run did, and what it cost the protocol."""

    deployment: str
    participants: int
    gpus: int
    discipline: str
    grants: int
    messages: int
    steps: int
    modelled_seconds: float
    wall_seconds: float
    events_in_flight: int
    longest_grant_seconds: float
    unfinished: tuple[str, ...]
    stopped_by: str

    @property
    def grants_per_participant(self) -> float:
        return self.grants / self.participants

    def __str__(self) -> str:
        return (
            f"{self.deployment:<28} {self.participants:>4} participants "
            f"{self.gpus:>4} GPUs {self.grants:>9} grants "
            f"{self.grants_per_participant:>8.0f}/participant "
            f"{self.steps:>7} steps {self.modelled_seconds:>7.1f}s modelled "
            f"{self.longest_grant_seconds:>7.2f}s longest grant "
            f"{self.wall_seconds:>7.1f}s wall"
        )


class SyntheticRun:
    """One deployment, one workload, one discipline, driven to the end."""

    def __init__(
        self,
        deployment,
        workload,
        discipline=Discipline.PARK_LAGGARD_LAST,
        grant_cap=None,
    ):
        registry, matrix = clock_parts(deployment)
        self.deployment = deployment
        self.workload = workload
        self.discipline = discipline
        self.grant_cap = grant_cap
        self.clock = ClockAuthority(registry, matrix)
        self.people = build(deployment, workload)
        self.ids = registry.ids()
        self.pending = {lp_id: [] for lp_id in self.ids}
        self.messages = 0
        self.events_in_flight = 0
        self.longest_grant_seconds = 0.0

    def send(self, source, target, when, payload):
        """Place a message on a peer: on the clock first, then in the harness."""
        self.clock.schedule_event(source, target, when)
        self.pending[target].append((when, payload))
        self.messages += 1
        self.events_in_flight = max(self.events_in_flight, len(self.pending[target]))

    def run(self):
        """Drive until the clock says nothing can move, and report."""
        started = time.perf_counter()
        stopped_by = "the driver ran out of participants to drive"
        try:
            while True:
                moved = False
                for lp_id in self._service_order():
                    if (
                        self.clock.state(lp_id).status
                        is not LpStatus.BLOCKED_ON_MESSAGE
                    ):
                        self._drive(lp_id)
                        moved = True
                if not moved:
                    break
        except ClockDeadlock as stop:
            stopped_by = stop.reason
        except GrantsExhausted:
            stopped_by = f"the grant cap of {self.grant_cap} was reached"
        return self._report(time.perf_counter() - started, stopped_by)

    def _service_order(self):
        """Who is offered a turn, and in what order.

        Furthest-behind last under the contract, because that participant is
        holding everyone else's bound and the others can only park while it
        stands still. Ties by name, so the order is the same on every run.
        """
        if self.discipline is Discipline.PARK_LAGGARD_LAST:
            return sorted(self.ids, key=lambda lp_id: (-self.clock.now(lp_id), lp_id))
        return self.ids

    def _drive(self, lp_id):
        """Give one participant its turn: take up, act, and ask again."""
        person = self.people[lp_id]
        if self.clock.state(lp_id).status is LpStatus.RUNNING:
            self.clock.request_advance(lp_id, person.horizon(self.clock.now(lp_id)))
        while self.clock.state(lp_id).status is LpStatus.GRANTED:
            self._check_cap()
            grant = self.clock.take_up_grant(lp_id)
            self.longest_grant_seconds = max(self.longest_grant_seconds, grant.seconds)
            self._refuse_step_over(lp_id, grant)
            person.on_time(grant.advance_to, self._hand_over(lp_id, grant), self)
            if self.discipline is Discipline.TAKE_UP_AND_REASK:
                return
            self.clock.request_advance(lp_id, person.horizon(grant.advance_to))

    def _check_cap(self):
        if self.grant_cap is not None and self.clock.grants_issued() >= self.grant_cap:
            raise GrantsExhausted

    def _refuse_step_over(self, lp_id, grant):
        """The one thing the clock cannot check about itself.

        It knows what it accepted for this participant; only the harness knows
        what the participant was actually handed. A grant reaching past an
        undelivered message is the silent failure the whole protocol exists to
        prevent, and it does not crash on its own.
        """
        missed = [when for when, _ in self.pending[lp_id] if when < grant.advance_to]
        if missed:
            raise SteppedOverEvent(
                f"{grant} moved past {len(missed)} message(s) it was never handed, "
                f"the earliest at {min(missed):.9g}s\n\n{self.clock.lp_table()}"
            )

    def _hand_over(self, lp_id, grant):
        """Give the participant every message whose time it has now reached."""
        held = self.pending[lp_id]
        due = [payload for when, payload in held if when <= grant.advance_to]
        if due:
            self.pending[lp_id] = [row for row in held if row[0] > grant.advance_to]
            for payload in due:
                self.people[lp_id].inbox.put(payload)
        return len(due)

    def _report(self, wall_seconds, stopped_by):
        return RunReport(
            deployment=self.deployment.name,
            participants=len(self.ids),
            gpus=self.deployment.gpus,
            discipline=self.discipline.value,
            grants=self.clock.grants_issued(),
            messages=self.messages,
            steps=sum(
                person.steps
                for person in self.people.values()
                if isinstance(person, EngineStage)
            ),
            modelled_seconds=max(self.clock.now(lp_id) for lp_id in self.ids),
            wall_seconds=wall_seconds,
            events_in_flight=self.events_in_flight,
            longest_grant_seconds=self.longest_grant_seconds,
            unfinished=tuple(
                str(lp_id) for lp_id in self.ids if not self.people[lp_id].finished
            ),
            stopped_by=stopped_by,
        )


def measure(workload=DESIGN_WORKLOAD, deployments=DEPLOYMENTS, grant_cap=None):
    """Run every deployment once under the contract, smallest first."""
    return tuple(
        SyntheticRun(deployment, workload, grant_cap=grant_cap).run()
        for deployment in deployments
    )


def disciplines(deployment, workload, grant_cap):
    """The same deployment under all three disciplines, so the gap is measured.

    The two that are not the contract are capped, because neither finishes on a
    deployment whose tightest floor is a microsecond.
    """
    return tuple(
        SyntheticRun(deployment, workload, discipline, grant_cap).run()
        for discipline in Discipline
    )


def main():
    """Print the grant table the design's sizing is compared against."""
    for report in measure():
        print(report, flush=True)
        print(f"    stopped by: {report.stopped_by}")
        print(
            f"    unfinished: {', '.join(report.unfinished) or 'none'}; "
            f"messages {report.messages}; most events in flight on one "
            f"participant {report.events_in_flight}"
        )
    print()
    for report in disciplines(DEPLOYMENTS[3], scaled(DESIGN_WORKLOAD, 4), 2_000_000):
        print(
            f"{report.discipline:<20} {report.grants:>9} grants "
            f"{report.steps:>6} of 660 steps "
            f"{report.modelled_seconds:>9.3f}s modelled "
            f"{'finished' if not report.unfinished else 'did not finish'}",
            flush=True,
        )


if __name__ == "__main__":
    main()

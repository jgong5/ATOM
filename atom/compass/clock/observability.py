# SPDX-License-Identifier: MIT
"""What a run of the clock says about itself: a timeline, a dump, a summary.

The clock is the only component that knows where every participant stands, so
it is the only place a global record of the run can be written from. Three
records come out of it and nothing else.

**A timeline.** One line per granted advance -- who moved, from when, to when,
what ended the advance, and the bound that decided it. It is off unless a run
asks for it, because a run handing out millions of grants would spend more time
describing itself than simulating, and it is the evidence a causality report
needs: when a message lands in a participant's past, both participants' clock
histories up to that moment are already written down.

**A dump, when nothing can move.** Every participant's clock, the state it is
in, what is holding it, and the whole row of floors that produced its bound,
with the term that binds marked. The clock decides when this is printed; this
module decides what it says.

**A summary, once, at the end.** It is split in two, and the split is the
point. One half is a function of the simulated schedule alone: the same
configuration produces it byte for byte however the participants happened to
interleave, and whatever machine ran it. The other half is what the run cost to
produce -- grants, wall seconds, watchdog warnings -- and every number in it
moves with the order requests arrived in, with how the participants drive the
clock, and with the host. A reader handed one number cannot tell which kind it
is, so the record says so per half rather than leaving it to be assumed.

Grant count in particular is not a property of a configuration. The same
topology and the same events cost three grants or two million depending on
whether a participant that has been refused waits for a peer to move or asks
again immediately. So a grant count is recorded with the discipline that
produced it, the discipline is a required argument rather than a defaulted one,
and where the timeline is on, the share of grants that ended on a peer's bound
is recorded beside it -- a run that crawls is almost all bound-terminated, and
one that steps over idle is not.

Nothing here reads a clock, opens a socket or touches a device. Wall seconds
arrive as a number somebody else measured, which is also what makes the summary
auditable away from the machine that produced it: every field is a plain value.
"""

import enum
import math
from dataclasses import dataclass, field

from .identity import LpId

#: A simulated run is worth having only if it is faster than the run it models.
#: This is the ratio the speed result is read against.
SPEED_TARGET_RATIO = 5.0


def _seconds(value: float) -> str:
    return "none" if value == math.inf else f"{value:.9g}s"


def _emission(value: float) -> str:
    return "never" if value == math.inf else f"{value:.9g}s"


# --- the timeline ------------------------------------------------------------


@dataclass(frozen=True)
class TimelineRecord:
    """One granted advance, as five columns.

    `event` says what ended the advance and is one word: `horizon` for an
    advance that stopped on an event the participant itself knew of, `bound`
    for one a peer cut short, `release` for a grant of no span, which tells a
    parked participant its wait is over rather than that time moved.
    """

    lp_id: LpId
    virtual_time_from: float
    virtual_time_to: float
    event: str
    detail: str

    def __str__(self) -> str:
        return (
            f"{self.lp_id} {self.virtual_time_from:.9g} "
            f"{self.virtual_time_to:.9g} {self.event} {self.detail}"
        )


class TimelineLog:
    """Every granted advance, in the order the clock issued them.

    Append-only: a record is never revised, because the value of the thing is
    that it says what was believed at the moment it was written. Held in memory
    and optionally handed line by line to `sink`, a callable somebody else
    supplies -- this module does not open a file for the same reason it does not
    open a socket.
    """

    def __init__(self, sink=None) -> None:
        if sink is not None and not callable(sink):
            raise TypeError(f"sink must be callable, got {type(sink).__name__}")
        self._records: list[TimelineRecord] = []
        self._sink = sink

    def record(self, grant, next_event: float) -> TimelineRecord:
        """Write one granted advance. `next_event` is the horizon it was cut against."""
        if grant.advance_to == grant.advance_from:
            event = "release"
        elif grant.advance_to == grant.bound:
            event = "bound"
        else:
            event = "horizon"
        pinned = "--" if grant.bound_from is None else str(grant.bound_from)
        entry = TimelineRecord(
            grant.lp_id,
            grant.advance_from,
            grant.advance_to,
            event,
            f"bound={_seconds(grant.bound)} from={pinned} next={_seconds(next_event)}",
        )
        self._records.append(entry)
        if self._sink is not None:
            self._sink(str(entry))
        return entry

    def records(self) -> tuple[TimelineRecord, ...]:
        """Every record written, oldest first."""
        return tuple(self._records)

    def lines(self) -> tuple[str, ...]:
        """The same records, rendered one line each."""
        return tuple(str(entry) for entry in self._records)

    def ended_at_peer_bound(self) -> int:
        """How many advances a peer cut short rather than the participant's own event.

        The fingerprint of how a run drove the clock. A participant that asks
        again the moment it is given time is cut by a peer almost every time;
        one that waits until a peer moves is not.
        """
        return sum(1 for entry in self._records if entry.event == "bound")

    def __len__(self) -> int:
        return len(self._records)


# --- the dump ----------------------------------------------------------------


class StallKind(enum.Enum):
    """What kind of state the clock is in when no participant can be granted."""

    #: At least one participant can still act, so this is not a stall at all.
    NOT_STALLED = "not-stalled"

    #: Everyone is parked and nobody knows of any future event. A run whose work
    #: is finished looks exactly like this, and so does one waiting forever.
    NO_FUTURE_EVENT = "no-future-event"

    #: Everyone is parked and somebody knows of a future event that nobody can
    #: be released to reach. No finished run looks like this.
    EVENT_UNREACHABLE = "event-unreachable"

    def __str__(self) -> str:
        return self.value


_HEADLINE = {
    StallKind.NOT_STALLED: (
        "at least one participant is able to act, so this is not a stall. If "
        "the clock printed this, the state below disagrees with the rule that "
        "decided to print it."
    ),
    StallKind.NO_FUTURE_EVENT: (
        "every participant is parked and none knows of a future event, so "
        "nothing exists anywhere that could release any of them. A run that has "
        "finished its work reaches exactly this state, and so does one waiting "
        "for a message that will never be sent. The protocol carries no way for "
        "a participant to say it has finished, so this dump reports the state "
        "and does not name the cause."
    ),
    StallKind.EVENT_UNREACHABLE: (
        "every participant is parked, and at least one knows of a future event "
        "that no participant can be released to reach. This is not a finished "
        "run: work remains and no participant was given the time to do it."
    ),
}


def stall_kind(authority) -> StallKind:
    """Classify the clock's current state, without changing it."""
    states = authority.states()
    if any(state.status.may_produce_events for state in states):
        return StallKind.NOT_STALLED
    if any(math.isfinite(state.next_event) for state in states):
        return StallKind.EVENT_UNREACHABLE
    return StallKind.NO_FUTURE_EVENT


def deadlock_dump(authority) -> str:
    """Every participant's clock, state, what holds it, and the row that bounds it.

    The blocked-on column is derived, not declared: the protocol records that a
    participant asked for time and was refused, never a reason in the
    participant's own words. So what this can honestly say is which peer, at
    which time, is holding it -- which is the actionable half -- and it says
    that rather than presenting a derivation as a declaration.
    """
    kind = stall_kind(authority)
    earliest = authority.earliest_emission_times()
    lines = [f"the clock can issue no grant ({kind})", "", _HEADLINE[kind], ""]
    ungranted = []
    for state in authority.states():
        row = authority.lookahead.inbound(state.lp_id)
        bound, pinned_by = _row_bound(row, earliest)
        grants = authority.grants_issued(state.lp_id)
        if not grants:
            ungranted.append(state.lp_id)
        lines.append(
            f"{state.lp_id} {state.status} clock {_seconds(state.now)} "
            f"next event {_seconds(state.next_event)} grants {grants} "
            f"blocked on {_blocked_on(row, bound, pinned_by)}"
        )
        for link in row:
            candidate = earliest[link.source] + link.floor_seconds
            binds = " <- binds" if link.source == pinned_by else ""
            lines.append(
                f"    from {link.source} floor {link.floor_seconds:.9g}s "
                f"earliest {_emission(earliest[link.source])} "
                f"gives {_emission(candidate)}{binds}"
            )
        if not row:
            lines.append("    no peers, so nothing bounds this participant")
    if kind is StallKind.NO_FUTURE_EVENT:
        lines += ["", _what_this_cannot_settle(authority, ungranted)]
    return "\n".join(lines)


def _row_bound(row, earliest: dict) -> tuple[float, LpId | None]:
    bound = math.inf
    pinned_by = None
    for link in row:
        candidate = earliest[link.source] + link.floor_seconds
        if candidate < bound:
            bound = candidate
            pinned_by = link.source
    return bound, pinned_by


def _blocked_on(row, bound: float, pinned_by) -> str:
    if not row:
        return "nothing: it has no peer, so nothing bounds it"
    if pinned_by is None:
        return "no peer, because none of them can produce an event at any time"
    return f"{pinned_by}, which could still produce an event at {_seconds(bound)}"


def _what_this_cannot_settle(authority, ungranted) -> str:
    ids = authority.registry.ids()
    furthest = max(authority.now(lp_id) for lp_id in ids)
    if ungranted:
        never = ", ".join(str(lp_id) for lp_id in ungranted)
        evidence = (
            f"{len(ungranted)} of {len(ids)} participants were never granted any "
            f"time at all ({never}), which is what a run that never started looks "
            "like; a run that finished would have moved all of them"
        )
    else:
        evidence = (
            "every participant was granted time and the furthest clock reached "
            f"{_seconds(furthest)}, which is what a run that finished looks like; "
            "a run stuck part way through would look the same"
        )
    return (
        "what the state above can and cannot settle: "
        + evidence
        + ". That is a reading of the same state and not a second source, so it "
        "narrows the question rather than answering it. Settling it needs a "
        "participant able to say that it has finished, so that a clock with no "
        "unfinished participant left is a completed run and one with an "
        "unfinished participant is a deadlock. Nothing in the protocol carries "
        "that today."
    )


# --- the summary -------------------------------------------------------------


class DriverDiscipline(enum.Enum):
    """How the participants in a run asked the clock for time.

    It decides the grant count by orders of magnitude, so a grant count without
    it is not a measurement of anything.
    """

    #: A participant refused time waits for a peer to move before asking again.
    #: Every participant parks, so the rule can look past the parked clocks to
    #: where the next event actually is, and a long idle stretch costs a grant
    #: or two rather than one per lookahead floor.
    PARK_WHEN_REFUSED = "park-when-refused"

    #: A participant takes up whatever time it is given and immediately asks for
    #: more. A grant cut short by a peer looks like progress and invites another
    #: request, so an idle stretch is walked one floor at a time -- and a tighter
    #: floor makes it finer, not cheaper.
    RE_ASK_AFTER_TAKE_UP = "re-ask-after-take-up"

    #: The run did not say. The grant count is then a number about this run and
    #: about nothing else.
    UNDECLARED = "undeclared"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class RefusalTally:
    """How much of a run the cost model declined to price, and for what reasons."""

    count: int = 0
    steps: int = 0
    refused_predicted_seconds: float = 0.0
    predicted_seconds: float = 0.0
    reasons: tuple[tuple[str, int], ...] = ()

    @classmethod
    def of(
        cls,
        reasons,
        steps: int,
        refused_predicted_seconds: float = 0.0,
        predicted_seconds: float = 0.0,
    ) -> "RefusalTally":
        """Tally one reason string per refusal into counts, ordered by reason.

        Sorted rather than kept in the order the refusals happened, because the
        tally is compared between runs and that order is not reproducible.
        """
        counted: dict[str, int] = {}
        for reason in reasons:
            counted[reason] = counted.get(reason, 0) + 1
        return cls(
            sum(counted.values()),
            steps,
            refused_predicted_seconds,
            predicted_seconds,
            tuple(sorted(counted.items())),
        )

    @property
    def fraction_of_steps(self) -> float:
        return self.count / self.steps if self.steps else 0.0

    @property
    def fraction_of_predicted_seconds(self) -> float:
        if not self.predicted_seconds:
            return 0.0
        return self.refused_predicted_seconds / self.predicted_seconds

    def record(self) -> dict:
        return {
            "count": self.count,
            "steps": self.steps,
            "fraction_of_steps": self.fraction_of_steps,
            "fraction_of_predicted_seconds": self.fraction_of_predicted_seconds,
            "reasons": [[reason, count] for reason, count in self.reasons],
        }


@dataclass(frozen=True)
class DetectorState:
    """What the causality detectors saw. A straggler above zero invalidates the run."""

    stragglers: int = 0
    watchdog_warnings: int = 0
    clock_lint: str = "not-run"


@dataclass(frozen=True)
class RunSummary:
    """The run, in two halves: what it simulated, and what it cost to simulate.

    `schedule_record` is a function of the simulated schedule and of nothing
    else, so two runs of one configuration produce it identically and it can be
    compared byte for byte. `cost_record` is the opposite, and says so in its
    own first field. `as_record` is both, and every value in it is a number, a
    string, or a list of those, so a result can be read without the machine that
    produced it.
    """

    final_clocks: tuple[tuple[str, float], ...]
    simulated_seconds: float
    wall_seconds: float
    grants: tuple[tuple[str, int], ...]
    discipline: DriverDiscipline
    lazy_traces: int = 0
    lazy_trace_wall_seconds: float = 0.0
    grants_recorded: int | None = None
    grants_ended_at_peer_bound: int | None = None
    detectors: DetectorState = field(default_factory=DetectorState)
    refusals: RefusalTally = field(default_factory=RefusalTally)

    @classmethod
    def of(
        cls,
        authority,
        wall_seconds: float,
        discipline: DriverDiscipline,
        lazy_traces: int = 0,
        lazy_trace_wall_seconds: float = 0.0,
        detectors: DetectorState | None = None,
        refusals: RefusalTally | None = None,
        timeline: TimelineLog | None = None,
    ) -> "RunSummary":
        """Read the clock once, at the end, and take everything else by value.

        `discipline` has no default. The grant count is orders of magnitude
        apart on one topology depending on it, so a summary that let it be
        omitted would be reporting a number nobody can read.
        """
        if not isinstance(discipline, DriverDiscipline):
            raise TypeError(
                "discipline must be a DriverDiscipline naming how this run asked "
                f"for time, got {type(discipline).__name__}; the grant count "
                "means nothing without it"
            )
        ids = authority.registry.ids()
        return cls(
            tuple((str(lp_id), authority.now(lp_id)) for lp_id in ids),
            max(authority.now(lp_id) for lp_id in ids) - authority.start_time,
            float(wall_seconds),
            tuple((str(lp_id), authority.grants_issued(lp_id)) for lp_id in ids),
            discipline,
            lazy_traces,
            lazy_trace_wall_seconds,
            None if timeline is None else len(timeline),
            None if timeline is None else timeline.ended_at_peer_bound(),
            detectors if detectors is not None else DetectorState(),
            refusals if refusals is not None else RefusalTally(),
        )

    @property
    def grants_issued(self) -> int:
        return sum(count for _, count in self.grants)

    @property
    def speed_ratio(self) -> float:
        """Simulated seconds per wall second. Unlimited if no wall time was spent."""
        if not self.wall_seconds:
            return math.inf
        return self.simulated_seconds / self.wall_seconds

    @property
    def meets_speed_target(self) -> bool:
        return self.speed_ratio >= SPEED_TARGET_RATIO

    def schedule_record(self) -> dict:
        """The half that is a function of the simulated schedule alone."""
        return {
            "final_clocks": [[name, clock] for name, clock in self.final_clocks],
            "simulated_seconds": self.simulated_seconds,
            "refusals": self.refusals.record(),
            "stragglers": self.detectors.stragglers,
            "clock_lint": self.detectors.clock_lint,
        }

    def cost_record(self) -> dict:
        """The half that says what the run cost, and what each number moves with."""
        return {
            "varies_with": ["arrival order", "driver discipline", "host"],
            "driver_discipline": str(self.discipline),
            "grants_issued": [[name, count] for name, count in self.grants],
            "grants_issued_total": self.grants_issued,
            "grants_recorded": self.grants_recorded,
            "grants_ended_at_peer_bound": self.grants_ended_at_peer_bound,
            "wall_seconds": self.wall_seconds,
            "speed_ratio": self.speed_ratio,
            "speed_target_ratio": SPEED_TARGET_RATIO,
            "meets_speed_target": self.meets_speed_target,
            "lazy_traces": self.lazy_traces,
            "lazy_trace_wall_seconds": self.lazy_trace_wall_seconds,
            "watchdog_warnings": self.detectors.watchdog_warnings,
        }

    def as_record(self) -> dict:
        """Both halves, kept apart, as plain values."""
        return {"schedule": self.schedule_record(), "cost": self.cost_record()}

# SPDX-License-Identifier: MIT
"""The rule that hands out simulated time, and the state it hands it out from.

One participant may move its clock to a time only when no other participant can
still produce an event earlier than that. The rule that expresses it is:

    bound(i)  = min over every peer j of ( earliest_emission(j) + L[j -> i] )
    advance(i) = min( bound(i), the next event i knows of )

and the whole of the correctness argument is in the first term. It is the peer's
**current** clock that enters the bound whenever the peer is executing, never
the peer's next event. Using the next event looks right and is not: a peer that
is executing at 0 while knowing of nothing until 10 can still schedule an event
at 0, and a participant granted up to 10 on the strength of that 10 receives it
ten seconds into its own past. Nothing crashes when that happens. The run
finishes and produces a plausible answer, which is why the check below is an
abort and not a warning.

A participant that has asked to advance and been refused is the one case where
the current clock is *not* the answer, because it cannot produce anything until
it is granted. Its earliest emission is the soonest it could do anything at all:
its own next event, or a message from a participant that can still move,
whichever comes first. Those bounds refer to each other -- a waiting participant
can be woken by another waiting participant -- so they are relaxed together to a
fixpoint. Floors are never negative, so that relaxation is a shortest-path
computation and settles.

Two shapes fall out rather than being special-cased.

* **One participant.** There is no peer, the bound is unlimited, and every grant
  is to the participant's own next event. That is a plain local clock.
* **Every floor at zero.** No participant is ever granted past the earliest
  event anywhere in the run, so exactly one event's worth of time is released at
  a time and the whole arrangement is a single event loop spread over several
  processes. It is correct, it is serialized, and it is not an error: the time
  this saves comes from stepping over idle stretches rather than from running
  participants at the same moment of real time.

The minimum is over the participants that exist, never over the delays that
happen to have been written down, and the reason is worth stating because it
reads backwards. **An absent floor is not a cautious version of a zero floor.**
A zero adds a term to the minimum and can only lower it; an absence removes a
term, which *raises* it, so the peer it belonged to stops constraining anything
at all. The mistake therefore hands out more time than the peer allows rather
than less, and it surfaces one event later and in a different participant from
the one that was mis-bounded. So the peer set comes from the registry, and an
unsized pair is refused while the run is being configured.

Nothing here opens a socket, resolves an address, starts a process or reads a
real clock. Carrying a request from a participant to this rule is somebody
else's job.
"""

import math

from .identity import LpId
from .lookahead import LookaheadMatrix
from .registry import LpRegistry
from .state import Grant, LpState, LpStatus


class ClockAbort(Exception):
    """The run cannot be trusted any further, and stops.

    Carries the table of every participant, because the value of one of these
    is entirely in being able to see which clock was where when it fired.
    """

    def __init__(self, reason: str, table: str) -> None:
        super().__init__(f"{reason}\n\n{table}")
        self.reason = reason
        self.table = table


class BackdatedEvent(ClockAbort):
    """An event was scheduled earlier than the recipient's clock already stands.

    The run is over at this point. There is no recovery: the recipient has
    already made decisions at times after the event, so the schedule it produced
    is not the schedule the modelled system would have produced.
    """


class ClockDeadlock(ClockAbort):
    """No participant can be given time, and none knows of a future event.

    Raised where a timeout would otherwise sit. A timeout that releases a run
    which never became valid is worse than no check at all: the run completes,
    reports no failures, and every number taken off it is wrong.
    """


class ClockAuthority:
    """Holds every participant's clock and decides who may move.

    Built once, after every participant is registered and every floor declared.
    Registering a participant afterwards is refused rather than picked up, so a
    participant that joined late cannot be silently left out of the bound that
    is supposed to be over all of them.
    """

    def __init__(
        self,
        registry: LpRegistry,
        lookahead: LookaheadMatrix,
        start_time: float = 0.0,
    ) -> None:
        if not len(registry):
            raise ValueError("a clock needs at least one participant")
        start = float(start_time)
        if not math.isfinite(start):
            raise ValueError(f"start_time must be a finite number, got {start_time!r}")
        self._registry = registry
        self._lookahead = lookahead
        self._ids = registry.ids()
        self._now = {lp_id: start for lp_id in self._ids}
        # Two records, and the horizon is the smaller of them. `_declared` is
        # what the participant itself last said, which is only ever the events
        # it has *seen*. `_accepted` is every event a peer has placed on it that
        # it has not yet been released to reach -- possibly still in the hands
        # of whatever carries messages, and therefore invisible to the
        # participant. Folding the two into one slot loses whichever was written
        # second, and losing the accepted one releases a participant straight
        # past a timestamp it has not reached. One slot is also not enough for
        # the accepted side on its own: two events in flight, and reaching the
        # first would forget the second.
        self._declared = {lp_id: math.inf for lp_id in self._ids}
        self._accepted = {lp_id: [] for lp_id in self._ids}
        self._next = {lp_id: math.inf for lp_id in self._ids}
        self._status = {lp_id: LpStatus.RUNNING for lp_id in self._ids}
        self._held = {lp_id: None for lp_id in self._ids}
        self._grants = {lp_id: 0 for lp_id in self._ids}
        self.require_sized_peers()
        # The row every walk below takes a minimum over, built once. Membership
        # is fixed when the clock is built and a floor is a declared constant,
        # so rebuilding the row per grant -- and re-hashing a pair of identities
        # per element while doing it -- is arithmetic paid for nothing. It is
        # the dominant cost at a large participant count, well ahead of the
        # relaxation it was assumed to be.
        self._row = {
            target: tuple(
                (source, self._lookahead.lookahead(source, target))
                for source in self.peers(target)
            )
            for target in self._ids
        }

    # --- reading the state ---------------------------------------------------

    @property
    def registry(self) -> LpRegistry:
        """Who is taking part, and in what order they are served."""
        return self._registry

    @property
    def lookahead(self) -> LookaheadMatrix:
        """The declared floors the bound is walked over."""
        return self._lookahead

    def peers(self, lp_id: LpId) -> tuple[LpId, ...]:
        """Every other participant, in the total order.

        The rule is a minimum over participants, so this is where the set it
        ranges over is decided, and it is decided by the registry. Deriving it
        from the pairs that happen to have been sized would quietly drop a peer,
        and dropping a term from a minimum lifts the bound rather than
        tightening it -- which releases more time than the dropped peer allows.
        """
        self._participant(lp_id)
        return tuple(member for member in self._ids if member != lp_id)

    def require_sized_peers(self) -> None:
        """Refuse to start unless every ordered pair has a declared floor.

        Once, here, while the run is being configured -- not on the path that
        hands out time, which then has nothing to re-check. A matrix with a hole
        in it is a set-up mistake, and it is worth the whole list of missing
        pairs at once rather than the first one to be walked over.
        """
        self._lookahead.require_complete()

    def state(self, lp_id: LpId) -> LpState:
        """One participant's clock, horizon and status."""
        self._participant(lp_id)
        return LpState(lp_id, self._now[lp_id], self._next[lp_id], self._status[lp_id])

    def states(self) -> tuple[LpState, ...]:
        """Every participant's row, in the registry's total order."""
        return tuple(self.state(lp_id) for lp_id in self._ids)

    def now(self, lp_id: LpId) -> float:
        """Where one participant's clock stands, in simulated seconds."""
        self._participant(lp_id)
        return self._now[lp_id]

    def held_grant(self, lp_id: LpId) -> Grant | None:
        """The grant a participant has been issued and not yet taken up."""
        self._participant(lp_id)
        return self._held[lp_id]

    def grants_issued(self, lp_id: LpId | None = None) -> int:
        """How many grants have been issued, in total or to one participant.

        The protocol's own cost, and the figure that says whether an arrangement
        with many participants is affordable at the floors it declared.
        """
        if lp_id is None:
            return sum(self._grants[known] for known in self._ids)
        self._participant(lp_id)
        return self._grants[lp_id]

    def earliest_emission_times(self) -> dict[LpId, float]:
        """The earliest simulated time each participant could produce an event.

        For one that is executing this is its current clock. For one that is
        parked it is the soonest it could do anything at all -- its own next
        event, or a message from a participant that can still move -- and never
        earlier than where its clock already stands.

        Relaxed to a fixpoint because parked participants bound each other. One
        pass per parked participant is enough: floors are not negative, so this
        is a shortest-path computation from the participants that can move, and
        only the parked ones are relaxed at all. A run in which most
        participants are executing therefore settles in a single pass.
        """
        earliest = {}
        waiting = []
        for lp_id in self._ids:
            if self._status[lp_id].may_produce_events:
                earliest[lp_id] = self._now[lp_id]
            else:
                earliest[lp_id] = max(self._now[lp_id], self._next[lp_id])
                waiting.append(lp_id)
        for _ in waiting:
            settled = True
            for target in waiting:
                best = earliest[target]
                for source, floor in self._row[target]:
                    best = min(best, earliest[source] + floor)
                best = max(best, self._now[target])
                if best < earliest[target]:
                    earliest[target] = best
                    settled = False
            if settled:
                break
        return earliest

    def grant_bound(self, lp_id: LpId) -> float:
        """How far this participant could be allowed to go, ignoring its own events.

        `+inf` when it has no peers, which is the single-participant case.
        """
        self._participant(lp_id)
        bound, _ = self._bound(lp_id, self.earliest_emission_times())
        return bound

    def lp_table(self) -> str:
        """Every participant's clock, horizon, status and current bound.

        The minimum an abort needs to be actionable. A richer record of a run
        belongs with whatever is keeping the run's history, not here.
        """
        earliest = self.earliest_emission_times()
        heading = (
            f"{'participant':<20} {'status':<19} {'clock':<14} "
            f"{'next event':<14} {'bound':<14} bound from"
        )
        rows = [heading]
        for lp_id in self._ids:
            bound, pinned_by = self._bound(lp_id, earliest)
            rows.append(
                f"{lp_id!s:<20} {self._status[lp_id]!s:<19} "
                f"{self._seconds(self._now[lp_id]):<14} "
                f"{self._seconds(self._next[lp_id]):<14} "
                f"{self._seconds(bound):<14} "
                f"{'--' if pinned_by is None else pinned_by!s}"
            )
        return "\n".join(rows)

    # --- what a participant asks for -----------------------------------------

    def request_advance(
        self, lp_id: LpId, next_event: float = math.inf
    ) -> tuple[Grant, ...]:
        """Ask to move past the current clock, declaring the next event known of.

        Returns every grant this made possible, for any participant, in the
        registry's total order -- one participant asking can release another
        that was waiting on it. The caller's own grant is in there if it got
        one; an empty result means it stays parked.

        Grants are returned rather than delivered, and they are ordered by
        participant rather than by who asked first, so the same run produces the
        same sequence however the requests happened to interleave in real time.

        The declared horizon is folded into what the clock already knows rather
        than replacing it. A participant declares only the events it has *seen*,
        and an event a peer has already placed on it may still be in the hands
        of whatever carries messages. Replacing would erase the only record of
        that event, and the participant would then be released straight past a
        timestamp it has not reached -- which is precisely the failure the rest
        of this module is built to make impossible, arriving through the
        bookkeeping instead of through the rule.
        """
        self._participant(lp_id)
        if self._status[lp_id] is LpStatus.GRANTED:
            raise ValueError(
                f"{lp_id} holds a grant to {self._seconds(self._now[lp_id])} that it "
                "has not taken up; take it up before asking for more time"
            )
        horizon = self._simulated_seconds(next_event, "next_event")
        if horizon < self._now[lp_id]:
            raise BackdatedEvent(
                f"{lp_id} declares its next event at "
                f"{self._seconds(horizon)}, behind its own clock at "
                f"{self._seconds(self._now[lp_id])}",
                self.lp_table(),
            )
        self._declared[lp_id] = horizon
        self._restate_horizon(lp_id)
        self._status[lp_id] = LpStatus.BLOCKED_ON_MESSAGE
        return self._resolve()

    def _restate_horizon(self, lp_id: LpId) -> None:
        """The earliest event the clock knows of for this participant."""
        horizon = self._declared[lp_id]
        for when in self._accepted[lp_id]:
            horizon = min(horizon, when)
        self._next[lp_id] = horizon

    def take_up_grant(self, lp_id: LpId) -> Grant:
        """Collect the grant issued to this participant and start executing.

        Taking up a grant consumes the events the grant reaches, and only those.
        Both halves matter. Keeping what the grant reached leaves the horizon
        pinned at a timestamp already passed, and the run freezes on grants of
        no span. Dropping what it did not reach throws away an event still in
        flight and re-opens, one grant further out, the gap this bookkeeping
        exists to close.
        """
        self._participant(lp_id)
        grant = self._held[lp_id]
        if grant is None:
            raise ValueError(
                f"{lp_id} holds no grant; it is {self._status[lp_id]} at "
                f"{self._seconds(self._now[lp_id])}"
            )
        self._held[lp_id] = None
        self._status[lp_id] = LpStatus.RUNNING
        self._accepted[lp_id] = [
            when for when in self._accepted[lp_id] if when > grant.advance_to
        ]
        if self._declared[lp_id] <= grant.advance_to:
            self._declared[lp_id] = math.inf
        self._restate_horizon(lp_id)
        return grant

    def schedule_event(
        self, source: LpId, target: LpId, timestamp: float
    ) -> tuple[Grant, ...]:
        """Place an event from one participant on another, and check it is legal.

        Two things have to hold, and both abort the run rather than warn. The
        event may not be earlier than the sender's own clock plus the floor
        between the pair -- that is the sender obeying the delay it declared.
        And it may not be earlier than the recipient's clock -- that is the rule
        above having done its job, and it is the one that fails when the bound
        was computed from the wrong quantity.

        Returns any grants the event released, since an event is exactly what a
        parked participant may have been waiting for.
        """
        self._participant(source)
        self._participant(target)
        if source == target:
            raise ValueError(
                f"{source} cannot schedule an event on itself; a participant's "
                "own future events are declared when it asks to advance"
            )
        when = self._simulated_seconds(timestamp, "timestamp")
        if not self._status[source].may_produce_events:
            raise ClockAbort(
                f"{source} is waiting for time to be granted and cannot produce "
                f"an event; it tried to schedule one on {target} at "
                f"{self._seconds(when)}",
                self.lp_table(),
            )
        floor = self._lookahead.lookahead(source, target)
        soonest = self._now[source] + floor
        if when < soonest:
            raise BackdatedEvent(
                f"{source} scheduled an event on {target} at "
                f"{self._seconds(when)}, earlier than the "
                f"{self._seconds(soonest)} it is allowed -- its own clock at "
                f"{self._seconds(self._now[source])} plus the "
                f"{self._seconds(floor)} floor on {source} -> {target}",
                self.lp_table(),
            )
        if when < self._now[target]:
            raise BackdatedEvent(
                f"{source} scheduled an event on {target} at "
                f"{self._seconds(when)}, but {target}'s clock already stands at "
                f"{self._seconds(self._now[target])} -- "
                f"{self._seconds(self._now[target] - when)} into its past. "
                f"{target} has already decided what it does after that event, "
                "so this run no longer describes the system being modelled",
                self.lp_table(),
            )
        self._accepted[target].append(when)
        self._restate_horizon(target)
        return self._resolve()

    # --- the rule ------------------------------------------------------------

    def _resolve(self) -> tuple[Grant, ...]:
        """Issue every grant the current state allows, in the total order.

        One pass is exact. A parked participant's earliest emission is already
        equal to the time it would be granted, so granting it changes nothing
        that a second pass would read differently; and a participant that is not
        parked is not a candidate.
        """
        earliest = self.earliest_emission_times()
        issued = []
        for lp_id in self._ids:
            if self._status[lp_id].may_produce_events:
                continue
            bound, pinned_by = self._bound(lp_id, earliest)
            horizon = self._next[lp_id]
            advance_to = min(horizon, bound)
            standing = self._now[lp_id]
            if advance_to < standing:
                raise ClockAbort(
                    f"{lp_id}'s clock stands at {self._seconds(standing)} but a "
                    f"peer could still produce an event for it at "
                    f"{self._seconds(advance_to)}; it was allowed past a time it "
                    "should have been held at",
                    self.lp_table(),
                )
            if not math.isfinite(advance_to):
                continue
            if advance_to == standing and horizon > standing:
                continue
            grant = Grant(lp_id, standing, advance_to, bound, pinned_by)
            self._now[lp_id] = advance_to
            self._status[lp_id] = LpStatus.GRANTED
            self._held[lp_id] = grant
            self._grants[lp_id] += 1
            issued.append(grant)
        if not issued:
            self._refuse_to_stall()
        return tuple(issued)

    def _bound(
        self, target: LpId, earliest: dict[LpId, float]
    ) -> tuple[float, LpId | None]:
        """`min over peers j of ( earliest_emission(j) + L[j -> target] )`.

        Ties keep the peer that comes first in the total order, so the name a
        stalled run reports is the same name on every run.
        """
        bound = math.inf
        pinned_by = None
        for source, floor in self._row[target]:
            candidate = earliest[source] + floor
            if candidate < bound:
                bound = candidate
                pinned_by = source
        return bound, pinned_by

    def _refuse_to_stall(self) -> None:
        """Abort if nothing can move. Never wait to see whether it frees up."""
        waiting = [
            lp_id for lp_id in self._ids if not self._status[lp_id].may_produce_events
        ]
        if len(waiting) != len(self._ids):
            return
        known = [lp_id for lp_id in waiting if math.isfinite(self._next[lp_id])]
        if not known:
            raise ClockDeadlock(
                "every participant is waiting and none knows of a future event, "
                "so no event exists anywhere that could release any of them",
                self.lp_table(),
            )
        pending = ", ".join(
            f"{lp_id} knows of an event at {self._seconds(self._next[lp_id])}"
            for lp_id in known
        )
        raise ClockDeadlock(
            "every participant is waiting and none can be granted time, though "
            f"{pending}. The rule is supposed to make this impossible, so the "
            "state below is the evidence that it did not",
            self.lp_table(),
        )

    # --- small shared helpers ------------------------------------------------

    def _participant(self, lp_id: LpId) -> None:
        if lp_id not in self._now:
            known = ", ".join(str(member) for member in self._ids)
            raise KeyError(
                f"{lp_id} is not a participant in this clock; it holds "
                f"{known}. Participants are fixed when the clock is built"
            )

    @staticmethod
    def _simulated_seconds(value: float, what: str) -> float:
        seconds = float(value)
        if math.isnan(seconds):
            raise ValueError(f"{what} must be a number of seconds, got {value!r}")
        if seconds == -math.inf:
            raise ValueError(f"{what} must not be minus infinity, got {value!r}")
        return seconds

    @staticmethod
    def _seconds(value: float) -> str:
        return "no limit" if value == math.inf else f"{value:.9g}s"

    def __repr__(self) -> str:
        return (
            f"ClockAuthority({', '.join(str(lp_id) for lp_id in self._ids)}; "
            f"{self.grants_issued()} grants issued)"
        )

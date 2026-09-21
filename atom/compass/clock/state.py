# SPDX-License-Identifier: MIT
"""What the clock knows about one logical process, and what a grant says.

Three values describe a participant: where its clock stands, the earliest
future event it knows of, and whether it is able to act. The first two are
numbers in simulated seconds; the third is the one the rule reads hardest,
because the bound a participant places on everybody else depends entirely on
whether it can still produce an event.

A participant that is executing can produce an event at its current time. One
that has asked for time and has not been given any cannot produce anything at
all until it is: it is parked, and the rule is allowed to use that. Collapsing
the two would either stall every run that reaches a quiet moment or let a
parked participant be treated as a source of events it cannot send.
"""

import enum
import math
from dataclasses import dataclass

from .identity import LpId


class LpStatus(enum.Enum):
    """Whether a logical process can produce an event at its current time."""

    #: Executing. Its clock stands where it stands, and it may schedule an
    #: event on a peer no earlier than that plus the floor between them.
    RUNNING = "running"

    #: Has been given time it has not yet taken up. Its clock already stands at
    #: the granted time, so for the purposes of the rule it is no different from
    #: `RUNNING`; the distinction exists so a participant can be told it may go
    #: and be seen not to have gone yet.
    GRANTED = "granted"

    #: Asked to advance and was not allowed to. Produces nothing until it is
    #: granted, which is what lets the rule look past it to the event it is
    #: waiting for instead of pinning everybody at its stale clock.
    BLOCKED_ON_MESSAGE = "blocked-on-message"

    def __str__(self) -> str:
        return self.value

    @property
    def may_produce_events(self) -> bool:
        """True for a participant that could schedule an event right now."""
        return self is not LpStatus.BLOCKED_ON_MESSAGE


@dataclass(frozen=True)
class LpState:
    """One participant's row: its clock, its horizon, and what it is doing.

    `next_event` is the earliest future event the participant knows of, and is
    `+inf` when it knows of none. That is not the same as having nothing to do
    -- a participant waiting for a message from a peer knows of no event of its
    own -- and the difference is what the deadlock rule turns on.
    """

    lp_id: LpId
    now: float
    next_event: float
    status: LpStatus

    def __str__(self) -> str:
        horizon = "none" if self.next_event == math.inf else f"{self.next_event:.9g}s"
        return f"{self.lp_id} {self.status} at {self.now:.9g}s, next event {horizon}"


@dataclass(frozen=True)
class Grant:
    """Permission for one participant to move its clock to `advance_to`.

    `bound` is the furthest the rule would have allowed regardless of what this
    participant asked for, and `bound_from` names the peer that produced it, so
    a run that is crawling can say which link is holding it back rather than
    only that it is slow. `bound_from` is `None` when there is no peer at all,
    where the bound is unlimited and the participant is simply a local clock.

    `advance_to == advance_from` is a real grant, not a no-op: it is issued to a
    participant that was parked and has since had an event scheduled on it at
    the time its clock already stands at. It says the wait is over, not that
    time has moved.
    """

    lp_id: LpId
    advance_from: float
    advance_to: float
    bound: float
    bound_from: LpId | None

    @property
    def seconds(self) -> float:
        """How much simulated time this grant covers."""
        return self.advance_to - self.advance_from

    def __str__(self) -> str:
        limit = "unlimited" if self.bound == math.inf else f"{self.bound:.9g}s"
        pinned = "" if self.bound_from is None else f" by {self.bound_from}"
        return (
            f"{self.lp_id} {self.advance_from:.9g}s -> {self.advance_to:.9g}s "
            f"(bound {limit}{pinned})"
        )

# SPDX-License-Identifier: MIT
"""Identity, order, lookahead, and the rule that hands out simulated time.

Who the participants are (`LpId`), the order they are served in (`LpRegistry`),
how far each may run ahead of each other one (`LookaheadMatrix`), and the rule
that reads all three and decides who may move (`ClockAuthority`). The transport
that carries a request from a participant to the rule is separate and sits
elsewhere. What a run says about itself -- a timeline of granted advances, a
dump of every clock when nothing can move, and a summary written once at the
end -- comes out of the same place, because the rule is the only thing that
sees every participant.

Nothing here imports a device runtime, reads a clock, or opens a socket, which
is what makes it testable on any machine.
"""

from .authority import (
    BackdatedEvent,
    ClockAbort,
    ClockAuthority,
    ClockDeadlock,
)
from .identity import LpId
from .lookahead import (
    TRAFFIC_TO_ENGINE_FLOOR_SECONDS,
    InterLpLink,
    LinkClass,
    LookaheadMatrix,
)
from .observability import (
    SPEED_TARGET_RATIO,
    DetectorState,
    DriverDiscipline,
    RefusalTally,
    RunSummary,
    StallKind,
    TimelineLog,
    TimelineRecord,
    deadlock_dump,
    stall_kind,
)
from .registry import LpRegistry
from .state import Grant, LpState, LpStatus

__all__ = [
    "SPEED_TARGET_RATIO",
    "TRAFFIC_TO_ENGINE_FLOOR_SECONDS",
    "BackdatedEvent",
    "ClockAbort",
    "ClockAuthority",
    "ClockDeadlock",
    "DetectorState",
    "DriverDiscipline",
    "Grant",
    "InterLpLink",
    "LinkClass",
    "LookaheadMatrix",
    "LpId",
    "LpRegistry",
    "LpState",
    "LpStatus",
    "RefusalTally",
    "RunSummary",
    "StallKind",
    "TimelineLog",
    "TimelineRecord",
    "deadlock_dump",
    "stall_kind",
]

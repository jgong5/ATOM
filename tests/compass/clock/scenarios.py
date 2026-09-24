# SPDX-License-Identifier: MIT
"""Three runs that break the time order on purpose, so the checks can be seen firing.

Each one is a defect this project can actually ship: a delay declared longer
than the path it stands for, a wait nobody declared, and a message the
coordinator was never told about. None of them raises anything on its own. Two
of them finish and print a latency row that looks like every other latency row,
and the third stops -- loudly, with nothing to report -- which is the half of
the arrangement the protocol was built around.

Every scenario therefore runs twice: once with its check off, which is what
produces the plausible row, and once with it on.

**Why these are driven by hand rather than by the synthetic harness beside them.**
That driver already refuses to step a participant over a message it has not
handed over, and it can refuse because it is a test harness and knows both
sides of every message. Nothing shipped knows that: a participant holds the
message and its own clock and nothing else, which is why the check that has to
work in production is on the receive side. A driver that cannot commit these
mistakes cannot be used to demonstrate catching them, so this one can.

The clock starts at the moment the prefill side finishes its step and hands the
request over, except where a scenario needs the step itself. Getting to that
moment is not what any of this is about.
"""

import collections
import math
import queue
import time
from dataclasses import dataclass

from atom.compass.clock import (
    ClockAuthority,
    LinkClass,
    LookaheadMatrix,
    LpRegistry,
    LpStatus,
)
from atom.compass.detect import Arrival, StragglerCheck

from .deployments import DEPLOYMENTS, ROLE_BOUNDARY_FLOOR_SECONDS
from .participants import DESIGN_WORKLOAD

#: The two participants, named by the deployment module rather than here.
ROLE_PAIR = next(
    deployment
    for deployment in DEPLOYMENTS
    if deployment.name == "tp8-role-disaggregated"
)
DECODE, PREFILL = sorted(replica.stage_ids()[0] for replica in ROLE_PAIR.replicas)

#: What the prefill side's one step costs, and what a decode step costs, from
#: the measured trace the harness beside this replays.
PREFILL_STEP_SECONDS = DESIGN_WORKLOAD.prefill_step_seconds
DECODE_STEP_SECONDS = DESIGN_WORKLOAD.decode_step_seconds

#: When the prefill side finishes its step and hands the request over.
HANDOFF_SECONDS = PREFILL_STEP_SECONDS

#: What the relay and the simulated transfer of the cached keys and values
#: actually cost on this path. A cost model produces this number.
TRANSFER_SECONDS = 2.0e-4

#: What the link was declared at instead -- five times the path. A second
#: number, in a second place, and nothing compares the two.
OVER_DECLARED_SECONDS = ROLE_BOUNDARY_FLOOR_SECONDS

#: How long the undeclared wait is allowed to sit before the test calls it hung.
#: It has to be far above the watchdog's threshold and far below any patience a
#: person would have, and the gap between the two is what the test reads.
HANG_BOUND_SECONDS = 0.5

#: The watchdog's threshold for these runs. The shipped default is longer; this
#: is the same separation on a scale a test tier can afford.
STALL_SECONDS = 0.05


@dataclass(frozen=True)
class Stall:
    """What a run has to show for itself once the decode side stopped waiting."""

    annotated: bool
    read: object
    waited_seconds: float
    prefill_now: float
    prefill_status: str
    decode_now: float
    decode_status: str
    bound_from: str
    table: str
    lp_table: str


class Injected:
    """Two participants, one declared floor, and a hand on every step.

    Messages go over the transport whether or not the coordinator is told about
    them, which is the one thing a correct driver would never do and the only
    way to arrange the failure the receive-side check exists for.
    """

    def __init__(
        self, role_floor, start_time=HANDOFF_SECONDS, straggler=None, watchdog=None
    ):
        self.registry = LpRegistry()
        for lp_id in (PREFILL, DECODE):
            self.registry.register(lp_id)
        self.matrix = LookaheadMatrix(self.registry)
        for source, target in ((PREFILL, DECODE), (DECODE, PREFILL)):
            self.matrix.declare(source, target, LinkClass.PREFILL_TO_DECODE, role_floor)
        self.clock = ClockAuthority(self.registry, self.matrix, start_time)
        self.role_floor = role_floor
        self.straggler = straggler or StragglerCheck(enabled=False)
        self.watchdog = watchdog
        self.inbox = {lp_id: collections.deque() for lp_id in self.registry.ids()}
        self.grants = []

    def send(self, sender, receiver, tell_the_clock):
        """Put one message on the transport, and declare it to the clock or not.

        The stamp on the message is when it takes effect, and it comes from what
        the path costs. What is declared to the clock comes from the floor. They
        are two numbers held in two places and only the receiver ever sees both.
        """
        sent_at = self.clock.now(sender)
        if tell_the_clock:
            self.clock.schedule_event(sender, receiver, sent_at + self.role_floor)
        self.inbox[receiver].append(
            Arrival(
                str(sender),
                str(receiver),
                sent_at,
                sent_at + TRANSFER_SECONDS,
                self.role_floor,
            )
        )

    def park(self, lp_id, horizon=math.inf):
        """Declare idle, take up whatever is given, and ask again until refused.

        A participant that stops asking after one grant holds every peer at its
        own clock plus one floor, so nothing ever reaches an event further out.
        It stops on a message, on a refusal, or at the horizon it declared.
        """
        handed = ()
        while not handed:
            if self.clock.state(lp_id).status is not LpStatus.GRANTED:
                if self.watchdog is not None:
                    self.watchdog.parked(lp_id)
                self.clock.request_advance(lp_id, horizon)
            if self.clock.state(lp_id).status is not LpStatus.GRANTED:
                break
            grant = self.take_up(lp_id)
            handed = self.hand_over(lp_id, grant)
            if grant.advance_to >= horizon:
                break
        return handed

    def take_up(self, lp_id):
        """Collect a grant and start executing again."""
        grant = self.clock.take_up_grant(lp_id)
        self.grants.append(grant)
        if self.watchdog is not None:
            self.watchdog.running(lp_id, grant.advance_to)
        return grant

    def hand_over(self, lp_id, grant):
        """Give the participant every message this grant reached, checked on the way in.

        The participant table is handed over as the function that builds it, not
        as the string: it is built once, on the message that fails, rather than
        on every message that does not.
        """
        held = self.inbox[lp_id]
        handed = []
        while held and held[0].takes_effect_at <= grant.advance_to:
            arrival = self.straggler.arriving(
                held.popleft(), self.clock.now(lp_id), self.clock.lp_table
            )
            handed.append(arrival)
        return tuple(handed)

    def table(self):
        """The one row the run reports. It is the thing that looks fine."""
        handed_over_at = self.clock.now(DECODE)
        return (
            f"request-0  handed over at {handed_over_at:.6f}s  "
            f"decode step {DECODE_STEP_SECONDS:.6f}s  "
            f"time to first token {handed_over_at + DECODE_STEP_SECONDS:.6f}s"
        )


def wrong_lookahead(straggler=None, role_floor=OVER_DECLARED_SECONDS):
    """A link declared five times longer than the path it stands for.

    Everything the coordinator is told is consistent with itself. The prefill
    side declares its handoff at the floor it promised, and the decode side is
    granted exactly that far -- which is further than the message, stamped from
    what the transfer really costs, has got to.
    """
    run = Injected(role_floor, straggler=straggler)
    run.send(PREFILL, DECODE, tell_the_clock=True)
    run.park(DECODE)
    return run.table()


def induced_straggler(straggler=None, tell_the_clock=False):
    """Floors that match the path, and a send the coordinator is never told about.

    The decode side is already mid-flight on another request, so it knows of an
    event of its own one decode step out. With the prefill side parked and
    nothing declared for the handoff, the rule looks straight past to that event
    and grants the whole step, over a message already sitting on the transport.
    """
    run = Injected(TRANSFER_SECONDS, straggler=straggler)
    run.send(PREFILL, DECODE, tell_the_clock=tell_the_clock)
    run.park(PREFILL)
    run.park(DECODE, HANDOFF_SECONDS + DECODE_STEP_SECONDS)
    return run.table()


def missing_annotation(watchdog, annotate, bound_seconds=HANG_BOUND_SECONDS):
    """The decode side blocks for real. With `annotate=False` it never says so.

    Undeclared, its clock stands still, and a clock that stands still is a bound
    every peer is held at. The prefill side is granted one floor and then
    refused, so it never finishes its step, never sends the handoff, and the
    blocking read ends at the bound with nothing in it.
    """
    run = Injected(TRANSFER_SECONDS, start_time=0.0, watchdog=watchdog)
    mail = queue.Queue()
    watchdog.running(DECODE, run.clock.now(DECODE))
    if annotate:
        run.park(DECODE)
    run.park(PREFILL, PREFILL_STEP_SECONDS)
    if run.clock.now(PREFILL) >= PREFILL_STEP_SECONDS:
        run.send(PREFILL, DECODE, tell_the_clock=True)
        mail.put(run.inbox[DECODE].popleft())
    started = time.perf_counter()
    with watchdog.sampling():
        try:
            read = mail.get(timeout=bound_seconds)
        except queue.Empty:
            read = None
    waited = time.perf_counter() - started
    return Stall(
        annotated=annotate,
        read=read,
        waited_seconds=waited,
        prefill_now=run.clock.now(PREFILL),
        prefill_status=str(run.clock.state(PREFILL).status),
        decode_now=run.clock.now(DECODE),
        decode_status=str(run.clock.state(DECODE).status),
        bound_from=str(run.grants[-1].bound_from) if run.grants else "",
        table=run.table() if read is not None else "",
        lp_table=run.clock.lp_table(),
    )

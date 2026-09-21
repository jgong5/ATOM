# SPDX-License-Identifier: MIT
"""A participant's whole view of the clock.

Four things a participant can do -- attach, ask to advance, take up a grant it
has been issued, and place an event on another participant -- and one thing it
can read, which is where its own clock stands. Every one of them is named in
simulated seconds and in participant names, and nothing else appears: not where
the clock is, not how it is reached, not how many clocks sit between this one
and whatever is keeping global order. A participant is handed one of these and
cannot tell which arrangement it is in.

That is not politeness, it is what makes an arrangement with a clock between
two others possible without rewriting either side. Such a clock is a clock
whose participants are other clocks: it answers requests through the same
surface it makes its own through, so it can be slotted in later without a
participant learning anything new. The moment this class grows a parameter that
says where or how deep, that stops being true and every participant has to be
told.

Where the clock stands is read off the reply, never guessed. A participant
starts outside simulated time, learns its clock when it attaches, and updates
it from every reply after that -- including the replies that carry someone
else's grants, because a grant to a peer can move nothing of this
participant's and the stamp says so. A refused request leaves the clock where
it was.

The carrier is anything that turns a frame into a frame. What it does in
between -- a call in this process, a socket to another container -- is not
visible here and must not become visible, because the two have to run the same
protocol for one to be evidence about the other.
"""

import math

from ..authority import BackdatedEvent, ClockAbort, ClockDeadlock
from ..identity import LpId
from ..state import Grant
from .message import BEFORE_THE_RUN, Message, MessageKind, decode, encode

#: What to raise again on this side for a refusal that came back on a frame.
#: Aborts carry a reason and the participant table; the rest carry a message.
ABORTED = {
    abort.__name__: abort for abort in (BackdatedEvent, ClockAbort, ClockDeadlock)
}
DECLINED = {
    declined.__name__: declined for declined in (KeyError, TypeError, ValueError)
}


class ClockSession:
    """One participant, talking to the clock through a carrier."""

    def __init__(self, lp_id: LpId, carrier) -> None:
        if not isinstance(lp_id, LpId):
            raise TypeError(f"a session belongs to an LpId, got {type(lp_id).__name__}")
        self._lp_id = lp_id
        self._carrier = carrier
        self._now = BEFORE_THE_RUN

    @property
    def lp_id(self) -> LpId:
        """Which participant this session speaks for."""
        return self._lp_id

    @property
    def now(self) -> float:
        """Where this participant's clock stands, as the clock last said.

        `-inf` until it has attached: a participant that has not joined the run
        has no simulated time to report, and reporting zero would be a guess
        that happens to be right only when the run starts at zero.
        """
        return self._now

    def attach(self) -> float:
        """Join the run and learn where this participant's clock stands.

        Refused if the clock does not hold this name. Membership is fixed when
        the clock is built, so a participant that is not in it will never be,
        and finding that out at the first message is far cheaper than finding it
        out as a missing term in a minimum halfway through a run.
        """
        self._exchange(MessageKind.ATTACH)
        return self._now

    def request_advance(self, next_event: float = math.inf) -> tuple[Grant, ...]:
        """Ask to move past the current clock, declaring the next event known of.

        Returns every grant the request released, for any participant, in the
        clock's total order -- one participant asking can release another that
        was waiting on it. An empty result means this one stays parked.
        """
        return self._exchange(MessageKind.ADVANCE, when=next_event).grants

    def take_up_grant(self) -> Grant:
        """Collect the grant issued to this participant and start executing."""
        return self._exchange(MessageKind.TAKE_UP).grants[0]

    def schedule_event(self, target: LpId, timestamp: float) -> tuple[Grant, ...]:
        """Place an event on another participant, named by identity.

        Returns any grants the event released, since an event is exactly what a
        parked participant may have been waiting for.
        """
        return self._exchange(MessageKind.EVENT, target=target, when=timestamp).grants

    def close(self) -> None:
        """Let go of the carrier. The clock keeps this participant's row."""
        self._carrier.close()

    def _exchange(
        self, kind: MessageKind, target: LpId | None = None, when: float = math.inf
    ) -> Message:
        """One frame out, one frame back, stamped with this participant's clock.

        The stamp goes on every request, in every arrangement. A carrier that
        skipped it in this process would make a single-container run cheap and
        worthless: it would stop exercising the thing a multi-container run
        depends on, and a defect would then live in one arrangement only.
        """
        request = Message(kind, self._lp_id, self._now, target=target, when=when)
        reply = decode(self._carrier.exchange(encode(request)))
        if reply.kind is MessageKind.REFUSAL:
            _refuse(reply)
        if reply.participant != self._lp_id:
            raise ValueError(
                f"{self._lp_id} asked and the reply is about {reply.participant}"
            )
        self._now = reply.sent_at
        return reply

    def __repr__(self) -> str:
        standing = (
            "not attached" if self._now == BEFORE_THE_RUN else f"{self._now:.9g}s"
        )
        return f"ClockSession({self._lp_id}, {standing})"


def _refuse(reply: Message) -> None:
    """Raise again, on this side, what the clock raised on its own.

    A run that has to stop stops the same way whichever carrier brought the
    news, which is the point: the abort's reason and its participant table
    travel on the frame rather than being reduced to a status code.

    A refusal this side does not recognise keeps both anyway. A participant
    table is the only thing that makes an abort actionable, and a refusal the
    clock grew after this module was written is exactly the case where somebody
    needs to see one. So an unrecognised name that arrived with a table is
    raised as an abort carrying it, with the original name in front of the
    reason; one that arrived without a table keeps the name in the message. The
    type is degraded, never the evidence.
    """
    named, reason, table = reply.detail
    abort = ABORTED.get(named)
    if abort is not None:
        raise abort(reason, table)
    declined = DECLINED.get(named)
    if declined is not None:
        raise declined(reason)
    if table:
        raise ClockAbort(f"{named}: {reason}", table)
    raise RuntimeError(f"{named}: {reason}")

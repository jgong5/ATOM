# SPDX-License-Identifier: MIT
"""The rule, wrapped so that a frame is the only way to reach it.

One frame in, one frame out, and `handle` is the whole public surface. That is
the point of the class rather than an accident of it: whichever carrier brought
the frame -- a socket from another container, or a direct call inside the
API-server process -- the same bytes are decoded by the same code and answered
by the same rule. There is no second entry point taking a request object, so a
co-hosted arrangement cannot quietly become a different implementation from a
standalone one.

Nothing here knows how the frame arrived, and nothing here knows what is on the
other side of it. A participant and another authority look identical from this
side, which is what an arrangement with an authority sitting between two others
needs: such an authority holds one of these over its own participants and talks
to whatever is above it through exactly the surface a participant uses.

A refusal is carried, not swallowed. Whatever the rule raises is encoded with
its type, its reason and the participant table, and the far side raises the same
type again -- so a run that must stop stops the same way through either carrier.

**What this can and cannot guarantee.** It holds the clock and hands nothing
back, so there is no way *through a carriage* to reach the rule without a
frame, and therefore no way to reach it without a stamp. That is the property
the two arrangements' equivalence rests on and it is structural. It is not a
guarantee that nothing in the hosting process can reach the clock directly:
whatever co-hosts a clock has to build it before it can serve it, so it holds a
reference by construction, and no arrangement of this class can take that away.
The rule there is about wiring -- a participant is given a session and not a
clock -- and it is a rule rather than an impossibility. Saying so is worth more
than a check that looks like it covers it and does not.
"""

import math

from ..authority import ClockAbort, ClockAuthority
from .message import Message, MessageKind, decode, encode


class ClockService:
    """Answers encoded requests from the clock it was built over."""

    def __init__(self, authority: ClockAuthority) -> None:
        self._authority = authority

    def handle(self, frame: bytes) -> bytes:
        """Answer one encoded request with one encoded reply.

        The only entry point, so that every arrangement carries the same bytes.
        """
        request = decode(frame)
        try:
            reply = self._answer(request)
        except ClockAbort as abort:
            reply = self._refusal(request, abort, abort.reason, abort.table)
        except (KeyError, TypeError, ValueError) as refused:
            reply = self._refusal(request, refused, str(refused), "")
        return encode(reply)

    def _answer(self, request: Message) -> Message:
        if not request.kind.is_a_request:
            raise ValueError(
                f"{request.kind} is something the clock says, not something a "
                "participant may ask for"
            )
        if request.kind is MessageKind.ATTACH:
            return self._reply(MessageKind.CLOCK, request.participant)
        if request.kind is MessageKind.ADVANCE:
            released = self._authority.request_advance(
                request.participant, request.when
            )
        elif request.kind is MessageKind.TAKE_UP:
            released = (self._authority.take_up_grant(request.participant),)
        else:
            released = self._authority.schedule_event(
                request.participant, request.target, request.when
            )
        return self._reply(MessageKind.GRANTS, request.participant, released)

    def _reply(self, kind: MessageKind, lp_id, grants=()) -> Message:
        return Message(kind, lp_id, self._authority.now(lp_id), grants=grants)

    def _refusal(
        self, request: Message, refused: Exception, reason: str, table: str
    ) -> Message:
        return Message(
            MessageKind.REFUSAL,
            request.participant,
            self._stamp(request.participant),
            detail=(type(refused).__name__, reason, table),
        )

    def _stamp(self, lp_id) -> float:
        """This participant's clock, or minus infinity if it has none here.

        A request from a name the clock does not hold is refused, and the
        refusal still has to be stamped with something. Minus infinity is the
        same value a participant uses before it has attached: outside the run.
        """
        try:
            return self._authority.now(lp_id)
        except KeyError:
            return -math.inf

    def __repr__(self) -> str:
        return f"ClockService({self._authority!r})"

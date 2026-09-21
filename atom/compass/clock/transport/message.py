# SPDX-License-Identifier: MIT
"""What crosses between a participant and the rule that hands out time.

Every exchange is one encoded frame out and one encoded frame back, and every
frame carries the virtual time at which it was formed. That stamp is the reason
this module exists as a wire format rather than as a pair of function
signatures: a participant reached through a socket and a participant reached
inside one process put the *same bytes* on their respective carriers, so a
defect in the protocol shows up in the cheap arrangement as readily as in the
expensive one.

The alternative -- a co-hosted arrangement that hands the rule a request object
directly and skips the stamp -- is the failure this shape is chosen to make
impossible. It looks like an optimisation and it costs almost nothing to write.
What it costs is the ability to trust a single-container run as evidence about
a multi-container one: the two would then exercise different code, and a bug
found in one would no longer be a bug in the other. A carrier here takes bytes
and returns bytes, so there is nowhere to put such a shortcut.

`sent_at` is the virtual clock of the participant the message concerns, read at
the moment the message was formed. A request carries the participant's own
clock; a reply carries the same participant's clock as the rule holds it, after
whatever the request did. One message that is not sent from inside simulated
time: the first one, by which a participant attaches and learns where its clock
stands. It has no clock to quote and stamps itself minus infinity.

The encoding is JSON with sorted keys and no spacing, so the same message
encodes to the same bytes on every run and in every process -- which is what
lets two arrangements be compared by comparing their traffic.

It is JSON a stranger can read, not merely JSON this interpreter accepts.
Python's encoder will happily write a bare `Infinity`, and most of this
protocol's numbers are simulated seconds whose ordinary value is exactly that:
a participant that knows of no future event declares one. A frame carrying a
bare `Infinity` parses here and is rejected by a conforming parser anywhere
else, which would quietly make the standalone arrangement Python-to-Python
only. Every simulated duration therefore travels as a number or as one of two
spelled-out strings, and a quantity that is not a number at all is refused at
the point it would have been written rather than encoded as one more bare word.
"""

import enum
import json
import math
from dataclasses import dataclass

from ..identity import LpId
from ..state import Grant

#: The stamp on a message formed before its participant has a clock to quote.
BEFORE_THE_RUN = -math.inf


class MalformedMessage(ValueError):
    """A frame is not a message. Refused rather than partially understood.

    A frame that decodes to something unexpected is a fault in whatever wrote
    it, and the only safe answer is to stop: a half-read grant is a grant to the
    wrong time, and a grant to the wrong time is the failure that produces a
    plausible answer rather than a crash.
    """


class MessageKind(enum.Enum):
    """What a message is for. Four a participant sends, three it receives."""

    #: Join the run and learn where this participant's clock stands.
    ATTACH = "attach"

    #: Ask to move past the current clock, declaring the next event known of.
    ADVANCE = "advance"

    #: Collect a grant already issued, and start executing.
    TAKE_UP = "take-up"

    #: Place an event on another participant.
    EVENT = "event"

    #: Where a participant's clock stands. The answer to `ATTACH`.
    CLOCK = "clock"

    #: Grants released by a request, for any participant, in the total order.
    GRANTS = "grants"

    #: The request was refused. Carries what to raise and the participant table.
    REFUSAL = "refusal"

    def __str__(self) -> str:
        return self.value

    @property
    def is_a_request(self) -> bool:
        """True for the kinds a participant sends to the rule."""
        return self in (
            MessageKind.ATTACH,
            MessageKind.ADVANCE,
            MessageKind.TAKE_UP,
            MessageKind.EVENT,
        )


@dataclass(frozen=True)
class Message:
    """One frame's worth of protocol, in either direction.

    `participant` is who the message is about, which is the sender on the way
    out and the same participant on the way back -- so `sent_at` means one thing
    in both directions and two arrangements' traffic can be compared field by
    field.
    """

    kind: MessageKind
    participant: LpId
    sent_at: float
    target: LpId | None = None
    when: float = math.inf
    grants: tuple[Grant, ...] = ()
    detail: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"{self.kind} {self.participant} at {self.sent_at:.9g}s"


def encode(message: Message) -> bytes:
    """The bytes that go on a carrier. Sorted keys, so it is the same every run."""
    if not isinstance(message, Message):
        raise MalformedMessage(
            f"only a message can be encoded, got {type(message).__name__}"
        )
    body = {
        "kind": message.kind.value,
        "participant": message.participant.name,
        "sent_at": _encode_seconds(message.sent_at),
        "target": None if message.target is None else message.target.name,
        "when": _encode_seconds(message.when),
        "grants": [_encode_grant(grant) for grant in message.grants],
        "detail": list(message.detail),
    }
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def decode(frame: bytes) -> Message:
    """The message a frame carries. Refuses anything that is not one.

    Takes bytes and nothing else, deliberately. A carrier that could be handed a
    message object would be a carrier that could skip the encoding, and skipping
    the encoding is how a co-hosted arrangement stops testing the protocol.
    """
    if not isinstance(frame, (bytes, bytearray)):
        raise MalformedMessage(
            f"a frame is bytes, got {type(frame).__name__}; a carrier moves "
            "encoded messages, never message objects"
        )
    try:
        body = json.loads(
            bytes(frame).decode("utf-8"), parse_constant=_refuse_bare_constant
        )
        kind = MessageKind(body["kind"])
        target = body["target"]
        return Message(
            kind,
            LpId(body["participant"]),
            _decode_seconds(body["sent_at"]),
            None if target is None else LpId(target),
            _decode_seconds(body["when"]),
            tuple(_decode_grant(row) for row in body["grants"]),
            tuple(str(item) for item in body["detail"]),
        )
    except (LookupError, TypeError, UnicodeDecodeError, ValueError) as fault:
        raise MalformedMessage(f"{frame!r} is not a message: {fault}") from fault


#: How the two unbounded durations are spelled on a frame. A participant that
#: knows of no future event declares `+inf`, so this is the ordinary case and
#: not an edge one.
UNBOUNDED = {math.inf: "+inf", -math.inf: "-inf"}
BOUNDS = {name: value for value, name in UNBOUNDED.items()}


def _encode_seconds(value: float):
    """A duration as a number, or as one of two words. Refuses anything else."""
    seconds = float(value)
    if seconds in UNBOUNDED:
        return UNBOUNDED[seconds]
    if math.isnan(seconds):
        raise MalformedMessage(
            "a simulated duration must be a number of seconds or unbounded, "
            "and this one is neither; nothing downstream can order it"
        )
    return seconds


def _decode_seconds(value) -> float:
    if isinstance(value, str):
        if value not in BOUNDS:
            raise ValueError(f"{value!r} is not a duration")
        return BOUNDS[value]
    return float(value)


def _refuse_bare_constant(name: str):
    raise ValueError(
        f"{name} is not valid JSON; a duration travels as a number or as "
        f"{sorted(BOUNDS)}"
    )


def _encode_grant(grant: Grant) -> list:
    pinned = grant.bound_from
    return [
        grant.lp_id.name,
        _encode_seconds(grant.advance_from),
        _encode_seconds(grant.advance_to),
        _encode_seconds(grant.bound),
        None if pinned is None else pinned.name,
    ]


def _decode_grant(row: list) -> Grant:
    name, advance_from, advance_to, bound, pinned = row
    return Grant(
        LpId(name),
        _decode_seconds(advance_from),
        _decode_seconds(advance_to),
        _decode_seconds(bound),
        None if pinned is None else LpId(pinned),
    )

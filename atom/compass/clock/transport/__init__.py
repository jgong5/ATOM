# SPDX-License-Identifier: MIT
"""Carrying a request from a participant to the rule that hands out time.

One protocol, one wire format, one piece of code that answers a request -- and
two ways of getting the frame there. Co-hosted in the API-server process, which
is what a single-container run uses and where the frame is handed straight over;
standalone behind a socket, for a run split across containers or nodes. Which
one a participant gets is decided by the endpoint it was given and by nothing
else, and the participant cannot tell the difference afterwards.

The in-process arrangement is deliberately not a shortcut. It builds and parses
the same frames, carrying the same virtual send times, and it reaches the rule
through the same single entry point. If it ever stops doing that, the cheap
runs stop being evidence about the expensive ones, and a defect found in one
arrangement is no longer a defect in the other.

Nothing a participant touches names where the clock is or how many clocks sit
above it. Two calls turn a written-down endpoint into something usable, and
they are the only place a location appears.
"""

from .message import (
    BEFORE_THE_RUN,
    MalformedMessage,
    Message,
    MessageKind,
    decode,
    encode,
)
from .resolve import (
    DEFAULT_ENDPOINT,
    IN_PROCESS_SCHEME,
    InProcessCarrier,
    InProcessServer,
    carrier_for,
    connect,
    serve,
)
from .service import ClockService
from .session import ClockSession
from .stream import STREAM_SCHEME

__all__ = [
    "BEFORE_THE_RUN",
    "DEFAULT_ENDPOINT",
    "IN_PROCESS_SCHEME",
    "STREAM_SCHEME",
    "ClockService",
    "ClockSession",
    "InProcessCarrier",
    "InProcessServer",
    "MalformedMessage",
    "Message",
    "MessageKind",
    "carrier_for",
    "connect",
    "decode",
    "encode",
    "serve",
]

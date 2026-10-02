# SPDX-License-Identifier: MIT
"""The two carriers of a simulated timestamp across HTTP: request and stream.

A request carries its ``(arrival, seq)`` as a ``compass`` entry of the W3C
``tracestate`` header, ``compass=a:<arrival>;s:<seq>``, written after any
entries already there. A streamed response carries one comment line before
each SSE event, ``: compass a=<arrival> s=<seq>``; SSE clients skip lines that
start with a colon. Both pass a router that forwards headers and the stream
unchanged. The arrival is written as ``repr(float)``, so it parses back exactly.
"""

import re

_NUM = r"(inf|\d[\d.e+-]*)"
_ENTRY = re.compile(rf"a:{_NUM};s:(\d+)")
_COMMENT = re.compile(rf": compass a={_NUM} s=(\d+)")


def _parse(pattern: re.Pattern, text: str, what: str) -> tuple[float, int]:
    m = pattern.fullmatch(text)
    try:
        return float(m[1]), int(m[2])
    except (TypeError, ValueError):
        raise ValueError(
            f"malformed compass {what} {text!r}; expected {pattern.pattern}"
        ) from None


def tracestate_stamp(header: str | None) -> tuple[float, int] | None:
    """The ``(arrival, seq)`` of the ``compass`` entry in `header`; None without one."""
    found = [
        value
        for key, _, value in (
            e.strip().partition("=") for e in (header or "").split(",")
        )
        if key == "compass"
    ]
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(f"tracestate {header!r} has more than one compass entry")
    return _parse(_ENTRY, found[0], "tracestate entry")


def tracestate_with(header: str | None, arrival: float, seq: int) -> str:
    """`header` with the ``compass`` entry for ``(arrival, seq)`` after its entries."""
    if tracestate_stamp(header) is not None:
        raise ValueError(f"tracestate {header!r} already has a compass entry")
    entry = f"compass=a:{arrival!r};s:{seq}"
    return f"{header},{entry}" if header and header.strip() else entry


def stamp_events(text: str, rt) -> str:
    """`text` with a stamped comment line before each SSE event in it.

    Each event is one send on the runtime's stream channel to the traffic LP.
    Every frame the endpoints write ends in a blank line, so splitting on one
    finds every event, and nothing after the last.
    """
    ch = f"{rt.me}->traffic:stream"
    *events, tail = text.split("\n\n")
    out = []
    for event in events:
        arrival, seq = rt.stamp_send(ch)
        out.append(f": compass a={arrival!r} s={seq}\n{event}\n\n")
    return "".join(out) + tail


def sse_stamp(line: str) -> tuple[float, int] | None:
    """The ``(arrival, seq)`` of a compass comment line; None for any other line."""
    line = line.rstrip("\r\n")
    if not line.startswith(": compass "):
        return None
    return _parse(_COMMENT, line, "SSE comment")

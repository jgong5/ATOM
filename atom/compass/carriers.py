# SPDX-License-Identifier: MIT
"""The three carriers of a simulated timestamp across HTTP: request, relay and stream.

A request carries its ``(arrival, seq)`` as a ``compass`` entry of the W3C
``tracestate`` header, ``compass=a:<arrival>;s:<seq>``, written before any
entries already there, as W3C Trace Context adds a new entry. A request the
traffic LP sends with a send-order seed adds its intake jitter, the bytes its
body is read as if it were longer, ``compass=a:<arrival>;s:<seq>;j:<bytes>``
(`compass_harness.traffic_lp`). A streamed response carries one comment line before
each SSE event, ``: compass a=<arrival> s=<seq>``; SSE clients skip lines that
start with a colon. Both pass a router that forwards headers and the stream
unchanged. A prefill response the router relays to decode carries its stamp as
the ``compass`` field of its ``kv_transfer_params``, ``a:<arrival>;s:<seq>``,
which the router copies onto the decode request. The arrival is written as
``repr(float)``, so it parses back exactly. Beside it, ``compass_relay_s``
holds the seconds the relay serves the response on decode's API server loop
(`RELAY_SERVICE`, `atom.utils.compass_loop.HttpChannel`), where prefill's host
entry prices a relay; the router restamps only ``compass`` and copies the rest
of the object, so the seconds reach decode as prefill wrote them.

A client's cancel of a request is itself a stamped request to the same URL,
whose body names the cancelled request's seq: ``{"compass_cancel": <seq>}``.
A router hands a cancel on along the request's path; on a relay it carries
its own stamp in the body, after the seq:
``{"compass_cancel": <seq>, "kv_transfer_params": {"compass": "a:<arrival>;s:<seq>"}}``.

A client's own records carry simulated time as aiperf's ns stamps, each the
simulated seconds in ns plus `EPOCH_NS`.
"""

import math
import re
from typing import Any

import msgspec

_NUM = r"(inf|\d[\d.e+-]*)"
#: The ``kv_transfer_params`` field of a relayed response holding the seconds
#: its relay serves it on decode's API server loop.
RELAY_SERVICE = "compass_relay_s"
_ENTRY = re.compile(rf"a:{_NUM};s:(\d+)")
_TRACE_ENTRY = re.compile(rf"a:{_NUM};s:(\d+)(?:;j:{_NUM})?")
_COMMENT = re.compile(rf": compass a={_NUM} s=(\d+)")
_MAX_ENTRIES = 32
#: Added to every simulated ns stamp a client records. Simulated time starts at
#: 0, and aiperf reads a 0 start stamp as missing.
EPOCH_NS = 10**9


def _parse(pattern: re.Pattern, text: str, what: str) -> tuple[float, int]:
    m = pattern.fullmatch(text)
    try:
        return float(m[1]), int(m[2])
    except (TypeError, ValueError):
        raise ValueError(
            f"malformed compass {what} {text!r}; expected {pattern.pattern}"
        ) from None


def _tracestate_entry(header: str | None) -> str | None:
    """The value of the ``compass`` entry in `header`; None without one."""
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
    return found[0]


def tracestate_stamp(header: str | None) -> tuple[float, int] | None:
    """The ``(arrival, seq)`` of the ``compass`` entry in `header`; None without one."""
    entry = _tracestate_entry(header)
    return None if entry is None else _parse(_TRACE_ENTRY, entry, "tracestate entry")


def tracestate_jitter(header: str | None) -> float:
    """The intake jitter, in bytes, of the ``compass`` entry in `header`; 0.0
    without one, or with an entry that names none."""
    entry = _tracestate_entry(header)
    if entry is None:
        return 0.0
    _parse(_TRACE_ENTRY, entry, "tracestate entry")
    jitter = _TRACE_ENTRY.fullmatch(entry)[3]
    return 0.0 if jitter is None else float(jitter)


def tracestate_with(
    header: str | None, arrival: float, seq: int, jitter: float | None = None
) -> str:
    """`header` with the ``compass`` entry for ``(arrival, seq)`` first, and
    `jitter`, the request's intake jitter in bytes, in it when given.

    The other entries keep their order; empty ones are dropped, and so are
    entries from the end past W3C Trace Context's limit of 32.
    """
    if tracestate_stamp(header) is not None:
        raise ValueError(f"tracestate {header!r} already has a compass entry")
    if jitter is not None and not (math.isfinite(jitter) and jitter >= 0):
        raise ValueError(f"intake jitter {jitter!r} is no finite count of bytes >= 0")
    entry = f"compass=a:{arrival!r};s:{seq}"
    if jitter is not None:
        entry += f";j:{float(jitter)!r}"
    others = [e.strip() for e in (header or "").split(",") if e.strip()]
    return ",".join([entry, *others][:_MAX_ENTRIES])


def stamp_split_events(events: list[str], tail: str, rt, ch: str) -> str:
    """The SSE events of a text its caller split on blank lines into
    ``[*events, tail]``, each with a stamped comment line before it.

    Each event is one send on `ch`, the runtime's stream channel, to the
    traffic LP or to the router LP in front of it.
    A write must end its last frame in a blank line: a `tail` is refused,
    since the next write's comment would land inside that frame.
    """
    if tail:
        raise ValueError(f"unterminated SSE frame {tail!r}: no blank line after it")
    out = []
    for event in events:
        arrival, seq = rt.stamp_send(ch)
        out.append(f": compass a={arrival!r} s={seq}\n{event}\n\n")
    return "".join(out)


def sse_stamp(line: str) -> tuple[float, int] | None:
    """The ``(arrival, seq)`` of a compass comment line; None for any other line."""
    line = line.rstrip("\r\n")
    if not line.startswith(": compass "):
        return None
    return _parse(_COMMENT, line, "SSE comment")


class _Params(msgspec.Struct):
    compass: Any = None
    compass_relay_s: Any = None


class _Relayed(msgspec.Struct):
    kv_transfer_params: _Params | None = None


# A relayed body carries the prompt's ids, beside its ``kv_transfer_params`` from
# prefill and inside it to decode. Reading or writing the stamp decodes neither:
# `_STAMP` reads the one entry, `_MEMBERS` leaves each member's value as its text.
_STAMP = msgspec.json.Decoder(_Relayed)
_MEMBERS = msgspec.json.Decoder(dict[str, msgspec.Raw])
_ENCODE = msgspec.json.Encoder()
#: msgspec's decoder of JSON with no type: `decode_relayed` decodes a relayed
#: body with it, and a simulated decode frontend's ``Request.json`` every body
#: (`atom.compass.run.handoff`), so a decoding handed on is the app's own.
JSON_DECODER = msgspec.json.Decoder()
#: The ASGI scope key under which `atom.utils.compass_loop.HttpChannel` hands a
#: relayed request's body and its `decode_relayed` decoding on to the app's
#: ``Request.json``, as ``(body, decoding)``.
RELAYED_JSON = "compass.relayed_json"


def relay_carried(body: bytes) -> tuple[tuple[float, int] | None, float]:
    """The ``(arrival, seq)`` a JSON request body's ``kv_transfer_params``
    carries, None without one, and the seconds it says its relay serves it on
    the receiving loop (`RELAY_SERVICE`), 0.0 without them or without a stamp.
    One decode reads both, past the prompt's ids a relayed body carries."""
    try:
        params = _STAMP.decode(body).kv_transfer_params
    except msgspec.DecodeError:
        return None, 0.0
    if params is None:
        return None, 0.0
    return _relay_entry(params.compass, params.compass_relay_s)


def decode_relayed(body: bytes) -> tuple[Any, tuple[float, int] | None, float]:
    """`body` decoded once, as JSON with no type, and the stamp and relay
    service `relay_carried` reads, read from that decoding; for a body msgspec
    does not decode so, None and `relay_carried`'s reading of it.

    A body naming a key twice is read as its decoding holds it, the last value
    of each, where `relay_carried` reads no stamp from one whose earlier
    ``kv_transfer_params`` is no object; a router's JSON object holds a key
    once."""
    try:
        doc = JSON_DECODER.decode(body)
    except (msgspec.DecodeError, ValueError, RecursionError):
        return (None, *relay_carried(body))
    params = doc.get("kv_transfer_params") if type(doc) is dict else None
    if type(params) is not dict:
        return doc, None, 0.0
    return (doc, *_relay_entry(params.get("compass"), params.get(RELAY_SERVICE)))


def _relay_entry(entry, value) -> tuple[tuple[float, int] | None, float]:
    """The ``(arrival, seq)`` of `entry`, a ``kv_transfer_params`` object's
    ``compass`` value, None for none, and the seconds of `value`, its
    `RELAY_SERVICE` value, 0.0 for none or without a stamp."""
    if entry is None:
        return None, 0.0
    stamp = _parse(_ENTRY, entry, "kv_transfer_params entry")
    if value is None:
        return stamp, 0.0
    # A JSON number, not a bool or a string of one.
    seconds = value if type(value) in (int, float) else math.nan
    if not (math.isfinite(seconds) and seconds >= 0):
        raise ValueError(
            f"malformed {RELAY_SERVICE} {value!r}; expected finite seconds >= 0"
        )
    return stamp, float(seconds)


def with_relay_stamp(body: bytes, stamp, service_s: float | None = None) -> bytes:
    """`body` with ``stamp()``'s ``(arrival, seq)`` in its ``kv_transfer_params``,
    and with `service_s` there under `RELAY_SERVICE` when given; `body`
    itself, and `stamp` not called, when it has no such object. Every other
    value keeps its JSON text."""
    if service_s is not None and not (math.isfinite(service_s) and service_s >= 0):
        raise ValueError(f"relay service {service_s!r} is no finite seconds >= 0")
    try:
        doc = _MEMBERS.decode(body)
        params = _MEMBERS.decode(doc["kv_transfer_params"])
    except (msgspec.DecodeError, KeyError):
        return body
    arrival, seq = stamp()
    params["compass"] = f"a:{arrival!r};s:{seq}"
    if service_s is not None:
        params[RELAY_SERVICE] = float(service_s)
    doc["kv_transfer_params"] = msgspec.Raw(_ENCODE.encode(params))
    return _ENCODE.encode(doc)


_CANCEL = re.compile(
    rb'\{"compass_cancel": (\d+)(?:, "kv_transfer_params": \{"compass": "[^"]*"\})?\}'
)


def cancel_body(seq: int) -> bytes:
    """The body of a stamped cancel of the request with http seq `seq`."""
    return b'{"compass_cancel": %d}' % seq


def cancel_seq(body: bytes) -> int | None:
    """The seq a cancel's body names; None for any other body."""
    m = _CANCEL.fullmatch(body)
    return None if m is None else int(m[1])

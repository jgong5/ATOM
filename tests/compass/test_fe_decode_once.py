# SPDX-License-Identifier: MIT
"""A simulated decode frontend's work per relayed request and per streamed
event, done once.

- A relayed hand-off behind `HttpChannel` on frontend-D, with the decode
  side's reading installed (`atom.compass.run.handoff`), is decoded once in
  all: `HttpChannel` reads its stamp from that decoding and ATOM's chat
  endpoint takes it, answering every hand-off as it does without
  `HttpChannel`, inside the run and outside it.
- A streamed response looks frontend-D's stream channel up once, however
  many frames it sends.
- `wrap_decode` decodes a repeated short span once and charges every call,
  and a stream detokenizes to the same text.
"""

import array
import asyncio
import json
import math
import threading
import uuid
from types import SimpleNamespace

import msgspec
import pytest
import starlette.requests
from test_handoff_codec import (
    HAND_OFFS,
    LONG,
    _client,
    _hand_off,
    pristine,  # noqa: F401  the fixture
)
from test_stream_events import (
    TABLE,
    _served,
    frontend_d,  # noqa: F401  the fixture
)

from atom.compass import carriers, clock_transport
from atom.compass.clock import NER, ClockAuthority, LpId
from atom.compass.run import handoff
from atom.entrypoints.openai.streaming_dispatch import IncrementalStreamDetokenizer
from atom.utils import clock, compass_loop
from atom.utils.clock import LPRuntime
from atom.utils.compass_loop import CompassEventLoop, HttpChannel, wrap_decode

RELAY = "router->frontend-D:relay"


def _scope(body: bytes) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "server": ("127.0.0.1", 8000),
        "client": ("127.0.0.1", 9000),
    }


async def _posted(channel: HttpChannel, body: bytes) -> tuple[int, object]:
    """`body` posted through `channel`: the reply's status and JSON."""
    parts = [{"type": "http.request", "body": body, "more_body": False}]
    sent = []

    async def receive():
        return parts.pop(0) if parts else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await channel(_scope(body), receive, send)
    return sent[0]["status"], json.loads(b"".join(m["body"] for m in sent[1:]))


def _counted(monkeypatch) -> list[str]:
    """Each decode of a body from here on, by who made it: frontend-D's own
    for its stamp (``stamp``, ``json``), the app's (``app``) or the standard
    library's (``loads``)."""
    calls = []

    def counted(name, decoder):
        return SimpleNamespace(decode=lambda b: calls.append(name) or decoder.decode(b))

    monkeypatch.setattr(carriers, "_STAMP", counted("stamp", carriers._STAMP))
    monkeypatch.setattr(carriers, "_JSON", counted("json", carriers._JSON))
    monkeypatch.setattr(handoff, "_DECODE", counted("app", handoff._DECODE))
    monkeypatch.setattr(
        starlette.requests,
        "json",
        SimpleNamespace(loads=lambda b: calls.append("loads") or json.loads(b)),
    )
    return calls


def _decodes(body: bytes) -> bool:
    try:
        msgspec.json.decode(body)
    except (msgspec.DecodeError, ValueError, RecursionError):
        return False
    return True


def test_a_hand_off_outside_the_run_is_decoded_once_and_answered_as_before(
    pristine, monkeypatch  # noqa: F811
):
    """Unstamped, outside the run: each hand-off passes `HttpChannel` straight
    to the app. One decode serves both where msgspec decodes the body; any
    other is read as it always was."""
    handoff.install("D")
    client, headers = _client(), {"content-type": "application/json"}
    before = {
        name: client.post("/v1/chat/completions", content=body, headers=headers)
        for name, body in HAND_OFFS.items()
    }
    rt = LPRuntime(LpId("frontend-D"), TABLE, None)
    monkeypatch.setattr(clock, "_installed", rt)
    channel = HttpChannel(client.app, lambda scope: None)
    calls = _counted(monkeypatch)
    for name, body in HAND_OFFS.items():
        calls.clear()
        got = asyncio.run(_posted(channel, body))
        assert got == (before[name].status_code, before[name].json()), name
        if _decodes(body):
            assert calls == ["json"], name
        else:
            assert calls == ["json", "stamp", "app", "loads"], name
    assert before["ids"].json() == {"ids": LONG}


def _in_the_run(app, bodies: list[bytes]) -> list[tuple[int, object]]:
    """frontend-D in its run behind the router LP, which stamps a relay send
    at LP time 0 for each of `bodies`; ``bodies[k]`` carries relay seq k's
    stamp, and `HttpChannel` serves it to `app`. Returns each reply."""
    router, me = LpId("router"), LpId("frontend-D")
    endpoint = f"inproc:test-fe-decode-once-{uuid.uuid4().hex}"
    server = clock_transport.serve(ClockAuthority(TABLE), endpoint)
    for lp in TABLE.registry.ids():
        if lp not in (router, me):
            clock_transport.connect(lp, endpoint).send((NER, math.inf, [], math.inf))
    rt = LPRuntime(me, TABLE, clock_transport.connect(me, endpoint))
    rt.start_run()
    clock.install(rt)

    def hop():
        rt = LPRuntime(router, TABLE, clock_transport.connect(router, endpoint))
        rt.start_run()
        for _ in bodies:
            rt.stamp_send(RELAY)
        rt.next_event(math.inf)
        rt.close()

    thread = threading.Thread(target=hop, name="router", daemon=True)
    thread.start()
    loop = CompassEventLoop()
    guard = threading.Timer(20, loop.call_soon_threadsafe, (loop.stop,))
    channel = HttpChannel(app, lambda scope: None)

    async def go():
        return await asyncio.gather(*(_posted(channel, b) for b in bodies))

    guard.start()
    try:
        replies = loop.run_until_complete(go())
        loop.run_forever()  # the run ends at the +inf grant
    finally:
        guard.cancel()
        thread.join(10)
        clock.install(None)
        server.close()
    loop.close()
    return replies


def test_a_stamped_hand_off_in_the_run_is_decoded_once_for_its_stamp_and_the_app(
    pristine, monkeypatch  # noqa: F811
):
    """Relays 0-2 carry their ids and a stamp, relay 1 a relay service too:
    each is decoded once in all, for its turn and for ATOM's chat endpoint,
    which resolves its ids from that decoding."""
    handoff.install("D")
    lookahead = TABLE.lookahead(RELAY)
    ids = [LONG[:5000], [0, 7], LONG[-300:]]
    bodies = [
        carriers.with_relay_stamp(
            _hand_off(json.dumps(i)), lambda k=k: (lookahead, k), 1e-3 if k else None
        )
        for k, i in enumerate(ids)
    ]
    calls = _counted(monkeypatch)
    replies = _in_the_run(_client().app, bodies)
    assert replies == [(200, {"ids": i}) for i in ids]
    assert calls == ["json"] * len(bodies)


def test_a_streamed_response_looks_its_stream_channel_up_once(
    frontend_d, monkeypatch  # noqa: F811
):
    """A 2-event stream and a 40-event one, a frame a body message: one
    lookup each, and every event stamped on the stream to the router."""
    lookups, channel_of = [], clock.channel_of

    def counted(rt, kind):
        lookups.append(kind)
        return channel_of(rt, kind)

    monkeypatch.setattr(clock, "channel_of", counted)
    streams = [
        [f"data: {k}\n\n" for k in range(n)] + ["data: [DONE]\n\n"] for n in (2, 40)
    ]
    got = _served(frontend_d, monkeypatch, streams)
    assert lookups == ["stream"] * len(streams)
    for served, frames in zip(got, streams):
        assert [e.split("\n", 1)[1] + "\n\n" for e in served["events"]] == frames
        assert [carriers.sse_stamp(e.split("\n", 1)[0]) for e in served["events"]] == [
            (a, s) for _, s, a in served["sends"]
        ]


RATE = {"fixed_s": 1e-5, "tokens_per_s": 1e6, "tokens": [0, 1000]}


def _decoder(decoded: list):
    """A tokenizer's decode, each call recorded in `decoded`."""

    def decode(ids, skip_special_tokens=False, **kwargs):
        decoded.append(list(ids))
        text = "".join(chr(0x61 + i % 26) for i in ids)
        return text if skip_special_tokens else f"<s>{text}"

    return decode


def test_a_repeated_short_decode_is_decoded_once_and_charged_every_time(monkeypatch):
    charged, decoded = [], []
    monkeypatch.setattr(compass_loop, "charge", charged.append)
    decode = wrap_decode(_decoder(decoded), RATE)
    span = [3, 3, 3, 3]
    calls = [
        (array.array("l", span), {"skip_special_tokens": True}),
        (span, {"skip_special_tokens": True}),
        (array.array("l", span), {"skip_special_tokens": True}),
        (list(span), {"skip_special_tokens": True}),
        (array.array("i", span), {"skip_special_tokens": True}),
        (array.array("l", span), {}),
        (array.array("l", span), {"skip_special_tokens": True}),
    ]
    texts = [decode(ids, **kwargs) for ids, kwargs in calls]
    assert texts == [_decoder([])(ids, **kwargs) for ids, kwargs in calls]
    assert texts[0] == "dddd" and texts[5] == "<s>dddd"
    # an array of each type code, a list, and an array without the argument
    assert len(decoded) == 4
    assert charged == [pytest.approx(1e-5 + 4 / 1e6)] * len(calls)


def test_a_decode_it_does_not_remember_is_made_every_time(monkeypatch):
    """More than `MEMO_IDS` ids, ids in a tuple, and an argument with no hash."""
    monkeypatch.setattr(compass_loop, "charge", lambda d: None)
    decoded = []
    decode = wrap_decode(_decoder(decoded), RATE)
    long = [5] * (compass_loop.MEMO_IDS + 1)
    calls = [(long, {}), ((5, 5), {}), ([5, 5], {"spaces": [1]})]
    for ids, kwargs in calls * 2:
        decode(ids, **kwargs)
    assert decoded == [list(ids) for ids, _ in calls * 2]


def test_a_full_memo_starts_over(monkeypatch):
    monkeypatch.setattr(compass_loop, "charge", lambda d: None)
    monkeypatch.setattr(compass_loop, "MEMO_DECODES", 2)
    decoded = []
    decode = wrap_decode(_decoder(decoded), RATE)
    for ids in ([1], [2], [1], [3], [1], [3]):
        decode(ids)
    assert decoded == [[1], [2], [3], [1]]


def test_a_stream_of_one_repeated_id_detokenizes_as_it_did(monkeypatch):
    """A simulated stream repeats one filler id: its deltas are the plain
    decode's, from fewer decodes."""
    monkeypatch.setattr(compass_loop, "charge", lambda d: None)
    plain, remembered = [], []

    def deltas(decode):
        state = IncrementalStreamDetokenizer(SimpleNamespace(decode=decode))
        return [state.update([11] * k, i == 29) for i, k in enumerate([4, 3, 5] * 10)]

    assert deltas(wrap_decode(_decoder(remembered), RATE)) == deltas(_decoder(plain))
    assert len(remembered) < len(plain) / 4

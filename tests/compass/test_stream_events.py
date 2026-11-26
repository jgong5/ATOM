# SPDX-License-Identifier: MIT
"""What a simulated decode frontend's loop does per streamed event and per relayed request.

Streams are served behind ATOM's own middleware, as the API server installs
it, through `HttpChannel` on a `CompassEventLoop`, with frontend-D's
`LPRuntime` on a co-hosted clock authority and an ``http_stream`` term in its
host entry: each event is charged that term and stamped on frontend-D's stream
to the router, and a charge places no tokenizer station while none is open. A
relayed request's stamp and service are read from one decode of its body.
"""

import json
import math
import threading
import uuid

import fastapi
import pytest
from starlette.responses import StreamingResponse

from atom.compass import carriers, clock_transport
from atom.compass import run as compass_run
from atom.compass.carriers import sse_stamp
from atom.compass.clock import NER, ClockAuthority, LpId, prefill_decode_table
from atom.entrypoints.openai import api_server
from atom.utils import clock
from atom.utils.clock import LPRuntime
from atom.utils.compass_loop import CompassEventLoop, HttpChannel, SimExecutor
from tests.compass.test_compass_event_loop import _relayed_to_decode

PER_EVENT = 1e-3
STREAM = "frontend-D->router:stream"
TABLE = prefill_decode_table(
    admission_s=0.54e-3,
    ipc_s=0.001,
    stream_s=0.002,
    router_s=2.0**-12,
    kv_write_req_s=2.0**-10,
    router_in_s=2.0**-14,
)
SCOPE = {
    "type": "http",
    "asgi": {"version": "3.0", "spec_version": "2.4"},
    "http_version": "1.1",
    "method": "GET",
    "scheme": "http",
    "path": "/stream",
    "raw_path": b"/stream",
    "root_path": "",
    "query_string": b"",
    "headers": [],
    "server": ("127.0.0.1", 8000),
    "client": ("127.0.0.1", 9000),
}


@pytest.fixture
def frontend_d(tmp_path, monkeypatch):
    """frontend-D in its run, its host entry charging `PER_EVENT` a streamed
    event; the router takes each event it is released and asks for nothing,
    and every other LP is idle. Each send frontend-D stamps is recorded in
    its ``sent``."""
    run = {"bound_s": 1.0, "host": {"http_stream": {"per_event_s": PER_EVENT}}}
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    endpoint = f"inproc:test-stream-events-{uuid.uuid4().hex}"
    server = clock_transport.serve(ClockAuthority(TABLE), endpoint)
    me, router = LpId("frontend-D"), LpId("router")
    idle = (NER, math.inf, [], math.inf)
    for lp in TABLE.registry.ids():
        if lp not in (me, router):
            clock_transport.connect(lp, endpoint).send(idle)
    sink = clock_transport.connect(router, endpoint)
    sink.send(idle)

    def take():
        while sink.recv()[0] < math.inf:
            sink.send(idle)

    thread = threading.Thread(target=take, name="router", daemon=True)
    thread.start()
    rt = LPRuntime(me, TABLE, clock_transport.connect(me, endpoint))
    rt.start_run()
    rt.sent, stamp_send = [], rt.stamp_send

    def recorded(ch, at=None):
        arrival, seq = stamp_send(ch, at)
        rt.sent.append((ch, seq, arrival))
        return arrival, seq

    rt.stamp_send = recorded
    clock.install(rt)
    try:
        yield rt
    finally:
        clock.install(None)
        thread.join(10)
        server.close()


def _atom_app(frames: list[str]) -> fastapi.FastAPI:
    """An app with the API server's middleware, streaming `frames` through the
    wrapper every streaming endpoint returns."""
    app = fastapi.FastAPI()
    app.user_middleware = list(api_server.app.user_middleware)

    async def replay():
        for f in frames:
            yield f

    @app.get("/stream")
    async def stream():
        return StreamingResponse(
            api_server._client_stream(replay(), "req-1"),
            media_type="text/event-stream",
        )

    return app


def _served(rt, monkeypatch, streams: list[list[str]]) -> list[dict]:
    """Each of `streams` served in turn behind `HttpChannel` on one loop: for
    each, the LP clock it starts at, the events the client is sent, their
    sends and the station placements made."""
    placed, out = [], []
    place = SimExecutor.place

    def counted_place(self, wait=False):
        placed.append(wait)
        return place(self, wait)

    monkeypatch.setattr(SimExecutor, "place", counted_place)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    sent = []

    async def send(message):
        if message["type"] == "http.response.body" and message["body"]:
            sent.append(message["body"].decode())

    loop = CompassEventLoop()
    for frames in streams:
        start, log = rt.now, len(rt.sent)
        sent.clear()
        placed.clear()
        channel = HttpChannel(_atom_app(frames), lambda scope: None)
        loop.run_until_complete(channel(dict(SCOPE), receive, send))
        out.append(
            {
                "start": start,
                "events": "".join(sent).split("\n\n")[:-1],
                "sends": rt.sent[log:],
                "placed": list(placed),
            }
        )
    loop.run_forever()  # the run ends at the +inf grant
    loop.close()
    return out


def test_each_event_behind_atoms_middleware_is_charged_then_stamped_placing_nothing(
    frontend_d, monkeypatch
):
    """Each event but ``[DONE]`` is charged the term before its stamp, on the
    stream to the router, in a 2-event stream and a 40-event one. With no
    tokenizer job open, a charge places nothing."""
    streams = [
        [f"data: {k}\n\n" for k in range(n)] + ["data: [DONE]\n\n"] for n in (2, 40)
    ]
    lookahead = TABLE.lookahead(STREAM)
    for got, frames in zip(_served(frontend_d, monkeypatch, streams), streams):
        n = len(frames) - 1
        charged = [(k + 1) * PER_EVENT for k in range(n)] + [n * PER_EVENT]
        assert [e.split("\n", 1)[1] + "\n\n" for e in got["events"]] == frames
        assert [sse_stamp(e.split("\n", 1)[0]) for e in got["events"]] == [
            (a, s) for _, s, a in got["sends"]
        ]
        assert [ch for ch, _, _ in got["sends"]] == [STREAM] * (n + 1)
        assert [a for _, _, a in got["sends"]] == pytest.approx(
            [got["start"] + c + lookahead for c in charged]
        )
        assert got["placed"] == []


def test_a_relayed_request_s_stamp_and_service_are_read_from_one_decode(monkeypatch):
    """Relays 0-2 arrive at one instant, serving 4, 2 and no ms, relay 3 a
    second later serving 3 ms; each enters the app after the services ahead
    of it. frontend-D decodes each body once for its stamp and service, and
    the app once more for the seq it records."""
    service = {0: 4e-3, 1: 2e-3, 3: 3e-3}
    decodes = []

    class Counted:
        def __init__(self, decoder):
            self.decoder = decoder

        def decode(self, body):
            decodes.append(len(body))
            return self.decoder.decode(body)

    monkeypatch.setattr(carriers, "_STAMP", Counted(carriers._STAMP))
    monkeypatch.setattr(carriers, "JSON_DECODER", Counted(carriers.JSON_DECODER))

    def bodies(k, arrival):
        params = {"compass": f"a:{arrival!r};s:{k}", "prompt_token_ids": [7] * 1000}
        if k in service:
            params[carriers.RELAY_SERVICE] = service[k]
        return json.dumps({"kv_transfer_params": params}).encode()

    entered, lookahead = _relayed_to_decode([0.0, 0.0, 0.0, 1.0], bodies)
    t0, t1 = lookahead, 1.0 + lookahead
    assert [k for k, _ in entered] == [2, 1, 0, 3]
    assert [t for _, t in entered] == pytest.approx(
        [t0, t0 + 2e-3, t0 + 2e-3 + 4e-3, t1 + 3e-3]
    )
    assert len(decodes) == 2 * len(entered)

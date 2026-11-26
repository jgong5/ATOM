# SPDX-License-Identifier: MIT
"""A prefill-decode run's ``relay`` term: a prefill response's time from
prefill's server to decode's app call through the router, beyond ``router_s``.

On w1 (pit2-p03-g08, #3720) P's last prefill step end to D's engine taking the
request in is p50 24 ms: the response carrying the prompt's token ids through
the router (~1.5 ms + 8-12 ns per byte of decode's request) and decode's API
server taking that body in (`http_intake`, ~1.2 ms + 12 ns/B). The term is
fitted by `step_linear_fit.fit_relay`; here it is read off the run file and
written on the relayed response as its service time, which decode's API
server loop charges (`tests/compass/test_compass_event_loop.py`), while the
relay send arrives at the channel's lookahead.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from atom.compass import run as compass_run
from atom.compass.carriers import RELAY_SERVICE, relay_carried
from atom.utils.compass_loop import _stamping

TERM = {"fixed_s": 1.5e-3, "per_byte_s": 1e-6, "bytes": [100, 10**7]}
RELAY = "frontend-P->router:relay"


def write(tmp_path, monkeypatch, **run):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"bound_s": 1.0, **run}))
    monkeypatch.setenv(compass_run.ENV, str(path))


@pytest.mark.parametrize(
    "term",
    [
        {"fixed_s": 1e-3},
        {"fixed_s": -1e-3, "per_byte_s": 1e-8},
        {"fixed_s": 1e-3, "per_byte_s": 1e-8, "bytes": [10, 1]},
        {"fixed_s": 1e-3, "per_byte_s": 1e-8, "idle_s": 1e-3},
    ],
)
def test_a_relay_term_the_frontend_would_misread_is_refused(
    tmp_path, monkeypatch, term
):
    write(tmp_path, monkeypatch, router_s=2e-3, host={"relay": term})
    with pytest.raises(ValueError, match="host's relay is"):
        compass_run.spec()


def test_a_prefill_decode_run_reads_its_prefill_entry_s_relay_with_router_s(
    tmp_path, monkeypatch
):
    write(tmp_path, monkeypatch, router_s=2e-3, host={"relay": TERM})
    assert compass_run.relay() == (1.5e-3, 1e-6, 2e-3)
    write(tmp_path, monkeypatch, router_s=2e-3, host={})
    assert compass_run.relay() is None
    # A run that is not prefill-decode relays nothing.
    write(tmp_path, monkeypatch, host={"relay": TERM})
    assert compass_run.relay() is None
    monkeypatch.delenv(compass_run.ENV)
    assert compass_run.relay() is None


def stamped(body: bytes, relay, request=0):
    """The arrival `_stamping` asks for a JSON response `body` sent on the
    relay, None for the channel's lookahead, and the body it sends, for a
    request of `request` bytes, the LP at 2 s and the lookahead 1 ms."""
    asked, sent = [], []
    lookahead = 1e-3

    def stamp_send(ch, at=None):
        asked.append(at)
        return (2.0 + lookahead if at is None else at), 0

    rt = SimpleNamespace(
        now=2.0,
        table=SimpleNamespace(lookahead=lambda ch: lookahead),
        stamp_send=stamp_send,
    )

    async def send(message):
        sent.append(message)

    async def go():
        relayed = _stamping(send, rt, RELAY, relay, [request])
        head = [(b"content-type", b"application/json")]
        await relayed({"type": "http.response.start", "status": 200, "headers": head})
        await relayed({"type": "http.response.body", "body": body})

    asyncio.run(go())
    assert b'"compass"' in sent[-1]["body"]
    return asked[0], sent[-1]["body"]


def test_a_relayed_response_carries_its_relay_time_past_router_s_as_its_service():
    """Decode's request carries the client's request and the response's token
    ids: the relay takes the time of both bodies' bytes. It is written on the
    response for decode's loop to charge; the send itself arrives at the
    channel's lookahead, so relays in flight together queue on that loop."""
    ids = list(range(100_000, 101_000))
    body = json.dumps({"kv_transfer_params": {"x": 1}, "prompt_token_ids": ids})
    body = body.encode()
    took = 1.5e-3 + (4000 + len(body)) * 1e-6
    at, sent = stamped(body, (1.5e-3, 1e-6, 2e-3), 4000)
    assert at is None
    assert relay_carried(sent)[1] == pytest.approx(took - 2e-3)
    assert json.loads(sent)["kv_transfer_params"]["x"] == 1
    # Shorter than router_s, or no term: no service, and the body as before.
    for relay in ((1e-4, 1e-9, 2e-3), None):
        at, sent = stamped(body, relay)
        assert at is None and relay_carried(sent)[1] == 0.0
        assert RELAY_SERVICE.encode() not in sent

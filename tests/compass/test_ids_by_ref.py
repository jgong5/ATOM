# SPDX-License-Identifier: MIT
"""A prompt's ids handed from a simulated prefill frontend to decode by
reference (`atom.compass.run.handoff`, ``COMPASS_IDS_BY_REF_DIR``).

Prefill's response, rendered as its server renders it, carries a reference
in place of the ids and leaves them in a file; ATOM's chat endpoint on
decode's side resolves the same ids from it and removes the file, so a
reference is read once. Every count of a body's bytes sees the ids inline:
the relay service prefill stamps on its response, and decode's intake
charges and turn order behind `HttpChannel` in its run. Ids an int32 array
does not hold go inline, and with no directory nothing changes.
"""

import collections
import json
import os

import pytest
from starlette.responses import JSONResponse
from test_fe_decode_once import RELAY, _in_the_run
from test_handoff_codec import (
    LONG,
    _client,
    _hand_off,
    _response,
    pristine,  # noqa: F401  the fixture
)
from test_pd_relay_term import stamped
from test_stream_events import TABLE

from atom.compass import carriers
from atom.compass import run as compass_run
from atom.compass.run import handoff
from atom.utils import compass_loop

JSON = {"content-type": "application/json"}


@pytest.fixture
def by_ref(pristine, monkeypatch, tmp_path):  # noqa: F811
    """Both sides installed, ids by reference through `tmp_path`."""
    monkeypatch.setattr(handoff, "_DIR", str(tmp_path))
    monkeypatch.setattr(handoff, "_READ_BACK", collections.OrderedDict())
    handoff.install("P")
    handoff.install("D")
    return tmp_path


def _text(ids) -> str:
    return json.dumps(ids, separators=(",", ":"))


def _prefill(ids) -> tuple[bytes, list]:
    """Prefill's response for `ids`, and what it carries in their place."""
    body = JSONResponse(_response(ids)).body
    return body, json.loads(body)[handoff.KEY]


def test_decode_takes_the_ids_prefill_handed_by_reference_and_removes_their_file(
    by_ref,
):
    _, ref = _prefill(LONG)
    assert ref[0] == handoff.REF and len(os.listdir(by_ref)) == 1
    reply = _client().post(
        "/v1/chat/completions", content=_hand_off(_text(ref)), headers=JSON
    )
    assert reply.json() == {"ids": LONG}
    # Their check skipped: the ids read back were taken as they are.
    assert os.listdir(by_ref) == [] and not handoff._READ_BACK


def test_a_reference_is_read_once_and_one_body_holds_one(by_ref):
    _, ref = _prefill(LONG)
    client, body = _client(), _hand_off(_text(ref))
    assert client.post("/v1/chat/completions", content=body, headers=JSON).json() == {
        "ids": LONG
    }
    again = client.post("/v1/chat/completions", content=body, headers=JSON)
    assert again.status_code == 400
    with pytest.raises(ValueError, match="read already"):
        handoff._take(json.loads(body), body)
    with pytest.raises(ValueError, match="two hand-off references"):
        handoff.inline_extra(_hand_off(_text(ref), f'"prompt_token_ids":{_text(ref)}'))


def test_a_prefill_response_and_its_relay_service_count_the_ids_inline(
    by_ref, monkeypatch
):
    body, _ = _prefill(LONG)
    monkeypatch.setattr(handoff, "_DIR", None)
    inline, _ = _prefill(LONG)
    monkeypatch.setattr(handoff, "_DIR", str(by_ref))
    assert len(body) < len(inline) // 100
    assert len(body) + handoff.inline_extra(body) == len(inline)
    relay = (1.5e-3, 1e-6, 2e-3)
    service = [
        carriers.relay_carried(stamped(b, relay, 4000)[1])[1] for b in (body, inline)
    ]
    assert service[0] == service[1] > 0


def test_decode_s_intake_and_turn_order_count_a_reference_s_ids_inline(
    by_ref, monkeypatch
):
    """Two hand-offs relayed at one instant, 60,000 ids by reference and
    1,000 inline: counted inline, the first body is the longer, so it takes
    the second turn, and each intake is charged for its bytes inline."""
    intake = (1e-4, 1e-8, 1e-4)
    monkeypatch.delenv(compass_run.ENV, raising=False)
    monkeypatch.setattr(compass_run, "http_intake", lambda: intake)
    charged, charge = [], compass_loop.charge
    monkeypatch.setattr(
        compass_loop, "charge", lambda d: charged.append(d) or charge(d)
    )
    lookahead = TABLE.lookahead(RELAY)

    def relayed(texts):
        return [
            carriers.with_relay_stamp(_hand_off(t), lambda k=k: (lookahead, k), None)
            for k, t in enumerate(texts)
        ]

    short = _text(LONG[:1000])
    inline = relayed([_text(LONG), short])
    replies = _in_the_run(_client().app, relayed([_text(_prefill(LONG)[1]), short]))
    assert replies == [(200, {"ids": LONG}), (200, {"ids": LONG[:1000]})]
    fixed, per_byte, _ = intake
    assert charged == [pytest.approx(fixed + per_byte * len(b)) for b in inline[::-1]]


@pytest.mark.parametrize("ids", [[1, 2**31], [5, -1], [True, 2], []])
def test_ids_an_int32_array_does_not_hold_go_inline_as_before(by_ref, monkeypatch, ids):
    content = {**_response([1]), handoff.KEY: ids}
    body = JSONResponse(content).body
    assert os.listdir(by_ref) == [] and handoff.inline_extra(body) == 0
    monkeypatch.setattr(handoff, "_DIR", None)
    assert JSONResponse(content).body == body


def test_with_no_directory_ids_go_inline_and_decode_refuses_a_reference(
    pristine, monkeypatch  # noqa: F811
):
    monkeypatch.setattr(handoff, "_DIR", None)
    handoff.install("P")
    handoff.install("D")
    body, ids = _prefill(LONG)
    assert ids == LONG and handoff.inline_extra(body) == 0
    ref = _text([handoff.REF, os.getpid(), 0, 100])
    reply = _client().post("/v1/chat/completions", content=_hand_off(ref), headers=JSON)
    assert reply.status_code == 400
    assert "non-negative integers" in reply.json()["refused"]

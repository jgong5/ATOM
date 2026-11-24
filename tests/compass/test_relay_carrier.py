# SPDX-License-Identifier: MIT
"""The relay carrier: the stamp in ``kv_transfer_params`` on the prefill-to-decode hop.

Prefill answers the PD router with ATOM's chat response, ``prompt_token_ids``
included, rendered by starlette's `JSONResponse` and sent behind `HttpChannel`,
which stamps it for the router. The router moves the ids into the decode
request's ``kv_transfer_params`` and restamps it there, where decode reads it.
"""

import asyncio
import json
import math

import pytest
from starlette.responses import JSONResponse

from atom.compass.carriers import relay_stamp, with_relay_stamp
from atom.compass.clock import LpId, prefill_decode_table
from atom.utils import clock
from atom.utils.clock import LPRuntime
from atom.utils.compass_loop import HttpChannel

RELAY = "frontend-P->router:relay"
IDS = [0, 7, 129279, 42] * 2500


def _response() -> dict:
    """A prefill response as ATOM's chat endpoint returns one to the PD router."""
    return {
        "id": "chatcmpl-0123456789abcdef0123456789abcdef",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"content": "é "}}],
        "usage": {"prompt_tokens": len(IDS), "ttft": 0.1 + 0.2},
        "kv_transfer_params": {"remote_block_ids": [3, 1, 2], "dp_rank": 5},
        "prompt_token_ids": IDS,
    }


def _decode_request(**params) -> bytes:
    """The decode request the router builds: the ids moved into ``kv_transfer_params``."""
    params = {"remote_block_ids": [3, 1, 2], "prompt_token_ids": IDS, **params}
    body = {
        "messages": [{"role": "user", "content": "x"}],
        "kv_transfer_params": params,
    }
    return json.dumps(body).encode()


def _sent_as_frontend_p(document: dict) -> tuple[LPRuntime, list]:
    """`document` served as a JSON response behind `HttpChannel` on frontend-P."""
    table = prefill_decode_table(
        admission_s=0.54e-3,
        ipc_s=0.001,
        stream_s=0.002,
        router_s=0.004,
        kv_write_req_s=0.001,
        router_in_s=0.001,
    )
    rt = LPRuntime(LpId("frontend-P"), table, conn=None)
    rt.now = 2.0
    rt.start_run()
    sent = []

    async def app(scope, receive, send):
        await JSONResponse(content=document)(scope, receive, send)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"spec_version": "2.4"},
        "method": "POST",
        "path": "/prefill",
        "headers": [],
    }
    clock.install(rt)
    try:
        asyncio.run(HttpChannel(app, lambda scope: None)(scope, receive, send))
    finally:
        clock.install(None)
    return rt, sent


class TestPrefillResponse:
    def test_the_stamp_is_the_relay_send_and_goes_last_in_kv_transfer_params(self):
        rt, sent = _sent_as_frontend_p(_response())
        start, body = sent[0], sent[1]["body"]
        arrival = 2.0 + rt.table.lookahead(RELAY)
        assert rt.send_log == [(RELAY, 0, arrival)]
        expected = _response()
        expected["kv_transfer_params"]["compass"] = f"a:{arrival!r};s:0"
        assert json.loads(body) == expected
        assert list(json.loads(body)["kv_transfer_params"])[-1] == "compass"
        assert relay_stamp(body) == (arrival, 0)
        assert dict(start["headers"])[b"content-length"] == str(len(body)).encode()

    def test_every_value_but_kv_transfer_params_keeps_the_text_it_was_rendered_as(
        self,
    ):
        rendered = JSONResponse(content=_response()).body
        _, sent = _sent_as_frontend_p(_response())
        body = sent[1]["body"]
        for member in ("prompt_token_ids", "choices", "usage"):
            text = json.dumps(
                _response()[member], ensure_ascii=False, separators=(",", ":")
            ).encode()
            assert b'"%s":%s' % (member.encode(), text) in rendered
            assert b'"%s":%s' % (member.encode(), text) in body

    def test_a_compass_entry_already_there_is_replaced_in_its_place(self):
        body = json.dumps({"kv_transfer_params": {"compass": "x", "dp_rank": 1}})
        calls = []

        def stamp():
            calls.append(None)
            return math.inf, 4

        out = with_relay_stamp(body.encode(), stamp)
        assert json.loads(out) == {
            "kv_transfer_params": {"compass": "a:inf;s:4", "dp_rank": 1}
        }
        assert len(calls) == 1
        assert relay_stamp(out) == (math.inf, 4)

    @pytest.mark.parametrize(
        "body",
        [
            b"",
            b"not json",
            b"[1, 2]",
            b'{"dp_rank": 1}',
            b'{"kv_transfer_params": null}',
            b'{"kv_transfer_params": [1]}',
            b'{"kv_transfer_params": {}, "prompt_token_ids": [1, 2,]}',
            b'{"kv_transfer_params": {}} trailing',
        ],
    )
    def test_a_body_without_a_kv_transfer_params_object_is_sent_as_it_is_unstamped(
        self, body
    ):
        def stamp():
            raise AssertionError("stamped")

        assert with_relay_stamp(body, stamp) is body
        assert relay_stamp(body) is None


class TestDecodeRequest:
    @pytest.mark.parametrize("arrival", [0.0, 0.1 + 0.2, 1e-05, math.inf])
    def test_the_stamp_the_router_wrote_beside_the_moved_ids_reads_back_exactly(
        self, arrival
    ):
        body = _decode_request(compass=f"a:{arrival!r};s:17")
        assert relay_stamp(body) == (arrival, 17)

    @pytest.mark.parametrize("params", [{}, {"compass": None}])
    def test_no_entry_reads_as_none(self, params):
        assert relay_stamp(_decode_request(**params)) is None

    def test_a_malformed_entry_is_refused_by_name(self):
        with pytest.raises(ValueError, match="malformed compass kv_transfer_params"):
            relay_stamp(_decode_request(compass="a:1.5"))

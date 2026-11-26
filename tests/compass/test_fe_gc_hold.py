# SPDX-License-Identifier: MIT
"""A decode frontend's in-flight hand-off holds its prompt's ids where the
cyclic garbage collector does not walk them (`atom.compass.run.handoff`).

ATOM's API server takes a relayed chat request on its own route, behind its
own middleware: FastAPI reads the body through Starlette's ``Request.json``,
and the request keeps that decoding until its response ends. Only the
engine's setup and the stream's chunks are stood in for; the stream stays
open while the test looks at what the request holds.
"""

import asyncio
import gc
import json
from types import SimpleNamespace

import pytest
from starlette.requests import Request
from test_handoff_codec import _client, _hand_off, pristine  # noqa: F401  fixture

from atom.compass.run import handoff
from atom.entrypoints.openai import api_server, protocol

IDS = list(range(70_000, 120_000))
KV = {"do_remote_prefill": True, "remote_engine_id": "p0", "remote_block_ids": [4, 5]}
BODY = json.dumps(
    {
        "model": "m",
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
        "kv_transfer_params": {**KV, "prompt_token_ids": IDS},
    }
).encode()


@pytest.fixture
def server(monkeypatch):
    """ATOM's chat route with the engine's setup and the stream's chunks
    stood in for: the setup records what the engine would be handed, and the
    stream waits for `release` before its one event."""
    seen = SimpleNamespace(engine=[], release=None)

    async def setup(prompt, _sampling, _request_id, **kwargs):
        seen.engine.append((prompt, kwargs["kv_transfer_params"]))
        return 7, object(), len(prompt)

    def stream(*_args, **_kwargs):
        async def events():
            await seen.release.wait()
            yield "data: [DONE]\n\n"

        return events()

    for name, value in {
        "model_name": "m",
        "_request_logger": None,
        "default_chat_template_kwargs": {},
        "reasoning_toggle": None,
        "custom_message_encoder": None,
        "tool_call_parser_cls": None,
        "tokenizer": SimpleNamespace(decode=lambda _ids, **_kw: ""),
        "setup_streaming_request": setup,
        "stream_chat_response": stream,
    }.items():
        monkeypatch.setattr(api_server, name, value)
    return seen


def _stream_held(seen) -> tuple[list, dict]:
    """Stream `BODY` through ATOM's app; return the response's messages, and
    the ids the request's decoding holds once the response has started, with
    whether the collector tracks them after a full collection then."""
    held = {}

    async def run():
        seen.release = asyncio.Event()
        sent, given = [], []

        async def receive():
            if not given:
                given.append(True)
                return {"type": "http.request", "body": BODY, "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.start":
                # The stream waits on `release` for its first event.
                gc.collect()
                (request,) = [
                    o
                    for o in gc.get_objects()
                    if isinstance(o, Request) and hasattr(o, "_json")
                ]
                ids = request._json["kv_transfer_params"]["prompt_token_ids"]
                held.update(ids=ids, tracked=gc.is_tracked(ids))
                seen.release.set()

        path = "/v1/chat/completions"
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json"), (b"host", b"d")],
            "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 80),
            "state": {},
        }
        await asyncio.wait_for(api_server.app(scope, receive, send), 30)
        return sent

    sent = asyncio.run(run())
    return [(m["type"], m.get("status"), m.get("body")) for m in sent], held


def test_a_streaming_hand_offs_held_ids_are_untracked_and_reach_the_engine_as_before(
    pristine, server  # noqa: F811
):
    before, held_before = _stream_held(server)
    handoff.install("D")
    after, held_after = _stream_held(server)

    assert after == before
    assert before[0][:2] == ("http.response.start", 200)
    # Starlette held the decoding's ids all along the stream: a list the
    # collector walks on every full collection; now a tuple it stops tracking.
    assert type(held_before["ids"]) is list and held_before["tracked"]
    assert type(held_after["ids"]) is tuple and not held_after["tracked"]
    assert list(held_after["ids"]) == held_before["ids"] == IDS
    # The engine is handed the same prompt, a list, and the same metadata.
    (prompt_before, kv_before), (prompt_after, kv_after) = server.engine
    assert type(prompt_after) is list and prompt_after == prompt_before == IDS
    assert kv_after == kv_before == KV


def test_ids_msgspec_refuses_reach_pydantic_as_the_list_the_body_held(
    pristine, monkeypatch  # noqa: F811
):
    validated = []
    adapter = protocol._PROMPT_TOKEN_IDS_ADAPTER
    monkeypatch.setattr(
        protocol,
        "_PROMPT_TOKEN_IDS_ADAPTER",
        SimpleNamespace(
            validate_python=lambda ids: validated.append(ids)
            or adapter.validate_python(ids)
        ),
    )
    handoff.install("D")
    reply = _client().post(
        "/v1/chat/completions",
        content=_hand_off("[true, 2]"),
        headers={"content-type": "application/json"},
    )
    assert reply.json() == {"ids": [1, 2]}
    assert validated == [[True, 2]] and type(validated[0]) is list

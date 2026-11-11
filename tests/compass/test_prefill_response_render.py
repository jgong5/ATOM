# SPDX-License-Identifier: MIT
"""A prefill response's body, as ATOM's chat and completion handlers render it.

The handlers are ATOM's own over a stubbed generation that answers as a prefill
server does: its prompt's ids when asked for them, KV transfer metadata holding
a UUID and a tuple, text with non-ASCII, quotes and a newline, and usage's
rounded seconds. Each body is held to what stdlib ``json`` renders of the same
dumped document, as starlette's ``JSONResponse`` rendered it before.
"""

import asyncio
import json
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
from starlette.responses import JSONResponse

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
)

IDS = list(range(100_000, 200_000))
PATH = {"chat": "/v1/chat/completions", "completion": "/v1/completions"}


@pytest.fixture
def served(monkeypatch):
    """The handlers over a stubbed prefill; `dumped` holds each response's
    document as the handler dumped it."""
    dumped = []

    async def generate(_prompt, _sampling, _rid, **kwargs):
        output = {
            "text": '你好，世界 🌍\n"quoted" \\ é',
            "finish_reason": "length",
            "num_tokens_input": len(IDS),
            "num_tokens_output": 1,
            "num_cached_tokens": 4096,
            "ttft": 1.23456789,
            "tpot": 0.0,
            "latency": 0.00012,
            "kv_transfer_output_meta_info": {
                "do_remote_decode": False,
                "remote_engine_id": UUID("b1382773-c171-4e07-b42d-e7ff18a825ab"),
                "remote_block_ids": (7, 8, 9),
                "remote_host": None,
            },
        }
        if kwargs.get("return_token_ids"):
            output["prompt_token_ids"] = list(IDS)
        yield output

    for name in ("build_chat_response", "build_completion_response"):
        build = getattr(api_server, name)

        def recording(*args, _build=build, **kwargs):
            response = _build(*args, **kwargs)
            dumped.append(response.model_dump(mode="json"))
            return response

        monkeypatch.setattr(api_server, name, recording)
    monkeypatch.setattr(api_server, "generate_async", generate)
    monkeypatch.setattr(api_server, "model_name", "m")
    monkeypatch.setattr(api_server, "_request_logger", None)
    monkeypatch.setattr(api_server, "default_chat_template_kwargs", {})
    monkeypatch.setattr(api_server, "reasoning_toggle", None)
    monkeypatch.setattr(api_server, "custom_message_encoder", None)
    monkeypatch.setattr(api_server, "tool_call_parser_cls", None)
    monkeypatch.setattr(api_server, "apply_chat_template", lambda *_a, **_k: "p")
    monkeypatch.setattr(
        api_server,
        "tokenizer",
        SimpleNamespace(encode=lambda *_a, **_k: [1, 2], decode=lambda *_a, **_k: ""),
    )
    return dumped


def _payload(endpoint: str, return_ids: bool) -> dict:
    payload = {"model": "m", "max_tokens": 1, "return_token_ids": return_ids}
    if endpoint == "chat":
        return payload | {"messages": [{"role": "user", "content": "hi"}]}
    return payload | {"prompt": "hi"}


def _post(endpoint: str, return_ids: bool) -> httpx.Response:
    payload = _payload(endpoint, return_ids)

    async def post():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_server.app), base_url="http://t"
        ) as client:
            return await client.post(PATH[endpoint], json=payload)

    return asyncio.run(post())


def _handle(endpoint: str, return_ids: bool):
    payload = _payload(endpoint, return_ids)
    if endpoint == "chat":
        payload["messages"] = [ChatMessage(role="user", content="hi")]
        request, handler = ChatCompletionRequest(**payload), api_server.chat_completions
    else:
        request, handler = CompletionRequest(**payload), api_server.completions
    return asyncio.run(handler(request, None))


def _members(body: bytes):
    """The document `body` holds, every object as its (key, value) pairs in order."""
    return json.loads(body, object_pairs_hook=list)


@pytest.mark.parametrize("endpoint", ["chat", "completion"])
@pytest.mark.parametrize("return_ids", [True, False])
def test_a_client_reads_the_bytes_stdlib_json_rendered(served, endpoint, return_ids):
    got = _post(endpoint, return_ids)

    assert got.status_code == 200
    [doc] = served
    assert got.content == JSONResponse(doc).body
    assert got.headers["content-type"] == "application/json"
    assert got.headers["content-length"] == str(len(got.content))
    body = got.json()
    assert body["prompt_token_ids"] == (IDS if return_ids else None)
    assert body["kv_transfer_params"] == {
        "do_remote_decode": False,
        "remote_engine_id": "b1382773-c171-4e07-b42d-e7ff18a825ab",
        "remote_block_ids": [7, 8, 9],
        "remote_host": None,
    }


@pytest.mark.parametrize("endpoint", ["chat", "completion"])
def test_the_response_has_jsonresponse_s_status_and_headers(served, endpoint):
    response = _handle(endpoint, return_ids=True)

    [doc] = served
    plain = JSONResponse(doc)
    assert isinstance(response, JSONResponse)
    assert (response.status_code, response.media_type) == (200, "application/json")
    assert response.raw_headers == plain.raw_headers
    assert response.body == plain.body


@pytest.mark.parametrize("endpoint", ["chat", "completion"])
def test_the_prompt_s_ids_are_not_rendered_by_stdlib_json(
    served, monkeypatch, endpoint
):
    rendered, render = [], JSONResponse.render

    def stdlib_render(self, content):
        rendered.append(list(content))
        return render(self, content)

    monkeypatch.setattr(JSONResponse, "render", stdlib_render)

    got = _post(endpoint, return_ids=True)

    assert (got.status_code, rendered) == (200, [])
    assert got.json()["prompt_token_ids"] == IDS


def test_a_float_spelled_otherwise_reads_back_as_the_same_number():
    usage = {"small": 1e-05, "tiny": -2.5e-07, "big": 1e16, "huge": 1.5e300}
    doc = {"usage": usage | {"seconds": 0.0001}, "prompt_token_ids": None}

    body = api_server._DumpedJSONResponse(doc).body

    assert body != JSONResponse(doc).body
    assert _members(body) == _members(JSONResponse(doc).body)
    assert json.loads(body)["usage"] == doc["usage"]

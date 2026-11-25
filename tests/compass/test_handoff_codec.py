# SPDX-License-Identifier: MIT
"""A simulated prefill-decode hand-off's prompt token ids, read and written
with msgspec (`atom.compass.run.handoff`).

A decode frontend's bootstrap installs the reading, a prefill frontend's the
writing. Each is held to what the server did without it:

- ATOM's `ChatCompletionRequest` behind a FastAPI app, as decode's server
  takes a hand-off, answers every body as before: it refuses the
  ``kv_transfer_params["prompt_token_ids"]`` that `resolve_prompt_token_ids`
  refuses, with the same detail, and resolves the same ids from the rest. A
  hand-off's ids reach neither ``json.loads`` nor pydantic.
- ATOM's `ChatCompletionResponse`, rendered by Starlette's `JSONResponse` as
  prefill's server renders it, has the same bytes, and its ids reach no
  standard library encoder; a response whose ids msgspec writes otherwise
  renders, or is refused, as before.
"""

import inspect
import json
import math
from types import SimpleNamespace

import msgspec
import pytest
import starlette.requests
from conftest import atom_config_double
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import TypeAdapter
from starlette.requests import Request
from starlette.responses import JSONResponse
from test_pd_slice import PD
from test_vertical_slice import _run_file

from atom.compass import clock_transport
from atom.compass import run as compass_run
from atom.compass.clock import ClockAuthority
from atom.compass.run import handoff
from atom.entrypoints.openai import protocol
from atom.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from atom.utils import clock

LONG = list(range(100_000, 160_000))


@pytest.fixture
def pristine(monkeypatch):
    """Starlette's and ATOM's own reading and writing, whatever an earlier
    test's bootstrap installed; each restored afterwards."""
    monkeypatch.setattr(Request, "json", inspect.unwrap(Request.json))
    monkeypatch.setattr(JSONResponse, "render", inspect.unwrap(JSONResponse.render))
    adapter = TypeAdapter(protocol.PromptTokenIds)
    monkeypatch.setattr(protocol, "_PROMPT_TOKEN_IDS_ADAPTER", adapter)


def _client() -> TestClient:
    """Decode's taking of a hand-off: the body read into ATOM's request
    model, then its ids resolved."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: ChatCompletionRequest):
        try:
            return {"ids": request.get_prompt_token_ids()}
        except ValueError as e:
            return JSONResponse({"refused": str(e)}, status_code=400)

    return TestClient(app)


def _hand_off(ids: str, *more: str) -> bytes:
    """A hand-off as the router writes it, its ids' JSON text `ids`, with
    each of `more` a further top-level member's text."""
    head = '{"model":"m","messages":[{"role":"user","content":"hi"}],'
    tail = '"kv_transfer_params":{"do_remote_prefill":true,"prompt_token_ids":%s}}'
    return (head + "".join(f"{m}," for m in more) + tail % ids).encode()


HAND_OFFS = {
    "ids": _hand_off(json.dumps(LONG)),
    "a zero": _hand_off("[0, 7]"),
    "past 64 bits": _hand_off("[18446744073709551616, 1]"),
    "the same ids twice": _hand_off("[1, 2]", '"prompt_token_ids":[1,2]'),
    "no ids": _hand_off("null"),
    "bools": _hand_off("[true, 2]"),
    "a whole float": _hand_off("[1.0, 2]"),
    "a string of digits": _hand_off('["3", 2]'),
    "a fraction": _hand_off("[1.5, 2]"),
    "a negative": _hand_off("[-1, 2]"),
    "none": _hand_off("[]"),
    "a string": _hand_off('"1,2"'),
    "an object": _hand_off('{"0":1}'),
    "a nested list": _hand_off("[[1], 2]"),
    "two prompts": _hand_off("[1, 2]", '"prompt_token_ids":[1,3]'),
    "NaN": _hand_off("[1, 2]", '"temperature":NaN'),
    "a lone surrogate": _hand_off("[1, 2]", '"user":"\\ud800"'),
    "a byte order mark": b"\xef\xbb\xbf" + _hand_off("[1, 2]"),
    "UTF-16": _hand_off("[1, 2]").decode().encode("utf-16"),
    "cut short": _hand_off("[1, 2]")[:-9],
}


def _answers(client: TestClient) -> dict[str, tuple[int, object]]:
    headers = {"content-type": "application/json"}
    replies = {
        name: client.post("/v1/chat/completions", content=body, headers=headers)
        for name, body in HAND_OFFS.items()
    }
    return {name: (r.status_code, r.json()) for name, r in replies.items()}


def test_a_decode_frontend_refuses_what_it_refused_and_takes_the_same_ids_from_the_rest(
    pristine,
):
    client = _client()
    before = _answers(client)
    handoff.install("D")
    assert _answers(client) == before
    assert before["ids"] == (200, {"ids": LONG})
    assert before["bools"] == (200, {"ids": [1, 2]})
    assert {n for n, (status, _) in before.items() if status == 400} == {
        "a fraction",
        "a negative",
        "none",
        "a string",
        "an object",
        "a nested list",
        "two prompts",
    }
    assert before["cut short"][0] == 422


def test_a_hand_offs_ids_reach_neither_json_loads_nor_pydantic(pristine, monkeypatch):
    loaded, validated = [], []
    monkeypatch.setattr(
        starlette.requests,
        "json",
        SimpleNamespace(loads=lambda body: loaded.append(body) or json.loads(body)),
    )
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
    handoff.install("D")
    assert inspect.unwrap(Request.json) is Request.json.__wrapped__
    client, headers = _client(), {"content-type": "application/json"}
    reply = client.post(
        "/v1/chat/completions", content=HAND_OFFS["ids"], headers=headers
    )
    assert reply.json() == {"ids": LONG}
    assert loaded == validated == []
    # What msgspec does not take, the standard library and pydantic still decide.
    body = _hand_off("[true, 2]", '"temperature":NaN')
    reply = client.post("/v1/chat/completions", content=body, headers=headers)
    assert reply.json() == {"ids": [1, 2]}
    assert loaded == [body] and validated == [[True, 2]]


def _response(ids) -> dict:
    """Prefill's response to the router, as its server hands it to `JSONResponse`."""
    return ChatCompletionResponse(
        id="chatcmpl-1",
        created=1,
        model="m",
        choices=[{"index": 0, "message": {"content": 'é\n"x"'}}],
        usage={"prompt_tokens": 3, "ttft_s": 0.0123},
        kv_transfer_params={"remote_block_ids": [4, 5], "remote_host": "h"},
        prompt_token_ids=ids,
    ).model_dump(mode="json")


RESPONSES = {
    "ids": _response(LONG),
    "none": _response([]),
    "no ids": _response(None),
    "past 64 bits": {**_response([1]), handoff.KEY: [2**70, 1]},
    "bools": {**_response([1]), handoff.KEY: [1, True]},
    "floats": {**_response([1]), handoff.KEY: [1.0, 1e16]},
    "a nested list": {**_response([1]), handoff.KEY: [[1], 2]},
    "a set": {**_response([1]), handoff.KEY: [1, {2}]},
    "raw JSON": {**_response([1]), handoff.KEY: [1, msgspec.Raw(b"7")]},
    "a tuple": {**_response([1]), handoff.KEY: (1, 2)},
    "the stand-in elsewhere": {"note": handoff._STAND_IN, **_response([1, 2])},
    "NaN elsewhere": {**_response([1, 2]), "ttft_s": math.nan},
    "a list": LONG,
}


def _rendered(content) -> bytes | str:
    try:
        return JSONResponse(content).body
    except (TypeError, ValueError) as e:
        return f"{type(e).__name__}: {e}"


def test_a_prefill_response_renders_as_before_and_its_ids_reach_no_standard_encoder(
    pristine, monkeypatch
):
    before = {name: _rendered(content) for name, content in RESPONSES.items()}
    seen = []
    render = JSONResponse.render
    monkeypatch.setattr(
        JSONResponse, "render", lambda self, c: seen.append(c) or render(self, c)
    )
    handoff.install("P")
    assert {name: _rendered(content) for name, content in RESPONSES.items()} == before
    assert before["a set"].startswith("TypeError")
    assert before["raw JSON"].startswith("TypeError")
    assert before["NaN elsewhere"].startswith("ValueError")
    seen.clear()
    _rendered(RESPONSES["ids"])
    assert seen and all(
        not isinstance(c, dict) or c[handoff.KEY] == handoff._STAND_IN for c in seen
    )


@pytest.mark.parametrize("role, side", [("kv_producer", "P"), ("kv_consumer", "D")])
def test_a_prefill_or_decode_frontends_bootstrap_installs_its_own_side(
    tmp_path, monkeypatch, pristine, role, side
):
    run = _run_file(tmp_path, **PD)
    authority = clock_transport.serve(
        ClockAuthority(compass_run.channel_table(json.loads(run.read_text()))),
        "tcp://127.0.0.1:0",
    )
    monkeypatch.setenv(compass_run.ENV, str(run))
    monkeypatch.setenv(compass_run.CLOCK_ENV, authority.endpoint)
    monkeypatch.setattr(clock, "_installed", None)
    monkeypatch.setattr(compass_run, "_authority", None)
    config = atom_config_double(
        kv_transfer_config={"kv_connector": "compass", "kv_role": role},
        runner_qualname=compass_run.ATOM_RUNNER,
    )
    try:
        with compass_run.frontend(config):
            pass
    finally:
        authority.close()
    installed = {
        "D": isinstance(protocol._PROMPT_TOKEN_IDS_ADAPTER, handoff.CheckedIds),
        "P": hasattr(JSONResponse.render, "__wrapped__"),
    }
    assert installed == {"P": side == "P", "D": side == "D"}
    assert hasattr(Request.json, "__wrapped__") == (side == "D")

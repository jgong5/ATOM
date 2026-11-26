# SPDX-License-Identifier: MIT
"""A simulated prefill frontend reads its request bodies with msgspec
(`atom.compass.run.handoff`). ATOM's request models behind a FastAPI app, as
prefill's server reads a request, take the same request object from every
body, defaults, ignored and kept extra fields and pydantic's coercions
included, or answer the same error, and a conversation reaches no
``json.loads``."""

import json
from types import SimpleNamespace

import starlette.requests
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_handoff_codec import pristine  # noqa: F401  the fixture

from atom.compass.run import handoff
from atom.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest

CHAT, TEXT = "/v1/chat/completions", "/v1/completions"
HEADERS = {"content-type": "application/json"}
TURN = (
    '{"role":"tool","tool_call_id":"c1","content":"def f():\\n    return {\\"é\\": 1}"}'
)


def _chat(*members: str, turn: str = TURN) -> bytes:
    """A chat request whose messages end in `turn`, with each of `members` a
    further top-level member's text."""
    head = '{"model":"m","messages":[{"role":"user","content":"hi"},' + turn + "]"
    return (head + "".join(f",{m}" for m in members) + "}").encode()


def _nested(depth: int) -> bytes:
    return _chat(turn='{"role":"user","content":' + "[" * depth + "]" * depth + "}")


BODIES = {
    "a conversation": _chat('"max_tokens":1', '"stream":false', '"seed":7'),
    "defaults": _chat(),
    "extra fields": _chat(
        '"vendor":{"x":1}', turn='{"role":"user","content":"a","x":[1]}'
    ),
    "coerced": _chat(
        '"max_tokens":"16"', '"temperature":1', '"stream":"true"', '"n":2.0'
    ),
    "refused types": _chat('"max_tokens":"x"', '"logprobs":1', '"stop":7'),
    "client ids": _chat('"kv_transfer_params":{"prompt_token_ids":[1,2]}'),
    "the same key twice": _chat('"seed":1', '"seed":2'),
    "past 64 bits": _chat('"seed":18446744073709551616'),
    "past the digit limit": _chat('"seed":' + "9" * 4301),
    "NaN": _chat('"temperature":NaN'),
    "past a float": _chat('"temperature":1e400'),
    "a lone surrogate": _chat().replace(b"hi", b"\\ud800"),
    "an encoded surrogate": _chat().replace(b"hi", b"\xed\xa0\x80"),
    "bad UTF-8": _chat().replace(b"hi", b"h\xff"),
    "a byte order mark": b"\xef\xbb\xbf" + _chat(),
    "UTF-16": _chat().decode().encode("utf-16"),
    "cut short": _chat()[:-7],
    "two values": _chat() + b",1",
    "a list": b"[" + _chat() + b"]",
    "whitespace": b" \n",
}


def _client(seen: list) -> TestClient:
    """Prefill's reading of a request, each request object it takes kept in `seen`."""
    app = FastAPI()

    @app.post(CHAT)
    async def chat(request: ChatCompletionRequest):
        seen.append(request)
        return {}

    @app.post(TEXT)
    async def text(request: CompletionRequest):
        seen.append(request)
        return {}

    return TestClient(app, raise_server_exceptions=False)


def _answers(bodies: dict) -> dict:
    seen: list = []
    client, answers = _client(seen), {}
    for name, body in bodies.items():
        seen.clear()
        path = TEXT if name == "a completion" else CHAT
        r = client.post(path, content=body, headers=HEADERS)
        taken = [(repr(q), repr(q.model_dump()), q.model_fields_set) for q in seen]
        answers[name] = (r.status_code, r.text, taken)
    return answers


def _deepest() -> int:
    """The deepest message content the server reads before ``json.loads``
    refuses its body as nested too deep."""
    client, lo, hi = _client([]), 1, 20_000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        refused = client.post(CHAT, content=_nested(mid), headers=HEADERS)
        lo, hi = (lo, mid - 1) if refused.status_code == 400 else (mid, hi)
    return lo


def test_a_prefill_frontend_takes_the_same_request_or_answers_the_same_error(
    pristine,  # noqa: F811
):
    deepest = _deepest()
    bodies = {
        **BODIES,
        "a completion": b'{"model":"m","prompt":[1,2],"max_tokens":"3"}',
        "the deepest": _nested(deepest),
        "one past the deepest": _nested(deepest + 1),
    }
    before = _answers(bodies)
    handoff.install("P")
    assert _answers(bodies) == before
    assert {n for n, (status, _, taken) in before.items() if taken} >= {
        "a conversation",
        "coerced",
        "client ids",
        "NaN",
        "a lone surrogate",
        "an encoded surrogate",
        "a byte order mark",
        "UTF-16",
    }
    refusals = {n: a[:2] for n, a in before.items() if a[0] in (400, 422)}
    assert refusals["cut short"][0] == refusals["refused types"][0] == 422
    assert refusals["bad UTF-8"][0] == refusals["one past the deepest"][0] == 400
    assert refusals.keys() >= {"past the digit limit", "two values", "a list"}


def test_a_conversation_reaches_no_json_loads(pristine, monkeypatch):  # noqa: F811
    loaded = []
    monkeypatch.setattr(
        starlette.requests,
        "json",
        SimpleNamespace(loads=lambda body: loaded.append(body) or json.loads(body)),
    )
    handoff.install("P")
    seen: list = []
    client = _client(seen)
    reply = client.post(CHAT, content=BODIES["a conversation"], headers=HEADERS)
    assert (
        reply.status_code == 200
        and loaded == []
        and seen[0].messages[1].content.endswith('{"é": 1}')
    )
    # What msgspec does not read, the standard library still decides.
    client.post(CHAT, content=BODIES["NaN"], headers=HEADERS)
    assert loaded == [BODIES["NaN"]]

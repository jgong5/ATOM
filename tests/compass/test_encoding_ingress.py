# SPDX-License-Identifier: MIT
"""The encoding ingress (`scripts/compass/encoding_ingress.py`) sends ATOM the
token ids ATOM's own chat endpoint would make of a request, and passes through
whatever it does not encode.

Encoding equivalence, on the checkpoint named by ``INGRESS_MODEL`` (default
``/models/dsv41``, DeepSeek-V4.1-Flash; skipped where it is absent):

- the checkpoint's own encoder goldens (``<model>/encoding/tests``) that an
  OpenAI chat request can state (system, user, assistant and tool turns, tools,
  thinking mode, effort): the ingress renders each to its golden text, and
  sends ``tokenizer.encode`` of that text;
- every chat body in ``INGRESS_BODIES`` (a directory ``--bodies`` filled
  during a run, AgentX requests as aiperf sent them), in order, so later turns
  hit the segment cache: the ids the ingress sends are the ids ATOM's engine
  makes of the server's render, through its segment tokenizer as the recipe
  runs it (``ATOM_FRONTEND_PREFIX_HASH=1``) and without it, and decode back to
  the rendered text.

Pass-through: a stamped cancel, a body that already carries
``prompt_token_ids`` and a body the server would refuse go on byte for byte.
"""

import json
import os
import sys
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TREE / "scripts" / "compass"))
MODEL = Path(os.environ.get("INGRESS_MODEL", "/models/dsv41"))
BODIES = os.environ.get("INGRESS_BODIES")
pytestmark = pytest.mark.skipif(
    not (MODEL / "encoding" / "tests").is_dir(),
    reason=f"no checkpoint with encoder goldens at {MODEL}",
)
#: The roles an OpenAI chat request can carry.
OPENAI_ROLES = {"system", "user", "assistant", "tool"}


@pytest.fixture(scope="module")
def encoder():
    import encoding_ingress

    return encoding_ingress.Encoder(str(MODEL), "dsml_v41", True)


def _golden_cases():
    """(case id, OpenAI request, golden text) of each golden an OpenAI request
    states: its messages less the tools the encoder's format hangs on the
    first one, which go in ``tools``; its thinking mode (the goldens' default
    ``chat``) and effort as template controls."""
    sys.path.insert(0, str(MODEL / "encoding"))
    import encoding as enc  # the checkpoint's own encoder

    out = []
    for path in sorted((MODEL / "encoding" / "tests").glob("test_input_*.json")):
        case_id = int(path.stem.rsplit("_", 1)[1])
        case = enc.load_cases(str(path))[0]
        messages = [dict(m) for m in case["messages"]]
        tools = messages[0].pop("tools", None) if messages else None
        plain = all(
            m.get("role") in OPENAI_ROLES
            and not ({"task", "mask", "tools"} & m.keys())
            and not isinstance(m.get("content"), list)
            for m in messages
        )
        if not plain:
            continue
        kwargs = {"thinking_mode": case.get("thinking_mode") or "chat"}
        if case.get("reasoning_effort") is not None:
            kwargs["reasoning_effort"] = case["reasoning_effort"]
        request = {"model": str(MODEL), "messages": messages, "stream": True}
        request["chat_template_kwargs"] = kwargs
        if tools:
            request["tools"] = tools
        golden = (path.parent / f"test_output_{case_id}.txt").read_text()
        out.append((case_id, request, golden))
    return out


def test_the_goldens_an_openai_request_states_render_to_their_text(encoder):
    cases = _golden_cases()
    assert cases, "no golden case is an OpenAI chat request"
    for case_id, request, golden in cases:
        assert encoder.prompt(request) == golden, f"golden case {case_id}"
        assert encoder.ids(golden) == encoder.srv.tokenizer.encode(golden)


def test_a_chat_body_goes_on_with_the_ids_and_everything_else_as_sent(encoder):
    import msgspec
    from encoding_ingress import encode_body

    _, request, golden = _golden_cases()[0]
    raw = json.dumps(request).encode()
    stamps = {}
    out, kind, tokens = encode_body(encoder, raw, stamps)
    sent = msgspec.json.decode(out)
    ids = sent.pop("prompt_token_ids")
    assert kind == "encoded" and tokens == len(ids)
    assert ids == encoder.srv.tokenizer.encode(golden)
    assert sent == request
    assert (
        stamps["start"]
        <= stamps["rendered"]
        <= stamps["encoded"]
        <= stamps["serialized"]
    )
    # A stamped cancel, ids already given, and a request the server refuses.
    for body in (
        b'{"compass_cancel": 7}',
        json.dumps({**request, "prompt_token_ids": [1, 2]}).encode(),
        json.dumps({**request, "messages": "not a list"}).encode(),
        b"not json",
    ):
        out, kind, tokens = encode_body(encoder, body, {})
        assert out == body and kind in ("passthrough", "fallback") and tokens == 0


@pytest.mark.skipif(not BODIES, reason="INGRESS_BODIES names no captured bodies")
def test_captured_agentx_bodies_encode_as_atom_s_engine_does(encoder, monkeypatch):
    from atom.model_engine.segment_tokenizer import SegmentTokenizer
    from atom.utils import envs

    tokenizer = encoder.srv.tokenizer
    segments = SegmentTokenizer.maybe_create(
        tokenizer, envs.ATOM_TOKENIZE_CACHE_MB << 20
    )
    assert segments is not None, "V4.1's tokenizer splits exactly at added tokens"
    paths = sorted(Path(BODIES).glob("*.json"))
    assert paths, f"{BODIES} holds no body"
    for path in paths:
        body = json.loads(path.read_bytes())
        text = encoder.prompt(body)
        ids = encoder.ids(text)
        assert ids == list(segments.encode_with_prefix(text, 128).token_ids), path.name
        assert ids == list(segments.encode(text)), path.name
        assert tokenizer.decode(ids, skip_special_tokens=False) == text, path.name


BACKEND = """
import json, sys, time, asyncio
import uvicorn
from starlette.applications import Starlette
from starlette.responses import StreamingResponse, JSONResponse
from starlette.routing import Route
out = sys.argv[2]

async def chat(request):
    raw = await request.body()
    body = json.loads(raw)
    seen = {"headers": dict(request.headers), "ids": len(body.get("prompt_token_ids") or []),
            "keys": sorted(body), "bytes": len(raw)}
    open(out + ".request", "w").write(json.dumps(seen))
    async def events():
        try:
            for i in range(300):
                yield f"data: {i}\\n\\n".encode()
                await asyncio.sleep(0.1)
        except BaseException:
            open(out + ".hung_up", "w").write(str(time.monotonic()))
            raise
    return StreamingResponse(events(), media_type="text/event-stream")

async def health(request):
    return JSONResponse({})

app = Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"]),
                        Route("/health", health)])
uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
"""


def _port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _until(predicate, seconds: float, what: str) -> None:
    import time

    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, what
        time.sleep(0.05)


def test_a_streamed_chat_goes_through_with_its_ids_and_headers_and_a_hang_up_reaches_the_backend(
    tmp_path, encoder
):
    """The ingress process forwards a chat request's ids and its headers (a
    simulated run's ``tracestate`` stamp, the request and session ids), relays
    the stream as it comes, and a client that hangs up mid-stream closes the
    backend's stream: the router sees the cancel."""
    import subprocess
    import time

    import httpx

    _, request, golden = _golden_cases()[0]
    backend, ingress = _port(), _port()
    out = str(tmp_path / "backend")
    command = [
        sys.executable,
        str(TREE / "scripts" / "compass" / "encoding_ingress.py"),
    ]
    command += ["--model", str(MODEL), "--backend", f"http://127.0.0.1:{backend}"]
    command += ["--host", "127.0.0.1", "--port", str(ingress), "--processes", "1"]
    command += ["--workers", "2", "--stamps", str(tmp_path / "stamps")]
    procs = [
        subprocess.Popen([sys.executable, "-c", BACKEND, str(backend), out]),
        subprocess.Popen(command, env={**os.environ, "PYTHONPATH": str(TREE)}),
    ]
    try:

        def up() -> bool:
            try:
                return (
                    httpx.get(f"http://127.0.0.1:{ingress}/health").status_code == 200
                )
            except httpx.HTTPError:
                return False

        _until(up, 120, "the ingress never answered /health")
        headers = {
            "tracestate": "compass=a:1.5;s:7",
            "X-Request-Id": "req-1",
            "X-Session-ID": "sess-1",
        }
        url = f"http://127.0.0.1:{ingress}/v1/chat/completions"
        with httpx.stream(
            "POST", url, json=request, headers=headers, timeout=60
        ) as got:
            assert got.status_code == 200
            chunks = got.iter_raw()
            assert next(chunks).startswith(b"data: 0")
        hung_up = time.monotonic()
        _until(
            lambda: Path(out + ".hung_up").exists(), 10, "the backend's stream ran on"
        )
        assert float(Path(out + ".hung_up").read_text()) - hung_up < 5
        seen = json.loads(Path(out + ".request").read_text())
        assert seen["ids"] == len(encoder.srv.tokenizer.encode(golden))
        assert seen["keys"] == sorted([*request, "prompt_token_ids"])
        for name, value in headers.items():
            assert seen["headers"][name.lower()] == value
        _until(lambda: list(tmp_path.glob("stamps.*")), 10, "no stamps")
        line = json.loads(next(tmp_path.glob("stamps.*")).read_text().splitlines()[-1])
        assert line["kind"] == "encoded" and line["cancelled"] is True
        assert line["recv"] <= line["sent"] <= line["head"] <= line["end"]
        assert line["sent_bytes"] == seen["bytes"] > line["bytes"]
    finally:
        for p in procs:
            p.terminate()
            p.wait(timeout=30)

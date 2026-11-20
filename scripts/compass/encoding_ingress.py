# SPDX-License-Identifier: MIT
"""An encoding ingress in front of atomesh: the C1228 recipe's AgentX front end,
rebuilt (the recipe's frozen script is not public).

    python3 scripts/compass/encoding_ingress.py --model MODEL --backend ROUTER_URL \
        --host HOST --port PORT --processes 4 --workers 16 --queued 1280 \
        --max-connections 2048 --max-keepalive-connections 512 \
        --timeout-keep-alive 600 --segment-cache-mib 0 [--stamps FILE] [--bodies DIR]

The argv is the recipe's frozen template. Each of `--processes` processes binds
PORT with SO_REUSEPORT and serves HTTP (uvicorn, `--timeout-keep-alive` its
keep-alive); a chat request (``POST /v1/chat/completions`` with ``messages``)
is rendered and tokenized on one of the process's `--workers` threads, exactly
as ATOM's server does it (`encode`: the server's own setup, request checks,
template controls and chat encoder, then `tokenizer.encode`), and forwarded to
`--backend` with the token ids as ``prompt_token_ids``, which ATOM's chat
endpoint takes in place of rendering and tokenizing again. The response is
streamed back as the router sends it, byte for byte, SSE comments included.

Everything else goes through unchanged: other paths and methods, a body that
is no chat request (a simulated run's stamped cancel, ``{"compass_cancel": N}``),
one that already carries ``prompt_token_ids``, and every request header but the
hop-by-hop ones (``tracestate``, which carries a simulated request's stamp,
``X-Request-Id`` and the session headers among them). A request whose render
the server would refuse is forwarded as sent, so the server answers it as it
would without an ingress. A client that hangs up closes the upstream request,
which is how atomesh, and ATOM behind it, see a cancel.

Capacity, as the recipe states it: `--workers` encode threads a process;
`--queued` encode jobs waiting for a thread, over all processes, beyond which a
request is answered 503; `--max-connections` and `--max-keepalive-connections`
each process's upstream connection pool (httpx's ``Limits``, whose names they
are), and `--max-connections` also each process's inbound bound (uvicorn's
``limit_concurrency``, 503 past it); `--timeout-keep-alive` both the inbound
keep-alive and the pooled upstream connections' expiry. `--segment-cache-mib`
must be 0: the recipe ran with the segment cache off, and this ingress has
none, so each prompt is tokenized whole.

`--stamps FILE` appends one JSON line a request to ``FILE.<pid>``, monotonic ns
(one clock for every process on a host): ``recv`` the handler's start,
``body`` its body read, ``start`` its encode job's start, ``parsed``,
``rendered``, ``encoded`` and ``serialized`` the job's stages, ``sent`` the
upstream request handed to the pool, ``head`` the upstream response's head,
``first`` its first body chunk, ``end`` its last relayed; with ``tokens``,
``bytes`` (body in), ``sent_bytes`` (body out), ``status`` and ``kind``
(``encoded``, ``passthrough`` or ``fallback``), ``cancelled`` when the client
left first. The ingress hop is ``sent - recv``. `compass fit ingress` fits a
simulated run's ``ingress`` term on these stamps (`atom.compass.fit.ingress`).
`--bodies DIR` keeps the first chat bodies it takes (100 a process) in DIR,
which the equivalence test replays.
"""

import argparse
import asyncio
import json
import multiprocessing
import os
import signal
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

#: Hop-by-hop headers, and those the forwarded body changes.
DROPPED = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "te",
    "trailer",
    "upgrade",
    "proxy-connection",
    "accept-encoding",
}
CHAT_PATH = "/v1/chat/completions"
#: Chat bodies `--bodies` keeps, a process.
KEEP_BODIES = 100


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--backend", required=True, help="the router's URL")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--processes", type=int, default=4)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--queued", type=int, default=1280)
    p.add_argument("--max-connections", type=int, default=2048)
    p.add_argument("--max-keepalive-connections", type=int, default=512)
    p.add_argument("--timeout-keep-alive", type=int, default=600)
    p.add_argument("--segment-cache-mib", type=int, default=0)
    p.add_argument("--tool-call-parser", default="dsml_v41")
    p.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--stamps", help="append each request's stamps to STAMPS.<pid>")
    p.add_argument("--bodies", help="keep the first chat bodies in this directory")
    return p


class Encoder:
    """ATOM's chat render and tokenize, set up as the server sets up a front end
    (`api_server._setup_frontend`, `_setup_reasoning_and_tools`, which its
    front-end workers share), and run as `api_server._chat_completions` runs it
    for a chat request without ``prompt_token_ids``."""

    def __init__(self, model: str, tool_call_parser: str, trust_remote_code: bool):
        from atom.entrypoints.openai import api_server as srv

        # The flags the two setups read, at the server's defaults but these.
        args = argparse.Namespace(
            model=model,
            trust_remote_code=trust_remote_code,
            tool_call_parser=tool_call_parser,
            chat_template=None,
            default_chat_template_kwargs=None,
            served_model_name=None,
            request_log=None,
        )
        srv._setup_frontend(args)
        srv._setup_reasoning_and_tools(args)
        self.srv = srv

    def prompt(self, body: dict) -> str:
        """The text ATOM's chat endpoint renders for `body`; raises as it
        refuses (ValueError, pydantic's ValidationError)."""
        srv = self.srv
        request = srv.ChatCompletionRequest.model_validate(body)
        request.tools = srv.normalize_chat_tools(request.tools)
        srv.validate_chat_request(request)
        messages = srv.check_chat_images(request.get_messages())
        if srv._has_multimodal_content(messages):
            raise ValueError("an image request is rendered by the server itself")
        merged = dict(srv.default_chat_template_kwargs)
        if request.chat_template_kwargs:
            merged.update(request.chat_template_kwargs)
        if request.response_format is not None:
            merged["response_format"] = request.response_format
        if isinstance(request.tool_choice, str):
            merged["tool_choice"] = request.tool_choice
        rules = srv._model_reasoning_rules()
        merged = srv.apply_reasoning_controls(
            merged,
            srv.reasoning_toggle,
            srv.resolve_thinking(request, rules=rules),
            rules=rules,
        )
        return srv.apply_chat_template(
            srv.tokenizer,
            srv.custom_message_encoder,
            messages,
            tools=request.tools,
            preserve_tool_call_arguments=srv.preserve_tool_call_arguments,
            **merged,
        )

    def ids(self, prompt: str) -> list[int]:
        """`prompt`'s token ids as ATOM's engine takes them without a segment
        tokenizer (`LLMEngine.encode_prompt`); the segment tokenizer's are the
        same ids (tests/compass/test_encoding_ingress.py)."""
        return self.srv.tokenizer.encode(prompt)


def encode_body(encoder: Encoder, raw: bytes, stamps: dict) -> tuple[bytes, str, int]:
    """`raw`, the client's body, as it goes on: with ``prompt_token_ids`` for a
    chat request ATOM renders (``encoded``), else as it came (``passthrough``,
    or ``fallback`` where the render refused); and the token count."""
    import msgspec

    stamps["start"] = time.monotonic_ns()
    try:
        body = msgspec.json.decode(raw)
    except msgspec.DecodeError:
        return raw, "passthrough", 0
    stamps["parsed"] = time.monotonic_ns()
    if (
        not isinstance(body, dict)
        or "messages" not in body
        or "prompt_token_ids" in body
    ):
        return raw, "passthrough", 0
    try:
        text = encoder.prompt(body)
    except Exception:  # noqa: BLE001 - the server answers what it refuses
        return raw, "fallback", 0
    stamps["rendered"] = time.monotonic_ns()
    ids = encoder.ids(text)
    stamps["encoded"] = time.monotonic_ns()
    body["prompt_token_ids"] = ids
    out = msgspec.json.encode(body)
    stamps["serialized"] = time.monotonic_ns()
    return out, "encoded", len(ids)


def serve_process(args, index: int, ready) -> None:
    """One ingress process: its encoder, its upstream pool and its server."""
    import httpx
    import uvicorn
    from starlette.applications import Starlette
    from starlette.requests import ClientDisconnect, Request
    from starlette.responses import JSONResponse, Response, StreamingResponse
    from starlette.routing import Route

    encoder = Encoder(args.model, args.tool_call_parser, args.trust_remote_code)
    pool = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="encode")
    waiting_cap = max(1, args.queued // args.processes)
    state = {"jobs": 0}
    # Open for the process's life; each line is flushed as it is written.
    stamps_file = (
        open(f"{args.stamps}.{os.getpid()}", "a", buffering=1)  # noqa: SIM115
        if args.stamps
        else None
    )
    stamps_lock = threading.Lock()
    kept = {"n": 0}
    bodies = Path(args.bodies) if args.bodies else None
    if bodies:
        bodies.mkdir(parents=True, exist_ok=True)
    backend = args.backend.rstrip("/")
    client = httpx.AsyncClient(
        base_url=backend,
        timeout=httpx.Timeout(None),
        limits=httpx.Limits(
            max_connections=args.max_connections,
            max_keepalive_connections=args.max_keepalive_connections,
            keepalive_expiry=args.timeout_keep_alive,
        ),
    )

    def write(line: dict) -> None:
        if stamps_file is not None:
            with stamps_lock:
                stamps_file.write(json.dumps(line) + "\n")

    async def forward(request: Request, body: bytes, line: dict) -> Response:
        headers = [
            (k, v) for k, v in request.headers.items() if k.lower() not in DROPPED
        ]
        upstream = client.build_request(
            request.method,
            request.url.path + (f"?{request.url.query}" if request.url.query else ""),
            headers=headers,
            content=body,
        )
        line["sent_bytes"] = len(body)
        line["sent"] = time.monotonic_ns()
        try:
            got = await client.send(upstream, stream=True)
        except httpx.HTTPError as e:
            line |= {"end": time.monotonic_ns(), "status": 502, "error": repr(e)}
            write(line)
            return JSONResponse({"error": f"ingress: backend: {e!r}"}, status_code=502)
        line["head"] = time.monotonic_ns()
        line["status"] = got.status_code

        async def relay():
            try:
                async for chunk in got.aiter_raw():
                    line.setdefault("first", time.monotonic_ns())
                    yield chunk
            except (asyncio.CancelledError, GeneratorExit):
                line["cancelled"] = True
                raise
            finally:
                await got.aclose()
                line["end"] = time.monotonic_ns()
                write(line)

        kept_headers = {
            k: v
            for k, v in got.headers.items()
            if k.lower() not in DROPPED | {"content-encoding"}
        }
        return StreamingResponse(
            relay(), status_code=got.status_code, headers=kept_headers
        )

    async def chat(request: Request) -> Response:
        line = {"recv": time.monotonic_ns(), "proc": index}
        try:
            raw = await request.body()
        except ClientDisconnect:
            return Response(status_code=499)
        line |= {"body": time.monotonic_ns(), "bytes": len(raw)}
        if state["jobs"] >= args.workers + waiting_cap:
            line |= {"status": 503, "end": time.monotonic_ns(), "kind": "refused"}
            write(line)
            return JSONResponse(
                {"error": "ingress: encode queue full"}, status_code=503
            )
        state["jobs"] += 1
        try:
            loop = asyncio.get_running_loop()
            out, kind, tokens = await loop.run_in_executor(
                pool, encode_body, encoder, raw, line
            )
        finally:
            state["jobs"] -= 1
        line |= {"kind": kind, "tokens": tokens}
        if bodies is not None and kind == "encoded" and kept["n"] < KEEP_BODIES:
            kept["n"] += 1
            (bodies / f"{index}-{kept['n']:03d}.json").write_bytes(raw)
        return await forward(request, out, line)

    async def other(request: Request) -> Response:
        line = {"recv": time.monotonic_ns(), "proc": index, "kind": "passthrough"}
        raw = await request.body()
        line |= {
            "body": time.monotonic_ns(),
            "bytes": len(raw),
            "path": request.url.path,
        }
        return await forward(request, raw, line)

    app = Starlette(
        routes=[
            Route(CHAT_PATH, chat, methods=["POST"]),
            Route("/{path:path}", other, methods=["GET", "POST", "PUT", "DELETE"]),
        ]
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((args.host, args.port))
    sock.listen(4096)
    config = uvicorn.Config(
        app,
        log_level="warning",
        access_log=False,
        timeout_keep_alive=args.timeout_keep_alive,
        limit_concurrency=args.max_connections,
        backlog=4096,
    )
    server = uvicorn.Server(config)
    ready.release()
    server.run(sockets=[sock])


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.segment_cache_mib != 0:
        print(
            "ingress: --segment-cache-mib must be 0: this ingress has no segment cache",
            file=sys.stderr,
        )
        return 2
    try:
        import resource

        hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except (ValueError, OSError):
        pass
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Semaphore(0)
    procs = [
        ctx.Process(target=serve_process, args=(args, i, ready), daemon=True)
        for i in range(args.processes)
    ]
    for p in procs:
        p.start()
    for _ in procs:
        while not ready.acquire(timeout=1):
            if any(not p.is_alive() for p in procs):
                print("ingress: a process exited before it served", file=sys.stderr)
                for q in procs:
                    q.terminate()
                return 1
    print(
        f"ingress: serving {args.host}:{args.port} with {args.processes} processes",
        flush=True,
    )

    def stop(*_):
        for p in procs:
            p.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for p in procs:
        p.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())

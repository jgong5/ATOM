# SPDX-License-Identifier: MIT
"""Fit the API server event loop's charge per HTTP request from one real burst.

    python3 scripts/compass/http_intake.py serve STAMPS API_SERVER_ARGS...
    python3 scripts/compass/http_intake.py fit STAMPS [N]
    python3 scripts/compass/http_intake.py floor STAMPS ARRIVALS
    python3 scripts/compass/http_intake.py idle STAMPS [PER_BYTE_S]
    python3 scripts/compass/http_intake.py hops STAMPS HANDOFF ARRIVALS
    python3 scripts/compass/http_intake.py pd STAMPS

`serve` runs `atom.entrypoints.openai.api_server` as ``python3 -m`` does and
appends ``<end ns> <thread id> body|tmpl|app|prep|send|out|head <start ns> <size>`` to STAMPS for
each request body first read (`Request.body`), chat template rendered
(`apply_chat_template`) and HTTP request handed to the ASGI app
(`Starlette.__call__`, size 0); `cctraces_sim.sh` runs it when HTTP_INTAKE is set.
`prep` is a tokenizer pool job (`InputOutputProcessor.preprocess_fanout`) and `send`
a request handed to the engine (`CoreManager.add_request`), `out` an engine
output its output thread took in (`CoreManager._mark_seq_prefill_complete` per
streamed output, `_deliver_terminal_output` per finished sequence) and `head` an
SSE response's head sent to the client, all size 0. Every process the server
spawns stamps into STAMPS too: `kvsend`, a Mooncake KV write request sent
(`MooncakeConnector._send_on_socket`), and `kvrecv`, one the prefill's listener
received, both size 0.

`floor` reads the transport from client send to ASGI app call off a server
`serve` ran and the parity workload's real-run ARRIVALS
(`atom.compass.parity.traffic`), sent one request in flight: a gap longer than
any request. Prints the median, p10 and p90 in seconds, N and both sha256s as JSON.
`idle` reads the same run's intake on a loop with no other: the event loop's time
from each ASGI app call to its pool job's start and from the job's end to its
engine send, less PER_BYTE_S (default 0) per body byte. Prints the
median as `idle_s`, p10, p90, PER_BYTE_S, N and the sha256 as JSON.
`hops` reads three hops off a parity workload run, the server run with
``ATOM_COMPASS_PARITY_RECORD`` set so its engine writes HANDOFF
(``dp0.handoff.jsonl``): ``request``, each engine send's return to the engine
input thread's receipt, from request 1 on, as request 0 (sent alone, or the
long one) finds the engine idle; ``output``, each engine output's put on the engine's
output queue to the server's output thread taking it in; ``stream``, each SSE
head sent to the client reading it (ARRIVALS' ``admitted_ns``). Prints each hop's median, p10,
p90 and N, and the three inputs' sha256, as JSON.
`pd` reads a prefill and a decode server's shared STAMPS (`parity_pd.sh` with
HTTP_INTAKE): ``kv_write_req``, each write request's send call on decode to prefill's
receipt, its median, p10, p90 and N, and the sha256, as JSON.

`fit` reads a burst of N first-turn requests (default 256), which the loop takes
in one at a time: each gap between template starts is the next request's
``fixed_s + body bytes x per_byte_s``. Some gaps also hold tens of milliseconds
of other loop work, so `per_byte_s` is the Theil-Sen slope and `fixed_s` keeps
the burst's span; least squares on the starts swings the slope's sign between
bursts. Prints the terms, a gap's rms residual, N, bytes and sha256 as JSON.
"""

import contextlib
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import runpy
import sys
import threading
import time
from multiprocessing import spawn
from pathlib import Path

import numpy as np


def serve(stamps: str, args: list[str]) -> None:
    install(stamps)
    sys.argv[1:], sys.path[0] = args, os.getcwd()
    main = "atom.entrypoints.openai.api_server"
    runpy.run_module(main, run_name="__main__", alter_sys=True)


def install(stamps: str) -> None:
    """Stamp into `stamps` in this process and every process it spawns."""
    out = open(stamps, "a", buffering=1)  # noqa: SIM115

    def stamp(kind: str, start: int, size: int) -> None:
        now, thread = time.monotonic_ns(), threading.get_ident()
        out.write(f"{now} {thread} {kind} {start} {size}\n")

    def requests(module) -> None:
        read = module.Request.body

        async def body(self):
            cached, start = hasattr(self, "_body"), time.monotonic_ns()
            got = await read(self)
            if not cached:
                stamp("body", start, len(got))
            return got

        module.Request.body = body

    def encoders(module) -> None:
        render = module.apply_chat_template

        def apply_chat_template(*a, **k):
            start = time.monotonic_ns()
            got = render(*a, **k)
            stamp("tmpl", start, len(got))
            return got

        module.apply_chat_template = apply_chat_template

    def apps(module) -> None:
        call = module.Starlette.__call__

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                stamp("app", time.monotonic_ns(), 0)
                send = heads(send)
            await call(self, scope, receive, send)

        module.Starlette.__call__ = __call__

    def heads(send):
        async def sent(message):
            start = time.monotonic_ns()
            await send(message)
            if message["type"] == "http.response.start":
                kind = dict(message.get("headers", [])).get(b"content-type", b"")
                if kind.startswith(b"text/event-stream"):
                    stamp("head", start, 0)

        return sent

    def stamped(owner: str, **kinds: str):
        def timed(call, kind: str):
            def timed(*a, **k):
                start = time.monotonic_ns()
                got = call(*a, **k)
                stamp(kind, start, 0)
                return got

            return timed

        def patch(module) -> None:
            cls = getattr(module, owner)
            for name, kind in kinds.items():
                setattr(cls, name, timed(getattr(cls, name), kind))

        return patch

    def mooncake(module) -> None:
        cls, kind = module.MooncakeConnector, module.MSG_WRITE_REQUEST
        send, open_socket = cls._send_on_socket, module.zmq_socket_ctx

        def _send_on_socket(self, addr, parts, *a, **k):
            start = time.monotonic_ns()
            send(self, addr, parts, *a, **k)
            if parts[0] == kind:
                stamp("kvsend", start, 0)

        class Received:
            def __init__(self, sock):
                self.sock = sock

            def __getattr__(self, name):
                return getattr(self.sock, name)

            def recv_multipart(self, *a, **k):
                parts = self.sock.recv_multipart(*a, **k)
                if parts[1:2] == [kind]:
                    stamp("kvrecv", time.monotonic_ns(), 0)
                return parts

        @contextlib.contextmanager
        def zmq_socket_ctx(*a, **k):
            with open_socket(*a, **k) as sock:
                yield Received(sock)

        cls._send_on_socket, module.zmq_socket_ctx = _send_on_socket, zmq_socket_ctx

    # Each module is patched once it has run, before anything imports from it.
    patches = {
        "starlette.requests": requests,
        "starlette.applications": apps,
        "atom.entrypoints.openai.chat_encoders": encoders,
        "atom.model_engine.llm_engine": stamped(
            "InputOutputProcessor", preprocess_fanout="prep"
        ),
        "atom.model_engine.engine_core_mgr": stamped(
            "CoreManager",
            add_request="send",
            _mark_seq_prefill_complete="out",
            _deliver_terminal_output="out",
        ),
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector": mooncake,
    }

    class Patcher(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in patches:
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            run = spec.loader.exec_module
            spec.loader.exec_module = lambda m: (run(m), patches[name](m))
            return spec

    sys.meta_path.insert(0, Patcher())
    # A spawned process installs this before it runs anything else.
    line, here = spawn.get_command_line, os.path.dirname(os.path.abspath(__file__))
    stamps = os.path.abspath(stamps)

    def get_command_line(**kwds) -> list[str]:
        argv = line(**kwds)
        i = argv.index("-c") + 1
        argv[i] = (
            f"import sys; sys.path.insert(0, {here!r}); import http_intake; "
            f"http_intake.install({stamps!r}); {argv[i]}"
        )
        return argv

    spawn.get_command_line = get_command_line


def fit(stamps: str, n: int = 256) -> dict:
    data = Path(stamps).read_bytes()
    rows = [line.split() for line in data.decode().splitlines()]
    reads = sorted((int(r[0]), int(r[3]), int(r[4])) for r in rows if r[2] == "body")
    t0 = min(start for _, start, _ in reads)
    sizes = [size for *_, size in reads[:n]]
    starts = sorted(int(r[3]) for r in rows if r[2] == "tmpl" and int(r[3]) >= t0)
    starts = starts[:n]
    if min(len(sizes), len(starts)) < n:
        raise SystemExit(
            f"{stamps}: {len(sizes)} bodies and {len(starts)} templates, not {n}"
        )
    gaps, nbytes = np.diff(starts) / 1e9, np.array(sizes[1:])
    i, j = np.triu_indices(len(gaps), 1)
    pair = nbytes[i] != nbytes[j]
    per_byte = np.median((gaps[j] - gaps[i])[pair] / (nbytes[j] - nbytes[i])[pair])
    fixed = gaps.mean() - per_byte * nbytes.mean()
    return {
        "fixed_s": float(fixed),
        "per_byte_s": float(per_byte),
        "rms_s": float(np.sqrt(np.mean((fixed + per_byte * nbytes - gaps) ** 2))),
        "requests": n,
        "bytes": sum(sizes),
        "record": f"sha256:{hashlib.sha256(data).hexdigest()}",
    }


def floor(stamps: str, arrivals: str) -> dict:
    data, sent_data = Path(stamps).read_bytes(), Path(arrivals).read_bytes()
    rows = [line.split() for line in data.decode().splitlines()]
    calls = sorted(int(r[3]) for r in rows if r[2] == "app")
    sent = sorted(json.loads(sent_data)["sent_ns"])
    # One request in flight: each call falls between its send and the next.
    if len(calls) != len(sent) or not all(
        s < c < nxt for s, c, nxt in zip(sent, calls, sent[1:] + [float("inf")])
    ):
        raise SystemExit(
            f"{stamps}: {len(calls)} app calls, not one after each of {len(sent)} sends"
        )
    p10, median, p90 = np.percentile((np.array(calls) - sent) / 1e9, [10, 50, 90])
    return {
        "median_s": float(median),
        "p10_s": float(p10),
        "p90_s": float(p90),
        "requests": len(sent),
        "record": f"sha256:{hashlib.sha256(data).hexdigest()}",
        "arrivals": f"sha256:{hashlib.sha256(sent_data).hexdigest()}",
    }


def idle(stamps: str, per_byte: float = 0.0) -> dict:
    data = Path(stamps).read_bytes()
    rows = [line.split() for line in data.decode().splitlines()]
    calls = sorted(int(r[3]) for r in rows if r[2] == "app")
    jobs = sorted((int(r[3]), int(r[0])) for r in rows if r[2] == "prep")
    sends = sorted(int(r[0]) for r in rows if r[2] == "send")
    # One request at a time: its pool job and engine send come before the next call.
    after = calls[1:] + [float("inf")]
    if not len(calls) == len(jobs) == len(sends) or not all(
        c < j < e < s < nxt for c, (j, e), s, nxt in zip(calls, jobs, sends, after)
    ):
        raise SystemExit(
            f"{stamps}: {len(jobs)} pool jobs and {len(sends)} engine sends, "
            f"not one each after each of {len(calls)} app calls"
        )
    nbytes = [
        sum(int(r[4]) for r in rows if r[2] == "body" and c < int(r[3]) < j)
        for c, (j, _) in zip(calls, jobs)
    ]
    loop = [j - c + s - e for c, (j, e), s in zip(calls, jobs, sends)]
    took = np.array(loop) / 1e9 - per_byte * np.array(nbytes)
    p10, median, p90 = np.percentile(took, [10, 50, 90])
    return {
        "idle_s": float(median),
        "p10_s": float(p10),
        "p90_s": float(p90),
        "per_byte_s": per_byte,
        "requests": len(jobs),
        "record": f"sha256:{hashlib.sha256(data).hexdigest()}",
    }


def _hop(sent, got, name: str, alone: bool = True) -> dict:
    """Message k's ``got[k] - sent[k]``, in order; with `alone`, no message
    is sent before the one ahead of it arrives."""
    sent, got = sorted(sent), sorted(got)
    after = sent[1:] + [float("inf")] if alone else [float("inf")] * len(sent)
    if len(got) != len(sent) or not all(
        s < g < nxt for s, g, nxt in zip(sent, got, after)
    ):
        raise SystemExit(
            f"{name}: {len(got)} arrivals, not one after each of {len(sent)} sends"
        )
    p10, median, p90 = np.percentile((np.array(got) - sent) / 1e9, [10, 50, 90])
    return {"median_s": median, "p10_s": p10, "p90_s": p90, "messages": len(sent)}


def hops(stamps: str, handoff: str, arrivals: str) -> dict:
    data, handoff_data = Path(stamps).read_bytes(), Path(handoff).read_bytes()
    sent_data = Path(arrivals).read_bytes()
    rows = [line.split() for line in data.decode().splitlines()]
    sends = [int(r[0]) for r in rows if r[2] == "send"]
    outs = [int(r[3]) for r in rows if r[2] == "out"]
    heads = [int(r[0]) for r in rows if r[2] == "head"]
    lines = [json.loads(line) for line in handoff_data.decode().splitlines()]
    recvs = [h["t_recv_ns"] for h in lines if "t_recv_ns" in h]
    # Outputs put from the first request on, each timed from its put: the
    # engine's send stamp is its output thread's next get, which can trail the
    # server's receipt. The server stamps each of a message's rows, and the
    # message arrived with its first.
    first = min(sends, default=0)
    put = sorted((h["t_out_put_ns"], h["rows"]) for h in lines if "rows" in h)
    put = [(t, n) for t, n in put if t > first]
    outs = sorted(t for t in outs if t > first)
    starts = np.cumsum([0] + [n for _, n in put])
    if starts[-1] != len(outs):
        raise SystemExit(
            f"output: {len(outs)} rows taken in, not the {starts[-1]} sent"
        )
    got = [outs[i] for i in starts[:-1]]
    admitted = json.loads(sent_data)["admitted_ns"]
    return {
        "request": _hop(sorted(sends)[1:], sorted(recvs)[1:], "request"),
        "output": _hop([t for t, _ in put], got, "output", alone=False),
        "stream": _hop(heads, admitted, "stream"),
        "record": f"sha256:{hashlib.sha256(data).hexdigest()}",
        "handoff": f"sha256:{hashlib.sha256(handoff_data).hexdigest()}",
        "arrivals": f"sha256:{hashlib.sha256(sent_data).hexdigest()}",
    }


def pd(stamps: str) -> dict:
    data = Path(stamps).read_bytes()
    rows = [line.split() for line in data.decode().splitlines()]
    # One decode rank writes to one prefill rank on one connection, which
    # delivers in order. Timed from the send call's start: its thread can
    # take the GIL back after prefill took the request in.
    sent = [int(r[3]) for r in rows if r[2] == "kvsend"]
    got = [int(r[0]) for r in rows if r[2] == "kvrecv"]
    return {
        "kv_write_req": _hop(sent, got, "kv_write_req", alone=False),
        "record": f"sha256:{hashlib.sha256(data).hexdigest()}",
    }


if __name__ == "__main__":
    verb, stamps, *rest = sys.argv[1:]
    if verb == "serve":
        serve(stamps, rest)
    elif verb == "floor":
        print(json.dumps(floor(stamps, *rest)))
    elif verb == "idle":
        print(json.dumps(idle(stamps, *map(float, rest))))
    elif verb == "hops":
        print(json.dumps(hops(stamps, *rest)))
    elif verb == "pd":
        print(json.dumps(pd(stamps)))
    else:
        print(json.dumps(fit(stamps, *map(int, rest))))

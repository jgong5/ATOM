# SPDX-License-Identifier: MIT
"""Fit the API server event loop's charge per HTTP request from one real burst.

    python3 scripts/compass/http_intake.py serve STAMPS API_SERVER_ARGS...
    python3 scripts/compass/http_intake.py fit STAMPS [N]
    python3 scripts/compass/http_intake.py floor STAMPS ARRIVALS

`serve` runs `atom.entrypoints.openai.api_server` as ``python3 -m`` does and
appends ``<end ns> <thread id> body|tmpl|app <start ns> <size>`` to STAMPS for
each request body first read (`Request.body`), chat template rendered
(`apply_chat_template`) and HTTP request handed to the ASGI app
(`Starlette.__call__`, size 0); `cctraces_sim.sh` runs it when HTTP_INTAKE is set.

`floor` reads the transport from client send to ASGI app call off a server
`serve` ran and the parity workload's real-run ARRIVALS
(`atom.compass.parity.traffic`), sent one request in flight: a gap longer than
any request. Prints the median, p10 and p90 in seconds, N and both sha256s as JSON.

`fit` reads a burst of N first-turn requests (default 256), which the loop takes
in one at a time: each gap between template starts is the next request's
``fixed_s + body bytes x per_byte_s``. Some gaps also hold tens of milliseconds
of other loop work, so `per_byte_s` is the Theil-Sen slope and `fixed_s` keeps
the burst's span; least squares on the starts swings the slope's sign between
bursts. Prints the terms, a gap's rms residual, N, bytes and sha256 as JSON.
"""

import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import runpy
import sys
import threading
import time
from pathlib import Path

import numpy as np


def serve(stamps: str, args: list[str]) -> None:
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
            await call(self, scope, receive, send)

        module.Starlette.__call__ = __call__

    def preprocessors(module) -> None:
        cls = module.InputOutputProcessor
        tokenize = cls.preprocess_fanout

        def preprocess_fanout(*a, **k):
            start = time.monotonic_ns()
            got = tokenize(*a, **k)
            stamp("prep", start, 0)
            return got

        cls.preprocess_fanout = preprocess_fanout

    # Each module is patched once it has run, before anything imports from it.
    patches = {
        "starlette.requests": requests,
        "starlette.applications": apps,
        "atom.entrypoints.openai.chat_encoders": encoders,
        "atom.model_engine.llm_engine": preprocessors,
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
    sys.argv[1:], sys.path[0] = args, os.getcwd()
    with open(stamps, "a", buffering=1) as out:
        main = "atom.entrypoints.openai.api_server"
        runpy.run_module(main, run_name="__main__", alter_sys=True)


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


if __name__ == "__main__":
    verb, stamps, *rest = sys.argv[1:]
    if verb == "serve":
        serve(stamps, rest)
    elif verb == "floor":
        print(json.dumps(floor(stamps, *rest)))
    else:
        print(json.dumps(fit(stamps, *map(int, rest))))

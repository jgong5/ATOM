"""Drive a served engine from a workload whose arrivals are declared, not raced.

`benchmark_serving` sends requests when the wall clock says to. Against a
simulated engine that advances a *virtual* clock by predicted step costs, the
two clocks race: the scheduler batches whatever has arrived by socket when a
step is decided, so the simulated run performs a different set of steps from the
run it stands for. Measured, on one workload: 189 decode steps against 127, in
buckets the real run never visits, and 7-20 prefill steps against 3 -- and two
runs at identical settings disagreed with each other.

So this client declares each request's arrival as an **offset into the run**
rather than delivering it at that moment. Requests are posted as fast as the
socket allows; `compass_arrival` says when the engine should treat each as having
arrived, and `compass_workload_size` tells the arrival barrier how many to expect
so it never advances virtual time past an arrival still in flight. The existing
workload row index travels as `compass_workload_index` to order equal-time
arrivals. Delivery order and socket latency then stop changing the schedule.

Against a real server there is no start-of-run to offset from, so declared
arrivals are ignored and "now" is used -- the same script measures both sides.

    # synthetic open-loop arrivals
    python scripts/compass/replay.py --port 8006 --num-requests 64 --rate 40 \
        --input-tokens 128 --output-tokens 32 --out replay.json

    # a recorded trace: one JSON object per line, with
    #   {"arrival_s": 0.0, "input_tokens": 512, "output_tokens": 64}
    python scripts/compass/replay.py --port 8006 --trace trace.jsonl --out replay.json

The trace form is the one that matters: a real arrival process is bursty in ways
no rate parameter reproduces, and burstiness is exactly what decides whether
requests batch together.
"""

import argparse
import asyncio
import json
import math
import os
import random
import resource
import sys
import time as _time
import urllib.error
import urllib.request


def _workload(args) -> list[dict]:
    """The requests to send, each with the instant it should count as arriving."""
    if args.trace:
        rows = []
        with open(args.trace, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        rows.sort(key=lambda r: float(r.get("arrival_s", 0.0)))
        if args.num_requests:
            rows = rows[: args.num_requests]
        base = float(rows[0].get("arrival_s", 0.0)) if rows else 0.0
        # Scaled here rather than at send time, so the workload written to the
        # output file is the workload that ran. It was scaled at send time and
        # saved unscaled, which meant a 40x compression that turned a 20-second
        # arrival process into a half-second burst left no trace in the
        # artifact -- the run looked paced and was not.
        scale = max(1e-9, float(args.time_scale))
        prefix_mode = bool(getattr(args, "prompt_encoding", None))
        if not prefix_mode and any("prompt_token_sha256" in r for r in rows):
            raise ValueError("prefix-aware rows require explicit --prompt-encoding")
        return [{**(r if prefix_mode else {}),
                 "arrival_s": (float(r.get("arrival_s", 0.0)) - base) / scale,
                 "input_tokens": int(r.get("input_tokens", args.input_tokens)),
                 "output_tokens": int(r.get("output_tokens", args.output_tokens))}
                for r in rows]

    # Poisson arrivals at --rate, or all at zero when the rate is infinite.
    rng = random.Random(args.seed)
    out, t = [], 0.0
    for _ in range(args.num_requests):
        out.append({"arrival_s": t,
                    "input_tokens": args.input_tokens,
                    "output_tokens": args.output_tokens})
        if args.rate > 0:
            t += rng.expovariate(args.rate)
    return out


DEFAULT_CLIENT_MEMORY_MIB = 4096


def _client_resource_plan(workload, prepare, memory_mib):
    """Refuse a workload that cannot be submitted whole, before warming a server.

    This is a conservative client allocation budget, not target KV accounting or
    an RSS guarantee. Allow 64 bytes per input/output token and 64 KiB per open
    request for Python/HTTP state. Actual prepared JSON bytes are also bounded.
    Never turn a resource shortage into a smaller completion-dependent pool.
    """
    if memory_mib <= 0 or prepare < 0:
        raise ValueError("client memory budget must be positive; preparation cannot be negative")
    n = len(workload)
    token_sum = sum(r["input_tokens"] + r["output_tokens"] for r in workload)
    cycles, remainder = divmod(prepare, n)
    prepare_tokens = cycles * token_sum + sum(
        r["input_tokens"] + r["output_tokens"] for r in workload[:remainder])
    estimate = max(64 * token_sum + 65536 * n,
                   64 * prepare_tokens + 65536 * prepare)
    budget = memory_mib * 1024 * 1024
    if estimate > budget:
        raise ValueError(
            f"client memory estimate {estimate} bytes exceeds the {budget}-byte "
            "budget; provide a sufficient --client-memory-budget-mib or a "
            "different explicitly registered workload (no requests were sent)")
    opened = len(os.listdir("/proc/self/fd"))
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = opened + max(n, prepare) + 64
    if soft_limit != resource.RLIM_INFINITY and needed > soft_limit:
        raise ValueError(
            f"whole-workload submission needs {needed} file descriptors including "
            f"reserve, above RLIMIT_NOFILE={soft_limit}; no requests were sent")
    return {"memory_budget_bytes": budget, "memory_estimate_bytes": estimate,
            "estimate_rule": "64 bytes/input-or-output token + 64 KiB/request; max of measured/preparation",
            "open_fds_before": opened, "fd_reserve": 64,
            "required_fds": needed, "fd_soft_limit": soft_limit}


#: `main` exits with this when the workload did not complete -- a request that
#: failed, never returned an outcome, or came back short. Its own code rather
#: than the refusal's 3 or the arrival barrier's 1, so a log line says which
#: boundary rejected the run without anyone having to read the artifact.
INCOMPLETE_EXIT = 4

WALL_WINDOW_SCHEMA = "compass.replay_wall_window/1"


def read_wall_window(value):
    """Validate the measured request interval, independently of virtual time."""
    if not isinstance(value, dict) or value.get("schema") != WALL_WINDOW_SCHEMA:
        raise ValueError("no explicit measured wall window from replay.py")
    if value.get("clock") != "wall":
        raise ValueError("the measured execution window is not on the wall clock")
    fields = ("started_at", "ended_at", "seconds")
    if not all(isinstance(value.get(k), (int, float))
               and not isinstance(value[k], bool) and math.isfinite(value[k])
               for k in fields):
        raise ValueError("the measured wall window has missing or non-finite times")
    start, end, seconds = (float(value[k]) for k in fields)
    if end < start or seconds < 0:
        raise ValueError("the measured wall window runs backwards")
    if not math.isclose(end - start, seconds, rel_tol=1e-4, abs_tol=0.005):
        raise ValueError("wall timestamps disagree with the monotonic execution duration")
    return dict(value)


def _completion_shortfall(response, want: int):
    """Why this reply is not the completion that was asked for, or None.

    Returned rather than raised: the *count* of these is the result. A run
    that produced three completions out of sixty-two is not a slow run, it is
    a different workload, and the artifact has to say so.

    `usage.completion_tokens` is the server's own count and the only one worth
    reading. `compare.py` already refuses a run whose replies do not carry it,
    so a client that accepts one has written an artifact nothing downstream
    will take.

    The count has to be the registered one **exactly**, and the finish reason
    does not excuse a difference. A replay's job is to run the output lengths
    the workload registered: each token is a decode step, so a request that
    produced three tokens where four were registered performed a different
    step sequence, whether it stopped because the engine abandoned it or
    because the model emitted its stop token. `finish_reason == "stop"` names
    the cause, not a dispensation -- a short "stop" is still a short request,
    and `compare.py` will refuse the pair for the same reason one step later.
    Overproduction is a mismatch in the other direction and is not waved
    through either.

    There is no EOS mode here to preserve: this client has never had one, and
    a request that may end early is not a request whose cost can be predicted
    from the workload. Both phases now send `ignore_eos`, so an early stop is
    not an option the client left open -- it is a reply that did not do what
    was asked.
    """
    if not isinstance(response, dict):
        return "the server's reply was not an object"
    choices = response.get("choices")
    if not choices:
        return "the reply carried no choices"
    got = (response.get("usage") or {}).get("completion_tokens")
    if not isinstance(got, int) or isinstance(got, bool):
        return "the reply carried no usage.completion_tokens"
    if got == want:
        return None
    reason = (choices[0] or {}).get("finish_reason")
    return (f"produced {got} output tokens where {want} were registered, "
            f"finishing as {reason!r}")


def _incomplete(results: list[dict], workload: list[dict]) -> dict:
    """Every request the run did not complete, grouped by why it did not.

    Three questions kept separate because they have different causes and
    different fixes: a request whose answer never came (`failed`), a declared
    request with no outcome recorded at all (`missing` -- the result list is
    not the workload, and a client that reports on the rows it holds says
    nothing about the ones it does not), and an answer that came back short
    (`truncated`). The three are disjoint by construction, so the counts add.
    """
    seen = {r.get("index") for r in results}
    failed = [{"index": r.get("index"), "why": r.get("error") or "no reason given"}
              for r in results if not r.get("ok")]
    missing = [{"index": i, "why": "no outcome was recorded for this request"}
               for i in range(len(workload)) if i not in seen]
    truncated = []
    for r in results:
        if not r.get("ok"):
            continue
        why = _completion_shortfall(
            r.get("response"), int(workload[r["index"]]["output_tokens"]))
        if why:
            truncated.append({"index": r["index"], "why": why})
    return {"failed": failed, "missing": missing, "truncated": truncated}


def _reasons(rows: list[dict]) -> list[dict]:
    """The distinct reasons, and how many requests each accounts for.

    "59 failed" names a number; "59 failed: TimeoutError: timed out" names a
    cause, and the two are a different amount of work to act on. Counted
    rather than sampled, because one failure standing for fifty-nine is only
    honest when all fifty-nine are the same one.
    """
    counts: dict[str, int] = {}
    for row in rows:
        why = str(row.get("why"))[:200]
        counts[why] = counts.get(why, 0) + 1
    return [{"reason": why, "requests": n}
            for why, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def _digest(path):
    """SHA-256 of a file, or None when there is no file to name."""
    if not path:
        return None
    import hashlib

    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def _revision():
    """The code this ran as, if the tree is a checkout."""
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is best effort
        return None


def _prompt(tokens: int, index: int) -> str:
    """Distinct text of exactly the requested token count.

    See `atom.compass.workload`, which run.py's sweep shares: a sweep that
    cannot target a token count cannot bracket a workload measured in tokens.
    """
    from atom.compass.workload import prompt_of_tokens

    return prompt_of_tokens(tokens, index)


def _load_prompt_tokenizer(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def _encoded_prompt(tokens: int, index: int, tokenizer):
    text = _prompt(tokens, index)
    if tokenizer is None:
        return text
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded) != tokens:
        raise ValueError(f"tokenizer produced {len(encoded)} tokens, expected {tokens}")
    return encoded


def _send(url: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        # The body says which field the server objected to; the status alone
        # does not, and "64 requests failed with 400" is not a diagnosis.
        try:
            detail = exc.read().decode()[:400]
        except Exception:  # noqa: BLE001
            detail = ""
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _encode_requests(workload, model, tokenizer, *, declared, byte_budget,
                     prompt_index_base=0, prefix_encoding=None, phase="measured"):
    """Construct identical per-request JSON before pacing; retain only bytes.

    Tokenization and JSON encoding are measured separately. The measured call
    belongs inside the execution wall window, after preparation/drain. This
    changes client encode jitter versus the old threaded sender and requires
    fresh paired runs; retained real references are not interchangeable.
    """
    payloads, total_bytes = [], 0
    token_seconds = json_seconds = 0.0
    token_digests = []
    for i, row in enumerate(workload):
        started = _time.monotonic()
        if prefix_encoding is None:
            prompt = _encoded_prompt(row["input_tokens"], prompt_index_base + i, tokenizer)
        else:
            from atom.compass.prefix_workload import token_digest
            prompt = prefix_encoding.tokens(
                row, phase=phase, warmup_index=i if phase == "warmup" else None)
            digest = token_digest(prompt)
            if phase == "measured" and digest != row.get("prompt_token_sha256"):
                raise ValueError("encoded prompt differs from the pinned row token digest")
            token_digests.append(digest)
        token_seconds += _time.monotonic() - started
        body = {"model": model, "prompt": prompt,
                "max_tokens": row["output_tokens"], "temperature": 0.0,
                "ignore_eos": True}
        if declared:
            body.update(compass_arrival=row["arrival_s"],
                        compass_workload_size=len(workload), compass_workload_index=i)
        started = _time.monotonic()
        data = json.dumps(body).encode()
        json_seconds += _time.monotonic() - started
        total_bytes += sys.getsizeof(data)
        if total_bytes > byte_budget:
            raise ValueError(
                f"prepared request bytes {total_bytes} exceed client memory "
                f"budget {byte_budget}; no requests from this phase were sent")
        payloads.append(data)
    encoding = {"prompt_construction_seconds": token_seconds,
                "json_encoding_seconds": json_seconds,
                "prepared_request_bytes": total_bytes}
    if prefix_encoding is not None:
        encoding["corpus_encoding"] = prefix_encoding.evidence(token_digests, phase)
    return payloads, encoding


def _reset_prefix_cache(base, timeout):
    from atom.compass.core.cache_boundary import reset_receipt_errors

    clock = _clock_of(base, timeout)
    if clock not in ("wall", "virtual"):
        raise ValueError("cache boundary requires a known wall or virtual server clock")
    receipt = _send(base + "/compass/cache/reset", {}, timeout)
    errors = reset_receipt_errors(
        receipt, expected_worker_kind=("modelled_no_device" if clock == "virtual"
                                       else "device_synchronize"),
        require_fresh_modelled=clock == "virtual")
    if errors:
        raise ValueError("; ".join(errors))
    return receipt


def _prefix_cache_snapshot(base, timeout):
    with urllib.request.urlopen(base + "/compass/cache", timeout=timeout) as response:
        snapshot = json.loads(response.read())
    if (not isinstance(snapshot, dict) or snapshot.get("schema") != "compass.cache_snapshot/1"
            or not isinstance(snapshot.get("ranks"), list) or not snapshot["ranks"]):
        raise ValueError("server returned no per-rank prefix-cache snapshot")
    return snapshot


async def _read_stream_response(response, timing):
    """Keep final usage and terminal framing without feeding generated history back."""
    request_id = finish_reason = usage = None
    done = False
    async for raw in response.content:
        line = raw.strip()
        if not line.startswith(b"data:"):
            continue
        value = line[5:].strip()
        if value == b"[DONE]":
            done = True
            timing["sse_done_wall_time"] = _time.time()
            continue
        chunk = json.loads(value)
        if chunk.get("error"):
            raise RuntimeError(f"stream error: {chunk['error']}")
        request_id = chunk.get("id", request_id)
        if chunk.get("usage") is not None:
            usage = chunk["usage"]
        for choice in chunk.get("choices", []):
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
                timing["sse_finish_wall_time"] = _time.time()
    timing["response_eof_wall_time"] = _time.time()
    if not done or not request_id or finish_reason is None or usage is None:
        raise ValueError("stream lacks its terminal finish, usage or DONE frame")
    return {"id": request_id, "usage": usage, "choices": [{"finish_reason": finish_reason}]}


async def _submit_requests(base, payloads, arrivals, *, pace, timeout,
                           endpoint="/v1/completions", streaming=False, response_gated=False,
                           expected_outputs=None):
    """Submit independent legacy rows, or a paced two-turn chat continuation.

    A declared workload needs all N connections until its closed registration
    barrier opens. aiohttp's default pool of 100 would deadlock at N>100.
    Coroutines remove the OS-thread ceiling; they do not remove socket costs.
    Paced requests wait on their own timers, never another request's response.
    """
    import aiohttp

    if len(payloads) != len(arrivals):
        raise ValueError("one arrival is required for every prepared request")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("network attempt timeout must be finite and positive")
    if response_gated and (not pace or len(payloads) != 2):
        raise ValueError("client response gating is the two-turn real-clock opening only")
    setup_started = _time.monotonic()
    trace = aiohttp.TraceConfig()
    epoch = epoch_wall = None
    start = asyncio.Event()
    ready = asyncio.Event()
    ready_count = 0
    finished = [asyncio.Event() for _ in payloads] if response_gated else None

    async def headers_callback(_session, context, _params):
        timing = context.trace_request_ctx
        timing["headers_callback_offset_s"] = _time.monotonic() - epoch
        timing["headers_callback_at"] = _time.time()

    async def body_chunk_callback(_session, context, params):
        timing = context.trace_request_ctx
        timing["body_chunk_callback_offset_s"] = _time.monotonic() - epoch
        timing["body_chunk_callback_at"] = _time.time()
        timing["body_chunk_callback_bytes"] = timing.get("body_chunk_callback_bytes", 0) + len(params.chunk)

    # Despite aiohttp's hook names, 3.13.5 invokes these before header buffering
    # and before chunk transport.write/drain. They do not witness wire delivery.
    trace.on_request_headers_sent.append(headers_callback)
    trace.on_request_chunk_sent.append(body_chunk_callback)
    results = [{"index": i, "ok": False, "error": "not submitted",
                "send_timing": {"source_arrival_s": arrivals[i]}}
               for i in range(len(payloads))]
    # Keep the old urllib connection-close policy. No per-host or global pool
    # limit may hold a declared row behind a response from an earlier row.
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0, force_close=True)
    request_timeout = aiohttp.ClientTimeout(total=timeout, sock_connect=timeout,
                                           sock_read=timeout, ceil_threshold=math.inf)
    async with aiohttp.ClientSession(connector=connector, timeout=request_timeout,
                                     trace_configs=[trace]) as session:
        async def one(i):
            nonlocal ready_count
            row = results[i]
            try:
                ready_count += 1
                if ready_count == len(payloads):
                    ready.set()
                await start.wait()
                if response_gated and i:
                    await finished[i - 1].wait()
                    if not results[i - 1]["ok"]:
                        raise RuntimeError("predecessor response did not complete")
                if pace:
                    await asyncio.sleep(max(0.0, arrivals[i] - (_time.monotonic() - epoch)))
                timing = row["send_timing"]
                timing["request_started_offset_s"] = _time.monotonic() - epoch
                timing["request_started_at"] = _time.time()
                if pace:
                    timing["request_start_lateness_s"] = timing["request_started_offset_s"] - arrivals[i]
                # sock_connect/sock_read alone leave upload backpressure
                # unbounded. The attempt deadline starts after the pacing wait
                # and covers connecting, body writes and response completion.
                async with session.post(
                    base + endpoint, data=payloads[i],
                    headers={"Content-Type": "application/json"},
                    trace_request_ctx=timing,
                ) as response:
                    if response.status >= 400:
                        raw = await response.read()
                        raise RuntimeError(f"HTTP {response.status}: {raw.decode(errors='replace')[:400]}")
                    if streaming:
                        result = await _read_stream_response(response, timing)
                        if expected_outputs is not None:
                            error = _completion_shortfall(result, expected_outputs[i])
                            if error:
                                raise ValueError(error)
                    else:
                        raw = await response.read()
                        result = json.loads(raw)
                    row.update(ok=True, response=result)
                    row.pop("error", None)
            except asyncio.CancelledError:
                row["ok"] = False
                row.pop("response", None)
                row["error"] = "CancelledError: request cancelled"
            except (aiohttp.ClientError, OSError, ValueError, RuntimeError) as exc:
                row["ok"] = False
                row.pop("response", None)
                row["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                row["send_timing"]["finished_offset_s"] = _time.monotonic() - epoch
                if streaming:
                    row["send_timing"]["client_response_returned_wall_time"] = _time.time()
                if finished is not None:
                    finished[i].set()

        tasks = [asyncio.create_task(one(i)) for i in range(len(payloads))]
        try:
            if tasks:
                await ready.wait()
            # All tasks and bodies exist before a common pacing epoch. Creating
            # thousands of tasks must not consume the first source intervals.
            epoch = _time.monotonic()
            epoch_wall = _time.time()
            start.set()
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            if epoch is None:
                epoch, epoch_wall = _time.monotonic(), _time.time()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    return results, {
        "schema": "compass.http_submission/1", "implementation": "aiohttp",
        "version": aiohttp.__version__, "connection_limit": None,
        "connection_reuse": False, "pacing_started_at": epoch_wall,
        "task_setup_seconds": epoch - setup_started,
        "network_attempt_timeout_s": timeout,
        "timeout_scope": "after pacing wait, through connection, upload and complete response",
        "requests_with_header_callback": sum("headers_callback_at" in r["send_timing"] for r in results),
        "timing_meaning": "request start and aiohttp pre-write trace callbacks; not wire completion, target ingress or engine arrival",
        "fresh_paired_runs_required": True,
    }


def _served_model(base: str, timeout: float) -> str | None:
    """Ask the server what it is serving.

    The completions endpoint validates the model name, so a wrong one fails
    every request identically and looks like a transport problem.
    """
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=timeout) as resp:
            listing = json.loads(resp.read())
        return listing["data"][0]["id"]
    except Exception:  # noqa: BLE001 - fall back to whatever was passed
        return None


def _drain_records(base: str, timeout: float) -> dict:
    """Read and clear the engine's record store.

    `GET /compass/requests` drains by default, which is what makes the
    boundary in the protocol real: after this returns, the store holds nothing
    a later read could pick up.
    """
    try:
        with urllib.request.urlopen(base + "/compass/requests",
                                    timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return {"count": 0, "requests": [], "error": f"{type(exc).__name__}: {exc}"}


#: Where preparation's prompt indices start, far enough above any workload
#: index that the two cannot collide.
_PREPARE_PROMPT_BASE = 1_000_000


#: Where preparation's prompt indices start, far enough above any workload
#: index that the two cannot collide.
_PREPARE_PROMPT_BASE = 1_000_000


def _clock_of(base: str, timeout: float) -> str | None:
    """Which clock this server reports on, before anything is sent to it.

    Read from `/compass/provenance` rather than by draining records, so asking
    the question does not consume the very store the drain boundary depends on.
    """
    try:
        with urllib.request.urlopen(base + "/compass/provenance",
                                    timeout=timeout) as resp:
            compass = json.loads(resp.read()).get("compass") or {}
    except Exception:  # noqa: BLE001 - absence is not a virtual clock
        return None
    if not compass.get("enabled") or not compass.get("virtual_clock"):
        return "wall"
    return "virtual" if compass.get("mode") == "predict" else "wall"


def _prepare(base: str, model: str, workload: list[dict], args) -> dict:
    """Warm the server, wait for it, and prove the engine forgot about it.

    By default, preparation traverses the measured workload's full shapes.
    An explicit diagnostic cap preserves full prompts but shortens warmup
    outputs; its policy and sent shapes travel with the drain evidence.
    Unpaced and concurrent -- preparation has no workload arrival process.

    For the PoC's native FULL graph, capture/replay keys are (batch, query
    length); GDN's capture metadata fixes max_seqlen_k to max_model_len and
    reads growing contexts from buffers. A cap of 32 exercises decode without
    repeating a long output walk. This is a different preparation policy,
    with no claim of full-sequence or thermal equivalence for acceptance.

    The drain is the point. `_drain_records` clears the engine's store, so the
    measured read that follows contains only measured requests, and the rows
    taken out are kept as the evidence that they were taken out.
    """
    shapes = [workload[i % len(workload)] for i in range(args.prepare)]
    cap = getattr(args, "diagnostic_prepare_output_cap", None)
    policy = {}
    if cap is not None:
        if type(cap) is not int or cap < 2:
            raise ValueError("diagnostic preparation output cap must be at least 2")
        policy = {
            "policy": {"purpose": "diagnostic", "output_tokens_cap": cap,
                       "prompt_tokens": "full", "measured_requests": "unchanged"},
            "source_shapes": [{"input_tokens": row["input_tokens"],
                               "output_tokens": row["output_tokens"]} for row in shapes],
        }
        shapes = [dict(row, output_tokens=min(row["output_tokens"], cap))
                  for row in shapes]
    began = _time.monotonic()
    # Preparation must never arm the measured workload's one-shot barrier or
    # reuse its prompt identities. Each HTTP request still reaches one native
    # CoreManager.add_request([seq]); this does not batch engine admission.
    payloads, encoding = _encode_requests(
        shapes, model, getattr(args, "_prompt_tokenizer", None), declared=False,
        byte_budget=getattr(args, "client_memory_budget_mib", DEFAULT_CLIENT_MEMORY_MIB) * 1024 * 1024,
        prompt_index_base=_PREPARE_PROMPT_BASE,
        **({"prefix_encoding": args._prefix_encoding, "phase": "warmup"}
           if getattr(args, "_prefix_encoding", None) is not None else {}))
    results, submission = asyncio.run(_submit_requests(
        base, payloads, [0.0] * len(shapes), pace=False, timeout=args.timeout))
    seconds = _time.monotonic() - began
    if cap is not None:
        policy["response_usage"] = [
            {"index": result["index"],
             "usage": (result.get("response") or {}).get("usage")}
            for result in results]
        for result in results:
            usage = (result.get("response") or {}).get("usage") or {}
            expected = shapes[result["index"]]
            if result["ok"] and (
                usage.get("prompt_tokens") != expected["input_tokens"]
                or usage.get("completion_tokens") != expected["output_tokens"]
            ):
                result.update(ok=False, error="diagnostic preparation served different token lengths")
    returned = sum(1 for r in results if r["ok"])

    drained = _drain_records(base, args.timeout)
    after = _drain_records(base, args.timeout)
    rows = drained.get("requests") or []
    boundary = max((r.get("finish_time") or 0.0) for r in rows) if rows else None
    ok = (returned == len(shapes) and not (after.get("requests") or []))
    print(f"  prepared {returned}/{len(shapes)} in {seconds:.1f}s, drained "
          f"{len(rows)} engine records, store empty after: "
          f"{not (after.get('requests') or [])}")
    return {"requested": len(shapes), "returned": returned,
            "wall_seconds": round(seconds, 3),
            "encoding": encoding, "submission": submission,
            "drained_records": len(rows),
            "store_empty_after_drain": not (after.get("requests") or []),
            "drained": bool(ok),
            "boundary_engine_time": boundary,
            "clock": drained.get("clock"),
            "declared_workload_size": False,
            "shapes": [{"input_tokens": r["input_tokens"],
                        "output_tokens": r["output_tokens"]} for r in shapes],
            **policy,
            "records": rows,
            "failures": [r for r in results if not r["ok"]]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", default=None,
                   help="defaults to whatever /v1/models reports")
    p.add_argument("--trace", help="JSONL of {arrival_s, input_tokens, output_tokens}")
    p.add_argument("--num-requests", type=int, default=64)
    p.add_argument("--rate", type=float, default=0.0,
                   help="Poisson arrivals per second; 0 means all arrive at once")
    p.add_argument("--input-tokens", type=int, default=128)
    p.add_argument("--output-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--client-memory-budget-mib", type=int,
                   default=DEFAULT_CLIENT_MEMORY_MIB,
                   help="client allocation planning budget (default 4096 MiB); "
                        "insufficient memory or file descriptors refuse the "
                        "whole workload, never throttle requests")
    p.add_argument("--out", required=True)
    p.add_argument("--pace", action="store_true",
                   help="deliver each request when its arrival really comes "
                        "round, instead of declaring it. Use against a real "
                        "engine: a real clock discards a declared arrival, so "
                        "without this the real side answers a burst while the "
                        "simulated side answers the trace")
    p.add_argument("--pretokenize", action="store_true",
                   help="send the same synthetic prompts as token IDs; tokenize "
                        "before the pacing epoch while retaining that work in "
                        "the measured wall window")
    p.add_argument("--prompt-encoding", help="explicit pinned corpus-prefix encoding; sends token IDs")
    p.add_argument("--prompt-encoding-sha256", help="required digest of --prompt-encoding")
    p.add_argument("--opening-plan", help="pinned two-turn AIPerf chat opening")
    p.add_argument("--opening-plan-sha256", help="required digest of --opening-plan")
    p.add_argument("--time-scale", type=float, default=1.0,
                   help="divide every arrival offset by this, to replay a "
                        "long trace in less time. 1.0 keeps the trace's own "
                        "timing; it changes how much requests batch, so it is "
                        "a property of the workload and not a free knob")
    p.add_argument("--prepare", type=int, default=0, metavar="N",
                   help="send N preparation requests of the measured shape and "
                        "wait for all of them before measuring anything, then "
                        "drain the engine's record store so no preparation row "
                        "can reach the result. See atom/compass/PROTOCOL.md.")
    p.add_argument("--prepare-out", default=None,
                   help="where to keep the drained preparation records; they "
                        "are evidence of the drain, not waste")
    p.add_argument("--diagnostic-prepare-output-cap", type=int, default=None,
                   help="diagnostic only: cap warmup outputs (at least 2), "
                        "preserving full prompts and every measured request; "
                        "default is the full preparation sequence")
    p.add_argument("--check-lengths", action="store_true",
                   help="compare the server's reported prompt_tokens against "
                        "what was asked for, and warn if they differ")
    args = p.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        p.error("--timeout must be finite and positive")
    if args.diagnostic_prepare_output_cap is not None:
        if args.diagnostic_prepare_output_cap < 2 or args.prepare < 1:
            p.error("--diagnostic-prepare-output-cap requires a cap of at least 2 and --prepare")

    if bool(args.prompt_encoding) != bool(args.prompt_encoding_sha256):
        p.error("--prompt-encoding and --prompt-encoding-sha256 are required together")
    if bool(args.opening_plan) != bool(args.opening_plan_sha256):
        p.error("--opening-plan and --opening-plan-sha256 are required together")
    if args.opening_plan and (args.trace or args.prompt_encoding or args.time_scale != 1):
        p.error("opening plan owns its payloads and unscaled source timing")
    args._prefix_encoding = None
    args._opening_plan = None
    try:
        if args.prompt_encoding:
            from atom.compass.prefix_workload import PrefixEncoding
            args._prefix_encoding = PrefixEncoding.load(
                args.prompt_encoding, args.prompt_encoding_sha256)
            args.pretokenize = True
        if args.opening_plan:
            from atom.compass.replay_plan import OpeningPlan
            args._opening_plan = OpeningPlan.load(args.opening_plan, args.opening_plan_sha256)
            args.pretokenize = True  # Tokenizer also supplies exact-length preparation prompts.
            workload = args._opening_plan.workload()
        else:
            workload = _workload(args)
        if args._prefix_encoding is not None:
            args._prefix_encoding.validate_rows(workload)
    except (OSError, ValueError) as exc:
        print(f"ATOMCompass refusing prompt encoding: {exc}", file=sys.stderr)
        return 3
    if not workload:
        print("empty workload", file=sys.stderr)
        return 2
    try:
        resource_plan = _client_resource_plan(
            workload, args.prepare, args.client_memory_budget_mib)
    except (OSError, ValueError) as exc:
        print(f"ATOMCompass refusing whole-workload submission: {exc}", file=sys.stderr)
        return 3
    base = f"http://{args.host}:{args.port}"
    model = args.model or _served_model(base, args.timeout)
    if model is None:
        print("could not determine the served model; pass --model",
              file=sys.stderr)
        return 2

    if args._opening_plan is not None:
        if model != args._opening_plan.model:
            print("opening model differs from the served model", file=sys.stderr)
            return 3
        clock = _clock_of(base, args.timeout)
        if clock == "wall":
            args.pace = True
        elif clock != "virtual":
            print("opening requires an identified wall or virtual clock", file=sys.stderr)
            return 3
        if clock == "virtual":
            with urllib.request.urlopen(base + "/compass/provenance", timeout=args.timeout) as response:
                runtime = json.loads(response.read())
            if runtime.get("compass", {}).get("opening_plan_sha256") != args.opening_plan_sha256:
                print("predictor has not loaded the same opening release plan", file=sys.stderr)
                return 3

    if args.pace and _clock_of(base, args.timeout) == "virtual":
        print("ATOMCompass WARNING: refusing to pace a predictor. --pace "
              "delivers each request when its arrival really comes round, on "
              "the wall clock; this server advances a virtual clock by "
              "predicted step costs, so the two race and the simulated run "
              "performs a different set of steps from the run it stands for. "
              "That is the failure declared arrivals exist to prevent. The "
              "predictive side is the unpaced one: every declared arrival is "
              "posted up front and honoured on the engine's own clock, so no "
              "arrival is clipped and none waits on the socket. Pace the real "
              "engine; declare arrivals to the predictor.", file=sys.stderr)
        return 3

    if args.prepare and _clock_of(base, args.timeout) == "virtual":
        print("ATOMCompass WARNING: refusing to warm a predictor. A declared "
              "arrival is an offset from the engine's epoch, and the process "
              "that stamps arrivals holds a virtual clock frozen there, so a "
              "preparation batch does not move the origin it is measured "
              "against: every measured request would be stamped as having "
              "arrived before the preparation that preceded it, and its whole "
              "duration would land inside their TTFT and latency. Measured on "
              "the first warmed 27B cell that was +71% TTFT from a 14.4s "
              "preparation. A predictor has no kernels to compile and no "
              "allocator to settle; it represents the warm target state by "
              "being given warmup_seconds=0, not by executing a warmup it "
              "would only have to model. Warm the real server; predict from a "
              "fresh empty run.", file=sys.stderr)
        return 3

    args._prompt_tokenizer = (_load_prompt_tokenizer(model)
                              if args.pretokenize else None)
    if args._opening_plan is not None:
        try:
            args._opening_plan.verify_tokenizer(args._prompt_tokenizer)
        except ValueError as exc:
            print(f"ATOMCompass refusing opening tokenizer: {exc}", file=sys.stderr)
            return 3
    if args._prefix_encoding is not None:
        try:
            args._prefix_encoding.verify_tokenizer(args._prompt_tokenizer, model)
        except ValueError as exc:
            print(f"ATOMCompass refusing tokenizer: {exc}", file=sys.stderr)
            return 3
    prepare = _prepare(base, model, workload, args) if args.prepare else None
    if prepare is not None and not prepare["drained"]:
        print("ATOMCompass WARNING: preparation did not drain -- the engine's "
              "record store was not empty at the boundary, so a preparation "
              "row could enter the measured result; refusing to measure",
              file=sys.stderr)
        return 3

    cache_reset = None
    if args._prefix_encoding is not None or args._opening_plan is not None:
        if prepare is not None and args.prepare_out:
            with open(args.prepare_out, "w", encoding="utf-8") as fh:
                json.dump(prepare, fh, indent=1)
        try:
            cache_reset = _reset_prefix_cache(base, args.timeout)
            if args._opening_plan is not None:
                from atom.compass.core.cache_policy import policy_errors
                errors = [error for rank in cache_reset["ranks"]
                          for error in policy_errors(rank["after"].get("policy"),
                                                     args._opening_plan.cache_policy)]
                if errors:
                    raise ValueError("; ".join(errors))
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"ATOMCompass refusing cache boundary: {exc}", file=sys.stderr)
            return 3

    execution_started_at = _time.time()
    began = _time.monotonic()
    # Keep prompt/token and JSON work inside measured execution but before the
    # pacing origin. Retain prepared bytes rather than every encoded token list.
    try:
        if args._opening_plan is not None:
            encoded_at = _time.monotonic()
            payloads = args._opening_plan.encode_payloads(declared=not args.pace)
            encoding = {"prompt_construction_seconds": 0.0,
                        "json_encoding_seconds": _time.monotonic() - encoded_at,
                        "prepared_request_bytes": sum(sys.getsizeof(data) for data in payloads)}
            if encoding["prepared_request_bytes"] > resource_plan["memory_budget_bytes"]:
                raise ValueError("opening payloads exceed the client memory budget")
        else:
            payloads, encoding = _encode_requests(
                workload, model, args._prompt_tokenizer, declared=not args.pace,
                byte_budget=resource_plan["memory_budget_bytes"],
                **({"prefix_encoding": args._prefix_encoding}
                   if args._prefix_encoding is not None else {}))
    except (MemoryError, ValueError) as exc:
        print(f"ATOMCompass refusing prepared workload: {exc}", file=sys.stderr)
        return 3
    results, submission = asyncio.run(_submit_requests(
        base, payloads, [row["arrival_s"] for row in workload],
        pace=args.pace, timeout=args.timeout,
        **({"endpoint": "/v1/chat/completions", "streaming": True,
            "response_gated": args.pace,
            "expected_outputs": [row["output_tokens"] for row in workload]}
           if args._opening_plan is not None else {})))
    execution_seconds = _time.monotonic() - began
    execution_ended_at = _time.time()
    cache_end, cache_error = None, None
    if args._prefix_encoding is not None or args._opening_plan is not None:
        try:
            cache_end = _prefix_cache_snapshot(base, args.timeout)
            if args._opening_plan is not None:
                from atom.compass.core.cache_policy import policy_errors
                errors = [error for rank in cache_end["ranks"]
                          for error in policy_errors(rank.get("policy"), args._opening_plan.cache_policy)]
                if errors:
                    raise ValueError("; ".join(errors))
        except (OSError, RuntimeError, ValueError) as exc:
            cache_error = str(exc)

    # What the run produced, against what it was asked to produce. This client
    # counted only the requests whose *send* raised and reported "0 failed"
    # for everything else, so a development run that answered 3 of 62 and
    # timed out on the rest exited zero and was read as a measurement. The
    # tally is written into the artifact and decides the exit code below.
    incomplete = _incomplete(results, workload)
    failed = incomplete["failed"]

    # What the server says it received, against what was asked for. The builder
    # is exact by construction under a tokenizer giving one token per word in
    # `_WORDS`; this checks the assumption held, and costs nothing because the
    # count is already in every response.
    length_check = "not requested"
    if args.check_lengths:
        off = []
        for r in results:
            if not r["ok"]:
                continue
            got = (r["response"].get("usage") or {}).get("prompt_tokens")
            want = workload[r["index"]]["input_tokens"]
            if got is not None and got != want:
                off.append((want, got))
        if off:
            worst = max(off, key=lambda pair: abs(pair[1] - pair[0]))
            length_check = f"{len(off)} of {len(results)} wrong"
            print(f"  WARNING: {len(off)} of {len(results)} prompts were not the "
                  f"requested length; worst asked {worst[0]} got {worst[1]}",
                  file=sys.stderr)
        else:
            length_check = "passed"
            print(f"  prompt lengths verified against the server for "
                  f"{len(results) - len(failed)} requests")
    engine = {}
    try:
        engine = _send(base + "/compass/requests", {}, args.timeout)
    except Exception:  # noqa: BLE001 - a real server has no such endpoint
        try:
            with urllib.request.urlopen(base + "/compass/requests",
                                        timeout=args.timeout) as resp:
                engine = json.loads(resp.read())
        except Exception:  # noqa: BLE001
            engine = {}

    # Everything needed to say what this run was, next to what it produced. A
    # result whose arrival process, calibration or code revision cannot be
    # recovered from its own artifact is not reproducible, and one of these
    # runs was read as a measurement for a day after its arrival protocol had
    # silently failed.
    # Who served this. The client's own git revision names the tree that *sent*
    # the requests, which against a remote server is a different machine from
    # the one that answered them -- so a manifest carrying only `revision` says
    # nothing about the build, model or calibration that produced the numbers.
    # `GET /compass/provenance` is read on the server side, including the digest
    # of the calibration file the server itself opened.
    server = {}
    try:
        with urllib.request.urlopen(base + "/compass/provenance",
                                    timeout=args.timeout) as resp:
            server = json.loads(resp.read())
    except Exception:  # noqa: BLE001 - an older or non-Compass server has none
        server = {}

    # Whether the engine's arrival barrier gave up waiting, straight from the
    # engine -- not inferred from the flags this client was given. A run can be
    # invoked correctly and still time out, if submission was slow enough, and
    # the client cannot see that from its own side: it reports "0 failed"
    # either way. `compare.py` refuses a run whose manifest says True; an
    # unknown barrier stays unknown and is reported rather than assumed.
    barrier = (engine.get("arrival_barrier") or {}) if isinstance(engine, dict) else {}

    manifest = {
        "revision": _revision(),
        "client_revision": _revision(),
        "arrival_barrier_timed_out": barrier.get("timed_out"),
        "arrival_barrier": barrier or None,
        "server_revision": server.get("server_revision"),
        "server_code_sha256": server.get("server_code_sha256"),
        "model_revision": server.get("model_revision"),
        "calibration_sha256": server.get("calibration_sha256"),
        "server": server or None,
        "paced": bool(args.pace),
        "time_scale": float(args.time_scale),
        "request_timeout_seconds": float(args.timeout),
        "client_resources": resource_plan,
        "submission": submission,
        "requests": len(workload),
        "arrival_span_s": (round(workload[-1]["arrival_s"], 6)
                           if workload else 0.0),
        "trace": args.trace,
        "trace_sha256": _digest(args.trace),
        "model": model,
        "failed": len(failed),
        "missing": len(incomplete["missing"]),
        "truncated": len(incomplete["truncated"]),
        # Named for what it is, so a reader does not have to add three
        # numbers up and hope they are disjoint.
        "completed": len(workload) - sum(len(v) for v in incomplete.values()),
        "complete": not any(incomplete.values()),
        # The reasons, counted. `compare.py` reads `failed`; a person reads
        # this, and a person is who decides whether to rerun.
        "incomplete_reasons": ({k: _reasons(v) for k, v in incomplete.items() if v}
                               or None),
        "prompt_lengths": length_check,
        "prepare": ({k: v for k, v in prepare.items() if k != "records"}
                    if prepare else None),
        "wall_execution": {
            "schema": WALL_WINDOW_SCHEMA,
            "clock": "wall",
            "started_at": execution_started_at,
            "ended_at": execution_ended_at,
            "seconds": execution_seconds,
            "includes": "measured request construction, async setup, pacing, dispatch, completion and client session teardown",
            "excludes": "preparation/drain and post-run reporting",
        },
        "prompt_encoding": {
            "kind": "token_ids" if args.pretokenize else "text",
            "tokenizer_model": model if args.pretokenize else None,
            "tokenizer_revision": (getattr(args._prompt_tokenizer, "init_kwargs", {})
                                   .get("_commit_hash") if args.pretokenize else None),
            "conversion_seconds": encoding["prompt_construction_seconds"] if args.pretokenize else None,
            "prompt_construction_seconds": encoding["prompt_construction_seconds"],
            "json_encoding_seconds": encoding["json_encoding_seconds"],
            "prepared_request_bytes": encoding["prepared_request_bytes"],
            "json_encoding_before_pacing": True,
            "pacing_started_at": submission["pacing_started_at"],
            "conversion_in_execution": bool(args.pretokenize),
        },
    }
    if args._prefix_encoding is not None:
        manifest["prompt_encoding"]["corpus_encoding"] = encoding["corpus_encoding"]
    if args._opening_plan is not None:
        manifest["aiperf_opening"] = args._opening_plan.evidence()
        manifest["prompt_encoding"].update(
            kind="chat_messages", conversion_in_execution=False,
            conversion_seconds=None, token_verification="before preparation; shared preprocessing verifies again")
        observed = {row["request_id"]: row for row in engine.get("requests", [])}
        try:
            errors = args._opening_plan.observation_errors(server, engine, results)
        except (KeyError, TypeError, ValueError) as exc:
            errors = [f"malformed opening evidence: {exc}"]
        manifest["aiperf_opening"]["observation_errors"] = errors
        manifest["aiperf_opening"]["timing_boundaries"] = {
            "engine_clock": engine.get("clock"), "client_clock": "wall",
            "requests": [{
                "index": result["index"],
                "engine": observed.get((result.get("response") or {}).get("id")),
                "client": result["send_timing"],
            } for result in results],
            "qualification": "Client wall times are observations, not predictor costs; cross-host clock offset is not calibrated here",
        }
        if errors:
            manifest["complete"] = False
            manifest["incomplete_reasons"] = {**(manifest["incomplete_reasons"] or {}),
                                               "opening_evidence": errors}
    if args._prefix_encoding is not None or args._opening_plan is not None:
        manifest["cache_boundary"] = cache_reset
        manifest["cache_state_after"] = cache_end
        if cache_error:
            manifest["cache_state_error"] = cache_error
            manifest["complete"] = False
            manifest["incomplete_reasons"] = {
                **(manifest["incomplete_reasons"] or {}), "cache_observation": cache_error}
    if prepare and args.prepare_out:
        with open(args.prepare_out, "w", encoding="utf-8") as fh:
            json.dump(prepare, fh, indent=1)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"run": manifest, "workload": workload, "results": results,
                   "engine": engine}, fh, indent=1)
    completed = len(workload) - sum(len(v) for v in incomplete.values())
    print(f"sent {len(workload)} requests, {completed} completed, "
          f"{len(failed)} failed, {len(incomplete['missing'])} missing, "
          f"{len(incomplete['truncated'])} short -> {args.out}")
    for kind, rows in incomplete.items():
        # Every reason with its own count, not the first one standing in for
        # the rest: fifty-nine timeouts and fifty-eight timeouts plus one 400
        # are different runs, and the second is the one worth reading.
        for entry in _reasons(rows):
            print(f"  {entry['requests']} {kind}: {entry['reason']}",
                  file=sys.stderr)
    if barrier.get("timed_out") is None:
        # Said out loud rather than passed over. The run may be perfectly good;
        # what is known is that nobody can tell from this artifact.
        print(f"  NOTE: the arrival barrier could not be read "
              f"({barrier.get('why') or 'no reading'}); this run's latencies "
              f"are unverified, not verified",
              file=sys.stderr)
    if barrier.get("timed_out"):
        # Written first and then failed: the artifact is the evidence of the
        # failure, and deleting it would leave only a log line. Non-zero so a
        # run that timed out cannot be read as a run that completed -- which is
        # exactly what happened when this was only a warning in a server log.
        detail = barrier.get("ranks") or barrier
        print(f"ATOMCompass WARNING: the engine's arrival barrier timed out, "
              f"so virtual time advanced past an arrival still in flight. "
              f"Every latency in this run is invalid: {json.dumps(detail)}",
              file=sys.stderr)
        return 1
    if cache_error:
        print(f"ATOMCompass refusing cache evidence: {cache_error}", file=sys.stderr)
        return 3
    if not manifest["complete"]:
        # After the artifact is written, for the same reason the barrier check
        # is: the file is the evidence, and a run that exits non-zero without
        # leaving one cannot be diagnosed.
        #
        # A replay is an expected-success run. Whether a *configuration* is
        # feasible is not asked here and never was -- the engine answers that
        # at startup by refusing to size a pool it cannot page with
        # (`InsufficientPoolBudget`, atom/compass/replay/runner.py), and a
        # server that refused never reaches a client. So there is no mode in
        # which these counts are the expected outcome, and nothing to exempt.
        print(f"ATOMCompass WARNING: {completed} of {len(workload)} requests "
              f"completed. The engine did not run the workload this artifact "
              f"describes, so its metrics are over a different one and its "
              f"step sequence is not the trace's: "
              f"{json.dumps(manifest['incomplete_reasons'])}",
              file=sys.stderr)
        return INCOMPLETE_EXIT
    return 0


if __name__ == "__main__":
    sys.exit(main())

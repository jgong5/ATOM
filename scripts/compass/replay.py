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
so it never advances virtual time past an arrival still in flight. Delivery order
and socket latency then stop mattering, which is the point.

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
import json
import random
import sys
import threading
import time as _time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


#: Requests to generate when there is no trace to replay. Not a default for the
#: trace path: a trace is a workload someone chose, and quietly keeping its
#: first N is a different workload wearing its name and sha256. This was that
#: default for the trace path once, and a 809-request replay ran 64 of them --
#: 55s of a 5726s timeline -- with the artifact naming the whole file.
_SYNTHETIC_REQUESTS = 64


def _workload(args) -> list[dict]:
    """The requests to send, each with the instant it should count as arriving.

    Sets ``args.trace_rows`` on the trace path, so the manifest can report what
    was on disk beside what ran and truncation cannot be silent.
    """
    if args.trace:
        rows = []
        with open(args.trace, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        rows.sort(key=lambda r: float(r.get("arrival_s", 0.0)))
        args.trace_rows = len(rows)
        if args.num_requests:
            rows = rows[: args.num_requests]
        base = float(rows[0].get("arrival_s", 0.0)) if rows else 0.0
        # Scaled here rather than at send time, so the workload written to the
        # output file is the workload that ran. It was scaled at send time and
        # saved unscaled, which meant a 40x compression that turned a 20-second
        # arrival process into a half-second burst left no trace in the
        # artifact -- the run looked paced and was not.
        scale = max(1e-9, float(args.time_scale))
        out = []
        for r in rows:
            row = {"arrival_s": (float(r.get("arrival_s", 0.0)) - base) / scale,
                   "input_tokens": int(r.get("input_tokens", args.input_tokens)),
                   "output_tokens": int(r.get("output_tokens", args.output_tokens))}
            # Carried, not dropped. `hash_ids` decides what the prompt builder
            # shares, so without it a cc-traces replay is the same lengths with
            # none of the reuse; `session` namespaces those ids, which are
            # session-local; and `api_time_s` is the recorded duration, which is
            # the only ground truth that ever reaches the artifact. They were
            # projected away here, so a trace could carry all three and the
            # output file would show no sign they had existed.
            for key in ("hash_ids", "session", "api_time_s", "source_ttft_s",
                        "session_id"):
                if key in r:
                    row[key] = r[key]
            out.append(row)
        return out

    # Poisson arrivals at --rate, or all at zero when the rate is infinite.
    rng = random.Random(args.seed)
    out, t = [], 0.0
    for _ in range(args.num_requests or _SYNTHETIC_REQUESTS):
        out.append({"arrival_s": t,
                    "input_tokens": args.input_tokens,
                    "output_tokens": args.output_tokens})
        if args.rate > 0:
            t += rng.expovariate(args.rate)
    return out


#: Most requests this client will hold open at once. One thread each, so this
#: is a thread count as much as a connection count. Not a tuning knob: past it
#: the declared-arrival protocol needs a bulk submission the server does not
#: have, and quietly posting fewer would reintroduce the deadlock this bounds.
MAX_IN_FLIGHT = 1024


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


def _prompt(row: dict, index: int) -> str:
    """Text of exactly the requested token count, sharing what the row shares.

    Two opposite constructions, picked by whether the trace says anything about
    sharing. See `atom.compass.workload`, which run.py's sweep shares: a sweep
    that cannot target a token count cannot bracket a workload measured in
    tokens.

    Without `hash_ids` the prompt is built to *defeat* the prefix cache, so
    every request pays for its own prefill and a synthetic workload measures
    what it says it measures.

    With `hash_ids` -- which is every cc-traces row -- each id names a 64-token
    block and consecutive turns of a session share their leading ids. On the
    256k corpus 96.2% of all input tokens are a re-send of a block the session
    already sent, so ignoring the ids would not be a small simplification: it
    would be a workload with 26x the prefill work of the one recorded.
    """
    from atom.compass.workload import prompt_of_hash_ids, prompt_of_tokens

    tokens = int(row["input_tokens"])
    ids = row.get("hash_ids")
    if ids:
        return prompt_of_hash_ids(ids, tokens, session=int(row.get("session", 0)))
    return prompt_of_tokens(tokens, index)


def _sessions(workload: list[dict]) -> list[list[int]]:
    """Indices of each session's rows, sessions in first-arrival order.

    A cc-traces session is the unit a user holds: its turns re-send the whole
    conversation, so they share prefix blocks with each other and with nothing
    else. Splitting one across clients would replay the same token counts with
    none of that sharing.
    """
    order: dict[int, list[int]] = {}
    for i, row in enumerate(workload):
        order.setdefault(int(row.get("session", 0)), []).append(i)
    return [idxs for _, idxs in
            sorted(order.items(), key=lambda kv: workload[kv[1][0]]["arrival_s"])]


def _dependencies(rows: list[dict]) -> list[list[int]]:
    """Which of a session's rows each row waited for, from the recorded timing.

    A session is not a straight line of turns. 43.5% of the corpus's requests
    overlap another request of their own session, because a turn can fan out
    into sub-agents -- and peak in-session concurrency runs from 1 to 23.

    But that concurrency is *recorded*, not structural. Only 8.4% of sub-agent
    window pairs actually overlap, and 218 of 393 sessions never have two
    sub-agent branches open at once, so treating every branch as parallel
    because the corpus nests it under a `subagent` wrapper would invent load
    the recording does not contain. The recorded windows say which requests
    were really in the server together, so that is what is read here: a row
    waits for exactly those rows that had already finished when it started, and
    runs alongside the rest.

    Time is used for ordering only; the gaps are dropped, which is what makes
    this a closed loop rather than a replay of the trace's arrival rate.
    """
    starts = [float(r.get("arrival_s", 0.0)) for r in rows]
    ends = [s + float(r.get("api_time_s") or 0.0) for s, r in zip(starts, rows)]
    return [[j for j in range(len(rows)) if j != k and ends[j] <= starts[k]]
            for k in range(len(rows))]


def _run_session(idxs: list[int], deps: list[list[int]], send, out: list) -> None:
    """Issue one session's requests, honouring the recorded happens-before."""
    done = [threading.Event() for _ in idxs]
    threads = []

    def go(k: int) -> None:
        try:
            out[idxs[k]] = send(idxs[k])
        finally:
            done[k].set()

    for k in range(len(idxs)):
        # Waiting here, before *starting* k, cannot deadlock: every member of
        # deps[k] is a row earlier in the session, and each was spawned before
        # this loop reached k.
        for j in deps[k]:
            done[j].wait()
        thread = threading.Thread(target=go, args=(k,), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


def _closed_loop(workload: list[dict], clients: int, per_client: int, send):
    """Run the workload as `clients` users, each holding a session to the end.

    Open-loop replay pins throughput to the trace's own arrival rate, so every
    client count lands on the same tokens/s and a saturation curve collapses to
    a single point. A closed loop asks the other question -- given N users who
    always have work outstanding, how much does the engine deliver in total and
    how fast does each user see its own tokens -- which is the curve with
    tokens/s/GPU against tokens/s/user.

    Sessions are dealt round-robin from the trace's own order rather than
    pulled from a shared queue as clients free up. A queue would hand
    different sessions to different clients on the real and the modelled side,
    because the two sides finish at different moments; dealing them up front
    makes both sides execute the same sessions in the same order, so the
    comparison stays paired per request instead of merely distributional.
    """
    groups = _sessions(workload)
    if per_client:
        groups = groups[: clients * per_client]
    assignment = [groups[c::clients] for c in range(clients)]
    out: list = [None] * len(workload)

    def client(c: int) -> None:
        for idxs in assignment[c]:
            _run_session(idxs, _dependencies([workload[i] for i in idxs]),
                         send, out)

    threads = [threading.Thread(target=client, args=(c,)) for c in range(clients)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return [r for r in out if r is not None], assignment


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


def _cache_stats(base: str, timeout: float) -> dict:
    """Cumulative prefix-cache counters, or {} on a server without them.

    Read either side of the run and differenced, this is the check that the
    replay actually reproduced the trace's reuse. It matters because failing to
    is silent: prompts that share no blocks still complete, still report 0
    failed, and just do 26x the prefill the recorded workload did. The trace's
    own `reusable_tokens` (see `cc_traces.py --out ...meta.json`) is what the
    delta should be compared against.
    """
    try:
        with urllib.request.urlopen(base + "/debug/cache_stats",
                                    timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:  # noqa: BLE001 - absent on a server built without it
        return {}


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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", default=None,
                   help="defaults to whatever /v1/models reports")
    p.add_argument("--trace", help="JSONL of {arrival_s, input_tokens, output_tokens}")
    p.add_argument("--num-requests", type=int, default=None,
                   help="how many requests to send. With --trace the default "
                        "is the whole trace; this only ever truncates it, and "
                        "the artifact records that it did. Without a trace it "
                        "is how many synthetic requests to generate "
                        f"(default {_SYNTHETIC_REQUESTS})")
    p.add_argument("--rate", type=float, default=0.0,
                   help="Poisson arrivals per second; 0 means all arrive at once")
    p.add_argument("--input-tokens", type=int, default=128)
    p.add_argument("--output-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--out", required=True)
    p.add_argument("--pace", action="store_true",
                   help="deliver each request when its arrival really comes "
                        "round, instead of declaring it. Use against a real "
                        "engine: a real clock discards a declared arrival, so "
                        "without this the real side answers a burst while the "
                        "simulated side answers the trace")
    p.add_argument("--time-scale", type=float, default=1.0,
                   help="divide every arrival offset by this, to replay a "
                        "long trace in less time. 1.0 keeps the trace's own "
                        "timing; it changes how much requests batch, so it is "
                        "a property of the workload and not a free knob")
    p.add_argument("--ignore-eos", action="store_true",
                   help="generate exactly --output-tokens rather than at most. "
                        "A trace records how many tokens a request produced, "
                        "and max_tokens is only a ceiling: a row asking for 376 "
                        "may stop at 3 on an EOS the recorded run never hit, "
                        "and the replay quietly stops being the workload while "
                        "still reporting 0 failed")
    p.add_argument("--check-lengths", action="store_true",
                   help="compare the server's reported prompt_tokens against "
                        "what was asked for, and warn if they differ")
    p.add_argument("--clients", type=int, default=0,
                   help="run closed-loop as this many users instead of "
                        "replaying the trace's arrivals. Each user holds one "
                        "session until every request in it -- sub-agents "
                        "included -- has finished, then takes the next. The "
                        "trace's inter-arrival gaps are dropped; its recorded "
                        "in-session overlap is kept. This is the mode that "
                        "produces a saturation curve: an open-loop replay "
                        "delivers the trace's own rate whatever N is")
    p.add_argument("--sessions-per-client", type=int, default=0,
                   help="with --clients, how many sessions each user works "
                        "through. 0 means every session in the trace. Fixing "
                        "it is what makes a sweep over client counts run "
                        "comparable amounts of work per user")
    args = p.parse_args()

    workload = _workload(args)
    if not workload:
        print("empty workload", file=sys.stderr)
        return 2
    base = f"http://{args.host}:{args.port}"
    model = args.model or _served_model(base, args.timeout)
    if model is None:
        print("could not determine the served model; pass --model",
              file=sys.stderr)
        return 2

    cache_before = _cache_stats(base, args.timeout)
    began = _time.monotonic()

    # Closed loop has no arrival process at all: a request is sent because the
    # one before it came back. So neither compass field is sent -- declaring an
    # arrival would contradict the loop, and declaring a workload size would
    # deadlock, since the barrier holds every request until all `size` of them
    # are waiting and a closed loop never has more than `clients` in flight.
    closed = args.clients > 0

    def one(i_row):
        i, row = i_row
        at = row["arrival_s"]  # already scaled when the workload was built
        if args.pace and not closed:
            # Hold the request until its moment really comes round. Against a
            # real engine this is what makes the arrival process real: the
            # queue is genuinely empty between arrivals, which declaring an
            # arrival cannot achieve -- a declared workload is posted up front
            # and sits in `waiting`, so the scheduler always sees a full queue
            # however the arrivals are stamped.
            delay = at - (_time.monotonic() - began)
            if delay > 0:
                _time.sleep(delay)
        body = {
            "model": model,
            "prompt": _prompt(row, i),
            "max_tokens": row["output_tokens"],
            "temperature": 0.0,
        }
        if not closed:
            body["compass_workload_size"] = len(workload)
        if args.ignore_eos:
            body["ignore_eos"] = True
        if not args.pace and not closed:
            # Declared rather than delivered: against a simulated engine the
            # wall clock and the virtual clock race, so the arrival is stated
            # and the engine honours it. Ignored by a server on a real clock,
            # which is why --pace exists for that side.
            body["compass_arrival"] = at
        try:
            return {"index": i, "ok": True, "response": _send(base + "/v1/completions",
                                                              body, args.timeout)}
        except (urllib.error.URLError, OSError, ValueError, RuntimeError) as exc:
            return {"index": i, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # Posted concurrently and as fast as the socket allows: *when* each lands is
    # deliberately not the arrival the engine uses.
    # One thread per request, in both modes, for two different reasons.
    #
    # Paced: each thread spends its wait sleeping, so a pool of 64 would
    # serialise the 65th arrival behind an earlier request's *generation*
    # rather than behind its arrival.
    #
    # Declared: the server holds every declared request until all of them have
    # arrived, so all of them must be in flight at once. A pool of 64 against a
    # 300-request workload is a deadlock -- 64 threads each blocked on a
    # response the server will not produce until 300 have been posted. It
    # resolves only when the arrival barrier times out, and then the run is not
    # the workload that was asked for: requests enter as earlier ones complete,
    # which is not the declared arrival process. This happened, went unnoticed
    # because the client still reported "0 failed", and a day's conclusions
    # were drawn from the result.
    assignment = None
    if closed:
        # No bound needed here: a closed loop holds at most one session per
        # client, and a session's own recorded peak concurrency is at most 23.
        results, assignment = _closed_loop(
            workload, args.clients, args.sessions_per_client,
            lambda i: one((i, workload[i])))
        if not results:
            print("closed loop executed no requests: the trace has fewer "
                  "sessions than it has clients, so some clients were dealt "
                  "nothing", file=sys.stderr)
            return 2
    else:
        workers = len(workload)
        if workers > MAX_IN_FLIGHT:
            raise SystemExit(
                f"{workers} requests needs {workers} concurrent connections, "
                f"over the {MAX_IN_FLIGHT} this client will open. A declared "
                f"workload cannot be posted in batches -- the server waits for "
                f"all of it before it starts -- so this needs a bulk "
                f"submission endpoint rather than a larger pool. Use "
                f"--num-requests to bound the workload meanwhile.")
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            results = list(pool.map(one, enumerate(workload)))

    failed = [r for r in results if not r["ok"]]

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
    cache_after = _cache_stats(base, args.timeout)
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
    manifest = {
        "revision": _revision(),
        "paced": bool(args.pace) and not closed,
        # A closed-loop artifact must be able to say so on its own. Its
        # `arrival_span_s` is the trace's, not the run's, and reading one as an
        # open-loop replay would attribute the trace's arrival rate to a run
        # that ignored it.
        "closed_loop": closed,
        "clients": int(args.clients) if closed else 0,
        "sessions_run": (sum(len(a) for a in assignment) if assignment else 0),
        "sessions_per_client": ([len(a) for a in assignment] if assignment
                                else None),
        "requests_executed": len(results),
        "time_scale": float(args.time_scale),
        "requests": len(workload),
        "arrival_span_s": (round(workload[-1]["arrival_s"], 6)
                           if workload else 0.0),
        "trace": args.trace,
        "trace_sha256": _digest(args.trace),
        # What was on disk, beside what ran. The sha256 above names the whole
        # file however few of its rows were sent, so without these two a
        # truncated replay is indistinguishable from a complete one.
        "trace_rows": getattr(args, "trace_rows", None),
        "trace_truncated": (args.trace is not None
                            and getattr(args, "trace_rows", None) is not None
                            and len(workload) < args.trace_rows),
        "model": model,
        "failed": len(failed),
        "prompt_lengths": length_check,
        "ignore_eos": bool(args.ignore_eos),
        # How many rows carried sharing, so a run with no reuse can be told
        # apart from a trace that never asked for any.
        "rows_with_hash_ids": sum(1 for r in workload if r.get("hash_ids")),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"run": manifest, "workload": workload, "results": results,
                   "engine": engine,
                   "cache_stats": {"before": cache_before, "after": cache_after}},
                  fh, indent=1)
    if closed:
        print(f"{args.clients} clients ran {manifest['sessions_run']} sessions, "
              f"{len(results)} requests, {len(failed)} failed -> {args.out}")
    else:
        print(f"sent {len(workload)} requests, {len(failed)} failed -> {args.out}")
    if failed:
        print("  first failure:", failed[0]["error"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

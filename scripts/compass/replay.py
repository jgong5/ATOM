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
import time as _time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


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
        return [{"arrival_s": float(r.get("arrival_s", 0.0)) - base,
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


#: Most requests this client will hold open at once. One thread each, so this
#: is a thread count as much as a connection count. Not a tuning knob: past it
#: the declared-arrival protocol needs a bulk submission the server does not
#: have, and quietly posting fewer would reintroduce the deadlock this bounds.
MAX_IN_FLIGHT = 1024


def _prompt(tokens: int, index: int) -> str:
    """Distinct text of exactly the requested token count.

    See `atom.compass.workload`, which run.py's sweep shares: a sweep that
    cannot target a token count cannot bracket a workload measured in tokens.
    """
    from atom.compass.workload import prompt_of_tokens

    return prompt_of_tokens(tokens, index)


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
    p.add_argument("--num-requests", type=int, default=64)
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
    p.add_argument("--check-lengths", action="store_true",
                   help="compare the server's reported prompt_tokens against "
                        "what was asked for, and warn if they differ")
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

    began = _time.monotonic()

    def one(i_row):
        i, row = i_row
        at = row["arrival_s"] / args.time_scale
        if args.pace:
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
            "prompt": _prompt(row["input_tokens"], i),
            "max_tokens": row["output_tokens"],
            "temperature": 0.0,
            "compass_workload_size": len(workload),
        }
        if not args.pace:
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
    workers = len(workload)
    if workers > MAX_IN_FLIGHT:
        raise SystemExit(
            f"{workers} requests needs {workers} concurrent connections, over "
            f"the {MAX_IN_FLIGHT} this client will open. A declared workload "
            f"cannot be posted in batches -- the server waits for all of it "
            f"before it starts -- so this needs a bulk submission endpoint "
            f"rather than a larger pool. Use --num-requests to bound the "
            f"workload meanwhile.")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(one, enumerate(workload)))

    failed = [r for r in results if not r["ok"]]

    # What the server says it received, against what was asked for. The builder
    # is exact by construction under a tokenizer giving one token per word in
    # `_WORDS`; this checks the assumption held, and costs nothing because the
    # count is already in every response.
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
            print(f"  WARNING: {len(off)} of {len(results)} prompts were not the "
                  f"requested length; worst asked {worst[0]} got {worst[1]}",
                  file=sys.stderr)
        else:
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

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"workload": workload, "results": results, "engine": engine},
                  fh, indent=1)
    print(f"sent {len(workload)} requests, {len(failed)} failed -> {args.out}")
    if failed:
        print("  first failure:", failed[0]["error"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

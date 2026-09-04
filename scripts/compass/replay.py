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


def _prompt(tokens: int, index: int) -> str:
    """Distinct text of roughly the requested token count.

    Distinct because identical prompts share prefix-cache blocks and every
    request after the first would skip prefill -- a real behaviour, but not the
    one being measured.
    """
    return f"Request {index}. " + " ".join(f"w{index}x{j}" for j in range(tokens))


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

    def one(i_row):
        i, row = i_row
        body = {
            "model": model,
            "prompt": _prompt(row["input_tokens"], i),
            "max_tokens": row["output_tokens"],
            "temperature": 0.0,
            # The two fields that make this a declared workload rather than a
            # raced one. Both are ignored by a server on a wall clock.
            "compass_arrival": row["arrival_s"],
            "compass_workload_size": len(workload),
        }
        try:
            return {"index": i, "ok": True, "response": _send(base + "/v1/completions",
                                                              body, args.timeout)}
        except (urllib.error.URLError, OSError, ValueError, RuntimeError) as exc:
            return {"index": i, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # Posted concurrently and as fast as the socket allows: *when* each lands is
    # deliberately not the arrival the engine uses.
    with ThreadPoolExecutor(max_workers=min(64, len(workload))) as pool:
        results = list(pool.map(one, enumerate(workload)))

    failed = [r for r in results if not r["ok"]]
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

"""A benchmark client that knows the server may be running on a virtual clock.

`benchmark_serving` times an HTTP stream with `perf_counter` and divides every
metric by its own wall-clock duration. Against a Compass server that measures
the simulator: the forward is predicted rather than performed, so the answers
come back several times faster than the system they stand for. Its numbers are
real and describe the wrong thing.

Two differences here, and both are about whose clock is authoritative.

*Arrival.* A simulated engine advances time only when it takes a step, so it
does not track the wall clock a client sends on. Every request therefore lands
at the same instant and its TTFT is measured from the start of the run rather
than from when it turned up -- worth 293ms of a 293ms error on the workload this
was built against. So the schedule is computed once and applied two ways: paced
in real time against a real server, and *declared* via ``compass_arrival``
against a simulated one, which stamps the arrival the engine will measure from.

*Measurement.* Results are read from ``GET /compass/requests``, which reports
what the engine measured on whichever clock it was running, rather than from
this process's stopwatch.

    python scripts/compass/serve_bench.py --base-url http://localhost:8000 \
        --model Qwen/Qwen3-0.6B --num-prompts 64 --request-rate 8
    python scripts/compass/serve_bench.py ... --declare-arrivals   # Compass server

CLOSED WORKLOADS ONLY. Requests are submitted concurrently and so reach the
engine out of declared order; an idle engine that jumped to the earliest arrival
it had *seen* could be overtaken by an earlier one still in flight, which then
looks retroactively late. So the client tells the engine how many requests are
coming and the engine runs nothing until all of them have arrived -- a one-off
wait bounded by submission, not a per-step sleep. An open-ended server has no
count to wait for and needs the schedule handed over up front instead.
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
import time

try:
    import aiohttp
except ImportError:  # pragma: no cover
    sys.exit("aiohttp is required: it is what the ATOM benchmarks already use")


def _arrival_schedule(n: int, rate: float, seed: int) -> list[float]:
    """Poisson arrivals, in seconds from the start of the run.

    ``inf`` means everything at once, which is the saturated case: it makes the
    run server-bound, and a paced run is arrival-bound and cannot tell a fast
    server from a slow one.
    """
    if rate == float("inf"):
        return [0.0] * n
    rng = random.Random(seed)
    out, t = [], 0.0
    for _ in range(n):
        out.append(t)
        t += rng.expovariate(rate)
    return out


def _prompts(model: str, n: int, length: int, seed: int) -> list[str]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    rng = random.Random(seed)
    vocab = tok.vocab_size
    # Decoded from random ids so each prompt is a distinct, uncacheable string
    # of about the right length. Prefix caching would otherwise make later
    # requests free and the comparison meaningless.
    return [tok.decode([rng.randint(100, vocab - 100) for _ in range(length)])
            for _ in range(n)]


async def _one(session, url, model, prompt, out_len, arrival, declare, total):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": out_len,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
    }
    if declare:
        body["compass_arrival"] = arrival
        # The engine may not advance virtual time past an arrival it has not
        # been told about, and these are submitted concurrently, so they reach
        # it out of order. Told the total, it holds until all have landed.
        body["compass_workload_size"] = total
    started = time.perf_counter()
    async with session.post(url, json=body) as resp:
        await resp.read()
    return time.perf_counter() - started


async def _run(args, prompts, schedule) -> float:
    url = args.base_url.rstrip("/") + "/v1/completions"
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    began = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for prompt, arrival in zip(prompts, schedule):
            if not args.declare_arrivals and arrival > 0:
                # Paced: the real server's arrivals are made real by waiting.
                delay = arrival - (time.perf_counter() - began)
                if delay > 0:
                    await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(
                _one(session, url, args.model, prompt, args.output_len,
                     arrival, args.declare_arrivals, len(prompts))))
        await asyncio.gather(*tasks)
    return time.perf_counter() - began


def _report(args, wall: float, engine: dict) -> int:
    rows = engine.get("requests", [])
    clock = engine.get("clock", "?")
    ttft = [r["ttft"] for r in rows if r.get("ttft") is not None]
    lat = [r["latency"] for r in rows if r.get("latency") is not None]

    print("=" * 70)
    print(f"  {len(rows)} requests, engine clock: {clock}")
    print(f"  arrivals: {'declared' if args.declare_arrivals else 'paced'} "
          f"at rate {args.request_rate}")
    print("=" * 70)
    if not lat:
        print("  No engine-side timings. Is /compass/requests on this server?")
        return 1

    negative = [t for t in ttft if t < 0]
    print("\n  ENGINE-SIDE (simulated time when the clock is virtual)")
    for name, vals in (("TTFT", ttft), ("latency", lat)):
        if vals:
            print(f"    {name:<8} mean {statistics.fmean(vals)*1000:8.2f}ms  "
                  f"median {statistics.median(vals)*1000:8.2f}  "
                  f"max {max(vals)*1000:8.2f}")
    print(f"\n  wall clock for the run itself: {wall*1000:.0f}ms "
          f"({'the simulator' if clock == 'virtual' else 'the system'})")
    if negative:
        print(f"\n  {len(negative)} requests finished before their declared "
              f"arrival.")
        print("  The engine does not yet hold a request until its arrival, so a")
        print("  declared schedule ahead of the engine's clock is not honoured.")
    print("=" * 70)
    (args.out and open(args.out, "w").write(json.dumps(
        {"clock": clock, "wall": wall, "requests": rows}, indent=2)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--input-len", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--request-rate", default="inf",
                    help="Poisson arrivals per second, or 'inf' for all at once")
    ap.add_argument("--declare-arrivals", action="store_true",
                    help="Send each request's arrival as compass_arrival "
                         "instead of pacing in real time. For a Compass server.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.request_rate = float(args.request_rate)

    prompts = _prompts(args.model, args.num_prompts, args.input_len, args.seed)
    schedule = _arrival_schedule(args.num_prompts, args.request_rate, args.seed)

    import urllib.request
    # Drain anything a warmup left behind, so the report covers this run only.
    urllib.request.urlopen(args.base_url.rstrip("/") + "/compass/requests").read()

    wall = asyncio.run(_run(args, prompts, schedule))
    with urllib.request.urlopen(
            args.base_url.rstrip("/") + "/compass/requests") as fh:
        engine = json.loads(fh.read())
    return _report(args, wall, engine)


if __name__ == "__main__":
    sys.exit(main())

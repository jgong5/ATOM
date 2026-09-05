"""Where a step's time actually goes, from a profile of the real run.

The cost model predicts a step as priced kernels plus modelled overhead, and
lands a few percent under the measured step every time. That residual has been
carried as one number, which hides that it is two:

* kernels that run slower in the step than they do on their own, and
* time when the device is running no kernel at all.

The first is a pricing problem and the second is not -- it is the gap between
launches, which the cost model *models* (`max(0, dispatch - kernel)`) rather
than measures. A profile has both directly: the step is a `gpu_user_annotation`
spanning it on the device timeline, and the kernels inside are its own events.
Summing them and subtracting says how much of the step was idle, without
assuming a dispatch constant.

    python scripts/compass/step_accounting.py <trace-dir-or-file> [--match prefill]
    ... [--prices p.json --graph g.json]   # also compare against what was priced
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import sys


def load_events(path: str) -> list:
    files = ([path] if os.path.isfile(path)
             else sorted(glob.glob(os.path.join(path, "**", "*.json*"),
                                   recursive=True)))
    if not files:
        raise SystemExit(f"no trace files under {path!r}")
    events = []
    for f in files:
        opener = gzip.open if f.endswith(".gz") else open
        with opener(f, "rt") as fh:
            events += json.load(fh).get("traceEvents", [])
    return events


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--match", default="prefill",
                    help="substring of the step annotation to account for")
    ap.add_argument("--prices")
    ap.add_argument("--graph")
    args = ap.parse_args()

    events = load_events(args.trace)
    steps = [e for e in events
             if e.get("cat") == "gpu_user_annotation"
             and args.match in str(e.get("name", ""))]
    if not steps:
        names = {str(e.get("name"))[:60] for e in events
                 if e.get("cat") == "gpu_user_annotation"}
        raise SystemExit(f"no step matched {args.match!r}; saw {sorted(names)[:8]}")
    step = max(steps, key=lambda e: float(e.get("dur", 0)))
    begin = float(step["ts"])
    end = begin + float(step["dur"])
    # One device. Under tensor parallelism every rank writes its own kernels
    # into the trace against its own pid, and summing across them counts two
    # GPUs' work against one GPU's window -- which came out as *negative* idle,
    # the arithmetic saying plainly that the input was wrong.
    device = step.get("pid")
    ranks = {e.get("pid") for e in steps}
    print(f"step: {step['name']}")
    if len(ranks) > 1:
        print(f"  {len(ranks)} ranks in this trace; accounting for pid {device}")
    print(f"  device window {float(step['dur']) / 1e3:.3f} ms")

    # Kernels of this step, by their own device timestamps. A kernel that
    # straddles the boundary is counted where it started, which matters for at
    # most one at each end.
    inside = [e for e in events
              if e.get("cat") in ("kernel", "gpu_memcpy")
              and e.get("pid") == device
              and begin <= float(e.get("ts", -1)) < end]
    busy = sum(float(e.get("dur", 0)) for e in inside)
    window = float(step["dur"])
    if busy > window:
        raise SystemExit(
            f"kernels sum to {busy / 1e3:.3f} ms inside a {window / 1e3:.3f} ms "
            "window, which is impossible on one device -- the trace is mixing "
            "devices or streams that run concurrently, and every number below "
            "would be wrong rather than merely imprecise")
    print(f"  {len(inside)} kernels running {busy / 1e3:.3f} ms")
    print(f"  idle between them {(window - busy) / 1e3:.3f} ms "
          f"({(window - busy) / window * 100:.1f}% of the step)")
    if len(inside):
        print(f"  average gap per kernel "
              f"{(window - busy) / len(inside):.2f} us")

    # What the idle actually looks like. The cost model charges an operator
    # `max(0, dispatch - kernel)`, which says a long kernel hides the gap after
    # it. If the gaps are the same size whatever ran before them, that model is
    # wrong in shape and not only in constant.
    ordered = sorted(inside, key=lambda e: float(e["ts"]))
    gaps = []
    for before, after in zip(ordered, ordered[1:]):
        finished = float(before["ts"]) + float(before["dur"])
        gaps.append((float(after["ts"]) - finished, float(before["dur"])))
    if gaps:
        sizes = sorted(g for g, _ in gaps)
        pick = lambda q: sizes[min(len(sizes) - 1, int(q * len(sizes)))]
        print(f"\n  gaps between consecutive kernels, {len(gaps)} of them")
        print(f"    median {pick(0.5):.2f} us   mean "
              f"{sum(sizes) / len(sizes):.2f} us   "
              f"p90 {pick(0.9):.2f} us   max {sizes[-1]:.2f} us")
        print(f"\n  {'kernel that ran before':>24}   {'gaps':>5} "
              f"{'median gap':>11}")
        edges = [(0, 10), (10, 50), (50, 200), (200, 1000), (1000, 1e9)]
        for low, high in edges:
            chosen = sorted(g for g, d in gaps if low <= d < high)
            if not chosen:
                continue
            label = (f"{low}-{high} us" if high < 1e9 else f"over {low} us")
            print(f"  {label:>24}   {len(chosen):5d} "
                  f"{chosen[len(chosen) // 2]:10.2f} us")
        # A per-launch overhead is many small gaps. A few enormous ones are
        # something else -- a compile, a host stall, an allocation -- and would
        # be modelled away wrongly as a constant charged to every operator.
        largest = sorted(gaps, key=lambda g: -g[0])[:10]
        share = sum(g for g, _ in largest) / max(sum(sizes), 1e-9)
        print(f"\n  the 10 largest gaps are {share * 100:.1f}% of all idle: "
              + ", ".join(f"{g / 1e3:.1f}ms" for g, _ in largest[:6]))
        small = [g for g in sizes if g < 100]
        print(f"  the {len(small)} gaps under 100us total "
              f"{sum(small) / 1e3:.3f} ms "
              f"({sum(small) / max(sum(sizes), 1e-9) * 100:.1f}% of idle)")

    if not (args.prices and args.graph):
        return 0

    sys.path.insert(0, os.getcwd())
    from atom.compass.runtime.microbench import signature_of

    prices = json.load(open(args.prices))["prices"]
    graph = json.load(open(args.graph))
    priced = sum(prices[signature_of(op)]["seconds"]
                 for op in graph["ops"] if signature_of(op) in prices) * 1e6
    # A price taken outside a graph is wall time per call, not kernel time, so
    # it carries launch overhead the in-situ kernel figure does not have.
    # Counting it here would attribute overhead to the kernels.
    captured = sum(prices[signature_of(op)]["seconds"]
                   for op in graph["ops"]
                   if prices.get(signature_of(op), {}).get("cache") == "graph"
                   ) * 1e6
    outside = sum(1 for op in graph["ops"]
                  if prices.get(signature_of(op), {}).get("cache") != "graph")
    print(f"\n  priced in isolation {priced / 1e3:.3f} ms")
    print(f"  same work in situ   {busy / 1e3:.3f} ms")
    print(f"  isolated is {(priced - busy) / busy * 100:+.2f}% of in situ")
    if outside:
        print(f"  ({outside} operators were timed outside a graph, so their "
              f"price carries launch\n   overhead; graph-captured alone is "
              f"{captured / 1e3:.3f} ms)")

    # Per kernel, where the price list recorded a breakdown. A kernel that is
    # slower in situ everywhere is a different finding from a few that are.
    got = collections.defaultdict(float)
    for e in inside:
        got[str(e.get("name"))] += float(e.get("dur", 0))
    want = collections.defaultdict(float)
    for op in graph["ops"]:
        entry = prices.get(signature_of(op))
        for name, seconds in (entry or {}).get("kernels", {}).items():
            want[name] += float(seconds) * 1e6
    both = sorted(set(got) & set(want), key=lambda k: -got[k])
    if not both:
        print("\n  no kernel names in common (breakdowns are off under "
              "parallelism -- see PRICE_KERNELS)")
        return 0
    print(f"\n  {len(both)} kernels priced and profiled by name, "
          f"slowest first")
    print(f"  {'in situ':>10} {'isolated':>10} {'diff':>8}  kernel")
    for name in both[:16]:
        diff = (want[name] - got[name]) / got[name] * 100
        print(f"  {got[name] / 1e3:9.3f}ms {want[name] / 1e3:9.3f}ms "
              f"{diff:+7.1f}%  {name[:52]}")
    covered = sum(got[k] for k in both)
    print(f"  those cover {covered / busy * 100:.1f}% of the step's kernel time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

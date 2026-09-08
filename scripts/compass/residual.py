"""Split a replayed step's residual into boundary and pricing error.

`step - priced` has been carried as one number and charged entirely to a
per-launch boundary. It is two things, and a profile of the same configuration
separates them without needing to assume either:

    step  =  device busy  +  idle between kernels  +  work outside the step's
                                                      own annotation
    residual  =  (busy - priced)   +   idle    +  outside
                  pricing error       boundary

The boundary is *measured* here rather than fitted -- it is the idle, which a
trace gives directly. Two properties make that sound.

* Idle survives the profiler. Profiling inflates every kernel's reported
  duration, which inflates the kernel sum and the step's window by the same
  amount, and idle is their difference. So an idle read off a profiled trace may
  be compared against a step measured without one.
* It is falsifiable at the scale in question. If each launch paid the modelled
  2.25us, a step would show hundreds of gaps in the 1-5us band. Whether it does
  is visible in `step_accounting`'s gap histogram.

`--prices`/`--graph` and the trace must come from **one machine**. The pricing
error differs between boxes by more than it differs between tensor-parallel
widths -- measured at -2.1% on one and +10.0% on another for the same model at
the same width -- so a residual assembled from two machines is measuring the
machines.

    python scripts/compass/residual.py <trace-dir> --graph g.json --prices p.json
    ... --steps measure.jsonl [--profile-steps other.jsonl] [--by-kernel]
"""
from __future__ import annotations

import argparse
import bisect
import collections
import glob
import gzip
import json
import os
import statistics
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


def decode_steps(path: str, prefill: bool = False) -> list:
    """Replayed steps of one kind, in the order the table recorded them."""
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return [r["seconds"] * 1e6 for r in rows
            if (((r.get("num_prefill_tokens") or 0) > 0) == prefill)
            and r.get("capture_bucket") is not None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--steps", required=True,
                    help="measure table from the run that produced the prices")
    ap.add_argument("--profile-steps",
                    help="measure table of the *profiled* run. Its rows from "
                         "before the profiler opened give the profiler's cost "
                         "per dispatch within one process, which is what makes "
                         "the per-kernel comparison honest")
    ap.add_argument("--match", default="decode")
    ap.add_argument("--by-kernel", action="store_true",
                    help="also show which kernels carry the pricing error")
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    from atom.compass.core.cost.priced import HOST_SYNC
    from atom.compass.runtime.microbench import signature_of

    events = load_events(args.trace)
    steps = [e for e in events
             if e.get("cat") == "gpu_user_annotation"
             and args.match in str(e.get("name", ""))
             and not str(e.get("name", "")).startswith("dummy")]
    if not steps:
        raise SystemExit(f"no {args.match!r} annotation in {args.trace!r}")
    # One device: under tensor parallelism every rank writes kernels against its
    # own pid, and summing across them counts several GPUs against one window.
    pid = max({e.get("pid") for e in steps},
              key=lambda p: sum(1 for e in steps if e.get("pid") == p))
    mine = sorted((e for e in steps if e.get("pid") == pid),
                  key=lambda e: float(e["ts"]))
    kernels = sorted((e for e in events
                      if e.get("cat") in ("kernel", "gpu_memcpy")
                      and e.get("pid") == pid), key=lambda e: float(e["ts"]))
    starts = [float(e["ts"]) for e in kernels]

    idles, counts, windows = [], [], []
    got, hits = collections.defaultdict(float), collections.Counter()
    for step in mine:
        begin = float(step["ts"])
        end = begin + float(step["dur"])
        inside = kernels[bisect.bisect_left(starts, begin):
                         bisect.bisect_left(starts, end)]
        busy = sum(float(k.get("dur", 0)) for k in inside)
        windows.append(float(step["dur"]))
        idles.append(float(step["dur"]) - busy)
        counts.append(len(inside))
        for k in inside:
            got[str(k.get("name"))] += float(k.get("dur", 0))
            hits[str(k.get("name"))] += 1
    idle = statistics.median(idles)
    nkern = int(statistics.median(counts))

    graph = json.load(open(args.graph))
    prices = json.load(open(args.prices))
    prices = prices.get("prices", prices)
    priced = 0.0
    launches = 0
    for op in graph["ops"]:
        # `launch` marks a custom operator's inner kernels, which the graph
        # holds alongside the operator that launched them; pricing both counts
        # the same kernel twice. `HOST_SYNC` operators run no kernel and their
        # price is a host wait, which a saturated step does not pay for.
        if op.get("launch") or op.get("name", "") in HOST_SYNC:
            continue
        entry = prices.get(signature_of(op))
        if entry is None:
            continue
        priced += float(entry.get("seconds", 0.0)) * 1e6
        launches += max(1, len(entry.get("kernels") or {}))

    measured = statistics.median(decode_steps(
        args.steps, prefill=args.match.startswith("prefill")))
    residual = measured - priced

    print(f"step: {mine[0]['name']}  ({len(mine)} of them on pid {pid})")
    print(f"  priced sum (work only)   {priced / 1e3:8.3f} ms  "
          f"over {launches} launches")
    print(f"  measured step            {measured / 1e3:8.3f} ms  "
          f"(no profiler, same run as the prices)")
    print(f"  residual                 {residual / 1e3:+8.3f} ms "
          f"({residual / measured * 100:+.1f}% of the step)")
    print(f"  of which boundary        {idle / 1e3:+8.3f} ms  measured, as idle "
          f"between {nkern} kernels")
    print(f"                           = {idle * 1e3 / max(nkern, 1):.1f} ns per "
          f"launch")
    print(f"  of which pricing error   {(residual - idle) / 1e3:+8.3f} ms  "
          f"({(residual - idle) / priced * 100:+.1f}% of the priced sum)")

    if not args.by_kernel:
        return 0

    delta = 0.0
    if args.profile_steps:
        seconds = decode_steps(args.profile_steps,
                               prefill=args.match.startswith("prefill"))
        shut = seconds[:max(0, len(seconds) - len(mine))]
        if shut:
            delta = ((statistics.median(windows) - statistics.median(shut))
                     / max(nkern, 1))
            print(f"\n  profiling cost {delta:.3f} us per dispatch, from the "
                  f"{len(shut)} rows that ran before it opened")
    if not delta:
        print("\n  no --profile-steps with unprofiled rows: in-situ times below "
              "are\n  profiled, and so overstated, which flatters the prices")

    want = collections.defaultdict(float)
    for op in graph["ops"]:
        entry = prices.get(signature_of(op))
        for name, seconds in (entry or {}).get("kernels", {}).items():
            want[name] += float(seconds) * 1e6
    n = len(mine)
    both = sorted(set(got) & set(want), key=lambda k: -got[k])
    if both:
        print(f"\n  {'in situ':>9} {'corrected':>10} {'priced':>9} {'error':>8}"
              f"  kernel")
        for name in both[:12]:
            situ = got[name] / n
            true = situ - hits[name] / n * delta
            err = (want[name] - true) / true * 100 if true > 0 else float("nan")
            print(f"  {situ / 1e3:8.3f}ms {true / 1e3:9.3f}ms "
                  f"{want[name] / 1e3:8.3f}ms {err:+7.1f}%  {name[:44]}")
    # A kernel with no priced breakdown is not unpriced -- its operator has a
    # price -- but it cannot be checked by name, so an error in it hides in the
    # total. Under parallelism the breakdowns are sparse, and that is where the
    # residual has been found to live, so what cannot be checked is worth as
    # much space as what can.
    missing = sorted((k for k in got if k not in want), key=lambda k: -got[k])
    if missing:
        unchecked = sum(got[k] - hits[k] * delta for k in missing) / n
        checked = sum(got[k] - hits[k] * delta for k in both) / n
        print(f"\n  no priced breakdown to check: {unchecked / 1e3:.3f} ms a "
              f"step ({unchecked / max(unchecked + checked, 1e-9) * 100:.0f}% "
              f"of kernel time)")
        for name in missing[:8]:
            situ = got[name] / n
            print(f"  {situ / 1e3:8.3f}ms {(situ - hits[name] / n * delta) / 1e3:9.3f}ms "
                  f"{hits[name] / n:7.0f}x  {name[:44]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

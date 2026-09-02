"""Does a priced kernel cost what it costs in a step?

Open problem 21. Every kernel measured so far comes out cheaper on its own than
the same kernel inside a forward, and the whole priced step lands about a
quarter under the replayed one. Whether that ratio is *constant* decides the
fix: constant means one calibration factor and the mechanism can stay
unexplained, while varying by kernel class means a factor is wrong and the
mechanism has to be found. Seven kernels could not tell those apart. This
measures it for every operator the price list reaches.

The in-situ side comes from a profile of the same workload the price list was
built from, attributed back from kernels to the operator that launched them:
each kernel event carries a correlation id shared with its launch, and the
launch sits inside the operator's own interval on the same thread.

**Two profiles, not one.** A single trace holds prefill as well as decode, and
the same operator name appears in both at different shapes. Differencing a long
run against a short one cancels prefill exactly -- it happens once in each --
and leaves whole decode steps:

    per step = (long_total - short_total) / (long_steps - short_steps)

    python scripts/compass/price_check.py prices.json \\
        --trace short/:8 --trace long/:32 [--step-seconds 0.003115]
"""

import argparse
import collections
import glob
import gzip
import json
import os
import sys


def _events(path: str) -> list:
    """Trace events from a file or from the newest trace under a directory."""
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, "**", "*.json*"),
                                 recursive=True), key=os.path.getmtime)
        if not found:
            raise SystemExit(f"no trace under {path}")
        path = found[-1]
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh).get("traceEvents", [])


def _launches_by_kernel(events: list) -> dict[str, int]:
    """How many times each kernel ran, which the additive fit needs."""
    count: dict[str, int] = collections.defaultdict(int)
    for e in events:
        if e.get("cat") in ("kernel", "Kernel"):
            count[e.get("name", "")] += 1
    return dict(count)


def _device_time_by_kernel(events: list) -> dict[str, float]:
    """Kernel time by kernel name, in microseconds.

    By kernel and not by operator, because a decode step is a replayed CUDA
    graph: the whole step is one host call, so the trace carries kernels with no
    operator around them to charge them to. The price list records which kernels
    each operator launched, for exactly this reason.
    """
    total: dict[str, float] = collections.defaultdict(float)
    for e in events:
        if e.get("cat") in ("kernel", "Kernel"):
            total[e.get("name", "")] += float(e.get("dur", 0.0))
    return dict(total)


def _priced_by_kernel(prices: dict) -> tuple[dict[str, float], list[str]]:
    """Apportion each operator's price across the kernels it launches.

    The measured price is the number that matters, so it is divided in the
    proportions the kernel breakdown gives rather than replaced by the
    breakdown's own durations -- which come from one eager call and carry its
    cold start.
    """
    priced: dict[str, float] = collections.defaultdict(float)
    missing = []
    for sig, entry in prices["prices"].items():
        kernels = entry.get("kernels") or {}
        total_us = entry["seconds"] * entry["occurrences"] * 1e6
        if not kernels:
            missing.append(entry["name"])
            continue
        share_total = sum(kernels.values()) or 1.0
        for name, seconds in kernels.items():
            priced[name] += total_us * (seconds / share_total)
    return dict(priced), missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prices")
    parser.add_argument("--trace", action="append", required=True,
                        metavar="PATH:DECODE_STEPS",
                        help="a profile and how many decode steps it ran; "
                             "give two, so prefill cancels")
    parser.add_argument("--step-seconds", type=float, default=0.0,
                        help="the measured replayed step, for the whole-step "
                             "ratio")
    args = parser.parse_args()

    if len(args.trace) != 2:
        print("two traces are needed so that prefill cancels", file=sys.stderr)
        return 2

    runs = []
    for spec in args.trace:
        path, _, steps = spec.rpartition(":")
        events = _events(path)
        runs.append((_device_time_by_kernel(events),
                     _launches_by_kernel(events), int(steps)))
    runs.sort(key=lambda r: r[2])
    (short, short_n, short_steps), (long_, long_n, long_steps) = runs
    if long_steps == short_steps:
        print("the two traces must differ in decode steps", file=sys.stderr)
        return 2

    span = long_steps - short_steps
    in_situ = {name: (long_.get(name, 0.0) - short.get(name, 0.0)) / span
               for name in set(long_) | set(short)}
    launches = {name: (long_n.get(name, 0) - short_n.get(name, 0)) / span
                for name in set(long_n) | set(short_n)}

    prices = json.loads(open(args.prices, encoding="utf-8").read())
    priced, missing = _priced_by_kernel(prices)

    print("=" * 78)
    print("  what a priced kernel costs, against the same kernel in a step")
    print(f"  in situ = ({long_steps} steps - {short_steps} steps) / {span}")
    print("=" * 78)
    print(f"    {'kernel':<48} {'priced':>9} {'in situ':>9} {'ratio':>7}")
    rows = sorted(priced.items(), key=lambda kv: -kv[1])
    comparable = []
    for name, us in rows:
        actual = in_situ.get(name, 0.0)
        if actual > 0.5 and us > 0.5:
            comparable.append((name, us, actual))
            ratio = f"{us / actual:>7.3f}"
        else:
            ratio = "      -"
        print(f"    {name[:48]:<48} {us:>8.1f}u {actual:>8.1f}u {ratio}")

    if comparable:
        total_priced = sum(p for _, p, _ in comparable)
        total_actual = sum(a for _, _, a in comparable)
        ratios = sorted(p / a for _, p, a in comparable)
        print(f"\n    {'-- comparable kernels':<48} "
              f"{total_priced:>8.1f}u {total_actual:>8.1f}u "
              f"{total_priced/total_actual:>7.3f}")
        print(f"    {len(ratios)} kernels, ratio {ratios[0]:.3f} to "
              f"{ratios[-1]:.3f}, median {ratios[len(ratios)//2]:.3f}")
        spread = ratios[-1] - ratios[0]
        print()
        if spread <= 0.15:
            print("  Tight enough to be one number: a single calibration factor")
            print("  corrects the price list and what is left is the spread.")
        else:
            print("  Wider than a calibration factor can absorb -- one number")
            print("  would be wrong at both ends. But a ratio is the wrong shape")
            print("  to look for if the gap is a fixed cost per kernel: it would")
            print("  vanish on a large kernel and dominate a small one, which is")
            print("  what the spread above looks like. So, additively:")

        # Per kernel *instance*, not per kernel kind: a fixed cost at a kernel
        # boundary is paid once per launch.
        print()
        print(f"    {'kernel':<48} {'n':>5} {'priced':>8} {'in situ':>8} "
              f"{'gap/call':>9}")
        gaps = []
        for name, us, actual in comparable:
            n = launches.get(name, 0.0)
            if n < 0.5:
                continue
            gap = (actual - us) / n
            gaps.append(gap)
            print(f"    {name[:48]:<48} {n:>5.0f} {us/n:>7.2f}u "
                  f"{actual/n:>7.2f}u {gap:>8.2f}u")
        if gaps:
            gaps.sort()
            total_launches = sum(launches.get(n, 0.0)
                                 for n, _, _ in comparable)
            flat = (total_actual - total_priced) / max(total_launches, 1)
            print(f"\n    gap per launch: {gaps[0]:.2f} to {gaps[-1]:.2f}us, "
                  f"median {gaps[len(gaps)//2]:.2f}us, "
                  f"{flat:.2f}us over all {total_launches:.0f} launches")

    if missing:
        print(f"\n  {len(missing)} priced operators carry no kernel breakdown "
              "and cannot be compared.")

    covered = sum(a for _, _, a in comparable) if comparable else 0.0
    everything = sum(in_situ.values())
    if everything > 0:
        print(f"\n  the compared kernels are {100*covered/everything:.1f}% of "
              f"the step's kernel time ({everything:.1f}us).")

    if args.step_seconds > 0:
        whole = sum(priced.values())
        every_launch = sum(launches.values())
        print(f"\n  the whole step, priced        : {whole/1000:.3f}ms")
        print(f"  the replayed step             : {args.step_seconds*1000:.3f}ms")
        print(f"  parts / whole                 : "
              f"{whole/1e6/args.step_seconds:.3f}")
        if every_launch:
            # Fitted against the step rather than against the profile, so no
            # instrument but the engine's own clock is involved. The profiled
            # kernel sum runs above the unprofiled step -- 3.54ms of kernels in
            # a 3.12ms step -- so the per-kernel gaps above are upper bounds and
            # this is the number to use.
            fitted = (args.step_seconds * 1e6 - whole) / every_launch
            print(f"  kernel launches in a step     : {every_launch:.0f}")
            print(f"  fitted cost per launch        : {fitted:.2f}us")
            if comparable and gaps:
                print(f"  (against a profiled median of {gaps[len(gaps)//2]:.2f}us "
                      f"per launch, arrived at independently)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

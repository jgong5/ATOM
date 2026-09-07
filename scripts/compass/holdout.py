"""Does the overhead constant survive being used on a shape it was not taken on?

Every good number this project has reported for the priced-graph oracle was
taken on the very step it was calibrated from. That is a fit residual, not a
prediction, and the difference is the whole question -- the constant is already
known not to transfer between models (9.71 us per launch on the 0.6B against
0.90 on the 27B), so whether it transfers between *shapes* decides if it is a
constant at all.

The kernel-pricing term and the overhead term are validated separately here,
deliberately. A step-level comparison cannot answer this: the overhead is under
half a percent of a prefill step, so any error in the kernel term swamps it.
What is cross-applied is the constant against the idle it claims to predict.

    python scripts/compass/holdout.py <trace-dir> \\
        --shape prefill:g.prefill.json:prices.json \\
        --shape decode:g.json:prices.json
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from step_accounting import load_events  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--shape", action="append", required=True,
                    metavar="MATCH:GRAPH:PRICES",
                    help="a step to measure, as match:graph:prices")
    args = ap.parse_args()

    import json
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    measured = []
    for spec in args.shape:
        match, graph, prices = spec.split(":", 2)
        out = "/tmp/holdout_%s.json" % match
        subprocess.run(
            [sys.executable, os.path.join(here, "step_accounting.py"), args.trace,
             "--match", match, "--graph", graph, "--prices", prices,
             "--calibrate", out],
            check=True, capture_output=True, text=True)
        with open(out, encoding="utf-8") as fh:
            measured.append(json.load(fh))

    print("  %-34s %10s %9s %12s" % ("step", "idle", "launches", "us/launch"))
    for m in measured:
        print("  %-34s %8.3fms %9d %12.4f" % (
            m["step"][:34], m["idle_seconds"] * 1e3, m["launches"],
            m["compiled_seconds_per_launch"] * 1e6))

    print("\n  cross-applied: each constant used on the step it was NOT taken on")
    print("  %-16s %-16s %10s %10s %9s"
          % ("calibrated on", "predicting", "predicted", "actual", "error"))
    for source in measured:
        for target in measured:
            if source is target:
                continue
            predicted = source["compiled_seconds_per_launch"] * target["launches"]
            actual = target["idle_seconds"]
            ratio = predicted / actual if actual else float("inf")
            print("  %-16s %-16s %8.3fms %8.3fms %8.1fx"
                  % (source["step"].split("[")[0], target["step"].split("[")[0],
                     predicted * 1e3, actual * 1e3, ratio))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

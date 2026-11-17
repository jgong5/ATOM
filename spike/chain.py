"""Each decode and prefill worker's main thread from split.py's JSON: pricing, the
wait for the group's slowest rank to price (the all_reduce in the step's charge),
the doorbell (waiting for the engine core's next call), the other collectives,
lookups, host recurrence and the rest, in seconds; and each engine core's waits.

usage: python3 chain.py SPLIT_JSON
"""

import collections
import json
import sys

with open(sys.argv[1]) as f:
    d = json.load(f)


def parts(x):
    out = collections.Counter()
    for k, where, s in x["main_all"]:
        if k == "wait" and "_group_step_seconds.<locals>.charge" in where:
            out["wait: group's slowest pricing (charge all_reduce)"] += s
        elif k == "wait" and where.startswith("Doorbell.spin"):
            out["wait: engine core's next call (doorbell)"] += s
        elif k == "wait" and "_Segment.exchange" in where:
            out["wait: other DP collectives"] += s
        elif k == "wait" and "AsyncIOProcManager.call_func" in where:
            out["wait: its worker"] += s
        elif k == "wait" and "_Remote.get < _Connection.recv" in where:
            out["wait: clock grant"] += s
        elif k == "wait":
            out["wait: other"] += s
        else:
            out[k] += s
    return out


for kind in ("worker-decode", "worker-prefill", "core-decode", "core-prefill"):
    rows = sorted(
        (x["role"], parts(x)) for x in d.values() if x["role"].startswith(kind)
    )
    keys = sorted(
        {k for _, p in rows for k in p}, key=lambda k: -sum(p[k] for _, p in rows)
    )
    print(f"{kind}: seconds of each rank's main thread")
    print(f"  {'part':<52}" + "".join(f"{r.split('-')[-1]:>8}" for r, _ in rows))
    for k in keys:
        print(f"  {k[:52]:<52}" + "".join(f"{p[k]:>8.0f}" for _, p in rows))
    print()

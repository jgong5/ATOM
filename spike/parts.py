"""The named result's split of a run's wall, from split.py's JSON: shares of the
decode workers' sampled main-thread time (the DP group's step chain), means over
the ranks, with the decode cores' clock-grant wait carved out of the workers' wait
for their core, and the prefill workers' lookups and lazy traces beside them.

usage: python3 parts.py SPLIT_JSON
"""

import json
import statistics
import sys

with open(sys.argv[1]) as f:
    d = json.load(f)


def share(role, test):
    xs = [x for x in d.values() if x["role"].startswith(role)]
    return statistics.mean(
        sum(s for k, w, s in x["main_all"] if test(k, w)) / sum(x["main"].values())
        for x in xs
    )


rows = {
    "pricing, own and the wait for the group's slowest rank": share(
        "worker-decode",
        lambda k, w: k == "pricing"
        or (k == "wait" and "step_seconds.<locals>.charge" in w),
    ),
    "decode cores' wait for a clock grant": share(
        "core-decode",
        lambda k, w: k == "wait" and "_Remote.get < _Connection.recv" in w,
    ),
    "workers' wait for their core, less that grant wait": share(
        "worker-decode", lambda k, w: k == "wait" and w.startswith("Doorbell.spin")
    ),
    "other DP collectives": share(
        "worker-decode",
        lambda k, w: k == "wait" and "_Segment.exchange" in w and "charge" not in w,
    ),
    "runner glue and other waits": share(
        "worker-decode",
        lambda k, w: k in ("rest", "clock/IPC shim")
        or (
            k == "wait"
            and "_Segment.exchange" not in w
            and not w.startswith("Doorbell.spin")
        ),
    ),
    "host recurrence": share("worker-decode", lambda k, w: k == "host recurrence"),
    "graph lookups and parsing, decode": share(
        "worker-decode", lambda k, w: k == "lookup"
    ),
    "graph lookups and parsing, prefill": share(
        "worker-prefill", lambda k, w: k == "lookup"
    ),
    "lazy traces, prefill": share("worker-prefill", lambda k, w: k == "lazy trace"),
}
rows["workers' wait for their core, less that grant wait"] -= rows[
    "decode cores' wait for a clock grant"
]
for k, v in rows.items():
    print(f"{v:7.1%}  {k}")

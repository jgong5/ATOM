# SPDX-License-Identifier: MIT
"""Print a real recording's engine IO-thread queue handoffs against its step time.

    PYTHONPATH=$T python3 scripts/compass/handoff.py RECORD_DIR

Reads each ``dp<r>.handoff.jsonl`` a recorded engine writes beside its step
record ``dp<r>.jsonl`` (`atom.compass.run.engine`). Segments: ``recv``, the
input thread's receipt of a request frame to its queue put; ``queue``, that
put to the busy loop's get; ``past pull``, the part of ``queue`` after the
busy loop's idle point in the pass that got it, since a simulated busy loop
also pulls only between steps; all bucketed by the frame's prompt tokens; and
``out``, an output's queue put to its send returning, bucketed by its rows.
A bucket ``b`` holds counts in ``[b, 2b)``. Per segment and bucket it prints
N, p50 and p99 in microseconds, and p99 as a share of the median step, the
period between consecutive forward entries: over the steps whose batch is in
the same bucket for ``out``, over every step for the others. Then whether
every p99 but ``queue``'s is under 1% of its step, and in the smallest prompt
bucket the p10 of ``recv`` + ``queue`` and of ``recv`` + ``past pull`` beside
the run file's ``ipc_s``.
"""

import sys
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np

from atom.compass.backends.json_file import read_jsonl
from atom.compass.drive.runfile import FIXED


def _bucket(n: int) -> int:
    return 1 << (max(n, 1).bit_length() - 1)


def report(record) -> list[str]:
    segments, periods = defaultdict(list), defaultdict(list)
    for path in sorted(Path(record).glob("dp*.handoff.jsonl")):
        for h in read_jsonl(path):
            if "t_recv_ns" in h:
                b, get = _bucket(h["prompt_tokens"]), h["t_in_get_ns"]
                past = get - max(h["t_in_put_ns"], h["t_pull_ns"])
                segments["recv", b].append(h["t_in_put_ns"] - h["t_recv_ns"])
                segments["queue", b].append(get - h["t_in_put_ns"])
                segments["past pull", b].append(past)
                segments["recv+queue", b].append(get - h["t_recv_ns"])
                segments["recv+past pull", b].append(
                    h["t_in_put_ns"] - h["t_recv_ns"] + past
                )
            else:
                b = _bucket(h["rows"])
                segments["out", b].append(h["t_out_sent_ns"] - h["t_out_put_ns"])
        steps = read_jsonl(path.with_name(path.name.replace(".handoff", "")))
        steps = [s for s in steps if not s.get("is_dummy_run")]
        for s, after in pairwise(steps):
            periods[_bucket(len(s["batch"]))].append(
                after["t_enter_ns"] - s["t_enter_ns"]
            )
    every = np.median([p for ps in periods.values() for p in ps])
    rows = ["| segment | bucket | N | p50 us | p99 us | median step us | p99 / step |"]
    rows.append("|---|---|---|---|---|---|---|")
    small = True
    for (seg, b), ns in sorted(segments.items()):
        if seg.startswith("recv+"):
            continue
        step = np.median(periods[b]) if seg == "out" and periods[b] else every
        p50, p99 = np.percentile(ns, [50, 99])
        small &= seg == "queue" or p99 < 0.01 * step
        rows.append(
            f"| {seg} | {b} | {len(ns)} | {p50 / 1e3:.1f} | {p99 / 1e3:.1f} "
            f"| {step / 1e3:.0f} | {p99 / step:.2%} |"
        )
    rows.append(
        f"every p99 but queue's under 1% of its step: {'yes' if small else 'no'}"
    )
    first = min((b for seg, b in segments if seg == "recv"), default=None)
    if first is not None:
        p10s = [
            np.percentile(segments[s, first], 10) / 1e3
            for s in ("recv+queue", "recv+past pull")
        ]
        rows.append(
            f"p10 at prompt bucket {first}: recv+queue {p10s[0]:.1f} us, "
            f"recv+past pull {p10s[1]:.1f} us; ipc_s {FIXED['ipc_s'] * 1e6:.1f} us"
        )
    return rows


if __name__ == "__main__":
    print("\n".join(report(sys.argv[1])))

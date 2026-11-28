# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""walls.py DESIGNED MEASURED CELL...: each short cell's profiling wall, from its aiperf log's
`profiling started` to `sending complete` at ms, per simulated second (120 s); the gap, designed less
measured; and what each other cell's change costs, designed less that cell, and its share of the gap.
A cell whose profiling did not complete has no wall and is left out."""
import datetime as dt
import re
import sys
from pathlib import Path

PHASE = re.compile(r"(\d\d:\d\d:\d\d\.\d+) +\w+ +Phase profiling \(profiling\) (started|sending complete)")


def wall(run: str) -> float | None:
    at = {}
    for line in open(Path(run) / "client/aiperf.log", errors="replace"):
        m = PHASE.match(line)
        if m:
            at.setdefault(m[2], dt.datetime.strptime(m[1], "%H:%M:%S.%f"))
    if len(at) < 2:
        return None
    s = (at["sending complete"] - at["started"]).total_seconds()
    return (s + 86400 if s < 0 else s) / 120


w = {run: wall(run) for run in sys.argv[1:]}
for run in [run for run, x in w.items() if x is None]:
    print(f"{run:12s} no profiling wall: the cell did not complete")
runs = [run for run in w if w[run] is not None]
designed, measured = w[runs[0]], w[runs[1]]
gap = designed - measured
for run in runs:
    print(f"{run:12s} {w[run] * 120:8.3f} s, {w[run]:.4f} wall s per simulated s")
print(f"gap, designed less measured: {gap:.4f}; designed / measured {designed / measured:.4f}")
for run in runs[2:]:
    cost = designed - w[run]
    print(f"{run:12s} costs {cost:+.4f} wall s per simulated s, {cost / gap:+.1%} of the gap")

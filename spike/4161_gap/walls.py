# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""walls.py DESIGNED[,DESIGNED2] MEASURED CELL...: each short cell's profiling wall, from its aiperf log's
`profiling started` to `sending complete` at ms, per simulated second (120 s); as designed, the mean of
its cells; whether the round counts; the gap, designed less measured; and what each other cell's change
costs, designed less that cell, and its share of the gap. A cell whose profiling did not complete has no
wall and is left out.

A round counts when designed over measured lies in RANGE, which holds #4266's six rounds (1.059 to 1.093,
at 8 and 16 CPUs a cell), all run before this spike, and, with two designed cells, when they differ by at
most AGREE of their mean: twice the 0.3% to 0.5% by which #4267's open-case cell, which changed nothing
measurable, differed from as designed in its first four rounds."""
import datetime as dt
import re
import sys
from pathlib import Path

PHASE = re.compile(r"(\d\d:\d\d:\d\d\.\d+) +\w+ +Phase profiling \(profiling\) (started|sending complete)")
RANGE, AGREE = (1.05, 1.10), 0.01


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


named = sys.argv[1].split(",")
w = {run: wall(run) for run in [*named, *sys.argv[2:]]}
for run in [run for run, x in w.items() if x is None]:
    print(f"{run:12s} no profiling wall: the cell did not complete")
w = {run: x for run, x in w.items() if x is not None}
for run, x in w.items():
    print(f"{run:12s} {x * 120:8.3f} s, {x:.4f} wall s per simulated s")
mine = [w[run] for run in named if run in w]
designed, measured = sum(mine) / len(mine), w[sys.argv[2]]
ok = RANGE[0] <= designed / measured <= RANGE[1]
print(f"designed / measured {designed / measured:.4f}: {'within' if ok else 'outside'} {RANGE[0]} to {RANGE[1]}")
if len(mine) == 2:
    spread = abs(mine[0] - mine[1]) / designed
    ok &= spread <= AGREE
    print(f"the designed cells differ by {spread:.2%}: {'within' if spread <= AGREE else 'beyond'} {AGREE:.0%}")
print(f"round counts: {'yes' if ok else 'no'}; gap, designed less measured: {designed - measured:.4f}")
for run in sys.argv[3:]:
    if run in w:
        cost = designed - w[run]
        print(f"{run:12s} costs {cost:+.4f} wall s per simulated s, {cost / (designed - measured):+.1%} of the gap")

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""walls.py RULED,... MEASURED,... OFF,...: each short cell's profiling wall, from its aiperf log's
`profiling started` to `sending complete` at ms, per simulated second (120 s); each setup's mean and the
spread of its cells; whether the round counts; and as ruled's and as measured's gain against off. A cell
whose profiling did not complete has no wall and is left out.

A round counts when as ruled over as measured lies in RANGE and the as-ruled cells differ by at most AGREE
of their mean. As ruled is the prototype with #4262's timer and TAR-hop fixes, whose median shares of
#4267's gap (10.8% and 2.9%, of 0.058 to 0.089 over as measured in its counted rounds) put it 1.006 to
1.012 times as measured; RANGE widens that by the 2.0% by which two identical cells differed in #4267, on
either side, rounded out. AGREE is #4267's."""
import datetime as dt
import re
import sys
from pathlib import Path

PHASE = re.compile(r"(\d\d:\d\d:\d\d\.\d+) +\w+ +Phase profiling \(profiling\) (started|sending complete)")
RANGE, AGREE, BAR = (0.98, 1.04), 0.01, 0.18


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


mean = {}
for name, arg in zip(("ruled", "measured", "off"), sys.argv[1:4]):
    walls = []
    for run in arg.split(","):
        x = wall(run)
        print(f"{run:12s} " + ("no profiling wall: the cell did not complete" if x is None else
                                f"{x * 120:8.3f} s, {x:.4f} wall s per simulated s"))
        walls += [] if x is None else [x]
    mean[name] = sum(walls) / len(walls)
    spread = (max(walls) - min(walls)) / mean[name]
    print(f"{name:9s} mean {mean[name]:.4f} of {len(walls)} cells, spread {spread:.2%}")
    if name == "ruled":
        agree = spread <= AGREE
ratio = mean["ruled"] / mean["measured"]
ok = RANGE[0] <= ratio <= RANGE[1]
print(f"ruled / measured {ratio:.4f}: {'within' if ok else 'outside'} {RANGE[0]} to {RANGE[1]}; "
      f"as-ruled cells {'within' if agree else 'beyond'} {AGREE:.0%}")
print(f"round counts: {'yes' if ok and agree else 'no'}")
for name in ("ruled", "measured"):
    gain = 1 - mean[name] / mean["off"]
    print(f"{name} against off: {mean[name] / mean['off']:.4f}, gain {gain:.1%}"
          + (f", {'clears' if gain >= BAR else 'short of'} {BAR:.0%}" if name == "ruled" else ""))

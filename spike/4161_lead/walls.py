# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""walls.py OFF MEASURED DESIGNED: each short cell's profiling wall, from its aiperf log's
`profiling started` to `sending complete` at ms, per simulated second (120 s), and the ratios."""
import datetime as dt
import re
import sys
from pathlib import Path

PHASE = re.compile(r"(\d\d:\d\d:\d\d\.\d+) +\w+ +Phase profiling \(profiling\) (started|sending complete)")


def wall(run: str) -> float:
    at = {}
    for line in open(Path(run) / "client/aiperf.log", errors="replace"):
        m = PHASE.match(line)
        if m:
            at.setdefault(m[2], dt.datetime.strptime(m[1], "%H:%M:%S.%f"))
    s = (at["sending complete"] - at["started"]).total_seconds()
    return s + 86400 if s < 0 else s


off, measured, designed = (wall(run) for run in sys.argv[1:4])
for run, w in zip(sys.argv[1:4], (off, measured, designed)):
    print(f"{run:12s} profiling {w:8.3f} s, {w / 120:.3f} wall s per simulated s")
print(
    f"measured/off {measured / off:.3f}  designed/off {designed / off:.3f}  "
    f"designed/measured {designed / measured:.3f}"
)

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""ccdload.py OUT CONTAINER [PERIOD]: on the host, until killed, every PERIOD s (default 30) append to OUT
the UTC time, then for each CCD of CPUs 64-127 (8 CPUs each, 64-71 first) the cores that threads outside
CONTAINER (a docker id prefix) kept busy there over the period: utime + stime of each thread last run on
it. ccdload.py --mean OUT: each CCD's mean over OUT's lines."""
import os
import sys
import time


def snap(mine: str) -> dict:
    got = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cgroup") as f:
                if mine in f.read():
                    continue
            tids = os.listdir(f"/proc/{pid}/task")
        except OSError:
            continue
        for tid in tids:
            try:
                with open(f"/proc/{pid}/task/{tid}/stat") as f:
                    rest = f.read().rpartition(")")[2].split()
            except OSError:
                continue
            got[tid] = (int(rest[11]) + int(rest[12]), int(rest[36]))  # utime + stime, processor
    return got


if sys.argv[1] == "--mean":
    rows = [[float(x) for x in line.split()[1:9]] for line in open(sys.argv[2])]
    means = [sum(col) / len(rows) for col in zip(*rows)] if rows else []
    print(f"outside busy cores per CCD 64-71 .. 120-127, mean of {len(rows)} samples:",
          " ".join(f"{m:.1f}" for m in means))
    sys.exit()
out, mine = sys.argv[1], sys.argv[2]
period, tick = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0, os.sysconf("SC_CLK_TCK")
before, t0 = snap(mine), time.time()
while True:
    time.sleep(period)
    after, t1 = snap(mine), time.time()
    busy = [0] * 8
    for tid, (cpu, proc) in after.items():
        if 64 <= proc <= 127 and tid in before:
            busy[(proc - 64) // 8] += max(cpu - before[tid][0], 0)
    with open(out, "a") as f:
        stamp = time.strftime("%H:%M:%S", time.gmtime(t1))
        f.write(stamp + " " + " ".join(f"{b / (tick * (t1 - t0)):.1f}" for b in busy) + "\n")
    before, t0 = after, t1

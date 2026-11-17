"""CPU seconds each process of a run spent between two of cpu.sh's snapshots.

usage: python3 cpudelta.py OUT [FIRST LAST] -- the first and last snapshots by default; prints each process's
utime + stime over that span (clock ticks at 100 a second), its share of the span, its comm and its role from
OUT.prof/procs.txt when sampled.
"""

import sys
from pathlib import Path

out = sys.argv[1]
snaps = sorted(Path(out + ".cpu").glob("*.txt"), key=lambda p: float(p.stem))
a, b = (
    (snaps[0], snaps[-1])
    if len(sys.argv) < 4
    else (Path(sys.argv[2]), Path(sys.argv[3]))
)
span = float(b.stem) - float(a.stem)
read = lambda p: {
    l.split()[0]: l.split() for l in p.read_text().splitlines() if len(l.split()) == 5
}
x, y = read(a), read(b)
procs = {}
pt = Path(out + ".prof/procs.txt")
if pt.exists():
    for l in pt.read_text().splitlines():
        f = l.split(" ")
        procs[f[0]] = " ".join(f[3:])[:60]
rows = []
for pid, r in y.items():
    if pid in x:
        d = (int(r[3]) + int(r[4]) - int(x[pid][3]) - int(x[pid][4])) / 100
        rows.append((d, pid, r[1], r[2], procs.get(pid, "")))
rows.sort(reverse=True)
print(
    f"span {span:.0f} s, {a.stem} to {b.stem}; total CPU {sum(r[0] for r in rows):.0f} s"
)
for d, pid, ppid, comm, cmd in rows[: int(sys.argv[4]) if len(sys.argv) > 4 else 40]:
    print(f"{d:8.1f} s {d / span:6.1%} {pid:>8} {ppid:>8} {comm:<18} {cmd}")

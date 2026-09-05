"""How few measured shapes does interpolating a price need?

If a handful suffice, serving's shape variety costs a handful of extra
benchmarks rather than a derived graph per shape -- which is the difference
between #6b being an M and an L.
"""
import json, sys

d = json.load(open('compass_ops/prices_m.json'))['prices']
pts = sorted((int(k.split('|')[1].split(';')[0].split(',')[0]), v['seconds'] * 1e6)
             for k, v in d.items())


def interp(known, m):
    lo = max((p for p in known if p[0] <= m), default=known[0])
    hi = min((p for p in known if p[0] >= m), default=known[-1])
    if hi[0] == lo[0]:
        return lo[1]
    t = (m - lo[0]) / (hi[0] - lo[0])
    return lo[1] + t * (hi[1] - lo[1])


def errors(known, allpts):
    out = []
    for m, us in allpts:
        if any(k[0] == m for k in known):
            continue
        out.append(abs(100 * (interp(known, m) - us) / us))
    return out


print('%-34s %5s %9s %9s' % ('measured shapes', 'n', 'median', 'worst'))
grids = [
    ("every shape but one in two", pts[::2]),
    ("powers of two only", [p for p in pts if p[0] & (p[0] - 1) == 0]),
    ("1, 8, 64, 512", [p for p in pts if p[0] in (1, 8, 64, 512)]),
    ("1, 32, 1024", [p for p in pts if p[0] in (1, 32, 1024)]),
    ("ends only: 1, 1024", [p for p in pts if p[0] in (1, 1024)]),
]
for label, known in grids:
    e = errors(known, pts)
    if not e:
        continue
    print('%-34s %5d %8.1f%% %8.1f%%'
          % (label, len(known), sorted(e)[len(e) // 2], max(e)))

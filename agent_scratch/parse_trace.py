import glob, gzip, json, sys, collections

paths = sorted(glob.glob("compass_ops/prof/**/*.json*", recursive=True))
print("traces:", paths)
events = []
for p in paths:
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as fh:
        events += json.load(fh).get("traceEvents", [])
print("events:", len(events))
print("categories:", collections.Counter(e.get("cat") for e in events).most_common(10))

k = {}
for e in events:
    if e.get("cat") not in ("kernel", "Kernel"):
        continue
    n = e.get("name", "")
    v = k.setdefault(n, [0, 0.0])
    v[0] += 1
    v[1] += float(e.get("dur", 0.0))
total = sum(v[1] for v in k.values())
print("\ntotal device time in kernels: %.1f us across %d launches"
      % (total, sum(v[0] for v in k.values())))
print("%12s %7s %10s  %s" % ("total us", "n", "us/call", "kernel"))
for n, (c, t) in sorted(k.items(), key=lambda kv: -kv[1][1])[:20]:
    print("%12.1f %7d %10.3f  %s" % (t, c, t / max(c, 1), n[:66]))

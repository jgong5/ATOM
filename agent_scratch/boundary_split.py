"""Is the per-launch constant one number or two?

2.25 us/launch fits TP=1 and TP=2 and three architectures, and TP=4 decode needs
3.33. What changes at TP=4 is the group: a collective's boundary includes waiting
for the slowest peer, which has more chances to be slow as the group grows. If
that is right, the non-collective part should stay put across TP while the
collective part grows.

Solve, per configuration:  step - priced = n_plain * b_plain + n_coll * b_coll
"""
import glob, json, sys
sys.path.insert(0, '/workspace/ATOM')
from atom.compass.runtime.microbench import signature_of

# (label, decode graph, price list, measured decode step in seconds)
CASES = [
    ("27B TP=1", "compass_ops/s1.json",        "compass_ops/sp1.json",         None),
    ("27B TP=2", "compass_ops/g27.tp0.json",   "compass_ops/prices27.tp0.json", 17.840e-3),
    ("27B TP=4", "compass_ops/s4.tp0.json",    "compass_ops/sp4.tp0.json",     None),
]

def terms(graph, prices):
    pl = json.load(open(prices))["prices"]
    ops = json.load(open(graph))["ops"]
    priced = n_plain = n_coll = 0
    total = 0.0
    for op in ops:
        e = pl.get(signature_of(op))
        if e is None:
            continue
        priced += 1
        total += e["seconds"]
        launches = max(1, len(e.get("kernels") or {}))
        if op.get("group"):
            n_coll += launches
        else:
            n_plain += launches
    return total, n_plain, n_coll, priced

print("%-10s %11s %8s %8s %8s" % ("", "priced", "plain", "coll", "ops"))
rows = []
for label, g, p, step in CASES:
    try:
        total, n_plain, n_coll, priced = terms(g, p)
    except (OSError, KeyError) as exc:
        print("%-10s  unavailable (%s)" % (label, type(exc).__name__))
        continue
    print("%-10s %10.3fms %8d %8d %8d" % (label, total * 1e3, n_plain, n_coll, priced))
    rows.append((label, total, n_plain, n_coll, step))

print()
print("with the 2.25us single constant, and what each step would need:")
for label, total, n_plain, n_coll, step in rows:
    pred = total + (n_plain + n_coll) * 2.25e-6
    line = "%-10s predicted %8.3fms" % (label, pred * 1e3)
    if step:
        line += "  measured %8.3fms  %+6.2f%%" % (step * 1e3, 100 * (pred - step) / step)
    print(line)

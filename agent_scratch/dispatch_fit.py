"""Is the eager overhead a per-operator dispatch that kernels hide?

Fitted per model it is 86.35 us/op on a 0.6B and 34.62 on a 27B, which is not a
constant. The proposal is that it is not meant to be: the host dispatches while
the device works, so only `max(0, dispatch - kernel)` is paid per operator. That
makes the *dispatch* the constant and the overhead a consequence of kernel sizes,
which the price list already holds.

Solve for the dispatch D that explains each model's measured overhead. If one D
fits both, the eager term can be computed instead of measured.
"""
import json, sys
sys.path.insert(0, '/workspace/ATOM')
from atom.compass.runtime.microbench import signature_of


def kernel_times(graph_path, prices_path):
    prices = json.load(open(prices_path))["prices"]
    ops = json.load(open(graph_path))["ops"]
    out = []
    for op in ops:
        e = prices.get(signature_of(op))
        if e is not None:
            out.append(e["seconds"])
    return out


def solve(ks, overhead):
    """Smallest D with sum(max(0, D - k)) == overhead. Monotone in D."""
    lo, hi = 0.0, overhead + max(ks) + 1e-3
    for _ in range(200):
        mid = (lo + hi) / 2
        if sum(max(0.0, mid - k) for k in ks) < overhead:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


cases = [
    ("Qwen3-0.6B  prefill", "compass_ops/gp.prefill.json", "compass_ops/prices9.json",
     42.640e-3, 15.093e-3),
    ("Qwen3.8-27B prefill", "compass_ops/g27.prefill.tp0.json",
     "compass_ops/prices27.tp0.json", 320.268e-3, 295.755e-3),
]
print("%-22s %5s %10s %12s %12s" % ("", "ops", "mean k", "overhead/op", "dispatch D"))
for name, g, p, step, priced in cases:
    ks = kernel_times(g, p)
    overhead = step - priced
    D = solve(ks, overhead)
    print("%-22s %5d %9.1fus %11.2fus %11.2fus"
          % (name, len(ks), 1e6 * sum(ks) / len(ks), 1e6 * overhead / len(ks), 1e6 * D))
    hidden = sum(1 for k in ks if k >= D)
    print("%-22s        %d of %d operators fully hide it" % ("", hidden, len(ks)))

print()
print("what a single shared D costs, on each model's prefill step:")
print("%-10s %12s %12s %12s %9s" % ("D", "0.6B pred", "0.6B real", "27B pred", "27B real"))
sets = [(kernel_times(g, p), step, priced) for _, g, p, step, priced in cases]
for D in (100e-6, 110e-6, 117e-6, 125e-6, 132e-6):
    line = ["%9.0fus" % (D * 1e6)]
    for ks, step, priced in sets:
        pred = priced + sum(max(0.0, D - k) for k in ks)
        line.append("%9.1fms (%+.1f%%)" % (pred * 1e3, 100 * (pred - step) / step))
    print("  ".join(line))

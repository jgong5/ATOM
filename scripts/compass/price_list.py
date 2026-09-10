"""What the kernel price list covers, and what a step costs according to it.

A price list is only useful if it reaches the operators that matter. Coverage by
operator *count* is the wrong measure — a step is hundreds of cheap slices and a
hundred-odd expensive gemms — so this reports both, and prices a graph with what
it has.

    python scripts/compass/price_list.py compass_ops/prices.json [graph.json]
"""

import collections
import json
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    prices = json.loads(open(sys.argv[1]).read())
    cov = prices["coverage"]

    print("=" * 72)
    print(f"  signatures priced : {cov['signatures_priced']} of {cov['signatures']}")
    print(f"  operators covered : {cov['operators_priced']} of {cov['operators']}"
          f"  ({100*cov['fraction_of_operators']:.1f}%)")
    print("=" * 72)

    by_name = collections.defaultdict(lambda: [0, 0.0])
    for entry in prices["prices"].values():
        row = by_name[entry["name"]]
        row[0] += entry["occurrences"]
        row[1] += entry["seconds"] * entry["occurrences"]
    ranked = sorted(by_name.items(), key=lambda kv: -kv[1][1])
    total = sum(v[1] for v in by_name.values())

    print(f"\n  priced kernels, by what they contribute to one step:")
    print(f"    {'kernel':<44} {'n':>4} {'each':>10} {'total':>10}")
    for name, (n, secs) in ranked[:12]:
        print(f"    {name[:44]:<44} {n:>4} {secs/n*1e6:>9.1f}us "
              f"{secs*1000:>9.3f}ms")
    print(f"    {'-- priced total':<44} {'':>4} {'':>10} {total*1000:>9.3f}ms")

    # How much of that total is the host waiting rather than the device working.
    #
    # `microbench` already records this per entry and says what it means --
    # "where this matches `seconds`, the device was idle waiting and the price is
    # the host's, not the kernel's" -- but nothing has ever added it up. On a
    # 0.6B graph one operator, `aten::is_nonzero`, came to 11.8% of the priced
    # step with 98% of its own price being host time, and it swung 53.4us to
    # 17.6us between runs of the same graph, which was the whole of an 8.3%
    # swing in the priced total. It forces a device-to-host transfer to read a
    # boolean, so what is timed is a wait for whatever the GPU had queued.
    #
    # It also could not be captured (`cache="over"`), and a production step
    # replays a captured graph -- so this is time spent on work the deployment
    # does not do, sitting inside a number meant to describe what it does.
    host_rows = []
    for entry in prices["prices"].values():
        host = float(entry.get("host_seconds") or 0.0)
        secs = float(entry.get("seconds") or 0.0)
        if secs > 0 and host / secs >= 0.5:
            host_rows.append((secs * entry["occurrences"], entry["name"],
                              host / secs, entry.get("cache"),
                              entry["occurrences"], secs))
    if host_rows:
        host_rows.sort(reverse=True)
        charged = sum(r[0] for r in host_rows)
        print(f"\n  of which the host was waiting, not the device "
              f"({charged*1000:.3f}ms, {100*charged/max(total, 1e-12):.1f}% "
              f"of the priced step):")
        print(f"    {'kernel':<38} {'n':>4} {'each':>10} {'host':>6}  cache")
        for contrib, name, frac, cache, n, secs in host_rows[:8]:
            print(f"    {name[:38]:<38} {n:>4} {secs*1e6:>9.1f}us "
                  f"{100*frac:>5.0f}%  {cache}")
        print("    A replayed step contains none of these. Treat the priced "
              "total as an upper bound until they are dealt with.")

    if prices["unpriced"]:
        reasons = collections.Counter(
            v.split(":")[0] for v in prices["unpriced"].values())
        print(f"\n  unpriced signatures ({len(prices['unpriced'])}), by reason:")
        for reason, n in reasons.most_common(6):
            print(f"    {n:>4}  {reason[:60]}")

    if len(sys.argv) > 2:
        graph = json.loads(open(sys.argv[2]).read())
        sys.path.insert(0, ".")
        from atom.compass.runtime.microbench import signature_of

        have = prices["prices"]
        missing = sum(1 for o in graph["ops"] if signature_of(o) not in have)
        print(f"\n  costing {sys.argv[2]}: {len(graph['ops'])-missing} of "
              f"{len(graph['ops'])} operators priced, "
              f"summing to {total*1000:.3f}ms")
        print("  Compare against the replayed step, not the eager one: this is")
        print("  a sum of kernel costs with no dispatch overhead in it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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

"""Do a step's operators account for the step?

The cost model predicts from a step's shape. Attributing cost per operator is
what would let the graph work contribute — and let Compass cost a configuration
nobody has run, by deriving its graph and pricing the operators in it. None of
that is worth building until the parts add up to the whole.

So this asks one question of one traced step: the operators were each timed
between their own pair of CUDA events, and the region containing them between
one more pair, in the same forward. If the sum matches the region, per-operator
attribution has something to stand on. If it does not, the gap is what has to be
explained first.

Note what these numbers are not. A traced step runs eagerly, because dispatch is
the only place an operator can be observed and a replayed CUDA graph is a single
submission. On this model an eager decode step takes 8.9x a replayed one, so
these are not production costs. They are a description of the work, which a
later phase has to map onto the replayed cost.

    python scripts/compass/op_cost.py compass_ops/timings.json [graph.json]
"""

import collections
import json
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    blob = json.loads(open(sys.argv[1]).read())
    ops = blob["operators"]
    summary = blob["summary"]

    total = summary["sum_of_operators"]
    region = summary["region"]
    print("=" * 70)
    print(f"  {summary['operators']} operators timed")
    print(f"  sum of operators : {total*1000:9.3f}ms")
    print(f"  region containing them: {region*1000:9.3f}ms")
    print(f"  covered          : {100*summary['covered']:9.1f}%")
    print("=" * 70)
    if summary["covered"] > 1.05:
        print("  The parts exceed the whole: operators overlap, so each is")
        print("  credited with time the others were also using. Summing them")
        print("  would over-count a step.")
    elif summary["covered"] < 0.95:
        print("  Time passed in the region that no operator claimed. Summing")
        print("  the operators would under-count a step by the difference.")
    else:
        print("  The operators account for the region. Per-operator costs can")
        print("  be summed into a step cost, at least eagerly.")

    by_name = collections.defaultdict(lambda: [0, 0.0])
    for op in ops:
        entry = by_name[op["name"]]
        entry[0] += 1
        entry[1] += op["seconds"]
    ranked = sorted(by_name.items(), key=lambda kv: -kv[1][1])

    print(f"\n  where the time goes ({len(by_name)} distinct operators):")
    print(f"    {'operator':<44} {'n':>4} {'total':>9} {'share':>7}")
    cum = 0.0
    for name, (n, secs) in ranked[:12]:
        cum += secs
        print(f"    {name[:44]:<44} {n:>4} {secs*1000:>8.3f}ms "
              f"{100*secs/total:>6.1f}%")
    print(f"    {'-- top 12 cumulative':<44} {'':>4} {cum*1000:>8.3f}ms "
          f"{100*cum/total:>6.1f}%")

    # A cost model needs the expensive operators to be few and stable. If cost
    # is spread thinly over hundreds of distinct operators, pricing each one is
    # a much larger undertaking than pricing a handful.
    half = 0.0
    for i, (_, (_, secs)) in enumerate(ranked, 1):
        half += secs
        if half >= total / 2:
            print(f"\n  half the time is in the top {i} of {len(by_name)} "
                  f"operator kinds.")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())

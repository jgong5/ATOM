"""The wall a DP group's pricing takes, from replay.py's per-step lines.

usage: python3 critical.py LABEL FILE... [--skip N --first N]

Each FILE is one rank's replay output (``--digest --out``: a line a step with its
step number, priced seconds and wall). The ranks of a DP group step together, so
a group step waits for its slowest rank: the group's pricing wall is the sum over
steps of the largest rank's wall. Prints, for the steps every file holds, that
sum, the mean rank's sum, and the group's simulated seconds (the largest rank's
priced seconds a step), whose ratio to the pricing wall bounds the speed a run
could reach were pricing its only cost.
"""

import json
import sys


def main():
    args = sys.argv[1:]
    label, files = args[0], args[1:]
    ranks = []
    for f in files:
        by = {}
        with open(f) as fh:
            for line in fh:
                x = json.loads(line)
                by[x["step"]] = (x["wall"], x["s"] or 0.0)
        ranks.append(by)
    steps = sorted(set.intersection(*(set(r) for r in ranks)))
    crit = sum(max(r[s][0] for r in ranks) for s in steps)
    mean = sum(sum(r[s][0] for r in ranks) for s in steps) / len(ranks)
    sim = sum(max(r[s][1] for r in ranks) for s in steps)
    print(
        f"{label}: {len(ranks)} ranks, {len(steps)} steps; pricing wall, slowest rank a step "
        f"{crit:.1f} s ({crit / len(steps) * 1e3:.2f} ms a step), mean rank {mean:.1f} s; "
        f"simulated {sim:.1f} s; simulated over pricing wall {sim / crit:.2f}"
    )


if __name__ == "__main__":
    main()

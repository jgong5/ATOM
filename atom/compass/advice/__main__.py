# SPDX-License-Identifier: MIT
"""`compass advice`: a user's configuration space, expanded.

    compass advice SPACE OUT

SPACE is the space file, expanded by `space.expand`; OUT, made when absent,
gets `CANDIDATES`, the expansion: SPACE's path, its model, machine, store,
frozen commit, workload and SLO, its candidates and their points; and
`check.REUSE`, the reuse check's verdicts on each point's laws and memory
readings (`check.check`); and `check.STATES`, each candidate's state and each
of its points', every command that makes what one lacks, after a start of each
point that could be searched (`check.states`). It prints the head the check
took, or why the tree is refused, then each candidate with its ATOM basis,
each departure with its reason, its verdicts and its state. Then each
searchable point is simulated into its run directory under `simulate.RUNS`, a
finished one skipped, and `simulate.SIMULATED` gets each point's outcome
(`simulate.simulate`), printed with the CPU seconds the points took. It reads
SPACE alone, and writes nothing when SPACE refuses.
"""

import argparse
import json
import sys
from pathlib import Path

from atom.compass.advice import simulate
from atom.compass.advice.check import REUSE, STATES, check, lines, state_lines, states
from atom.compass.advice.space import expand
from atom.compass.cli import refusing

PROG = "compass advice"
#: The expansion, in OUT.
CANDIDATES = "candidates.json"


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog=PROG)
    parser.add_argument("space", type=Path, help="the space file")
    parser.add_argument("out", type=Path, help="the search's directory")
    args = parser.parse_args(argv)
    expansion = expand(args.space)
    args.out.mkdir(parents=True, exist_ok=True)
    written = args.out / CANDIDATES
    written.write_text(json.dumps(expansion, indent=1) + "\n")
    judged, verdicts = check(expansion, args.out), args.out / REUSE
    found = states(expansion, judged, args.out)
    if head := judged["head"]:
        print(
            f"the reuse check at {head['commit']} ({head['ref']}), ATOM digest "
            f"{head['atom']}: verdicts in {verdicts}"
        )
    else:
        print(f"the reuse check refuses: {judged['refused']['reason']}")
    for candidate in expansion["candidates"]:
        print(f"{candidate['name']}: {candidate['deployment']}, {candidate['basis']}")
        for what, reason in candidate["departures"].items():
            print(f"  {what} departs: {reason}")
        if not judged["refused"]:
            print("\n".join(lines(judged["candidates"][candidate["name"]])))
        print("\n".join(state_lines(found[candidate["name"]])))
    searchable = sum(
        v["state"] == "searchable" for s in found.values() for v in s["points"].values()
    )
    print(f"{searchable} points searchable, each state in {args.out / STATES}")
    outcome = simulate.simulate(expansion, judged, found, args.out)
    print(f"each simulated point in {args.out / simulate.SIMULATED}:")
    print("\n".join(simulate.lines(outcome)))
    candidates, points = len(expansion["candidates"]), len(expansion["points"])
    print(f"{candidates} candidates and {points} points in {written}")


if __name__ == "__main__":
    refusing(PROG, main, sys.argv[1:])

# SPDX-License-Identifier: MIT
"""`compass advice`: a user's configuration space, expanded.

    compass advice SPACE OUT

SPACE is the space file, expanded by `space.expand`; OUT, made when absent,
gets `CANDIDATES`, the expansion: SPACE's path, its model, machine, store,
frozen commit, workload and SLO, its candidates and their points. It prints
each candidate with its ATOM basis and each departure with its reason. It
reads SPACE alone, and writes nothing when SPACE refuses.
"""

import argparse
import json
import sys
from pathlib import Path

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
    for candidate in expansion["candidates"]:
        print(f"{candidate['name']}: {candidate['deployment']}, {candidate['basis']}")
        for what, reason in candidate["departures"].items():
            print(f"  {what} departs: {reason}")
    candidates, points = len(expansion["candidates"]), len(expansion["points"])
    print(f"{candidates} candidates and {points} points in {written}")


if __name__ == "__main__":
    refusing(PROG, main, sys.argv[1:])

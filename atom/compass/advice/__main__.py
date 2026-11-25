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
(`simulate.simulate`), printed with the CPU seconds the points took. Last,
`results.RESULTS` gets the results (`results.results`), printed: the SLO, the
result table, each command that makes what a point lacks, and the
recommendation's text; and `dashboard.FILE` gets the dashboard
(`dashboard.write`), with the reference layer in the file ``--reference``
names, when given (`dashboard.reference`). When SPACE differs from the search
OUT holds only in its SLO, the results and the dashboard alone are made again,
from the verdicts, states and outcomes OUT holds, and nothing is checked or
simulated; a search that did not end holds no outcomes, so the next is whole.
Beside SPACE and the store and laws the check reads, it reads OUT's last
search: an SLO-only rerun takes its `CANDIDATES`, `check.REUSE`,
`check.STATES` and `simulate.SIMULATED` as they are, and each point's run
directory is read to skip a finished point, for its metrics and for the
dashboard. It writes nothing when SPACE or the reference layer refuses.
"""

import argparse
import json
import sys
from pathlib import Path

from atom.compass.advice import dashboard, results, simulate
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
    parser.add_argument(
        "--reference",
        type=Path,
        help="a reference layer for the dashboard: its source and its curves",
    )
    args = parser.parse_args(argv)
    # Whole, so each path and command naming OUT holds from any directory.
    args.out = args.out.resolve()
    expansion = expand(args.space)
    layer = dashboard.reference(args.reference) if args.reference else None
    args.out.mkdir(parents=True, exist_ok=True)
    earlier = _only_slo_changed(expansion, args.out)
    written = args.out / CANDIDATES
    written.write_text(json.dumps(expansion, indent=1) + "\n")
    if earlier:
        print(
            f"{args.space} differs from the search in {args.out} only in its SLO: "
            "its results again, nothing checked or simulated"
        )
        judged, found, outcome = earlier
    else:
        judged, found, outcome = _search(expansion, args.out)
    held = results.results(expansion, judged, found, outcome, args.out)
    print("\n".join(results.lines(held)))
    print(f"the results in {args.out / results.RESULTS}")
    print(f"the dashboard in {dashboard.write(args.out, written, layer)}")
    candidates, points = len(expansion["candidates"]), len(expansion["points"])
    print(f"{candidates} candidates and {points} points in {written}")


def _search(expansion: dict, out: Path) -> tuple[dict, dict, dict]:
    """Check and simulate `expansion` into `out`, printing each step: its
    verdicts, states and outcomes."""
    # A search stopped before its outcomes holds none, so the next is whole.
    for name in (simulate.SIMULATED, results.RESULTS, dashboard.FILE):
        (out / name).unlink(missing_ok=True)
    judged, verdicts = check(expansion, out), out / REUSE
    found = states(expansion, judged, out)
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
    print(f"{searchable} points searchable, each state in {out / STATES}")
    outcome = simulate.simulate(expansion, judged, found, out)
    print(f"each simulated point in {out / simulate.SIMULATED}:")
    print("\n".join(simulate.lines(outcome)))
    return judged, found, outcome


def _only_slo_changed(expansion: dict, out: Path) -> tuple | None:
    """The verdicts, states and outcomes of the search `out` holds, when
    `expansion` differs from it in its SLO alone; else None."""
    held = [out / n for n in (CANDIDATES, REUSE, STATES, simulate.SIMULATED)]
    if not all(path.is_file() for path in held):
        return None
    was, now = json.loads(held[0].read_text()), json.loads(json.dumps(expansion))
    if was.pop("slo") == now.pop("slo") or was != now:
        return None
    return tuple(json.loads(path.read_text()) for path in held[1:])


if __name__ == "__main__":
    refusing(PROG, main, sys.argv[1:])

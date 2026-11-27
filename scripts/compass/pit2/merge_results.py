# SPDX-License-Identifier: MIT
"""One space's results and dashboard from two searches of it, when one
candidate was searched apart, as when its calibration ended after the full
search had run.

    PYTHONPATH=$PIT2_TREE python3 merge_results.py OUT FULL_OUT PART_OUT CANDIDATE...

FULL_OUT is a search of the whole space; PART_OUT a search of a space holding
CANDIDATE... (the same candidates' blocks, fewer of them). The CANDIDATEs'
verdicts, states and point outcomes are PART_OUT's, every other candidate's
FULL_OUT's, and the results and the dashboard are computed by advice's own
`results.results` and `dashboard.write` over them into OUT, which then holds
reuse.json, states.json, simulated.json, candidates.json, results.json and
dashboard.html, as a search's OUT does. Prints the results' lines and where the
dashboard is.

Searching the whole space again into FULL_OUT, once every calibration has
ended, gives the same answer with no merge: a rerun skips each finished point.
"""

import json
import sys
from pathlib import Path


def merge(full: Path, part: Path, candidates: list[str], out: Path) -> dict:
    """The merged search's results, written into `out` with its inputs."""
    from atom.compass.advice import results
    from atom.compass.advice.check import REUSE, STATES
    from atom.compass.advice.simulate import SIMULATED

    def read(p: Path):
        return json.loads(p.read_text())

    out.mkdir(parents=True, exist_ok=True)
    expansion = read(full / "candidates.json")
    judged, found, outcome = (
        read(full / REUSE),
        read(full / STATES),
        read(full / SIMULATED),
    )
    pj, pf, po = read(part / REUSE), read(part / STATES), read(part / SIMULATED)
    names = {c["name"] for c in expansion["candidates"]}
    for c in candidates:
        if c not in names or c not in pj["candidates"]:
            raise SystemExit(f"candidate {c!r} is not in both searches")
        judged["candidates"][c] = pj["candidates"][c]
        found[c] = pf[c]
    for point, held in po["points"].items():
        if point.split("-c")[0] in candidates:
            outcome["points"][point] = held
    for name, data in ((REUSE, judged), (STATES, found), (SIMULATED, outcome)):
        (out / name).write_text(json.dumps(data, indent=1) + "\n")
    (out / "candidates.json").write_text(json.dumps(expansion, indent=1) + "\n")
    return results.results(expansion, judged, found, outcome, out)


def main(argv: list[str]) -> None:
    from atom.compass.advice import dashboard, results

    if len(argv) < 4:
        raise SystemExit(__doc__)
    out, full, part = map(Path, argv[:3])
    held = merge(full, part, argv[3:], out)
    print("\n".join(results.lines(held)))
    print(f"the dashboard in {dashboard.write(out, out / 'candidates.json')}")


if __name__ == "__main__":
    main(sys.argv[1:])

# SPDX-License-Identifier: MIT
"""A real run's stalled forwards: `python -m atom.compass.validate.stalls RUN [OUT]`

RUN is a real run's directory, a `compass record` `--out` with its client's
export (`client.sh`). A forward stalls when one DP rank's host takes far longer
to launch it (`t_launch_ns`, from the first launch stamp to the last) than its
group's other ranks: under DP attention every rank waits for it at the first
collective, so the whole group's step stretches by the difference. No law
prices that: the fit and `validate.cost` leave the worst out as warm steps, and
a simulated run charges none.

Each forward that is not a dummy run is compared with the slowest other rank
that ran a forward at its step; one with no such rank, the only rank that ran a
forward, with the median launch of its own forwards of the same kind (prefill
or decode, output or not). Its excess is its launch less that reference, and it
stalls when the excess is above `MIN_S`. A step counts once, at its largest
excess.

The report gives the stalls' count and seconds over the whole record and over
the client's graded window, from its first profiling request's send to its
last's end on the server's clock (`report.split`'s), with their share of that
window and, per `BIN_S` of it, their seconds. Given OUT, it writes the stalls
there as `COMPASS_REPLAY_STALLS` reads them (`runner.overrides.replayed_stall`):
each its rank, its seconds and its time from the record's first forward.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from atom.compass.backends.json_file import read_jsonl
from atom.compass.report.split import _client

#: The least excess, in seconds, a stalled forward's launch holds.
MIN_S = 0.2
#: The graded window's bins, in seconds.
BIN_S = 300.0


def _kind(line: dict) -> tuple[bool, bool]:
    return any(not d for _, _, d in line["rows"]), bool(line["produces_output"])


def stalls(steps: Path, min_s: float = MIN_S) -> tuple[list[dict], float]:
    """The stalled forwards of the step record `steps` (`dp<r>.jsonl`), each its
    ``step``, ``rank``, ``t_s`` (its entry on the record's clock, seconds),
    ``launch_s`` and ``excess_s``; and the record's first entry, in seconds."""
    lines = defaultdict(list)
    own = defaultdict(list)
    first = None
    for path in sorted(steps.glob("dp*[0-9].jsonl")):
        rank = int(path.stem[2:])
        for line in read_jsonl(path):
            enter = line["t_enter_ns"]
            first = enter if first is None else min(first, enter)
            launch = line.get("t_launch_ns")
            if line["is_dummy_run"] or not launch or not line["rows"]:
                continue
            seconds = (launch[-1] - launch[0]) * 1e-9
            lines[line["step"]].append((seconds, rank, line))
            own[rank, _kind(line)].append(seconds)
    median = {k: statistics.median(v) for k, v in own.items()}
    out = []
    for step, ranks in sorted(lines.items()):
        ranks.sort(key=lambda r: r[0])
        seconds, rank, line = ranks[-1]
        ref = ranks[-2][0] if len(ranks) > 1 else median[rank, _kind(line)]
        if seconds - ref > min_s:
            out.append(
                {
                    "step": step,
                    "rank": rank,
                    "t_s": line["t_enter_ns"] * 1e-9,
                    "launch_s": seconds,
                    "excess_s": seconds - ref,
                }
            )
    return out, (first or 0) * 1e-9


def window(run: Path) -> tuple[float, float] | None:
    """The graded window of `run`'s client on its server's clock, in seconds:
    its first profiling request's send to its last one's end; None without one."""
    done, _ = _client(run, simulated=False)
    ends = [e for r, e in done.values() if r.get("phase") == "profiling"]
    if not ends:
        return None
    return min(e[0] for e in ends), max(e[2] for e in ends)


def report(run: Path, min_s: float = MIN_S) -> dict:
    """`run`'s stalls, as the module says."""
    found, first = stalls(run / "steps", min_s)
    out = {
        "record": str(run),
        "min_s": min_s,
        "first_s": first,
        "stalls": len(found),
        "seconds": sum(s["excess_s"] for s in found),
        "events": found,
    }
    graded = window(run)
    if graded is not None:
        lo, hi = graded
        inside = [s for s in found if lo <= s["t_s"] < hi]
        bins = defaultdict(float)
        for s in inside:
            bins[int((s["t_s"] - lo) // BIN_S)] += s["excess_s"]
        out["graded"] = {
            "window_s": hi - lo,
            "stalls": len(inside),
            "seconds": sum(s["excess_s"] for s in inside),
            "bins": [bins[i] for i in range(int((hi - lo) // BIN_S) + 1)],
        }
    return out


def replay_file(found: dict) -> dict:
    """The stalls of `report`'s `found` as `COMPASS_REPLAY_STALLS` reads them."""
    return {
        "record": found["record"],
        "stalls": [
            {
                "rank": s["rank"],
                "t_s": s["t_s"] - found["first_s"],
                "seconds": s["excess_s"],
            }
            for s in found["events"]
        ],
    }


def text(found: dict) -> str:
    lines = [
        f"record {found['record']}: {found['stalls']} stalled forwards, "
        f"{found['seconds']:.1f} s past their group's launch"
        f" (excess > {found['min_s']} s)"
    ]
    if g := found.get("graded"):
        bins = ", ".join(f"{b:.1f}" for b in g["bins"])
        lines.append(
            f"graded window {g['window_s']:.0f} s: {g['stalls']} stalls, "
            f"{g['seconds']:.1f} s ({g['seconds'] / g['window_s']:.1%}); "
            f"per {BIN_S:.0f} s: {bins}"
        )
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if not argv or len(argv) > 2:
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        return 2
    found = report(Path(argv[0]))
    print(text(found))
    if len(argv) == 2:
        Path(argv[1]).write_text(json.dumps(replay_file(found)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

# SPDX-License-Identifier: MIT
"""How much of a simulated run its cost sources priced past their support
domains, on the rank that sets each step's time as well as on every rank:
`python -m atom.compass.report.extrapolation OUT [T0 T1]`

The run summary's ``extrapolated_steps`` counts every rank's step holding a
term its source extrapolated (`Provenance.extrapolated`). Under DP that counts
the ranks whose price the group's max over its ranks ignores too, its dummy
forwards among them, so it says little about the time a run predicts. A DP
group step's time is set by its critical rank: of the ranks running the step,
the one whose cost line states the most device seconds, ``device_s`` plus
``draft_s``, the lowest DP rank among equals. With one rank, that rank.

`of_run` reads each simulated step record under OUT (`RECORDS`): each DP
rank's cost lines, ``dp<r>.cost.jsonl`` (`runner.overrides`'
`_group_step_seconds`), beside its step lines, ``dp<r>.jsonl`` (`parity`). A
group step is a step number every rank's cost lines hold, since a DP group's
ranks step together, and it enters at the earliest of its ranks' ``t_s``. It
gives, under `FIELD` in the run summary's ``schedule``:

- ``critical_rank``: one step per group step, the critical rank's, and under
  ``by_kind`` the same split by that step's kind: ``prefill``, ``mixed``,
  ``decode/rung<n>`` (``decode`` on a line with no rung) or ``dummy``, as its
  step line states it, ``unrecorded`` where it has none;
- ``every_rank``: every rank's steps, those ``extrapolated_steps`` counts;
- ``by_rank``: each DP rank's own steps, by ``dp<r>``, or above one record by
  ``<instance>/dp<r>``, its directory under OUT less the ``steps`` part.

Each holds its ``steps`` and their ``device_seconds``, and three shares of
them by what was extrapolated: ``extrapolated``, the steps holding any term its
source extrapolated; ``target_extrapolated``, a target term; and
``draft_extrapolated``, a draft term. A share states its ``steps``, their whole
``device_seconds``, and its ``fraction_of_steps`` and
``fraction_of_device_seconds``, None over none. A refused step counts as a step
of 0 s holding no term. The terms a line's ``target_terms`` counts first are
the target's; on a line written before lines stated it, a draft term is one
named as the step-linear, step-stub, op-graph and roofline sources name theirs
(`DRAFT_NAMES`).

Given a window ``(t0, t1)`` on the run's clock, only the group steps entering
in ``[t0, t1)`` count, and each rank's own steps entering there: a client run's
summary counts its own span, and ``T0 T1`` on the command line a graded window.
A line with no ``t_s``, a run with no host laws, has no place in a window.

A record still being written is read up to its last whole line. `of_run`
raises ValueError naming why there is no tally: OUT holds no step record, a
record is a pipeline's, whose stages each price their own layers, or a real
run's, whose ``dp<r>.cost.jsonl`` holds labels (`step_labels`), or a line is
not JSON or cannot be placed in the window.
"""

import json
import sys
from pathlib import Path

from atom.compass.backends.provenance import EXTRAPOLATED

#: The run summary's ``schedule`` key `of_run` fills, and the one saying why it
#: cannot, each None where the other is not.
FIELD, REFUSED = "extrapolation", "extrapolation_refused"
#: Where a run's step records lie under OUT: one server's ``steps``, a
#: RapidServe engine's below it (`parity.record_dir`), and a deployment
#: instance's (`deploy.sh`), and a RapidServe engine's below that.
RECORDS = ("steps", "steps/*", "*/steps", "*/steps/*")
#: Each share a tally states, by whether a target term and a draft term count.
SHARES = {
    "extrapolated": (True, True),
    "target_extrapolated": (True, False),
    "draft_extrapolated": (False, True),
}
#: How a draft term is named, on a line that states no ``target_terms``.
DRAFT_NAMES = ("draft.", "draft ")
USAGE = "usage: python -m atom.compass.report.extrapolation OUT [T0 T1]"


class _Tally:
    """Steps and their device seconds, and each of `SHARES`' of them."""

    def __init__(self) -> None:
        self.steps, self.seconds = 0, 0.0
        self.shares = {name: [0, 0.0] for name in SHARES}

    def add(self, seconds: float, target: bool, draft: bool) -> None:
        self.steps += 1
        self.seconds += seconds
        for name, (by_target, by_draft) in SHARES.items():
            if (by_target and target) or (by_draft and draft):
                self.shares[name][0] += 1
                self.shares[name][1] += seconds

    def merge(self, other: "_Tally") -> None:
        self.steps += other.steps
        self.seconds += other.seconds
        for name, (steps, seconds) in other.shares.items():
            self.shares[name][0] += steps
            self.shares[name][1] += seconds

    def as_record(self) -> dict:
        out = {"steps": self.steps, "device_seconds": self.seconds}
        for name, (steps, seconds) in self.shares.items():
            out[name] = {
                "steps": steps,
                "device_seconds": seconds,
                "fraction_of_steps": steps / self.steps if self.steps else None,
                "fraction_of_device_seconds": (
                    seconds / self.seconds if self.seconds > 0 else None
                ),
            }
        return out


def _lines(path: Path):
    """Each whole line of the JSON-lines file at `path`, parsed, with its
    number; a last line not yet ended is left out."""
    with open(path, "rb") as lines:
        for n, line in enumerate(lines):
            if not line.endswith(b"\n"):
                return
            try:
                yield n, json.loads(line)
            except ValueError as e:
                raise ValueError(f"{path} line {n + 1} is not JSON: {e}") from None


def _costs(path: Path) -> dict[int, tuple]:
    """Each step's ``(t_s, device seconds, target extrapolated, draft
    extrapolated)`` off the cost lines at `path`, by its step number."""
    out = {}
    for n, line in _lines(path):
        if n == 0:
            if "version" not in line:
                raise ValueError(f"{path} starts with no version line")
            continue
        if "terms" not in line:
            raise ValueError(
                f"{path} holds a real record's labels, not a simulated run's cost lines"
            )
        target = draft = False
        targets = line.get("target_terms")
        for k, term in enumerate(line["terms"]):
            if EXTRAPOLATED in term["source"]:
                if targets is None:
                    is_draft = term["name"].startswith(DRAFT_NAMES)
                else:
                    is_draft = k >= targets
                draft, target = draft or is_draft, target or not is_draft
        seconds = line["device_s"] + line["draft_s"]
        out[line["step"]] = (line.get("t_s"), seconds, target, draft)
    return out


def kind(step: dict) -> str:
    """A step line's kind, as `by_kind` keys it."""
    rows = step.get("rows") or []
    if step.get("is_dummy_run") or not rows:
        return "dummy"
    decodes = sum(1 for row in rows if row[2])
    if decodes == len(rows):
        rung = step.get("rung")
        return "decode" if rung is None else f"decode/rung{rung}"
    return "prefill" if decodes == 0 else "mixed"


def _kinds(path: Path) -> dict[int, str]:
    """Each step's `kind` off the step lines at `path`, by its step number;
    none where there is no such file."""
    if not path.is_file():
        return {}
    return {line["step"]: kind(line) for _, line in _lines(path)}


def _within(t, window, path: Path, step: int) -> bool:
    if window is None:
        return True
    if t is None:
        raise ValueError(
            f"{path} step {step} states no time (t_s), so it has no place in a "
            f"window [{window[0]}, {window[1]})"
        )
    return window[0] <= t < window[1]


def _record(record: Path, window) -> tuple[_Tally, dict, dict]:
    """The tallies of the step record in `record`: its critical rank's, by
    kind, and each rank's, by ``dp<r>``."""
    if any(record.glob("pp*.dp*.cost.jsonl")):
        raise ValueError(
            f"{record} is a pipeline's record, whose stages each price their own "
            "layers, so no rank's line prices a whole step"
        )
    paths = {}
    for path in record.glob("dp*.cost.jsonl"):
        name = path.name.split(".")[0]
        if name[2:].isdigit():
            paths[int(name[2:])] = path
    ranks = sorted(paths)
    if not ranks:
        raise ValueError(f"{record} holds no DP rank's cost lines, dp<r>.cost.jsonl")
    costs = {r: _costs(paths[r]) for r in ranks}
    critical, by_kind, by_rank = _Tally(), {}, {}
    for r in ranks:
        mine = by_rank[f"dp{r}"] = _Tally()
        for step, (t, seconds, target, draft) in costs[r].items():
            if _within(t, window, paths[r], step):
                mine.add(seconds, target, draft)
    kinds = {r: _kinds(record / f"dp{r}.jsonl") for r in ranks}
    for step in sorted(set.intersection(*(set(costs[r]) for r in ranks))):
        lines = [costs[r][step] for r in ranks]
        times = [line[0] for line in lines]
        entered = None if None in times else min(times)
        if not _within(entered, window, record, step):
            continue
        k = max(range(len(ranks)), key=lambda i: lines[i][1])
        _, seconds, target, draft = lines[k]
        critical.add(seconds, target, draft)
        of_kind = kinds[ranks[k]].get(step, "unrecorded")
        by_kind.setdefault(of_kind, _Tally()).add(seconds, target, draft)
    return critical, by_kind, by_rank


def of_run(out, window: tuple[float, float] | None = None) -> dict:
    """The tallies of the simulated run in `out`, over the group steps
    entering in `window` if given, as the module says."""
    out = Path(out)
    records = sorted(
        {p.parent for where in RECORDS for p in out.glob(f"{where}/dp*.cost.jsonl")}
    )
    if not records:
        raise ValueError(
            f"{out} holds no step record with cost lines, in any of {', '.join(RECORDS)}"
        )
    critical, every, by_kind, by_rank = _Tally(), _Tally(), {}, {}
    for record in records:
        mine, kinds, ranks = _record(record, window)
        critical.merge(mine)
        for name, tally in kinds.items():
            by_kind.setdefault(name, _Tally()).merge(tally)
        where = [p for p in record.relative_to(out).parts if p != "steps"]
        for name, tally in ranks.items():
            every.merge(tally)
            by_rank["/".join([*where, name])] = tally.as_record()
    return {
        "critical_rank": critical.as_record()
        | {"by_kind": {name: t.as_record() for name, t in sorted(by_kind.items())}},
        "every_rank": every.as_record(),
        "by_rank": by_rank,
    }


def main(argv: list[str]) -> None:
    if len(argv) not in (1, 3):
        sys.exit(USAGE)
    try:
        window = (float(argv[1]), float(argv[2])) if len(argv) == 3 else None
        print(json.dumps(of_run(argv[0], window), indent=1, sort_keys=True))
    except (OSError, ValueError) as refused:
        sys.exit(f"extrapolation: {argv[0]}: refused: {refused}")


if __name__ == "__main__":
    main(sys.argv[1:])

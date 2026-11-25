# SPDX-License-Identifier: MIT
"""A real run's stalled forwards and whole-group pauses:
`python -m atom.compass.validate.stalls RUN [OUT] [--only stalls|pauses] [--kfd EVICTED]`

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

A whole-group pause stops every rank for seconds while no launch stretches
(#3778): on a node whose GPU queues the kernel evicts, every rank waits on its
device. Each rank of the group waits where its step leaves it: one whose last
forward produced output inside its next forward's wait for it (`t_mtp_ns`, or
the staging gate's, `t_gate_ns`), one with nothing to wait for between its
forwards, for the group. `group_pauses` reads it from the group's clock: a
step's span runs from the latest exit over the ranks of the step before to the
latest exit of its own, less any part where every rank sat between forwards
(an idle engine). The span of a step after a forward of one kind (decode, or
prefill of as many tokens, to a power of two, on its largest rank) is mostly
that forward's device time, so the median span after that kind is its
reference (`REF_STEPS` or more steps, else the median after any forward of its
prefill-or-decode kind); a step whose span exceeds it by more than `GROUP_MIN_S`
holds a pause of that excess.

A rank that arrived at the paused step late by half the pause or more, between
its previous forward's exit and this one's entry, held the group: the others
wait for it at their forward's first collective. The pause is that rank's, the
latest, unless every rank's device window (`d_busy_ns`) at that step or the one
before exceeds its kind's median by half the pause or more: the device stopped
on every rank, and a host that waited on it arrived late too. With no late
rank it is device-side, on all ranks: every rank waited where its step left
it, released together, so the rank that held them is not in host stamps. A
rank between its forwards after the paused one, as one with no output to wait
for is, waits for its group, not the other way round (#3778's second shape:
in g08's records every such pause is in every rank's device window).

The report gives the stalls' and the pauses' count and seconds over the whole
record and over the client's graded window, from its first profiling request's
send to its last's end on the server's clock (`report.split`'s), with their
share of that window and, per `BIN_S` of it, their seconds. Given a KFD
sampler's `evicted_ms` file (`kfd`: lines of wall time, pid, KFD stats node
and the process's cumulative `evicted_ms` there), each pause also names the
processes whose GPU queues the kernel evicted over it, placed on the wall clock
by the client export's sends, which carry both clocks.

Given OUT, it writes the stalls and the pauses there as `COMPASS_REPLAY_STALLS`
reads them (`runner.overrides.replayed_stall`): each its rank, its seconds and
its time from the record's first forward; a pause, marked `any_step`, on its
rank, or a device-side one on every rank. `--only` writes one of the two.
"""

from __future__ import annotations

import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

from atom.compass.backends.json_file import read_jsonl
from atom.compass.backends.step_linear_fit import PAUSE_S
from atom.compass.parity.client_run import RAW
from atom.compass.report.split import _client

#: The least excess, in seconds, a stalled forward's launch holds.
MIN_S = 0.2
#: The graded window's bins, in seconds.
BIN_S = 300.0
#: The least seconds a whole-group pause holds past its step's reference, the
#: excess a step law's fit takes for one (`step_linear_fit.PAUSE_S`).
GROUP_MIN_S = PAUSE_S
#: The fewest steps whose median span a kind of forward takes as its reference.
REF_STEPS = 20
#: A forward's stamped waits, any of which may hold a pause, by the name a
#: report gives them.
WAITS = {
    "gate": "t_gate_ns",
    "mtp": "t_mtp_ns",
    "recv": "t_recv_ns",
    "draft_recv": "t_draft_recv_ns",
    "launch": "t_launch_ns",
}
#: Where a rank that waited out a pause between its forwards, after the paused
#: one, waited: for its group.
BETWEEN = "between steps"
#: Where a rank that arrived late at the paused forward was: it held the group.
LATE = "late"


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


class _Forward(NamedTuple):
    """One rank's line, as `group_pauses` reads it."""

    enter: int
    exit: int
    busy: int | None
    #: Its prefill rows' tokens, None for a dummy run or a decode.
    prefill: int | None
    #: Its longest stamped wait: its `WAITS` name and nanoseconds.
    wait: tuple[str, int]


def _forward(line: dict) -> _Forward:
    rows = [] if line["is_dummy_run"] else line.get("rows") or []
    tokens = [n for n, _, decode in rows if not decode]
    waits = [
        (name, line[key][-1] - line[key][0])
        for name, key in WAITS.items()
        if len(line.get(key) or ()) >= 2
    ]
    return _Forward(
        line["t_enter_ns"],
        line["t_exit_ns"],
        line.get("d_busy_ns"),
        sum(tokens) if tokens else None,
        max(waits, key=lambda w: w[1], default=("", 0)),
    )


def _key(forwards) -> tuple[str, int]:
    """The kind of a group's forwards at one step: prefill, by its largest
    rank's tokens to a power of two, or decode."""
    tokens = [f.prefill for f in forwards if f.prefill is not None]
    return ("prefill", max(tokens).bit_length()) if tokens else ("decode", 0)


def _medians(values: dict) -> dict:
    """The median of each key's values, a key with fewer than `REF_STEPS`
    taking its kind's (prefill or decode)."""
    kinds = defaultdict(list)
    for (kind, _), v in values.items():
        kinds[kind] += v
    kinds = {k: statistics.median(v) for k, v in kinds.items()}
    return {
        k: statistics.median(v) if len(v) >= REF_STEPS else kinds[k[0]]
        for k, v in values.items()
    }


def _overlap(a: int, b: int, lo: int, hi: int) -> float:
    """Seconds [a, b] and [lo, hi], nanoseconds, share."""
    return max(0, min(b, hi) - max(a, lo)) * 1e-9


def group_pauses(
    steps: Path, min_s: float = GROUP_MIN_S
) -> tuple[list[dict], float, list[int]]:
    """The whole-group pauses of the step record `steps`, as the module says,
    each its ``step``, ``t_s`` (its span's start on the record's clock,
    seconds), ``seconds`` (its excess), ``span_s``, ``rank`` (None for a
    device-side pause, on all ranks), ``waits`` (where each rank waited:
    a `WAITS` name, `LATE` or `BETWEEN`) and ``device_s`` (the least device excess
    over the ranks, None without device windows); the record's first entry,
    in seconds; and its ranks."""
    record = {}
    for path in sorted(steps.glob("dp*[0-9].jsonl")):
        record[int(path.stem[2:])] = {
            line["step"]: _forward(line)
            for line in read_jsonl(path)
            if "t_enter_ns" in line
        }
    ranks = sorted(record)
    entries = [f.enter for mine in record.values() for f in mine.values()]
    if not entries:  # no rank, or a simulated record, which stamps no entry
        return [], 0.0, ranks
    first = min(entries)
    common = sorted(set.intersection(*(set(m) for m in record.values())))
    group = {s: [record[r][s] for r in ranks] for s in common}
    keys = {s: _key(fs) for s, fs in group.items()}
    spans = {}
    for s in common:
        if s - 1 not in group:
            continue
        start = max(max(f.exit for f in group[s - 1]), min(f.enter for f in group[s]))
        spans[s] = (start, max(f.exit for f in group[s]) - start)
    after = defaultdict(list)
    for s, (_, span) in spans.items():
        after[keys[s - 1]].append(span)
    reference = _medians(after)
    windows = all(f.busy is not None for fs in group.values() for f in fs)
    busy = (
        _medians(
            {
                k: [min(f.busy for f in group[s]) for s in common if keys[s] == k]
                for k in dict.fromkeys(keys.values())
            }
        )
        if windows
        else {}
    )
    out = []
    for s, (start, span) in sorted(spans.items()):
        excess = (span - reference[keys[s - 1]]) * 1e-9
        if excess <= min_s:
            continue
        end = start + span
        waits, late, device = {}, {}, {}
        for r, now, before in zip(ranks, group[s], group[s - 1]):
            inside = _overlap(now.enter, now.exit, start, end)
            late[r] = _overlap(before.exit, now.enter, start, end)
            waits[r] = (
                now.wait[0]
                if inside >= excess / 2
                else LATE if late[r] >= excess / 2 else BETWEEN
            )
            if windows:
                device[r] = (
                    max(now.busy - busy[keys[s]], before.busy - busy[keys[s - 1]])
                    * 1e-9
                )
        held_device = windows and min(device.values()) >= excess / 2
        latest = max(late, key=late.get)
        if held_device or late[latest] < excess / 2:
            latest = None
        out.append(
            {
                "step": s,
                "t_s": start * 1e-9,
                "seconds": excess,
                "span_s": span * 1e-9,
                "rank": latest,
                "waits": waits,
                "device_s": min(device.values()) if windows else None,
            }
        )
    return out, first * 1e-9, ranks


def window(run: Path) -> tuple[float, float] | None:
    """The graded window of `run`'s client on its server's clock, in seconds:
    its first profiling request's send to its last one's end; None without one."""
    done, _ = _client(run, simulated=False)
    ends = [e for r, e in done.values() if r.get("phase") == "profiling"]
    if not ends:
        return None
    return min(e[0] for e in ends), max(e[2] for e in ends)


def wall_offset(run: Path) -> float | None:
    """Seconds from `run`'s clock to the wall clock: the median, over its
    client's sends, of each send's wall time (`request_start_ns`) less its
    time on the run's clock (`start_perf_ns`); None without both."""
    path = run / "artifacts" / RAW
    if not path.exists():
        path = Path(f"{path}.gz")
    if not path.exists():
        return None
    offsets = []
    with (gzip.open if path.suffix == ".gz" else open)(path, "rb") as lines:
        for line in lines:
            raw = json.loads(line.decode())
            wall = raw.get("metadata", {}).get("request_start_ns")
            if wall is not None and raw.get("start_perf_ns") is not None:
                offsets.append(wall - raw["start_perf_ns"])
    return statistics.median(offsets) * 1e-9 if offsets else None


def kfd(path: Path, pauses: list[dict], offset: float) -> None:
    """Add to each of `pauses` its ``kfd``: the processes whose `evicted_ms`
    grew from the last sample of the `path` file at or before its start to the
    first at or after its end, each its pid and largest growth over its GPUs,
    in ms, largest first; and those samples' wall times."""
    if not pauses:
        return
    lo = min(p["t_s"] for p in pauses) + offset - 60
    hi = max(p["t_s"] + p["span_s"] for p in pauses) + offset + 60
    series = defaultdict(list)
    with open(path) as lines:
        for line in lines:
            t, pid, node, value = line.split("\t")
            if lo <= (t := float(t)) <= hi:
                series[pid, node].append((t, float(value)))
    for p in pauses:
        start, end = p["t_s"] + offset, p["t_s"] + p["span_s"] + offset
        grew, seen = defaultdict(float), set()
        for (pid, _), samples in series.items():
            before = [s for s in samples if s[0] <= start]
            later = [s for s in samples if s[0] >= end]
            if not before or not later:
                continue
            a, b = before[-1], later[0]
            seen.add((a[0], b[0]))
            grew[pid] = max(grew[pid], b[1] - a[1])
        p["kfd"] = {
            "evicted_ms": sorted(
                ([int(pid), ms] for pid, ms in grew.items() if ms > 0),
                key=lambda x: -x[1],
            ),
            "samples": [min(seen)[0], max(seen)[1]] if seen else None,
        }


def _graded(events: list[dict], lo: float, hi: float, at, seconds) -> dict:
    inside = [e for e in events if lo <= e[at] < hi]
    bins = defaultdict(float)
    for e in inside:
        bins[int((e[at] - lo) // BIN_S)] += e[seconds]
    return {
        "count": len(inside),
        "seconds": sum(e[seconds] for e in inside),
        "bins": [bins[i] for i in range(int((hi - lo) // BIN_S) + 1)],
    }


def report(
    run: Path,
    min_s: float = MIN_S,
    group_min_s: float = GROUP_MIN_S,
    evicted: Path | None = None,
) -> dict:
    """`run`'s stalls and pauses, as the module says, with `evicted`'s
    alignment when given."""
    found, first = stalls(run / "steps", min_s)
    paused, _, ranks = group_pauses(run / "steps", group_min_s)
    out = {
        "record": str(run),
        "min_s": min_s,
        "first_s": first,
        "ranks": ranks,
        "stalls": len(found),
        "seconds": sum(s["excess_s"] for s in found),
        "events": found,
        "group_min_s": group_min_s,
        "pauses": len(paused),
        "pause_seconds": sum(p["seconds"] for p in paused),
        "pause_events": paused,
    }
    if evicted is not None and (offset := wall_offset(run)) is not None:
        kfd(evicted, paused, offset)
        out["wall_offset_s"] = offset
    graded = window(run)
    if graded is not None:
        lo, hi = graded
        stalled = _graded(found, lo, hi, "t_s", "excess_s")
        pauses = _graded(paused, lo, hi, "t_s", "seconds")
        out["graded"] = {
            "window_s": hi - lo,
            "start_s": lo,
            "stalls": stalled["count"],
            "seconds": stalled["seconds"],
            "bins": stalled["bins"],
            "pauses": pauses["count"],
            "pause_seconds": pauses["seconds"],
            "pause_bins": pauses["bins"],
        }
    return out


def replay_file(found: dict, only: str | None = None) -> dict:
    """The stalls and pauses of `report`'s `found` as `COMPASS_REPLAY_STALLS`
    reads them; `only` ``stalls`` or ``pauses`` keeps those alone."""
    first = found["first_s"]
    out = []
    if only != "pauses":
        out += [
            {"rank": s["rank"], "t_s": s["t_s"] - first, "seconds": s["excess_s"]}
            for s in found["events"]
        ]
    if only != "stalls":
        out += [
            {
                "rank": rank,
                "t_s": p["t_s"] - first,
                "seconds": p["seconds"],
                "any_step": True,
                "step": p["step"],
            }
            for p in found.get("pause_events", ())
            for rank in (found["ranks"] if p["rank"] is None else [p["rank"]])
        ]
    return {"record": found["record"], "stalls": out}


def _pause(p: dict, first: float) -> str:
    who = (
        "device-side, all ranks"
        if p["rank"] is None
        else f"rank {p['rank']}'s, late to its forward"
    )
    waits = ", ".join(f"{r} {w}" for r, w in p["waits"].items())
    line = (
        f"  step {p['step']} at {p['t_s'] - first:.1f} s: {p['seconds']:.2f} s "
        f"({p['span_s']:.2f} s span), {who}; waits: {waits}"
    )
    if (k := p.get("kfd")) is not None and k["samples"] is None:
        line += "; KFD: no sampler sample either side of it"
    elif k is not None:
        top = ", ".join(f"{pid} {ms / 1e3:.1f} s" for pid, ms in k["evicted_ms"][:6])
        line += f"; KFD evicted over it: {len(k['evicted_ms'])} processes" + (
            f" ({top})" if top else ""
        )
    return line


def text(found: dict) -> str:
    lines = [
        (
            f"record {found['record']}: {found['stalls']} stalled forwards, "
            f"{found['seconds']:.1f} s past their group's launch"
            f" (excess > {found['min_s']} s)"
        )
    ]
    g = found.get("graded")
    if g:
        bins = ", ".join(f"{b:.1f}" for b in g["bins"])
        lines.append(
            f"graded window {g['window_s']:.0f} s: {g['stalls']} stalls, "
            f"{g['seconds']:.1f} s ({g['seconds'] / g['window_s']:.1%}); "
            f"per {BIN_S:.0f} s: {bins}"
        )
    if "pauses" in found:
        line = (
            f"whole-group pauses (> {found['group_min_s']} s past their step's "
            f"median): {found['pauses']}, {found['pause_seconds']:.1f} s"
        )
        if g:
            bins = ", ".join(f"{b:.1f}" for b in g["pause_bins"])
            line += (
                f"; graded window: {g['pauses']}, {g['pause_seconds']:.1f} s "
                f"({g['pause_seconds'] / g['window_s']:.1%}); per {BIN_S:.0f} s: {bins}"
            )
        lines.append(line)
        lines += [_pause(p, found["first_s"]) for p in found["pause_events"]]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    argv, only, evicted = list(argv), None, None
    try:
        if "--only" in argv:
            i = argv.index("--only")
            only = argv.pop(i + 1)
            argv.pop(i)
            if only not in ("stalls", "pauses"):
                raise ValueError(only)
        if "--kfd" in argv:
            i = argv.index("--kfd")
            evicted = Path(argv.pop(i + 1))
            argv.pop(i)
    except (IndexError, ValueError):
        argv = []
    if not argv or len(argv) > 2:
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        return 2
    found = report(Path(argv[0]), evicted=evicted)
    print(text(found))
    if len(argv) == 2:
        Path(argv[1]).write_text(json.dumps(replay_file(found, only)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

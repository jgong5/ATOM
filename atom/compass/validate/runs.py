# SPDX-License-Identifier: MIT
"""Schedule agreement and end-to-end results: a simulated cell against N real runs of it.

    python -m atom.compass.validate.runs SIMULATED REAL...

Each argument is a run directory: its step record in `steps/` and aiperf's
export in `artifacts/profile_export_aiperf.json`. SIMULATED is the cell at its
backend, and REAL two or more real runs of it, spaced in time.

A directory `cctraces_sim.sh` wrote is graded as below when it holds
`artifacts/client_run.json`, whole when it does not. SIMULATED is refused
when its run.json, the run file it ran, skips host segments, has no host laws,
or turns the step roofline on with no step grid; a real run's server never
reads the run file it was handed. It is also refused when an engine's hand-in
in it, `lp-engine*.json`, states an assumed `memory_readings` member, naming
each assumption. In one `record` or `run` wrote, whose run.json
states a `format_version`, the graded client run is the one whose export is
`artifacts/client_run.json`, split from the record by `parity.served`: only
its steps are graded, and every run's client must have been given the same
`traces` and `arguments`, as its export states them; which traces its finished
requests came from moves with its schedule. A step holding only requests its
export names as its warmup's (`warmup`) is its warmup, and not graded. It is
refused, naming every reason, when steps of any other request precede its own:
it is then not its server's first client run, and meets the prefix cache an
earlier one left; when its export names no warmup, since its first steps may
then be cold; when no step before its own first holds a request the export
names as the warmup's, since its graded steps may then precede it; when any
step from its own first on, its own or not, holds such a request, since its
warmup has then not ended when it starts; when its export lacks traces or
arguments; when its export lists `unnamed` requests, naming each by its
`x_request_id`, since no step can be matched to them; and when its run.json
names `aiter_jit_builds`, the aiter JIT modules its server built or began while
it served, since a step may have waited on a build (`compass record` and a real
`cctraces_sim.sh` cell write it). A simulated run, whose run.json holds
`run_file`, is also refused when it skips host segments, or its run file has no
host laws and so prices no host time; when the step roofline was on in a run of
another backend than the step grid, which always prices below it; when its
first client run refused anything, or has no summary to say, or priced no step,
since the graded client run then came later; when it ran the step grid without
a coverage check; when its backend's fit inputs were profiled or served a trace
it served; and when its engines read an assumed `memory_readings` member,
naming each assumption its `provenance` lists, since its KV blocks rest on
readings no start took. The step grid's plan, an input of `source` `plan`, is
not one: it placed the points from shapes alone, so it may have read any trace,
a graded one included.
Beside the table, the command states how many of that client run's steps its
backend extrapolated and a fallback below it priced.

A deployment `deploy.sh` wrote holds a server's directory per side,
`prefill/` and `decode/`, and its client's export beside them. Each side is
checked and graded as a run `record` or `run` wrote, against that export, and
the step families below are per side (`counts prefill`, `counts decode`, ...);
end to end is the deployment's, once. A record of several DP ranks is graded
over every rank's steps: counts summed over ranks, distributions over every
rank's steps, and the warmup checks on its steps in step order, every rank's
step k one forward.

The tolerance of a metric is the real runs' own spread: its distance between
every ordered pair of real runs, at most. A metric's result is its distance
from the simulated cell to each real run. The families:

- counts, from the step record and the export: steps and tokens per kind,
  preemptions, prefill streaks, requests completed. Distance `|a - b|`.
  `exact` where every real run has the simulated cell's value. `prefill
  tokens` counts only the requests both runs issued, since which turns a
  cc-traces run issues before its cutoff moves with its end-to-end time; the
  values it reports are the cell totals.
- one run, `prefill tokens`: the prefill tokens of the requests only the first
  run issued less those only the second did, `not gated`. A request is its
  prompt length and its place in arrival order among the run's requests of
  that length, which a prefix-cache hit does not move; the requests a record
  never names, their prefill unfinished, are one request.
- distributions, per step from the step record: batch width, scheduled
  tokens, context tokens, and the cached fraction of a prefill step's context.
  Distance: the two-sample KS statistic.
- end to end, from the export: TTFT, TPOT (aiperf's inter-token latency) mean
  and p50, request and output-token throughput. Distance `|a / b - 1|`.
  `ungradeable` where the real spread exceeds the target (15% TTFT, 10% the
  others); else `within target` where every result is inside it.

Every metric is `inside the real spread` where each result is at most the
spread, else `outside`.

A step is prefill when any of its rows is. A preemption is a request's row
that is prefill after one of its decode rows, or that reads no more context
than its previous row did.
"""

from __future__ import annotations

import argparse
import collections
import json
from itertools import permutations
from pathlib import Path

from scipy.stats import ks_2samp

from atom.compass.backends.json_file import read_json, read_jsonl
from atom.compass.drive import aiter_jit
from atom.compass.parity.served import served
from atom.compass.run import (
    CLIENT_DIR,
    ENGINE,
    LP_FILE,
    STEP_GRID,
    SUMMARY_FILE,
    entries,
)

EXPORT = "artifacts/profile_export_aiperf.json"
#: The export of the client run a `record` or `run` directory grades.
CLIENT_RUN = "artifacts/client_run.json"
#: A deployment's sides, each its servers' directory in it.
SIDES = ("prefill", "decode")
TARGET = {"ttft": 0.15}
DEFAULT_TARGET = 0.10
ASSUMED = "its engines read assumed memory_readings members: {}"
END_TO_END = {
    "ttft mean": ("time_to_first_token", "avg"),
    "ttft p50": ("time_to_first_token", "p50"),
    "tpot mean": ("inter_token_latency", "avg"),
    "tpot p50": ("inter_token_latency", "p50"),
    "request throughput": ("request_throughput", "avg"),
    "output token throughput": ("output_token_throughput", "avg"),
}


def sides(run) -> list[Path]:
    """Where `run`'s servers wrote their records: the prefill/ and decode/ of a
    deployment `deploy.sh` wrote, else `run` itself."""
    run = Path(run)
    return [run / s for s in SIDES] if (run / SIDES[0]).is_dir() else [run]


def _steps(cell) -> list[dict]:
    """Every DP rank's steps, rank by rank, their dummy runs left out."""
    # Each rank's record, and not the cost lines a simulated run writes beside it.
    ranks = sorted((Path(cell) / "steps").glob("dp*[0-9].jsonl"))
    if not ranks:
        raise ValueError(f"{cell} holds no DP rank's record")
    return [s for r in ranks for s in read_jsonl(r) if not s["is_dummy_run"]]


def _requests(steps) -> collections.Counter:
    """Prefill tokens per request, keyed by (prompt length, arrival place)."""
    tokens, length = collections.Counter(), {}
    for step in steps:
        rank = step.get("dp_rank")  # each rank numbers its own requests
        named = {request for request, _key, *_response_id in step["named"]}
        for (request, _n, context), (q, _c, decode) in zip(step["batch"], step["rows"]):
            tokens[rank, request] += 0 if decode else q
            if request in named:  # its final chunk: the context is the prompt
                length[rank, request] = context
    out, seen = collections.Counter(), collections.Counter()
    for request in sorted(tokens):  # request ids number a rank's arrivals
        n = length.get(request)
        out[None if n is None else (n, seen[n])] += tokens[request]
        seen[n] += 1
    return out


def _both(a, b) -> int:
    return abs(sum(n - b[k] for k, n in a.items() if k in b))


def _one_run(a, b) -> int:
    only_a = sum(n for k, n in a.items() if k not in b)
    return abs(only_a - sum(n for k, n in b.items() if k not in a))


def _prefill(step) -> bool:
    return not all(decode for _q, _c, decode in step["rows"])


def _counts(steps, export) -> dict:
    out = dict.fromkeys(
        (
            "prefill steps",
            "decode steps",
            "prefill tokens",
            "decode tokens",
            "preemptions",
            "prefill streaks",
        ),
        0,
    )
    # Summed over DP ranks: each rank's streaks and requests are its own.
    last, was_prefill = {}, {}
    for step in steps:
        prefill, rank = _prefill(step), step.get("dp_rank")
        out["prefill steps" if prefill else "decode steps"] += 1
        out["prefill streaks"] += int(prefill and not was_prefill.get(rank))
        was_prefill[rank] = prefill
        for (request, _n, _c), (q, context, decode) in zip(step["batch"], step["rows"]):
            out[f"{'decode' if decode else 'prefill'} tokens"] += q
            before = last.get((rank, request))
            if before and not decode and (before[1] or context <= before[0]):
                out["preemptions"] += 1
            last[rank, request] = (context, decode)
    out["requests completed"] = export["request_count"]["avg"]
    return out


def _distributions(steps) -> dict:
    cached = [
        sum(c - q for q, c, _ in s["rows"]) / sum(c for _, c, _ in s["rows"])
        for s in steps
        if _prefill(s)
    ]
    return {
        "batch width": [len(s["rows"]) for s in steps],
        "scheduled tokens": [sum(q for q, _, _ in s["rows"]) for s in steps],
        "context tokens": [sum(c for _, c, _ in s["rows"]) for s in steps],
        "cached fraction": cached,
    }


def _end_to_end(export) -> dict:
    return {name: export[k][stat] for name, (k, stat) in END_TO_END.items()}


def _client_run(cell, run, client) -> tuple[list[dict], list[str], dict]:
    """The steps of the client run `cell` grades, the traces its client was
    given, and its warmup: how many requests its export, in `client`, names as
    the warmup's, and how many steps hold only those; refuses, naming every
    reason, a run grading does not take. Its DP ranks step together, so their
    steps are taken in step order, every rank's step k one."""
    steps = _steps(cell)
    export = read_json(Path(client) / CLIENT_RUN)
    warmup_names = set(export.get("warmup") or ())
    record = served(Path(cell) / "steps", [export["requests"]])
    split = [s for rank in record.values() for s in rank if not s["is_dummy_run"]]
    order = sorted(range(len(split)), key=lambda i: split[i]["step"])
    steps, split = [steps[i] for i in order], [split[i] for i in order]
    ids = {k: v for s in split for k, v in s.get("response_ids", {}).items()}
    held = [{ids.get(r[0], r[0]) for r in s["batch"]} for s in split]
    warm = [bool(names) and names <= warmup_names for names in held]
    mine = [s["client_run"] == 0 for s in split]
    traces = sorted({t for s, m in zip(split, mine) if m for t in s["traces"]})
    first = mine.index(True)
    before = warm[:first]
    why = [] if all(before) else ["steps of requests it did not send precede its own"]
    if not warmup_names:
        why.append("its export names no warmup, so its first steps may be cold")
    elif not any(names & warmup_names for names in held[:first]):
        why.append(
            "no step before its own first holds its warmup, so its first steps may be cold"
        )
    late = [s["step"] for s, n in zip(split[first:], held[first:]) if n & warmup_names]
    if late:
        why.append(
            "steps from its own first on hold a request of its warmup, "
            f"{len(late)} from step {late[0]}, so its warmup has not ended when it starts"
        )
    if "traces" not in export:
        why.append("its export names no traces its client was given")
    if "arguments" not in export:
        why.append("its export names no arguments its client was given")
    if unnamed := [u["x_request_id"] for u in export.get("unnamed") or ()]:
        why.append(f"its export lists requests no step can be matched to: {unnamed}")
    if built := aiter_jit.refusal(run):
        why.append(built)
    if "run_file" in run:
        why += _not_graded(run, traces)
    if why:
        raise ValueError(f"{cell} is not graded: " + "; ".join(why))
    warmup = {"requests": len(warmup_names), "steps": sum(warm)}
    return [s for s, m in zip(steps, mine) if m], export["traces"], warmup


def _run_file_not_graded(run_file, roofline) -> list[str]:
    """Why a simulated run of `run_file`, the step roofline on if `roofline`,
    is not graded, as its run file says."""
    why = []
    if "host" not in run_file:
        why.append("its run file has no host laws, so it prices no host time")
    if roofline and STEP_GRID not in run_file:
        why.append("the step roofline was on")
    return why


def _not_graded(run, traces) -> list[str]:
    """Why a simulated run, `run` its run.json, whose client run served
    `traces`, is not graded."""
    why, refused, coverage = [], run.get("refusals"), run["coverage"]
    if run["skip"]:
        why.append(f"it skips host segments {run['skip']}, a what-if run")
    why += _run_file_not_graded(run["run_file"], run["step_roofline"])
    if refused is None:
        why.append("its first client run has no summary, so its refusals are unknown")
    elif refused["count"]:
        why.append(f"it refused {refused['count']}: {refused['reasons']}")
    if refused and refused.get("steps") == 0:
        why.append("its first client run priced no step, so it is not the graded one")
    if STEP_GRID in run["run_file"] and not coverage:
        why.append("it ran the step grid without a coverage check")
    provenance = run["provenance"]["provenance"] or {}
    inputs = [i for i in provenance.get("inputs") or () if i.get("source") != "plan"]
    if fitted := sorted({t for i in inputs for t in i["traces"]} & set(traces)):
        why.append(f"its backend was fitted on traces it served: {fitted}")
    if profiled := [i["run"] for i in inputs if i.get("profiled")]:
        why.append(f"its backend was fitted on profiled inputs: {profiled}")
    if assumed := run["provenance"].get("memory_readings_assumed"):
        why.append(ASSUMED.format(assumed))
    return why


def graded(cell, client=None) -> tuple[list[dict], list[str] | None, dict | None]:
    """The steps `cell` grades, every DP rank's; the traces its client was
    given, and its warmup, as `_client_run` gives them, both None where it is
    graded whole. `client` holds the client's export, `cell` itself unless
    `cell` is a deployment's side."""
    client = client or cell
    path = Path(cell) / "run.json"
    run = read_json(path) if path.exists() else {}
    if "format_version" in run or (Path(client) / CLIENT_RUN).is_file():
        return _client_run(cell, run, client)
    return _steps(cell), None, None


def _refuse_by_run_file(cell) -> None:
    """Refuse the simulated `cell`, a directory `cctraces_sim.sh` wrote, when
    the run file it ran, its run.json, says it is not graded, or when an
    engine's hand-in in it, of either side of a 1P1D cell, states an assumed
    `memory_readings` member."""
    path = Path(cell) / "run.json"
    if not path.exists():
        return
    run = read_json(path)
    # `_client_run` checks a `record` or `run` cell, and a deployment's sides.
    if "format_version" in run or len(sides(cell)) > 1:
        return
    if any(h.get("skip") for h in entries(run, "host")):
        raise ValueError(f"{cell} skips host segments; a what-if run is not graded")
    engines = Path(cell).glob(LP_FILE.format(f"{ENGINE.name}*"))
    memory = [read_json(p).get("memory") or {} for p in engines]
    why = _run_file_not_graded(run, run.get("step_roofline"))
    if assumed := sorted({m["assumed"] for m in memory if m.get("assumed")}):
        why.append(ASSUMED.format(assumed))
    if why:
        raise ValueError(f"{cell} is not graded: " + "; ".join(why))


def _read(cell) -> dict:
    export = read_json(Path(cell) / EXPORT)
    out = {"end to end": _end_to_end(export), "sides": {}}
    for side in sides(cell):
        steps, traces, _warmup = graded(side, cell)
        out["sides"]["" if side == Path(cell) else f" {side.name}"] = {
            "counts": _counts(steps, export),
            "distributions": _distributions(steps),
            "requests": _requests(steps),
        }
    # A run graded whole states its arguments in run.json.
    client = {} if traces is None else read_json(Path(cell) / CLIENT_RUN)
    return out | {"traces": traces, "arguments": client.get("arguments", {})}


DISTANCE = {
    "counts": lambda a, b: abs(a - b),
    "distributions": lambda a, b: ks_2samp(a, b).statistic,
    "end to end": lambda a, b: abs(a / b - 1.0),
}


def _grade(family, name, real, simulated, d=None) -> dict:
    d = d or DISTANCE[family]
    spread = max(d(a, b) for a, b in permutations(real, 2))
    result = [d(simulated, r) for r in real]
    out = {"spread": spread, "result": result}
    if family != "distributions":
        out.update(real=real, simulated=simulated)
    target = TARGET.get(name.split()[0], DEFAULT_TARGET)
    if family == "counts" and max(result) == 0:
        verdict = "exact"
    elif family == "end to end" and spread > target:
        verdict = "ungradeable"
    elif max(result) <= spread:
        verdict = "inside the real spread"
    elif family == "end to end" and max(result) <= target:
        verdict = "within target"
    else:
        verdict = "outside"
    out["verdict"] = verdict
    return out


def grade(real_cells, simulated_cell) -> dict:
    """Every metric's grade: {family: {metric: grade}}, as the module says."""
    if len(real_cells) < 2:
        names = [str(c) for c in real_cells]
        raise ValueError(f"{names}: a real spread needs two or more real runs")
    _refuse_by_run_file(simulated_cell)
    real = [_read(c) for c in real_cells]
    simulated = _read(simulated_cell)
    given = {str(c): r["traces"] for c, r in zip(real_cells, real)}
    given[str(simulated_cell)] = simulated["traces"]
    if len({json.dumps(t) for t in given.values()}) > 1:
        every = set.intersection(*(set(t or ()) for t in given.values()))
        odd = {c: sorted(set(t or ()) - every) for c, t in given.items()}
        raise ValueError(
            f"the runs were given different traces; beyond those all were: {odd}"
        )
    args = {c: r["arguments"] for c, r in zip(given, (*real, simulated))}
    first, *rest = args.values()
    odd = {
        k: {c: a.get(k) for c, a in args.items()}
        for k in sorted(set().union(*args.values()))
        if any(a.get(k) != first.get(k) for a in rest)
    }
    if odd:
        raise ValueError(f"the runs' clients were given different arguments: {odd}")
    named = {c: sorted(r["sides"]) for c, r in zip(given, (*real, simulated))}
    if len({json.dumps(s) for s in named.values()}) > 1:
        raise ValueError(f"the runs' servers wrote different sides: {named}")

    def every(family, theirs, mine) -> dict:
        return {
            name: _grade(family, name, [t[name] for t in theirs], mine[name])
            for name in mine
        }

    # A deployment's schedule per side, `counts prefill` and up; its end to end once.
    report, one_run = {}, {}
    for at, mine in simulated["sides"].items():
        theirs = [r["sides"][at] for r in real]
        for family in ("counts", "distributions"):
            report[family + at] = every(
                family, [t[family] for t in theirs], mine[family]
            )
        counts = report["counts" + at]
        totals = {k: counts["prefill tokens"][k] for k in ("real", "simulated")}
        requests = [t["requests"] for t in theirs], mine["requests"]
        counts["prefill tokens"] = (
            _grade("counts", "prefill tokens", *requests, _both) | totals
        )
        once = _grade("counts", "prefill tokens", *requests, _one_run) | totals
        one_run["one run" + at] = {"prefill tokens": once | {"verdict": "not gated"}}
    end = [r["end to end"] for r in real], simulated["end to end"]
    report["end to end"] = every("end to end", *end)
    return report | one_run


def _text(report: dict) -> str:
    out = ["family | metric | spread | result | verdict"]
    for family, metrics in report.items():
        for name, g in metrics.items():
            result = " ".join(f"{r:.4g}" for r in g["result"])
            out.append(
                f"{family} | {name} | {g['spread']:.4g} | {result} | {g['verdict']}"
            )
    return "\n".join(out)


def priced_apart(cell) -> str | None:
    """How many steps of the simulated `cell`'s graded client run, or of the
    whole run, its backend extrapolated and a fallback below it priced, as that
    run's summary counts them; None for a cell with no summary."""
    for path in (Path(cell) / CLIENT_DIR.format(0), Path(cell)):
        if (path / SUMMARY_FILE).is_file():
            schedule = read_json(path / SUMMARY_FILE)["schedule"]
            n = [schedule.get(f"{k}_steps") for k in ("extrapolated", "fallback")]
            total = schedule["refusals"]["steps"]
            return f"of {total} steps priced: extrapolated {n[0]}, fallback {n[1]}"
    return None


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.validate.runs",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("simulated", help="the simulated cell at its backend")
    parser.add_argument("real", nargs="+", help="two or more real runs of the cell")
    args = parser.parse_args(argv)
    print(_text(grade(args.real, args.simulated)))
    if apart := priced_apart(args.simulated):
        print(apart)


if __name__ == "__main__":
    try:
        main()
    except json.JSONDecodeError:  # it names no file, so it keeps its traceback
        raise
    except ValueError as refused:  # a refusal: its reason, exit 1, no traceback
        raise SystemExit(str(refused)) from refused

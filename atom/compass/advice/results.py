# SPDX-License-Identifier: MIT
"""The results of the search `compass advice` makes: each point's metrics, the
frontier, the recommendation, the runner-up, the result table and the
recommendation's text, written into OUT as `RESULTS`.

A simulated point's metrics come from its run directory. InferenceX's are
`validate.curves.metrics`': throughput per GPU of the input and output tokens,
p90 interactivity, p90 E2E-normalized interactivity and p90 TTFT. The search
adds two over the same records (`curves.profiled`): the throughput per GPU of
the tokens the engine computed, the input and output tokens less
``usage_prompt_cache_read_tokens``, with no value unless every record states
it; and the p90 end-to-end latency, of ``request_latency``. Its run summary
gives the pricing-method shares of its predicted seconds under
`backends.cost.GROUPS`, its steps a law extrapolated as a share of its steps,
and whether it is a coverage report. No search point is graded: one with a
refusal is a coverage report, every other ungraded.

- The frontier: the points no other dominates on throughput per GPU and p90
  interactivity, each higher better.
- The recommendation: the most throughput per GPU among the points that meet
  SPACE's SLO, p90 interactivity at least ``min_interactivity_p90`` and p90
  TTFT at most ``max_ttft_p90_s``, a bound left out bounding nothing and one
  with no value missed; the runner-up, by the same rule, another
  configuration's point at its concurrency. Of equals, the first in SPACE's
  order.
- A point with a refusal, a coverage report, is out of all three: its row
  names what it lacks, each refusal with its count, and the commands that
  calibrate its configuration anew, as the check names them for a missing law,
  or, once the calibration into its ``cal/<candidate>`` has ended, why none can.
  At an earlier candidate's law key, by model, platform and engine config, the
  row names that candidate's calibration instead, as the check names it, with
  the command measuring its readings into the store that calibration publishes
  into, unless the point was looked up there; there, once that calibration has
  ended, why none can. So is a point the search did not simulate, with its
  candidate's state or its own and why, and the commands that make what it
  lacks; a point whose simulation failed, with why; and one whose run directory
  cannot be read.

The table has a row per point, in SPACE's order. The text states the
recommendation's metrics, the runner-up's beside them, its confidence (its
grading state, method shares and extrapolated share of steps), the laws it ran
under, and each point left out with its reason, every number in it shown as the
table shows it.
"""

import json
from pathlib import Path

from atom.compass import run as compass_run
from atom.compass.advice import check
from atom.compass.artifacts import machine_platform
from atom.compass.backends.cost import GROUPS
from atom.compass.backends.json_file import read_json, read_jsonl
from atom.compass.drive import lacking, serve
from atom.compass.drive import run as drive_run
from atom.compass.run.step_grid import keyed
from atom.compass.validate import curves

#: The results, in OUT.
RESULTS = "results.json"
#: Each metric of a point: its column's label and how the table shows it.
METRICS = {
    "throughput per GPU": ("throughput per GPU (tok/s/GPU)", "{:.1f}"),
    "computed throughput per GPU": (
        "computed throughput per GPU (tok/s/GPU)",
        "{:.1f}",
    ),
    "p90 interactivity": ("p90 interactivity (tok/s/user)", "{:.2f}"),
    "p90 E2E-normalized interactivity": (
        "p90 E2E-normalized interactivity (tok/s/user)",
        "{:.2f}",
    ),
    "p90 TTFT": ("p90 TTFT (s)", "{:.3f}"),
    "p90 E2E latency": ("p90 E2E latency (s)", "{:.3f}"),
}
#: InferenceX's metrics, as `curves.metrics` gives them.
INFERENCEX = (
    "throughput per GPU",
    "p90 interactivity",
    "p90 E2E-normalized interactivity",
    "p90 TTFT",
)
#: What the frontier and the recommendation rank on, each higher better.
RANKED = ("throughput per GPU", "p90 interactivity")
#: Each SLO bound: the metric it bounds, whether from below, and its unit.
BOUNDS = {
    "min_interactivity_p90": ("p90 interactivity", True, "tok/s/user"),
    "max_ttft_p90_s": ("p90 TTFT", False, "s"),
}
#: The tokens a record served, and those of its prompt the prefix cache held.
SERVED = ("input_sequence_length", "output_sequence_length")
CACHED = "usage_prompt_cache_read_tokens"
#: The table's columns, by their row fields, and their labels.
COLUMNS = {
    "configuration": "configuration",
    "concurrency": "concurrency",
    **{name: label for name, (label, _) in METRICS.items()},
    "slo_met": "SLO met",
    "frontier": "on the frontier",
    "grading": "grading state",
    "method_shares": "pricing-method shares",
    "extrapolated_share": "extrapolated share of steps",
    "refusal": "refusal reason",
}


def results(expansion: dict, judged: dict, found: dict, outcome: dict, out: Path):
    """The results of the search in `out` of `expansion`'s points, from the
    `check.check` verdicts `judged`, the `check.states` entries `found` and
    the `simulate.simulate` outcome `outcome`: its ``slo``, its ``rows``, the
    ``recommendation`` and the ``runner_up`` by point, and the ``text``,
    written into `out` as `RESULTS`."""
    candidates = {c["name"]: c for c in expansion["candidates"]}
    slo, rows, calibrations = expansion["slo"], [], {}
    for point in expansion["points"]:
        row = _row(
            expansion, judged, found, outcome, candidates, calibrations, point, out
        )
        if row["grading"]:
            row["slo_met"] = all(
                _meets(row[m], slo, b) for b, (m, *_) in BOUNDS.items()
            )
        rows.append(row)
    ranked = [r for r in rows if not r["refusal"] and None not in _ranked(r)]
    for r in ranked:
        r["frontier"] = not any(_dominates(o, r) for o in ranked)
    met = [
        r
        for r in rows
        if not r["refusal"] and r["slo_met"] and r["throughput per GPU"] is not None
    ]
    best = _most(met)
    runner = best and _most(
        r
        for r in met
        if r["concurrency"] == best["concurrency"]
        and r["configuration"] != best["configuration"]
    )
    written = {
        "slo": slo,
        "rows": rows,
        "recommendation": best and best["point"],
        "runner_up": runner and runner["point"],
        "text": _text(rows, best, runner),
    }
    (out / RESULTS).write_text(json.dumps(written, indent=1) + "\n")
    return written


def lines(written: dict) -> list[str]:
    """`results`' `written` as lines: the SLO, the table, each command that
    makes what a point lacks with the points it serves, then the text."""
    slo = written["slo"]
    bounds = [
        f"{metric} {'at least' if below else 'at most'} {slo[bound]:g} {unit}"
        for bound, (metric, below, unit) in BOUNDS.items()
        if bound in slo
    ]
    out = [f"SLO: {', '.join(bounds) or 'none'}", " | ".join(COLUMNS.values())]
    out += [" | ".join(_shown(r, c) for c in COLUMNS) for r in written["rows"]]
    made = {}
    for r in written["rows"]:
        for command in r["made_by"]:
            made.setdefault((r["configuration"], command), []).append(r)
    out += [f"{_named(held)}, made by: {c}" for (_, c), held in made.items()]
    return out + written["text"].splitlines()


def _row(
    expansion, judged, found, outcome, candidates, calibrations, point, out: Path
) -> dict:
    """`point`'s row: its metrics once its run directory reads, or why it is
    out, with the commands that make what it lacks, a coverage report's from
    the calibrations of earlier rows in `calibrations` (`_calibrating`)."""
    name, configuration = point["name"], point["candidate"]
    laws = judged["candidates"][configuration]["points"][name]["law"]
    row = {
        "point": name,
        "configuration": configuration,
        "concurrency": point["concurrency"],
        **dict.fromkeys(METRICS),
        "slo_met": None,
        "frontier": False,
        "grading": None,
        "method_shares": None,
        "extrapolated_share": None,
        "refusal": None,
        "made_by": [],
        "laws": {side: v["run"] for side, v in laws.items() if "run" in v},
        "run": None,
    }
    state = found[configuration]
    held = state["points"].get(name, state)
    if held["state"] != "searchable":
        why = f"{held['state']}: {'; '.join(held['why'])}"
        return row | {"refusal": why, "made_by": held["made_by"]}
    ran = outcome["points"][name]
    if ran["state"] != "finished":
        tried = f"failed after {ran['tries']} tries" if ran["tries"] else "not run"
        return row | {"refusal": f"{tried}: {'; '.join(ran['why'])}"}
    run = Path(ran["run"])
    try:
        row |= _measured(run)
    except (OSError, ValueError) as unread:
        return row | {"refusal": f"no metrics in {run}: {unread}"}
    if row["refusal"]:
        candidate = candidates[configuration]
        row["made_by"] = _calibrating(
            expansion, judged, candidate, point, run, out, calibrations
        )
    return row | {"run": str(run)}


def _measured(run: Path) -> dict:
    """The metrics and the trust of the simulated point run into `run`, from
    its client export and its run summary, and its refusals as its reason."""
    got = curves.metrics(run, simulated=True)
    records = curves.profiled(read_jsonl(run / curves.RECORDS))
    served = sum(
        n
        for r in records
        for k in SERVED
        if (n := curves._metric(r, k, int)) is not None
    )
    cached = [curves._metric(r, CACHED, int) for r in records]
    throughput, computed = got["throughput per GPU"], None
    if throughput is not None and None not in cached:
        # The computed tokens' share of the served ones, so curves' window and GPUs.
        computed = throughput * (1 - sum(cached) / served) if served else throughput
    latency = curves._positive(curves._metric(r, "request_latency") for r in records)
    summary = read_json(run / compass_run.SUMMARY_FILE)
    schedule = summary["schedule"]
    by_method, steps = schedule["seconds_by_method"], schedule["refusals"]["steps"]
    extrapolated = schedule["extrapolated_steps"]
    refused = [f"{why}, {n} times" for why, n in schedule["refusals"]["reasons"]]
    return {
        **{k: got[k] for k in INFERENCEX},
        "computed throughput per GPU": computed,
        "p90 E2E latency": curves.percentile([v / 1000 for v in latency], 90),
        "method_shares": (
            {g: by_method[g]["fraction_of_predicted_seconds"] for g in GROUPS}
            if by_method
            else None
        ),
        "extrapolated_share": (
            extrapolated / steps if steps and extrapolated is not None else None
        ),
        "grading": "coverage report" if summary["coverage_report"] else "ungraded",
        "refusal": (
            f"coverage report: {'; '.join(refused)}"
            if summary["coverage_report"]
            else None
        ),
    }


def _calibrating(
    expansion, judged, candidate, point, run: Path, out: Path, calibrations: dict
) -> list[str]:
    """The commands that calibrate `candidate`'s law anew, as the check names
    them for a missing one: on 1P1D each side's fits after a record of the
    deployment; else the `calibrate.sh` command of `point`'s instance into
    OUT's ``cal/<candidate>``, its clients on SPACE's traces, resolved under
    its variables on the device arch of the machine its run file states,
    publishing into the store a calibration started there names, as its resume
    must. Once that calibration has ended, why no command serves instead.

    That calibration joins `calibrations` at its law key, by model, platform
    and engine config, unless an earlier row's calibration is there, since a
    store refuses a second law at a key: the point is then made by that one,
    named as the check names it (`check.BORROWED`), and by the command
    measuring its readings into the store it publishes into
    (`check._readings_by`), unless the check's verdicts `judged` looked the
    candidate up in that store; there, once that calibration has ended, why no
    command serves instead."""
    if candidate["deployment"] != "single":
        fits = lacking.FITS.values()
        return [
            lacking.SIDE_LAW.format(fit=f, side=side)
            for side in point["instances"]
            for f in fits
        ]
    (one,) = point["instances"].values()
    machine = compass_run._machine(read_json(run / serve.RUN_FILE))
    traces = {"TRACES": expansion["workload"]["traces"]}
    with check._env(one["env"] | traces):
        config = drive_run._config(one["args"], machine.value("device.arch"))
        key = [config.model, machine_platform(machine), keyed(config)]
        made = out / check.CAL / candidate["name"]
        lender = calibrations.setdefault(json.dumps(key, sort_keys=True), made)
        started = check.started_store(lender)
        into = started or str(lender / "store")
        # The point's own calibration, or one whose store it was looked up in.
        store = judged["candidates"][candidate["name"]]["store"]
        reads = lender == made or Path(into) == Path(store)
        if reads and _ended(lender / check.STATUS):
            why = (
                f"the calibration in {lender} has ended, so calibrate.sh records "
                "nothing more there, and its records taken again in a new OUT fit "
                "the same vectors, which refuse these steps again"
            )
            return [f"no calibrate.sh command: {why}"]
        if lender == made:
            return [lacking.calibration(made, one["args"], config, started)]
        borrowed = [check.BORROWED.format(lender.name, into)]
        return borrowed + ([] if reads else [check._readings_by(lender, into, one)])


def _ended(status: Path) -> bool:
    """Whether the calibration whose status file is `status` has ended: its
    last line is the ``end calibration`` a run that fits both laws writes."""
    said = status.read_text().splitlines() if status.is_file() else []
    return bool(said) and said[-1].split()[1:3] == ["end", "calibration"]


def _meets(value, slo: dict, bound: str) -> bool:
    """Whether `value` meets `slo`'s `bound`: always when `slo` states no such
    bound, never when there is no value."""
    if bound not in slo:
        return True
    _, below, _ = BOUNDS[bound]
    return value is not None and (value >= slo[bound] if below else value <= slo[bound])


def _ranked(row: dict) -> tuple:
    return tuple(row[k] for k in RANKED)


def _dominates(a: dict, b: dict) -> bool:
    """Whether point `a` is at least as good as `b` on both, and better on one."""
    ours, theirs = _ranked(a), _ranked(b)
    return ours != theirs and all(x >= y for x, y in zip(ours, theirs))


def _most(rows) -> dict | None:
    """The row of `rows` with the most throughput per GPU, the first of equals."""
    return max(rows, key=lambda r: r["throughput per GPU"], default=None)


def _shown(row: dict, column: str) -> str:
    """`row`'s `column` as the table shows it."""
    value = row[column]
    if value is None:
        return "-"
    if column in METRICS:
        return METRICS[column][1].format(value)
    if column == "method_shares":
        shares = [f"{g} {s:.1%}" for g, s in value.items() if s]
        return ", ".join(shares) or "none"
    if column == "extrapolated_share":
        return f"{value:.1%}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _named(rows: list[dict]) -> str:
    """One configuration's `rows` by name: ``S1 at concurrency 1, 2 and 4``."""
    at = [str(r["concurrency"]) for r in rows]
    listed = at[0] if len(at) == 1 else f"{', '.join(at[:-1])} and {at[-1]}"
    return f"{rows[0]['configuration']} at concurrency {listed}"


def _metrics(row: dict) -> str:
    """`row`'s metrics as the text states them."""
    shown = {k: _shown(row, k) for k in METRICS}
    return (
        f"{shown['throughput per GPU']} tok/s/GPU, "
        f"{shown['computed throughput per GPU']} computed; p90 interactivity "
        f"{shown['p90 interactivity']} tok/s/user, "
        f"{shown['p90 E2E-normalized interactivity']} E2E-normalized; p90 TTFT "
        f"{shown['p90 TTFT']} s; p90 E2E latency {shown['p90 E2E latency']} s"
    )


def _text(rows: list[dict], best: dict | None, runner: dict | None) -> str:
    """The recommendation's text."""
    if best is None:
        said = ["No recommendation: no point meets the SLO with no refusal."]
    else:
        tpg = "throughput per GPU"
        said = [f"Recommended: {_named([best])}: {_metrics(best)}."]
        if runner:
            said.append(
                f"Runner-up: {_named([runner])}: {_metrics(runner)}; the "
                f"recommendation's {_shown(best, tpg)} tok/s/GPU against its "
                f"{_shown(runner, tpg)}."
            )
        else:
            said.append(
                "No runner-up: no other configuration's point at concurrency "
                f"{best['concurrency']} meets the SLO with no refusal."
            )
        said.append(
            f"Confidence: {best['grading']}; priced "
            f"{_shown(best, 'method_shares')}; {_shown(best, 'extrapolated_share')}"
            " of steps extrapolated."
        )
        laws = [f"{s} {', '.join(dict.fromkeys(r))}" for s, r in best["laws"].items()]
        said.append(f"Laws: {'; '.join(laws)}.")
    left = {}
    for r in rows:
        if r["refusal"]:
            left.setdefault((r["configuration"], r["refusal"]), []).append(r)
    said += [f"Left out: {_named(held)}: {why}" for (_, why), held in left.items()]
    return "\n".join(said)

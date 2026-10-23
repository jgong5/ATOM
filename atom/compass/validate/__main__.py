# SPDX-License-Identifier: MIT
"""Grade cells from every source, in one table: `python -m atom.compass.validate CELL...`

A CELL holds runs of one workload: `step_linear/` (at its run file's step linear law, or
of several the one fitted under the engine config each real run states) or `step_grid/`
(at its run file's bundle, which `run.step_grid.read` reads by key with no check against
this process, refused unless its digest is the one the run summary records)
and two or more `real*/`. `cctraces_sim.sh` writes each run.json as the run file with
the `workload` every run must share; `record` and `run` write their own, the run file
as `run_file`. `validate.runs` decides which client run is graded, and whether one is,
in a run `record` or `run` wrote and in a `cctraces_sim.sh` run holding
`artifacts/client_run.json`. Rows: `validate.runs`' grades, TTFT p50 per band of
uncached prompt tokens graded alike; per such run, its warmup: how many requests
its export names as such and how many steps hold only those; `validate.cost` on the
steps graded (with the steps it leaves out, declines with their most frequent
reasons, or extrapolates) and `memory.check` per real run; and the simulated run's refusals by source, its steps
extrapolated and priced by a fallback, shares and speed, from its first client run's
summary when it has one. Any refusal makes a
`cctraces_sim.sh` cell a coverage report; the shares grade nothing, and neither do the
counts but preemptions and prefill tokens.

A run `deploy.sh` wrote, a deployment, holds `prefill/` and `decode/` with its client's
export beside them: its warmup, cost and memory rows are per side (`cost real0 prefill`),
each side costed at the law fitted under its engine config and checked against its
engine's hand-in, `lp-engine-P.json` or `lp-engine-D.json`, at DP width above 1
each rank's (`lp-engine-P.dp0.json`, a row per rank); `validate.runs` grades its
schedule per side, its end to end once.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median

from atom.compass.backends.cost import GROUPS
from atom.compass.backends.json_file import read_json, read_jsonl
from atom.compass.backends.step_linear_fit import TOKEN_BANDS, token_band
from atom.compass.memory.check import MemoryRefusal, parts
from atom.compass.run import (
    CLIENT_DIR,
    ENGINE,
    LP_FILE,
    STEP_GRID,
    STEP_LINEAR,
    SUMMARY_FILE,
    _fitted_here,
    entries,
    refuse_renamed_keys,
)
from atom.compass.run.step_grid import keyed_of, read
from atom.compass.validate import cost, runs

BANDS = [token_band(edge) for edge in (0, *TOKEN_BANDS)]
HULL = {"validated": ("validated", "-"), "unvalidated": ("unvalidated", "not gated")}
GATED_COUNTS = {"preemptions", "prefill tokens"}
#: A deployment side's engine LP, whose hand-in its memory is checked against.
HAND_IN = {"prefill": "engine-P", "decode": "engine-D"}


def _ttft(run: Path) -> dict[str, float]:
    by = {}
    for r in read_jsonl(run / "artifacts/profile_export.jsonl"):
        m, phase = r["metrics"], r["metadata"]["benchmark_phase"]
        if phase == "profiling" and "time_to_first_token" in m:
            n = m["usage_prompt_tokens"]["value"]
            n -= m["usage_prompt_cache_read_tokens"]["value"]
            by.setdefault(token_band(n), []).append(m["time_to_first_token"]["value"])
    return {band: median(v) for band, v in by.items()}


def _graded(family: str, name: str, g: dict) -> tuple:
    result = " ".join(f"{r:.4g}" for r in g["result"])
    value, verdict = f"spread {g['spread']:.4g}; result {result}", g["verdict"]
    if family.split()[0] == "counts" and name not in GATED_COUNTS:
        verdict += ", not gated"
    return family, name, value, verdict


def _runs(real: list[Path], sim: Path):
    for family, metrics in runs.grade(real, sim).items():
        for name, g in metrics.items():
            yield _graded(family, name, g)
    every = (*real, sim)
    ttft = [_ttft(r) for r in every]
    for band in BANDS:
        v, name = [t.get(band) for t in ttft], f"ttft p50 {band} uncached"
        if None in v:
            missing = [r.name for r, x in zip(every, v) if x is None]
            yield "end to end", name, f"no request in {missing}", "-"
        else:
            g = runs._grade("end to end", "ttft", v[:-1], v[-1])
            yield _graded("end to end", name, g)


def _warmup(name: str, warmup: dict | None):
    if warmup is not None:  # None: a run graded whole
        value = f"requests {warmup['requests']}, steps {warmup['steps']}"
        yield f"warmup {name}", "warmup", value, "not graded"


def _cost(name: str, run: Path, backend, steps: list[dict]):
    numbers = {s["step"] for s in steps}
    report = cost.cost_accuracy(run / "steps", backend, numbers)
    where = f"cost {name}"
    for group, (hull, gated) in HULL.items():
        for kind, s in report[group].items():
            held = "-" if s["held_out"] is None else f"{s['held_out']:+.2%}"
            value = (
                f"{s['steps']} steps, real {s['real_s']:.2f} s, priced {s['priced_s']:.2f} s;"
                f" on totals {s['on_totals']:+.2%}, holding out step {s['held_out_step']} {held},"
                f" median {s['median']:+.2%}"
            )
            yield where, f"{kind} {hull}", value, gated
            for band, b in s["bands"].items():
                value = f"{b['steps']} steps; p10 {b['p10']:+.1%} p50 {b['p50']:+.1%} p90 {b['p90']:+.1%}"
                yield where, f"{band} {hull}", f"{value}; within 10% {b['within_10pct']:.1%}", gated
    yield where, "left out", f"not labelled {json.dumps(report['cut'])}; warm steps, real s { {w['step']: round(w['real_s'], 2) for w in report['warm']} }", "-"
    for kind in ("fallback", "refused"):
        for band, r in report[kind].items():
            yield where, f"{kind} {band}", cost.declined_text(r), "-"
    yield where, "extrapolated", f"{report['extrapolated']} steps", "-"


def _memory(name: str, sim: Path, engine: str, real: Path):
    """`real`'s memory against `engine`'s hand-in in `sim`, per rank at DP width
    above 1, where each rank hands in as `<engine>.dp<r>`."""
    ranks = sorted(sim.glob(LP_FILE.format(f"{engine}.dp*")))
    for hand_in in ranks or [sim / LP_FILE.format(engine)]:
        rank = f" {hand_in.stem.rsplit('.', 1)[1]}" if ranks else ""
        yield from _checked(f"memory {name}{rank}", hand_in, real, len(ranks) or 1)


def _checked(where: str, hand_in: Path, real: Path, ranks: int):
    try:
        comparison, *_, graph = parts(hand_in, real / "server.log", ranks)
    except MemoryRefusal as e:
        yield where, "memory", e.what, "refused"
        return
    for t in comparison.compared:
        yield where, t.name, f"{t.relative:+.2%} ({t.delta_bytes:+,} B)", str(t.verdict)
    for r in comparison.refused:
        yield where, r.name, r.what, "refused"
    if not isinstance(graph, str):
        graph = f"reserves() {graph.reserves_relative:+.2%}, predicts() {graph.predicts_relative:+.2%}"
    yield where, "graph pool", graph, "-"


def _summary(run: Path, summary: dict):
    schedule, c, where = summary["schedule"], summary["cost"], f"summary {run.name}"
    n = schedule["refusals"]
    verdict = "coverage report" if summary["coverage_report"] else "none"
    cost = (
        "steps not counted"
        if n["steps"] is None
        else f"cost refused {n['fraction_of_steps']:.1%} of {n['steps']} steps"
    )
    yield where, "refusals", f"{n['count']} {json.dumps(dict(n['by_source']))}; {cost}", verdict
    for k in ("extrapolated", "fallback"):
        yield where, f"{k} steps", f"{schedule.get(f'{k}_steps')} of {n['steps']}", "not gated"
    by_method = schedule["seconds_by_method"] or {}
    for g in GROUPS:
        yield where, g, json.dumps(by_method.get(g)), "not gated"
    ratio = c["speed_refused"] or f"{c['speed_ratio']:.1f}x"
    value = f"{ratio}: {c['simulated_seconds']:.0f} simulated s in {c['wall_seconds']:.0f} wall s"
    met = {True: "meets", False: "below"}.get(c["meets_speed_target"], "no result for")
    yield where, "speed", value, f"{met} {c['speed_target_ratio']:g}x"


def _law(run_file: dict, real: Path) -> dict:
    """The run file's step linear law for `real`: of several, the one fitted
    under the engine config its record states."""
    laws = entries(run_file, STEP_LINEAR)
    if len(laws) > 1:
        here = keyed_of(read_json(real / "steps/dp0.engine.json"))
        return _fitted_here(laws, here, lambda e: e.get("engine_config"), STEP_LINEAR)
    return laws[0]


def _grid(sim: Path, run: dict, summary: dict):
    """The backend of the bundle the run file of `run`, `sim`'s run.json, names,
    read with the digest `summary`, its run summary, records for it."""
    recorded = summary.get(STEP_GRID, {}).values()
    digests = sorted({g["digest"] for g in recorded if "digest" in g})
    if len(digests) != 1:
        raise ValueError(
            f"{sim}'s run summary records the step grid bundle digests {digests}, "
            "and grading reads the one bundle its run priced with"
        )
    return read(run.get("run_file", run)[STEP_GRID], *digests)


def rows(cell) -> list[tuple]:
    """The cell's rows, (result, metric, value, verdict), its verdict first."""
    cell = Path(cell)
    real = sorted(r for r in cell.glob("real*") if r.is_dir())
    if (cell / "law").is_dir():
        raise ValueError(f"{cell} holds law/, a run directory now named step_linear/")
    simulated = [cell / k for k in (STEP_LINEAR, STEP_GRID) if (cell / k).is_dir()]
    if len(simulated) != 1:
        raise ValueError(
            f"{cell} holds {[s.name for s in simulated]} of step_linear/ and "
            "step_grid/; a cell holds one simulated run"
        )
    if len(real) < 2:
        raise ValueError(
            f"{cell} holds {len(real)} of real*/; a cell holds two or more real runs"
        )
    (sim,) = simulated
    for needed in ("run.json", runs.EXPORT):
        if bare := [r.name for r in (*real, sim) if not (r / needed).is_file()]:
            raise ValueError(
                f"{cell}'s {bare} hold no {needed}; every run of a cell holds one"
            )
    # A run `run` wrote grades its first client run, so its summary is that run's.
    first = sim / CLIENT_DIR.format(0) / SUMMARY_FILE
    summary_file = first if first.is_file() else sim / SUMMARY_FILE
    if not summary_file.is_file():
        raise ValueError(
            f"{cell}'s {sim.name} holds no run summary, {SUMMARY_FILE} or "
            f"{first.relative_to(sim)}; a simulated run holds one"
        )
    run = {r.name: read_json(r / "run.json") for r in (*real, sim)}
    for name, f in run.items():
        refuse_renamed_keys(f, cell / name / "run.json")
    if len({json.dumps(f.get("workload"), sort_keys=True) for f in run.values()}) != 1:
        raise ValueError(f"{cell}'s runs {sorted(run)} ran different workloads")
    summary = read_json(summary_file)
    verdict = "coverage report" if summary["coverage_report"] else "graded"
    out = [("cell", "verdict", f"{len(real)} real runs", verdict)]
    out += _runs(real, sim)
    grid = _grid(sim, run[STEP_GRID], summary) if sim.name == STEP_GRID else None
    # A deployment's runs per side, each at its law and against its engine's hand-in.
    for r in real:
        for side in runs.sides(r):
            name = " ".join(side.relative_to(cell).parts)
            backend = grid or _law(
                run[STEP_LINEAR].get("run_file", run[STEP_LINEAR]), side
            )
            steps, _traces, warmup = runs.graded(side, r)
            out += [*_warmup(name, warmup), *_cost(name, side, backend, steps)]
            out += _memory(name, sim, HAND_IN.get(side.name, ENGINE.name), side)
    for side in runs.sides(sim):
        name = " ".join(side.relative_to(cell).parts)
        out += _warmup(name, runs.graded(side, sim)[2])
    out += _summary(sim, summary)
    return out


def main(argv=None) -> None:
    print("cell | result | metric | value | verdict")
    for cell in sys.argv[1:] if argv is None else argv:
        for row in rows(cell):
            print(" | ".join((Path(cell).name, *row)))


if __name__ == "__main__":
    try:
        main()
    except json.JSONDecodeError:  # it names no file, so it keeps its traceback
        raise
    except ValueError as refused:  # a refusal: its reason, exit 1, no traceback
        raise SystemExit(str(refused)) from refused

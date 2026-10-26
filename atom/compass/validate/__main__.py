# SPDX-License-Identifier: MIT
"""Grade cells from every source, in one table: `python -m atom.compass.validate CELL...`

A CELL holds runs of one workload: `step_linear/` (at its run file's step linear law, or
of several the one fitted under the engine config each real run states) or `step_grid/`
(at its run file's bundle, which `run.step_grid.read` reads by key with no check against
this process, refused unless its digest is the one the run summary records)
or `op_graph/` (`compass run --op-graph`'s, at the `OpGraphBackend` its run file
and price list give under the ATOM arguments its command line served, built as
`compass run` builds it) and two or more `real*/`. `cctraces_sim.sh` writes each run.json as the run file with
the `workload` every run must share; `record` and `run` write their own, the run file
as `run_file`, and `client.sh` the workload in `artifacts/workload.json`; a refusal
names each field the runs do not share. `validate.runs` decides which client run is
graded, and whether one is, in a run `record` or `run` wrote and in a `cctraces_sim.sh`
run holding `artifacts/client_run.json`, and grades every run whole when a real run
holds none. Rows: `validate.runs`' grades, TTFT p50 per band of
uncached prompt tokens graded alike; per such run, its warmup: how many requests
its export names as such and how many steps hold only those; `validate.cost` on the
steps graded (with the steps it leaves out, declines with their most frequent
reasons, or extrapolates), its graded rows gated and its validated, unvalidated
and declared rows not, and `memory.check` per real run; and the simulated run's refusals by source, its steps
extrapolated and priced by a fallback (with each reason and its steps), shares and speed, from its first client run's
summary when it has one. An `op_graph/` cell's cost rows also give, per phase,
the priced seconds by method and each op kind's seconds over the measured ones,
and the cell `measure ops`'s warm time over its cold one per op kind, over the
points its store holds at the price list's key. Any refusal makes a
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

from atom.compass.artifacts import ArtifactStore, Key, Kind
from atom.compass.artifacts.store import ENTRY_FILE
from atom.compass.backends.cost import GROUPS
from atom.compass.backends.json_file import read_json, read_jsonl
from atom.compass.backends.step_linear_fit import TOKEN_BANDS, token_band
from atom.compass.memory.check import MemoryRefusal, parts
from atom.compass.run import (
    CLIENT_DIR,
    ENGINE,
    LP_FILE,
    OP_GRAPH,
    PRICE_LIST,
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
#: Cost rows' verdict per report group: the graded ones carry the gate.
GATED = {
    "graded": "-",
    "validated": "not gated",
    "unvalidated": "not gated",
    "declared": "not gated",
}
GATED_COUNTS = {"preemptions", "prefill tokens"}
#: A deployment side's engine LP, whose hand-in its memory is checked against.
HAND_IN = {"prefill": "engine-P", "decode": "engine-D"}
#: Where `client.sh` states the workload of the client run it drove.
WORKLOAD = "artifacts/workload.json"


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


def _cost(name: str, run: Path, backend, steps: list[dict], fallback, op=False):
    numbers = {s["step"] for s in steps}
    report = cost.cost_accuracy(run / "steps", backend, numbers, fallback)
    where = f"cost {name}"
    for kind, methods in report["methods"].items() if op else ():
        total = sum(methods.values())
        shares = ", ".join(f"{g} {s / total:.1%}" for g, s in methods.items())
        yield where, f"{kind} by method", f"{total:.2f} s priced: {shares}", "not gated"
        for term, t in report["terms"][kind].items():
            value = f"on totals {t['on_totals']:.3f}; per step p10 {t['p10']:.3f} p50 {t['p50']:.3f} p90 {t['p90']:.3f}"
            yield where, f"{kind} {term} over measured", value, "not gated"
    for group, gated in GATED.items():
        for kind, s in report[group].items():
            held = "-" if s["held_out"] is None else f"{s['held_out']:+.2%}"
            value = (
                f"{s['steps']} steps, real {s['real_s']:.2f} s, priced {s['priced_s']:.2f} s;"
                f" on totals {s['on_totals']:+.2%}, holding out step {s['held_out_step']} {held},"
                f" median {s['median']:+.2%}"
            )
            yield where, f"{kind} {group}", value, gated
            for band, b in s["bands"].items():
                value = f"{b['steps']} steps; p10 {b['p10']:+.1%} p50 {b['p50']:+.1%} p90 {b['p90']:+.1%}"
                yield where, f"{band} {group}", f"{value}; within 10% {b['within_10pct']:.1%}", gated
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
        value = f"{schedule.get(f'{k}_steps')} of {n['steps']}"
        if k == "fallback" and schedule.get("fallback_reasons"):
            value += f"; reasons {json.dumps(schedule['fallback_reasons'])}"
        yield where, f"{k} steps", value, "not gated"
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


def _op_graph(side: Path):
    """The `OpGraphBackend` a `compass run --op-graph` server at `side`
    priced with: its run file under the ATOM arguments its command line served."""
    from atom.compass import run as compass_run
    from atom.compass.drive import run as drive_run

    run = read_json(side / "run.json")
    argv = run.get("argv", [])
    if "--" not in argv:
        raise ValueError(
            f"{side}'s run.json names no ATOM arguments after -- in its argv; "
            "an op graph is read under the ones its server ran"
        )
    atom_args = argv[argv.index("--") + 1 :]
    config = drive_run._config(atom_args)
    machine = compass_run._machine(run["run_file"])
    ep = drive_run._ep_width(atom_args, machine) if config.enable_expert_parallel else 1
    return compass_run.op_graph(run["run_file"], config, machine, ep)


def _warm_over_cold(run_file: dict):
    """`measure ops`'s warm time over its cold one per op kind, over the
    points the store holds at the key of the run file's price list."""
    store, fields, by = ArtifactStore(run_file["store"]), run_file.get(PRICE_LIST), {}
    folder = store.root / Kind.OP_MEASUREMENT.value
    for path in sorted(folder.glob(f"[!.]*/{ENTRY_FILE}")) if fields else ():
        key = read_json(path)["key"]
        if {k: v for k, v in key.items() if k != "point"} == fields:
            entry = store.read(Key.of(Kind.OP_MEASUREMENT, **key))
            point = json.loads(
                entry.read_member("point", entry.topology.ranks()[0], "json")
            )
            by.setdefault(point["op"], []).append(point.get("warm_over_cold"))
    for op, ratios in sorted(by.items()):
        r = [x for x in ratios if x is not None]
        value = f"{len(r)} of {len(ratios)} points timed warm"
        if r:
            value += f"; min {min(r):.3f}, median {median(r):.3f}, max {max(r):.3f}"
        yield "warm over cold", op, value, "not gated"


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


def _odd_workloads(cell: Path, run: dict) -> dict:
    """Each workload field the runs of `cell`, `run` their run.json by name,
    do not all share, with each run's value, its trace digests as how many are
    not every run's. A run states its workload in its run.json, or in
    `artifacts/workload.json` as `client.sh` writes it."""
    path = {n: cell / n / WORKLOAD for n in run}
    stated = {
        n: f.get("workload") or (read_json(path[n]) if path[n].is_file() else {})
        for n, f in run.items()
    }
    odd = {}
    for k in sorted({k for w in stated.values() for k in w}):
        v = {n: w.get(k) for n, w in stated.items()}
        if len({json.dumps(x, sort_keys=True) for x in v.values()}) > 1:
            odd[k] = v
    if digests := odd.get("traces_sha256"):
        every = set.intersection(*(set(d or ()) for d in digests.values()))
        odd["traces_sha256"] = {
            n: len(set(d or ()) - every) for n, d in digests.items()
        }
    return odd


def rows(cell) -> list[tuple]:
    """The cell's rows, (result, metric, value, verdict), its verdict first."""
    cell = Path(cell)
    real = sorted(r for r in cell.glob("real*") if r.is_dir())
    if (cell / "law").is_dir():
        raise ValueError(f"{cell} holds law/, a run directory now named step_linear/")
    simulated = [
        cell / k for k in (STEP_LINEAR, STEP_GRID, OP_GRAPH) if (cell / k).is_dir()
    ]
    if len(simulated) != 1:
        raise ValueError(
            f"{cell} holds {[s.name for s in simulated]} of step_linear/, "
            "step_grid/ and op_graph/; a cell holds one simulated run"
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
    if odd := _odd_workloads(cell, run):
        raise ValueError(f"{cell}'s runs {sorted(run)} ran different workloads: {odd}")
    summary = read_json(summary_file)
    verdict = "coverage report" if summary["coverage_report"] else "graded"
    out = [("cell", "verdict", f"{len(real)} real runs", verdict)]
    out += _runs(real, sim)
    run_file = run[sim.name].get("run_file", run[sim.name])
    grid = _grid(sim, run[STEP_GRID], summary) if sim.name == STEP_GRID else None
    op = sim.name == OP_GRAPH
    # Each side of a deployment served under its own ATOM arguments.
    graphs = {s.relative_to(sim): _op_graph(s) for s in runs.sides(sim)} if op else {}
    fallback = run_file.get("step_roofline", False)
    whole = runs.graded_whole(real)
    # A deployment's runs per side, each at its law and against its engine's hand-in.
    for r in real:
        for side in runs.sides(r):
            name = " ".join(side.relative_to(cell).parts)
            backend = grid or graphs.get(side.relative_to(r)) or _law(run_file, side)
            steps, _traces, warmup = runs.graded(side, r, whole)
            out += _warmup(name, warmup)
            out += _cost(name, side, backend, steps, fallback, op)
            out += _memory(name, sim, HAND_IN.get(side.name, ENGINE.name), side)
    out += _warm_over_cold(run_file) if op else ()
    for side in runs.sides(sim):
        name = " ".join(side.relative_to(cell).parts)
        out += _warmup(name, runs.graded(side, sim, whole, simulated=True)[2])
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

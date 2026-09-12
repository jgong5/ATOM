"""Does a simulated run reproduce the real one's latency?

This is the question the whole project exists to answer, and until it is asked
every claim about Compass is structural — graphs matching graphs, not time
matching time. A structural claim cannot be ranked against another structural
claim, which is why nothing in DESIGN_NOTES.md had a priority attached until
this ran.

Four phases, a process each, because a run's own timings are what the next phase
is fitted to:

    calibrate -- a sweep of shapes, real forward, recording how long each took
    real      -- the evaluation workload, unmodified
    modelled  -- the same workload, forward replaced by a cost fitted to the sweep
    compare   -- per-request TTFT and TPOT, real against modelled

Calibrating on a sweep rather than on the evaluation workload is deliberate twice
over: a fixed workload prefills everything in one or two steps, too few to fit
against, and evaluating on shapes the model was not shown makes the reported
number a generalisation error rather than a fit residual.

The comparison is deliberately tight: same engine, same scheduler, same
admission decisions, same prompts. Only the forward differs. So the error is
attributable to the cost model rather than to two benchmarks disagreeing about
what they ran.

    python scripts/compass/validate.py --model M --num-prompts 8
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


#: Prefix every Compass warning carries, so a captured phase can be scanned for
#: them without depending on the log format naming the level.
MARKER = "ATOMCompass WARNING:"


def _run(cmd: list[str]) -> None:
    """Run one phase, surfacing anything Compass wanted to say about it.

    Phases are captured rather than streamed, because an engine start-up is
    thousands of lines and none of them are the point. But capturing them
    swallowed the warnings too, and those *are* the point: the oracle warns when
    asked to cost a step outside the range it was calibrated over, and that
    warning exists precisely so a bad number announces itself rather than being
    read off the table as fact. Swallowed, the safeguard was inert in the one
    workflow anybody actually uses.
    """
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-4000:])
        raise SystemExit(f"phase failed ({proc.returncode}): {' '.join(cmd[:6])} ...")

    seen = set()
    for line in (proc.stdout + proc.stderr).splitlines():
        # Matched on the message, not the log level: ATOM's format prints
        # "[atom.compass.x 00:00:00] ..." and never names the level, so
        # filtering on "WARNING" as a field matches nothing at all. Compass
        # warnings carry the word in the message itself for exactly this reason;
        # tests/compass enforces that they all do.
        if MARKER not in line:
            continue
        message = line[line.index(MARKER):]
        if message not in seen:
            seen.add(message)
            print(f"  ! {message}")


def _steps_fitted(table: Path) -> str:
    """Rows the oracle was fitted to, *per rank*.

    Each rank writes its own table under parallelism, so the single path this
    script asked for may not be the file that exists — counting it
    unconditionally is how a TP=2 run first failed here. Summing across ranks
    instead would be worse than failing: every rank fits to its own table, so a
    total overstates the evidence behind any one prediction by the world size.
    """
    own = sorted(table.parent.glob(f"{table.stem}.*{table.suffix}"))
    files = own or ([table] if table.exists() else [])
    if not files:
        return "none"
    counts = [sum(1 for _ in f.open()) for f in files]
    if len(counts) == 1:
        return str(counts[0])
    lo, hi = min(counts), max(counts)
    span = str(lo) if lo == hi else f"{lo}-{hi}"
    return f"{span} per rank, {len(counts)} ranks"


def _percent_error(real: float, modelled: float) -> float:
    return 100.0 * (modelled - real) / real if real else float("nan")


def _summarise(name: str, real: list[float], modelled: list[float]) -> dict:
    r, m = statistics.fmean(real), statistics.fmean(modelled)
    per_request = [
        abs(_percent_error(a, b)) for a, b in zip(real, modelled) if a
    ]
    return {
        "metric": name,
        "real_mean": r,
        "modelled_mean": m,
        "mean_error_pct": _percent_error(r, m),
        "median_abs_request_error_pct": (
            statistics.median(per_request) if per_request else float("nan")
        ),
        "worst_request_error_pct": max(per_request) if per_request else float("nan"),
    }


def _measure_admission(real_out: Path, real_table: Path) -> float:
    """How long a request takes to become schedulable, from the real run.

    A simulated engine advances its clock by the forward durations it predicts,
    so the hops between a request arriving and a worker having it in hand cost
    nothing -- and that was the whole of the TTFT error. It was a hand-passed
    constant until now, which is a guess dressed as a measurement.

    Measured instead as ``TTFT - the prefill step that produced the first
    token``. Exact for an offline workload, where every request arrives at once
    and one prefill step serves them all, so there is no queueing between the two
    to confuse for admission. It degrades where requests queue, which is why it
    stays overridable.

    Deriving it from the *modelled* run would be circular -- it would absorb
    whatever TTFT error the oracle has and report zero. This uses only the real
    run, so the comparison stays honest.
    """
    try:
        requests = json.loads(real_out.read_text())["requests"]
        steps = [json.loads(line) for line in
                 real_table.read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return 0.0
    prefill = [s["seconds"] for s in steps if s.get("num_prefill_tokens")]
    ttfts = [r["ttft"] for r in requests if r.get("ttft")]
    if not prefill or not ttfts:
        return 0.0
    return max(0.0, statistics.fmean(ttfts) - prefill[0])


def _one_pass(args, work: Path) -> tuple[list[dict], dict, dict, str, float]:
    """Calibrate, run for real, run modelled, compare. One draw."""
    work.mkdir(parents=True, exist_ok=True)
    table = work / "steps.jsonl"
    real_table = work / "real_steps.jsonl"
    real_out = work / "real.json"
    modelled_out = work / "modelled.json"

    common = [
        args.python, "scripts/compass/run.py",
        "--model", args.model, "-tp", str(args.tp),
        "--num-prompts", str(args.num_prompts),
        "--max-tokens", str(args.max_tokens),
        "--prompt-tokens", str(args.prompt_tokens),
    ]

    print("  phase 1/4  calibrating on a sweep of shapes ...", flush=True)
    _run(common + ["--out", str(work / "sweep.json"), "--sweep",
                   "--compass", "--compass-mode", "measure",
                   "--compass-measure-out", str(table),
                   # One of each kind: drops the launch that autotunes, keeps
                   # every other prefill sample the sweep produced.
                   "--compass-measure-warmup-steps", "1"])

    print("  phase 2/4  running the evaluation workload for real ...", flush=True)
    # Recorded rather than merely run, so admission can be measured from it.
    # Recording is free: over three repeats a measured run and a plain one agree
    # on TTFT to 2% and on TPOT and latency to well under 1%, all inside the
    # run-to-run spread of a shared GPU.
    _run(common + ["--out", str(real_out), "--compass",
                   "--compass-mode", "measure",
                   "--compass-measure-out", str(real_table)])

    print("  phase 3/4  running it again, modelled ...", flush=True)
    modelled_cmd = common + [
        "--out", str(modelled_out), "--compass",
        "--compass-oracle",
        "atom.compass.core.cost.calibrated.CalibratedCostOracle",
        "--compass-oracle-option", f"table={table}",
    ]
    admission = args.admission_seconds or _measure_admission(real_out, real_table)
    if admission:
        modelled_cmd += ["--compass-admission-seconds", str(admission)]
    _run(modelled_cmd)

    print("  phase 4/4  comparing", flush=True)
    real = json.loads(real_out.read_text())
    modelled = json.loads(modelled_out.read_text())

    if len(real["requests"]) != len(modelled["requests"]):
        raise SystemExit("the two runs served different numbers of requests")

    rows = [
        _summarise("TTFT", [r["ttft"] for r in real["requests"]],
                   [r["ttft"] for r in modelled["requests"]]),
        _summarise("TPOT", [r["tpot"] for r in real["requests"]],
                   [r["tpot"] for r in modelled["requests"]]),
        _summarise("latency", [r["latency"] for r in real["requests"]],
                   [r["latency"] for r in modelled["requests"]]),
    ]
    return rows, real, modelled, _steps_fitted(table), admission


def main() -> int:
    ap = argparse.ArgumentParser(description="ATOMCompass end-to-end validation")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--workdir", default="compass_artifacts")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument(
        "--admission-seconds", type=float, default=0.0,
        help="Seconds a request takes to become schedulable, passed to the "
             "modelled run. A simulated engine advances its clock by predicted "
             "forward durations, so the two process hops between preprocess and "
             "a worker cost nothing -- which is the whole of a -20%% TTFT error "
             "and nothing to do with the cost model. Measure it: run the "
             "workload under --compass-mode=measure and take mean TTFT minus "
             "the prefill step that produced the first token. It is specific to "
             "the entry path, so the offline number (~13ms here) is not the "
             "serving one (~9ms). Default 0 leaves it unmodelled.",
    )
    ap.add_argument(
        "--repeats", type=int, default=1,
        help="Repeat the whole pipeline this many times and report a spread. "
             "One run cannot distinguish an improvement from a lucky draw: "
             "five repeats of the identical command have moved TPOT error from "
             "-5.4%% to +2.4%%. Three or more before believing a difference.",
    )
    args = ap.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")

    work = Path(args.workdir)
    passes = []
    for i in range(args.repeats):
        target = work if args.repeats == 1 else work / f"rep{i + 1}"
        if args.repeats > 1:
            print(f"repeat {i + 1}/{args.repeats}", flush=True)
        passes.append(_one_pass(args, target))

    rows0, real, modelled, fitted, admission = passes[0]
    print()
    print("ATOMCompass end-to-end validation")
    print("=" * 78)
    print(f"  model      : {args.model.rstrip('/').split('/')[-1]}, tp={args.tp}")
    print(f"  requests   : {len(real['requests'])}, "
          f"{args.prompt_tokens} prompt tokens, {args.max_tokens} output tokens")
    print(f"  steps fitted: {fitted}")
    print(f"  repeats    : {args.repeats}")
    print(f"  admission  : {admission*1000:.1f}ms"
          + (" (given)" if args.admission_seconds else
             " (measured from the real run as TTFT minus its prefill step)"))
    print()

    if args.repeats == 1:
        print(f"  {'metric':<9} {'real':>10} {'modelled':>10} {'mean err':>10} "
              f"{'median req':>11} {'worst req':>10}")
        for row in rows0:
            print(f"  {row['metric']:<9} {row['real_mean']*1000:>9.2f}ms "
                  f"{row['modelled_mean']*1000:>9.2f}ms "
                  f"{row['mean_error_pct']:>+9.1f}% "
                  f"{row['median_abs_request_error_pct']:>10.1f}% "
                  f"{row['worst_request_error_pct']:>9.1f}%")
        print()
        print(f"  wall clock : real {real['wall']:.2f}s, "
              f"modelled {modelled['wall']:.2f}s "
              f"({real['wall'] / modelled['wall']:.1f}x faster)")
        print("=" * 78)
        print("  One run. It cannot tell an improvement from a lucky draw --")
        print("  pass --repeats 3 or more before believing a difference.")
    else:
        print(f"  {'metric':<9} {'mean err':>10} {'sd':>8} {'min':>9} {'max':>9}"
              f"   per run")
        for k, name in enumerate(("TTFT", "TPOT", "latency")):
            errs = [p[0][k]["mean_error_pct"] for p in passes]
            mean = statistics.fmean(errs)
            sd = statistics.stdev(errs) if len(errs) > 1 else 0.0
            each = " ".join(f"{e:+.1f}" for e in errs)
            print(f"  {name:<9} {mean:>+9.1f}% {sd:>7.1f} {min(errs):>+8.1f}% "
                  f"{max(errs):>+8.1f}%   {each}")
        print("=" * 78)
        flagged = [
            n for k, n in enumerate(("TTFT", "TPOT", "latency"))
            if (lambda e: abs(statistics.fmean(e)) < statistics.stdev(e))(
                [p[0][k]["mean_error_pct"] for p in passes])
        ]
        if flagged:
            print(f"  Inside the noise: {', '.join(flagged)} -- the spread is "
                  f"wider than the error.")
            print("  Do not quote these as accuracy without more repeats.")
        if args.repeats < 3:
            print("  Two draws do not estimate a spread. Use 3 or more.")
        print("  Repeats run back-to-back share whatever the machine was doing")
        print("  at the time, so this sd is a lower bound on the real one.")

    (work / "validation.json").write_text(json.dumps(
        {"repeats": args.repeats, "tp": args.tp,
         "passes": [p[0] for p in passes]}, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())

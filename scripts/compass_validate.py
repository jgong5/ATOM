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

    python scripts/compass_validate.py --model M --num-prompts 8
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-4000:])
        raise SystemExit(f"phase failed ({proc.returncode}): {' '.join(cmd[:6])} ...")


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


def main() -> int:
    ap = argparse.ArgumentParser(description="ATOMCompass end-to-end validation")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--workdir", default="compass_artifacts")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    table = work / "steps.jsonl"
    real_out = work / "real.json"
    modelled_out = work / "modelled.json"

    common = [
        args.python, "scripts/compass_run.py",
        "--model", args.model, "-tp", str(args.tp),
        "--num-prompts", str(args.num_prompts),
        "--max-tokens", str(args.max_tokens),
        "--prompt-tokens", str(args.prompt_tokens),
    ]

    print("phase 1/4  calibrating on a sweep of shapes ...")
    _run(common + ["--out", str(work / "sweep.json"), "--sweep",
                   "--compass", "--compass-mode", "measure",
                   "--compass-measure-out", str(table),
                   # One of each kind: drops the launch that autotunes, keeps
                   # every other prefill sample the sweep produced.
                   "--compass-measure-warmup-steps", "1"])

    print("phase 2/4  running the evaluation workload for real ...")
    _run(common + ["--out", str(real_out)])

    print("phase 3/4  running it again, modelled ...")
    _run(common + ["--out", str(modelled_out), "--compass",
                   "--compass-oracle",
                   "atom.compass.core.cost.calibrated.CalibratedCostOracle",
                   "--compass-oracle-option", f"table={table}"])

    print("phase 4/4  comparing\n")
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

    print("ATOMCompass end-to-end validation")
    print("=" * 78)
    print(f"  model      : {args.model.rstrip('/').split('/')[-1]}")
    print(f"  requests   : {len(real['requests'])}, "
          f"{args.prompt_tokens} prompt tokens, {args.max_tokens} output tokens")
    print(f"  steps fitted: {sum(1 for _ in table.open())}")
    print()
    print(f"  {'metric':<9} {'real':>10} {'modelled':>10} {'mean err':>10} "
          f"{'median req':>11} {'worst req':>10}")
    for row in rows:
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
    print("  Calibrated on a sweep of shapes, evaluated on a workload it did not")
    print("  see. The error is a generalisation error, not a fit residual.")

    (work / "validation.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Sweep closed-loop client counts, real and modelled, into one saturation curve.

One rung per client count, each a full `replay_validate.py` run in its own
directory, then `saturation.py` over all of them.

The whole sweep shares one calibration table and one trace, and that is the
point of driving it from here rather than from a shell loop. A rung calibrated
against a different table is not a point on the same curve -- it is a different
model -- and the difference does not show up anywhere in the rung's own output.

    python scripts/compass/cc_sweep.py --model Qwen/Qwen3.8-27B \
        --trace out/cc/trace.jsonl --table out/cc/steps.jsonl \
        --out-dir out/sweep --clients 1 4 8 16 --sessions-per-client 1

Rungs run smallest first. The small ones are cheap and exercise the same
plumbing, so a mistake that would waste the 16-client rung's GPU hour surfaces
in the first few minutes.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_for_sweep",
                                                  _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--trace", required=True)
    p.add_argument("--table", required=True,
                   help="the calibration every rung is priced by. Required, "
                        "not optional: a sweep that calibrated per rung would "
                        "compare points measured against different models")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--clients", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--sessions-per-client", type=int, default=1)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--keep-going", action="store_true",
                   help="run the remaining rungs after one fails. The curve is "
                        "still withheld; this is for getting the rest of the "
                        "data out of an expensive night")
    # Everything else is replay_validate's, passed through untouched.
    p.add_argument("--rest", nargs=argparse.REMAINDER, default=[],
                   help="further flags for replay_validate.py, after --rest")
    args = p.parse_args(argv)

    validate = _load("replay_validate")
    saturation = _load("saturation")

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    rungs, failures = [], []
    for n in sorted(args.clients):
        work = root / f"c{n}"
        print(f"\n=== rung: {n} client(s) -> {work} ===", flush=True)
        rc = validate.main([
            "--model", args.model, "--trace", args.trace,
            "--table", args.table, "--out-dir", str(work),
            "--clients", str(n),
            "--sessions-per-client", str(args.sessions_per_client),
            *args.rest])
        rungs.append(str(work))
        if rc != 0:
            # Recorded and carried, not raised. A rung can fail its accuracy
            # verdict and still hold two usable artifacts, and the curve is
            # about throughput, which is a different question from whether the
            # per-request error passed.
            failures.append({"clients": n, "rc": rc, "dir": str(work)})
            print(f"rung {n} returned {rc}", file=sys.stderr)
            if not args.keep_going:
                break

    print("\n=== curve ===", flush=True)
    rc = saturation.main([*rungs, "--gpus", str(args.gpus),
                          "--out", str(root / "saturation.json"),
                          "--plot", str(root / "saturation.png")])
    (root / "sweep.json").write_text(json.dumps(
        {"clients": args.clients, "sessions_per_client": args.sessions_per_client,
         "table": args.table, "trace": args.trace, "rungs": rungs,
         "rung_failures": failures}, indent=1))
    if failures:
        print(f"\n{len(failures)} rung(s) did not pass their own verdict; see "
              f"{root / 'sweep.json'}", file=sys.stderr)
    return rc or (1 if failures else 0)


if __name__ == "__main__":
    sys.exit(main())

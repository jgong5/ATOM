"""Does a calibration table reach where a run actually goes?

A sweep's rounds say what was asked for. The table says what was measured, and
they are not the same thing: rounds are clamped to the model's context window
silently, so a sweep asked for 262144 tokens against a 262144 window once
produced coverage that stopped 90k short with nothing in the log to say so. A
commit here claimed the coverage from the rounds and was wrong.

So this reads the table, per CUDA-graph rung, and lays a real run's steps beside
it. The oracle already warns per step when it extrapolates, but a warning during
a run is easy to lose in a log and says nothing about how far outside it is or
how many steps went there. This answers both before the run.

    python scripts/compass/coverage.py sweep.tp0.jsonl [run_steps.tp0.jsonl]

Decode is reported per rung because that is how it is fitted -- one small model
per bucket -- so a gap in one rung is not covered by evidence in another. Prefill
is reported over the chunk shapes, which share a single fit.

Exits non-zero when a run goes outside the table, so it can gate a comparison
rather than only describe one.
"""

import argparse
import json
import sys
from collections import defaultdict


def _rows(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _decode_by_rung(path):
    out = defaultdict(list)
    for row in _rows(path):
        if (row.get("num_prefill_tokens") or 0) > 0:
            continue
        bucket = row.get("capture_bucket")
        if bucket:
            out[int(bucket)].append(sum(row.get("context_lens") or []))
    return out


def _prefill(path):
    tokens, context = [], []
    for row in _rows(path):
        n = row.get("num_prefill_tokens") or 0
        if n <= 0:
            continue
        tokens.append(n)
        context.append(sum(row.get("context_lens") or []))
    return tokens, context


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", help="calibration table written by --compass-mode=measure")
    ap.add_argument("steps", nargs="?", help="a run's step table to check against it")
    args = ap.parse_args(argv)

    sweep = _decode_by_rung(args.table)
    run = _decode_by_rung(args.steps) if args.steps else {}

    print("decode, by CUDA-graph rung (total context across the batch)")
    print(f"{'rung':>6} {'sweep n':>8} {'sweep max':>12} "
          f"{'run n':>7} {'run max':>12}   verdict")
    outside = 0
    for rung in sorted(set(sweep) | set(run)):
        s, r = sweep.get(rung, []), run.get(rung, [])
        smax, rmax = (max(s) if s else 0), (max(r) if r else 0)
        if not r:
            verdict = "not used by the run"
        elif not s:
            verdict = "NO SAMPLES AT THIS RUNG"
            outside += 1
        elif rmax <= smax:
            verdict = "covered"
        else:
            verdict = f"EXTRAPOLATES {rmax / max(smax, 1):.1f}x on "
            verdict += f"{sum(1 for v in r if v > smax)} of {len(r)} steps"
            outside += 1
        print(f"{rung:>6} {len(s):>8} {smax:>12} {len(r):>7} {rmax:>12}   {verdict}")

    st, sc = _prefill(args.table)
    if st:
        print("\nprefill (one fit, so reported over all chunk shapes)")
        print(f"  sweep: {len(st)} chunks, tokens {min(st)}-{max(st)}, "
              f"context {min(sc)}-{max(sc)}")
        if args.steps:
            rt, rc = _prefill(args.steps)
            if rt:
                print(f"  run:   {len(rt)} chunks, tokens {min(rt)}-{max(rt)}, "
                      f"context {min(rc)}-{max(rc)}")
                for name, sv, rv in (("tokens", st, rt), ("context", sc, rc)):
                    if max(rv) > max(sv):
                        print(f"  ^ run exceeds the table's {name} by "
                              f"{max(rv) / max(max(sv), 1):.2f}x")
                        outside += 1

    print(f"\nrungs or axes the run pushes past the table: {outside}")
    return 1 if outside else 0


if __name__ == "__main__":
    sys.exit(main())

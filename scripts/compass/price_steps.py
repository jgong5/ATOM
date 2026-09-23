"""Price a recorded step table, and report the error so it cannot cancel.

The headline for the 27B's prefill was +0.10%. It was arithmetically correct and
it described a model that was +3.06% on every step but one, and -98% on the one:
the run's first step took 6.87 s and was priced at 0.13 s, and that single
-6.75 s offset +6.99 s spread over the other 105 chunks.

Nothing in the number gave that away, and it mattered. A scheduler consumes the
*running* sum, not the total, so a bias the total cancels still moves every
decision after it -- that 3% is what put the 27B's TTFT 90% high while its
totals stayed within 1%. This is the second time a metric here has looked right
because two errors offset, so the total is no longer reported on its own:

* the total, and the total with the single largest contributor held out
* the median per-step relative error, which no single step can move
* the steps that contribute most of the error, named

If holding one step out moves the headline by more than a couple of points, the
headline was describing that step and not the model, and this says so.

    python scripts/compass/price_steps.py sweep.jsonl run_steps.jsonl

Both arguments are JSONL step tables written by ``--compass-mode=measure``: the
first is fitted, the second is priced against the fit. Passing the same file
twice reports a fit residual, not a prediction, and the run is labelled as such.

``--oracle`` chooses what does the pricing. This matters because the other way
to read a cost model's error -- replay the workload and compare the result --
measures cost error and schedule divergence together, and those are separable:
a step cost that is 10% high moves which sequences are resident at each step,
and the scheduler then makes different decisions, so the replayed difference is
not the model's error. Here the steps are given, so nothing can diverge and what
comes out is the cost model alone.

    python scripts/compass/price_steps.py sweep.jsonl run_steps.jsonl \
        --oracle priced \
        --oracle-option prices=bench.json \
        --oracle-option graph=graph.json \
        --oracle-option prefill_graph=graph.prefill.json
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from atom.compass.core.cost.base import StepShape  # noqa: E402
from atom.utils import resolve_obj_by_qualname  # noqa: E402
from replay_validate import ORACLES, TABLE_KEY  # noqa: E402


def _rows(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _shape(row):
    return StepShape(
        num_scheduled_tokens=tuple(row["num_scheduled_tokens"]),
        context_lens=tuple(row["context_lens"]),
        num_prefill_tokens=row["num_prefill_tokens"],
        prefix_cache_hit_tokens=tuple(row.get("prefix_cache_hit_tokens") or ()),
        topology=row.get("topology") or {},
        rank_coords=row.get("rank_coords") or {},
        capture_bucket=row.get("capture_bucket"),
    )


def _options(items):
    """``K=V`` strings to constructor keywords, typed as the engine types them.

    Int, then float, then left a string -- the same coercion ``EngineArgs``
    applies to ``--compass-oracle-option``, so an oracle configured here and
    the same oracle configured on an engine command line are handed the same
    values rather than a float in one place and "0.5" in the other.
    """
    options = {}
    for item in items:
        key, _, value = item.partition("=")
        try:
            options[key] = int(value)
        except ValueError:
            try:
                options[key] = float(value)
            except ValueError:
                options[key] = value
    return options


def _configure(oracle, table, items):
    """(qualname, constructor keywords) for a named oracle and a step table.

    The positional table is the default for whichever keyword the chosen oracle
    reads it through: ``table`` for the ones that fit it, ``fallback`` for the
    priced oracle, which fits nothing and only falls back to it for the steps
    its graphs do not cover. An oracle absent from ``TABLE_KEY`` is handed no
    table at all, because a constructor that does not take one raises on it.

    Named explicitly, ``--oracle-option`` wins: the default exists so the
    common case needs no flag, not to stop a table being pointed elsewhere.
    """
    qualname = ORACLES.get(oracle, oracle)
    options = _options(items)
    key = TABLE_KEY.get(qualname)
    if key and key not in options:
        options[key] = table
    return qualname, options


def _report(name, samples, held_out_threshold):
    """samples: (index, measured, priced) in the order the steps ran."""
    if not samples:
        print(f"  {name}: no steps")
        return
    measured = sum(m for _, m, _ in samples)
    priced = sum(p for _, _, p in samples)
    if measured <= 0:
        print(f"  {name}: {len(samples)} steps, nothing measurable")
        return
    total_err = 100.0 * (priced - measured) / measured

    # Held out: the one step whose signed error contributes most in seconds.
    # Signed, not absolute, because the failure being guarded against is
    # cancellation -- an offsetting error is exactly the one that hides.
    worst = max(samples, key=lambda s: abs(s[2] - s[1]))
    rest = [s for s in samples if s[0] != worst[0]]
    rest_m = sum(m for _, m, _ in rest)
    rest_p = sum(p for _, _, p in rest)
    rest_err = 100.0 * (rest_p - rest_m) / rest_m if rest_m > 0 else float("nan")

    per_step = [100.0 * (p - m) / m for _, m, p in samples if m > 0]
    median = statistics.median(per_step)

    print(f"  {name}: {len(samples)} steps  "
          f"measured {measured:.2f}s  priced {priced:.2f}s")
    print(f"    on totals            {total_err:+7.2f}%")
    print(f"    holding out step {worst[0]:<5} {rest_err:+7.2f}%   "
          f"(measured {worst[1]:.2f}s, priced {worst[2]:.2f}s, "
          f"{worst[2] - worst[1]:+.2f}s)")
    print(f"    median per step      {median:+7.2f}%")

    if abs(rest_err - total_err) > held_out_threshold:
        print(f"    ^ one step moves the total by "
              f"{abs(rest_err - total_err):.1f} points. The total is describing "
              f"that step, not the model.")

    ranked = sorted(samples, key=lambda s: -abs(s[2] - s[1]))[:5]
    print("    largest contributors (step, measured, priced, error):")
    for i, m, p in ranked:
        print(f"      {i:>6} {m:>9.3f}s {p:>9.3f}s {p - m:>+9.3f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table",
                    help="step table the oracle reads: fitted by the "
                         "calibrated oracle, used as the priced oracle's "
                         "fallback")
    ap.add_argument("steps", nargs="?",
                    help="step table to price; defaults to the calibration one, "
                         "which reports a fit residual rather than a prediction")
    ap.add_argument("--oracle", default="calibrated",
                    help=f"what prices the steps: one of "
                         f"{', '.join(sorted(ORACLES))}, or a qualname. "
                         f"Default calibrated, which is what every rung so far "
                         f"was measured under. A qualname, and constant, are "
                         f"handed no step table unless you name one with "
                         f"--oracle-option.")
    ap.add_argument("--oracle-option", action="append", default=[],
                    metavar="K=V",
                    help="a keyword argument for the oracle's constructor, "
                         "repeatable. --oracle priced needs prices= and "
                         "graph=, and reaches prefill only with "
                         "prefill_graph=.")
    ap.add_argument("--held-out-threshold", type=float, default=2.0,
                    help="how many points one step may move the total before "
                         "that is called out (default 2.0)")
    args = ap.parse_args(argv)

    steps_path = args.steps or args.table
    qualname, options = _configure(args.oracle, args.table, args.oracle_option)
    oracle = resolve_obj_by_qualname(qualname)(**options)
    print(oracle.describe())
    if options.get("table") == steps_path:
        # Only the oracles that *fit* the table can report a residual against
        # it. The priced oracle reads the same file as a fallback, and pricing
        # a step table it falls back to is still a prediction.
        print("  NOTE: priced against its own calibration table. This is a fit "
              "residual, not a prediction.")

    prefill, decode = [], []
    for i, row in enumerate(_rows(steps_path)):
        measured = float(row.get("seconds") or 0.0)
        priced = oracle.estimate(_shape(row)).seconds
        target = prefill if (row.get("num_prefill_tokens") or 0) > 0 else decode
        target.append((i, measured, priced))

    print(f"priced {steps_path} with {qualname}")
    _report("prefill", prefill, args.held_out_threshold)
    _report("decode", decode, args.held_out_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Price a real run's own steps with the oracle that predicted the simulated one.

A simulated run can be wrong about a workload in three separable ways, and the
end-to-end comparison in `compare.py` cannot tell them apart:

  1. **The oracle prices a step wrongly.** Given the exact shape the real engine
     computed, it returns the wrong duration.
  2. **The simulated engine runs a different set of steps.** The oracle drives
     the clock, the clock drives the scheduler, so a small per-step error
     changes what gets batched and the two runs stop being comparable step for
     step.
  3. **Time passes that no step accounts for.** Between one forward returning
     and the next beginning, the engine detokenizes, samples, posts outputs and
     schedules. A simulated run advances its clock by predicted forward
     durations only, so all of that is free.

This replays the *real* run's step table through the *same* oracle the simulated
run used, which holds the step set fixed and isolates (1). What is left over --
the real run's wall time minus its own forwards -- is (3), and it is already
recorded per step as `gap_seconds`. The difference between the two step tables
is (2).

    python scripts/compass/price_steps.py --cell agent_scratch/poc/matrix/tp1_short_idle

Nothing here is fitted. It reads the artifacts a cell already wrote and prices
them with the oracle named in the modelled run's own provenance, so a correction
cannot be tuned against the numbers it is being judged by.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from atom.compass.core.cost.base import StepShape  # noqa: E402
from atom.utils import resolve_obj_by_qualname  # noqa: E402


def load_steps(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a table flushed per step can end mid-row
    return rows


def shape_of(row: dict) -> StepShape:
    return StepShape(
        num_scheduled_tokens=tuple(row.get("num_scheduled_tokens") or ()),
        context_lens=tuple(row.get("context_lens") or ()),
        num_prefill_tokens=int(row.get("num_prefill_tokens") or 0),
        topology=dict(row.get("topology") or {}),
        rank_coords=dict(row.get("rank_coords") or {}),
        capture_bucket=row.get("capture_bucket"),
        compiled=row.get("compiled"),
    )


def oracle_from(modelled_path: str, rank_coords: dict | None = None):
    """The oracle the simulated run declared, rebuilt from its own manifest.

    ``rank_coords`` matters under parallelism: each rank wrote its own table and
    the oracle resolves the suffix itself, so pricing rank 0's steps has to ask
    for rank 0's table rather than the unsuffixed name that does not exist.
    """
    blob = json.loads(open(modelled_path, encoding="utf-8").read())
    compass = (((blob.get("run") or {}).get("server") or {}).get("compass") or {})
    qualname = compass.get("oracle")
    options = dict(compass.get("oracle_options") or {})
    if rank_coords:
        options["rank_coords"] = rank_coords
    if not qualname:
        raise SystemExit(f"{modelled_path} does not say which oracle served it")
    cls = resolve_obj_by_qualname(qualname)
    return cls(**options), qualname, options, compass


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    s = sorted(values)
    return {"n": len(s), "median": s[len(s) // 2], "mean": statistics.fmean(s),
            "p10": s[int(0.1 * len(s))], "p90": s[min(len(s) - 1, int(0.9 * len(s)))],
            "min": s[0], "max": s[-1], "sum": sum(s)}


def _fmt(label: str, st: dict, unit: str = "%") -> str:
    if not st.get("n"):
        return f"    {label:<22} none"
    return (f"    {label:<22} n={st['n']:<5} med {st['median']:+8.2f}{unit}  "
            f"mean {st['mean']:+8.2f}{unit}  p10 {st['p10']:+8.2f}{unit}  "
            f"p90 {st['p90']:+8.2f}{unit}")


def _dump(priced: list, kind: str, limit: int) -> None:
    """The individual steps, so a bad aggregate can be attributed to shapes."""
    rows = [(r, w, g) for r, w, g in priced
            if (int(r.get("num_prefill_tokens") or 0) > 0) == (kind == "prefill")]
    rows.sort(key=lambda t: abs((t[2] - t[1]) / t[1]) if t[1] else 0,
              reverse=True)
    if not rows:
        return
    print(f"\n    worst {min(limit, len(rows))} {kind} steps of {len(rows)}")
    print(f"      {'tokens':>7} {'ctx':>8} {'bs':>4} {'bucket':>7} "
          f"{'real s':>9} {'priced s':>9} {'err':>9}")
    for r, was, got in rows[:limit]:
        shape = shape_of(r)
        print(f"      {shape.total_tokens:>7} {sum(shape.context_lens):>8} "
              f"{shape.batch_size:>4} {str(r.get('capture_bucket')):>7} "
              f"{was:>9.4f} {got:>9.4f} {_pct(got, was):>9}")


def report(cell: str, dump: int = 0) -> dict:
    real_steps = load_steps(_resolve(os.path.join(cell, "real_steps.jsonl")))
    modelled_steps = load_steps(_resolve(os.path.join(cell,
                                                      "modelled_steps.jsonl")))
    rank_coords = dict((real_steps[0].get("rank_coords") or {})) \
        if real_steps else {}
    oracle, qualname, options, compass = oracle_from(
        os.path.join(cell, "modelled.json"), rank_coords)

    priced, errors = [], []
    by_kind = {"prefill": [], "decode": []}
    seconds_by_kind = {"prefill": [0.0, 0.0], "decode": [0.0, 0.0]}
    for row in real_steps:
        shape = shape_of(row)
        got = float(oracle.estimate(shape).seconds)
        was = float(row.get("seconds") or 0.0)
        priced.append((row, was, got))
        kind = "prefill" if shape.is_prefill else "decode"
        seconds_by_kind[kind][0] += was
        seconds_by_kind[kind][1] += got
        if was > 0:
            err = (got - was) / was * 100.0
            errors.append(err)
            by_kind[kind].append(err)

    gaps = [float(r["gap_seconds"]) for r in real_steps
            if r.get("gap_seconds") is not None]
    forward_s = sum(w for _, w, _ in priced)
    priced_s = sum(g for _, _, g in priced)
    gap_s = sum(gaps)

    # Elapsed time is observed, not summed. Forward seconds and host gaps are
    # each measured against a different pair of instants and can overlap, so
    # adding them is not the run's duration; the first step's start to the last
    # step's end is.
    stamped = [(r["started_at"], float(r.get("seconds") or 0.0))
               for r in real_steps if r.get("started_at")]
    span = (max(t + d for t, d in stamped) - min(t for t, _ in stamped)) \
        if len(stamped) > 1 else 0.0

    mod_prefill = sum(1 for r in modelled_steps
                      if int(r.get("num_prefill_tokens") or 0) > 0)
    real_prefill = sum(1 for r in real_steps
                       if int(r.get("num_prefill_tokens") or 0) > 0)

    tail = priced[1:]
    steady_real = sum(w for _, w, _ in tail)
    steady_priced = sum(g for _, _, g in tail)

    out = {
        "cell": cell,
        "first_step_real_s": priced[0][1] if priced else 0.0,
        "first_step_priced_s": priced[0][2] if priced else 0.0,
        "steady_real_forward_s": steady_real,
        "steady_priced_forward_s": steady_priced, "oracle": qualname, "oracle_options": options,
        "oracle_option_sha256": compass.get("oracle_option_sha256"),
        "real_steps": len(real_steps), "modelled_steps": len(modelled_steps),
        "real_prefill_steps": real_prefill, "modelled_prefill_steps": mod_prefill,
        "step_error_pct": _stats(errors),
        "prefill_error_pct": _stats(by_kind["prefill"]),
        "decode_error_pct": _stats(by_kind["decode"]),
        "real_forward_s": forward_s, "priced_forward_s": priced_s,
        "real_gap_s": gap_s, "gap_samples": len(gaps),
        "real_step_span_s": span,
        "prefill_seconds": seconds_by_kind["prefill"],
        "decode_seconds": seconds_by_kind["decode"],
    }

    print(f"  {cell}")
    print(f"    oracle {qualname}")
    print(f"    options {options}")
    print(f"    digests {compass.get('oracle_option_sha256')}")
    print("\n  (1) the same shapes, priced by the same oracle")
    print(_fmt("every step", out["step_error_pct"]))
    print(_fmt("prefill steps", out["prefill_error_pct"]))
    print(_fmt("decode steps", out["decode_error_pct"]))
    if dump:
        _dump(priced, "prefill", dump)
        _dump(priced, "decode", dump)
    print("\n  (2) the step sets")
    print(f"    real     {len(real_steps):>6} steps, {real_prefill} with prefill")
    print(f"    modelled {len(modelled_steps):>6} steps, {mod_prefill} with prefill")
    print("\n  (3) the real run's own totals -- descriptive, not additive")
    print(f"    observed elapsed{span:9.3f}s  (first step start to last step end)")
    print(f"    forwards        {forward_s:9.3f}s  "
          f"(priced at {priced_s:9.3f}s, {_pct(priced_s, forward_s)})")
    print(f"      prefill       {seconds_by_kind['prefill'][0]:9.3f}s  "
          f"(priced {seconds_by_kind['prefill'][1]:9.3f}s)")
    print(f"      decode        {seconds_by_kind['decode'][0]:9.3f}s  "
          f"(priced {seconds_by_kind['decode'][1]:9.3f}s)")
    print(f"    between forwards{gap_s:9.3f}s  ({len(gaps)} gaps) -- host work "
          f"a simulated clock does not advance for. Measured against a "
          f"different pair of instants than the forwards, so the two are "
          f"reported apart rather than added.")
    print("\n    first use vs steady state, kept apart")
    print(f"      first step    {out['first_step_real_s']:9.3f}s  "
          f"(priced {out['first_step_priced_s']:9.3f}s, "
          f"{_pct(out['first_step_priced_s'], out['first_step_real_s'])}) -- "
          f"paid once")
    print(f"      steps 2..n    {steady_real:9.3f}s  "
          f"(priced {steady_priced:9.3f}s, {_pct(steady_priced, steady_real)})")
    real_tl = _timeline(priced, False)
    priced_tl = _timeline(priced, True)
    out["timeline_real"], out["timeline_priced"] = real_tl, priced_tl
    print("\n  (4) COUNTERFACTUAL: the real step sequence with its costs "
          "substituted")
    print("      Not a reconstruction of the real run -- it holds the schedule "
          "fixed on purpose,\n      which the simulated engine does not, and "
          "advances a clock by step costs alone.")
    print(f"    requests seen   {real_tl['requests']}")
    print(f"    TPOT median     real {real_tl['tpot'].get('median', 0):8.5f}s  "
          f"priced {priced_tl['tpot'].get('median', 0):8.5f}s  "
          f"{_pct(priced_tl['tpot'].get('median', 0), real_tl['tpot'].get('median', 0))}")
    print(f"    first token med real {real_tl['first_token'].get('median', 0):8.4f}s  "
          f"priced {priced_tl['first_token'].get('median', 0):8.4f}s  "
          f"{_pct(priced_tl['first_token'].get('median', 0), real_tl['first_token'].get('median', 0))}")
    print("    (compare with the reported errors for this cell; a match means "
          "the step costs explain it)")

    # And the simulated engine against its own step table. Section (4) holds
    # the step set fixed to isolate the oracle; this asks the separate question
    # of whether the run the simulated engine actually reported agrees with the
    # steps it actually took. A disagreement here is not a cost-model error at
    # all -- it is the engine attributing its own virtual time differently.
    mod_tl = _timeline([(r, float(r.get("seconds") or 0.0), 0.0)
                        for r in modelled_steps], False)
    out["timeline_modelled_own"] = mod_tl
    print("\n  (5) the simulated engine against its own step table")
    print(f"    steps {len(modelled_steps)}, requests {mod_tl['requests']}, "
          f"virtual span {mod_tl['span']:.3f}s")
    print(f"    TPOT median      {mod_tl['tpot'].get('median', 0):8.5f}s")
    print(f"    first token med  {mod_tl['first_token'].get('median', 0):8.4f}s")

    return out


def _timeline(priced: list, use_priced: bool) -> dict:
    """Per-request token times, rebuilt from the step table alone.

    A step names the requests it computed, so walking the table with a clock
    that advances by each step's duration reproduces when every request's
    tokens became available -- under the real durations, or under the oracle's.
    TPOT is the mean gap between a request's own emissions, which needs no
    arrival times and so cannot absorb a pacing difference.
    """
    clock, emissions = 0.0, {}
    for row, was, got in priced:
        clock += got if use_priced else was
        for rid in (row.get("req_ids") or ()):
            emissions.setdefault(rid, []).append(clock)
    tpot, first = [], []
    for rid, times in emissions.items():
        if len(times) > 1:
            tpot.append((times[-1] - times[0]) / (len(times) - 1))
        first.append(times[0])
    return {"requests": len(emissions), "span": clock,
            "tpot": _stats(tpot), "first_token": _stats(first)}


def _reported(cell: str) -> dict:
    """What `compare.py` said about this cell, for the reconstruction to hit."""
    path = os.path.join(cell, "summary.json")
    if not os.path.exists(path):
        return {}
    blob = json.loads(open(path, encoding="utf-8").read())
    return blob.get("errors") or blob.get("error") or blob


def _pct(got: float, was: float) -> str:
    return f"{(got - was) / was * 100.0:+.2f}%" if was else "n/a"


def _resolve(path: str) -> str:
    """The rank's own table, or the unsuffixed one at world size 1."""
    if os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    for candidate in sorted(__import__("glob").glob(f"{stem}.*{ext}")):
        return candidate
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell", action="append", required=True,
                    help="a matrix cell directory; repeatable")
    ap.add_argument("--json-out")
    ap.add_argument("--dump", type=int, default=0,
                    help="also print the N worst-priced steps of each kind")
    args = ap.parse_args(argv)
    out = []
    for cell in args.cell:
        out.append(report(cell, args.dump))
        print()
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

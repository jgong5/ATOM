"""Compare a real serving run against a simulated one, and refuse an invalid one.

This is the maintained replacement for `agent_scratch/cc_compare.py`. The
analysis is the same; what is different is that a run which did not complete as
asked now **fails the process** instead of printing a warning above numbers that
read as a result. Every check below was added because something once passed it
by accident:

  * a client reported "0 failed" while the server had given up waiting for 236
    requests and said so only in its own log;
  * a comparison with two requests declared and one engine record printed
    metrics and exited zero;
  * a workload compressed 40x at send time was saved unscaled, so the artifact
    described a paced run that never happened.

Metric definitions are fixed in `atom/compass/POC_STATUS.md` §0 and implemented
here once, so that a gate row and a diagnostic cannot drift apart. Everything is
read from the engine's own clock -- a client stopwatch against a simulated
engine times the simulator.

    python scripts/compass/compare.py --real real.json --modelled modelled.json \
        [--expect-requests N] [--summary-out summary.json]

Exit status is 0 only when both runs are reportable and the two describe the
same experiment.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# loading


@dataclass
class Run:
    """One side of the comparison, as it was written by `replay.py`."""

    path: str
    label: str
    manifest: dict = field(default_factory=dict)
    workload: list = field(default_factory=list)
    #: workload index -> engine record
    joined: dict = field(default_factory=dict)
    #: workload index -> the client's `usage` block, which carries token counts
    usage: dict = field(default_factory=dict)
    engine_records: int = 0
    clock: str = "?"


def load_run(path: str, label: str) -> Run:
    blob = json.loads(open(path, encoding="utf-8").read())
    engine = blob.get("engine") or {}
    records = engine.get("records") or engine.get("requests") or []
    by_id = {r.get("request_id"): r for r in records}
    run = Run(path=path, label=label, manifest=blob.get("run") or {},
              workload=blob.get("workload") or [],
              engine_records=len(records), clock=engine.get("clock", "?"))
    for result in blob.get("results") or []:
        if not result.get("ok"):
            continue
        response = result.get("response") or {}
        record = by_id.get(response.get("id"))
        if record is not None:
            run.joined[result["index"]] = record
            run.usage[result["index"]] = response.get("usage") or {}
    return run


# --------------------------------------------------------------------------
# validity


def check_run(run: Run, *, expect_requests: int | None = None,
              require_length_check: bool = True,
              require_provenance: bool = True) -> list[str]:
    """Reasons this run may not be reported. Empty means reportable."""
    bad: list[str] = []
    m = run.manifest
    if not m:
        bad.append("no run manifest: the workload, pacing and time scale of "
                   "this run are unrecorded")
        return bad
    if m.get("failed"):
        bad.append(f"{m['failed']} requests failed")

    lengths = m.get("prompt_lengths")
    if require_length_check and lengths != "passed":
        bad.append(f"prompt lengths were not verified against the workload "
                   f"({lengths!r}); replay with --check-lengths")
    elif lengths not in (None, "passed", "not requested"):
        bad.append(f"prompt lengths: {lengths}")

    declared = int(m.get("requests") or 0)
    if declared and len(run.workload) != declared:
        bad.append(f"manifest declares {declared} requests, the saved workload "
                   f"has {len(run.workload)}")
    if expect_requests is not None and declared and declared != expect_requests:
        bad.append(f"expected {expect_requests} requests, the run declares "
                   f"{declared}")

    # The join is the check that used to be a printed count. A request with no
    # engine record contributes to no metric, and a comparison over the
    # remainder is a comparison over a workload nobody asked for.
    missing = [i for i in range(len(run.workload)) if i not in run.joined]
    if missing:
        bad.append(f"{len(missing)} of {len(run.workload)} requests have no "
                   f"engine record (first: {missing[:5]})")
    # Preparation must be *before* the measurement, on the engine's clock and
    # not merely in the client's sending order. It is possible for the drain to
    # succeed -- no preparation row in the result -- while every measured
    # request is still stamped as having arrived before preparation finished:
    # a declared arrival is an offset from the engine's epoch, and the process
    # that stamps arrivals holds a virtual clock frozen there, so warming a
    # predictor does not move the origin its workload is measured against.
    # Measured on the first warmed 27B cell: a 14.4s preparation, 64 arrivals
    # stamped at the epoch, and TTFT read +71% for work the engine did in 4.7s.
    prepared = (m.get("prepare") or {}).get("boundary_engine_time")
    if prepared and run.joined:
        first = min(r.get("arrive_time") or 0.0 for r in run.joined.values())
        if first < float(prepared) - 1e-6:
            bad.append(
                f"preparation ran until engine time {prepared:.3f} but the "
                f"earliest measured arrival is stamped {float(prepared) - first:.3f}s "
                f"before that: preparation did not move the origin these "
                f"arrivals were declared against, so its whole duration is "
                f"inside every measured TTFT and latency")
    # Token counts come from the server's own `usage`, and only from there.
    # The workload says how many tokens were *asked* for; a run that stopped
    # early, or a simulator that produced a different number, would look
    # perfect if the requested count were substituted for the produced one.
    no_usage = [i for i in sorted(run.joined)
                if not isinstance((run.usage.get(i) or {}).get(
                    "completion_tokens"), int)]
    if no_usage:
        bad.append(f"{len(no_usage)} of {len(run.joined)} responses carry no "
                   f"usage.completion_tokens (first: {no_usage[:5]}); token "
                   f"counts and throughput cannot be computed from what the "
                   f"workload asked for")

    if run.engine_records != len(run.workload):
        bad.append(f"{run.engine_records} engine records against "
                   f"{len(run.workload)} requests: the trace was not fully "
                   f"drained, or the engine served something else")

    # Time ordering, per request, on one clock. `arrival <= first <= finish`
    # cannot be assumed: it is what tells a mixed clock domain from a slow run.
    for i, rec in sorted(run.joined.items()):
        a = rec.get("arrive_time")
        f = rec.get("first_token_time")
        z = rec.get("finish_time")
        if a is None or f is None or z is None:
            bad.append(f"request {i} is missing a timestamp "
                       f"(arrive={a}, first={f}, finish={z})")
            continue
        if not (a <= f <= z):
            bad.append(f"request {i} out of order: arrive {a:.6f}, "
                       f"first {f:.6f}, finish {z:.6f}")
        for name, derived, computed in (("ttft", rec.get("ttft"), f - a),
                                        ("latency", rec.get("latency"), z - a)):
            if derived is not None and abs(derived - computed) > 1e-3:
                bad.append(f"request {i}: reported {name} {derived:.6f} does "
                           f"not match its own timestamps ({computed:.6f}); "
                           f"the two are on different clocks")

    if m.get("arrival_barrier_timed_out"):
        bad.append("the engine's arrival barrier timed out: virtual time "
                   "advanced past an arrival still in flight")

    if require_provenance:
        # Which code served it. A git revision where the tree is a checkout, a
        # source digest where it is an rsync copy -- the GPU nodes are the
        # latter, and requiring the revision alone made every run served from
        # one unattributable.
        if not (m.get("server_revision") or m.get("server_code_sha256")):
            bad.append("no server_revision or server_code_sha256 in the "
                       "manifest: this run cannot be attributed to a server "
                       "build")
        if not m.get("model_revision"):
            bad.append("no model_revision in the manifest: this run cannot be "
                       "attributed to a checkpoint")
        # A calibration digest only when there was a calibration. A measured
        # run has no table, and demanding one there made the real half of every
        # comparison unreportable for a reason that was not a defect.
        # Only when there was something to digest. A measured run's oracle is
        # the declared stub and names no table; demanding a calibration digest
        # from it refused every ground-truth run for a reason that was not a
        # defect.
        oracle = ((m.get("server") or {}).get("compass") or {})
        named = {k: v for k, v in (oracle.get("oracle_options") or {}).items()
                 if isinstance(v, str) and v}
        digests = oracle.get("oracle_option_sha256") or {}
        missing = sorted(k for k in named if not digests.get(k))
        if missing:
            bad.append(f"the server ran oracle {oracle.get('oracle')} with "
                       f"{', '.join(f'{k}={named[k]}' for k in missing)} but "
                       f"read no such file: what it was fitted to is "
                       f"unrecorded")
    return bad


def check_pair(real: Run, modelled: Run) -> list[str]:
    """Reasons these two runs are not the same experiment."""
    bad: list[str] = []
    a, b = real.manifest, modelled.manifest
    for key, what in (("trace_sha256", "workload"),
                      ("time_scale", "time scale"),
                      ("model", "model")):
        if a.get(key) != b.get(key):
            bad.append(f"different {what}: {a.get(key)!r} against {b.get(key)!r}")
    if len(real.workload) != len(modelled.workload):
        bad.append(f"different request counts: {len(real.workload)} against "
                   f"{len(modelled.workload)}")
    else:
        for i, (x, y) in enumerate(zip(real.workload, modelled.workload)):
            if (x.get("input_tokens") != y.get("input_tokens")
                    or x.get("output_tokens") != y.get("output_tokens")):
                bad.append(f"request {i} has different lengths on the two "
                           f"sides: {x} against {y}")
                break
    shared = set(real.joined) & set(modelled.joined)
    if len(shared) != len(real.workload):
        bad.append(f"only {len(shared)} of {len(real.workload)} requests are "
                   f"present on both sides")

    # Produced tokens, not requested ones. This used to be a warning printed
    # under the throughput row, which is the wrong place: throughput is
    # tokens/second, so two runs that produced different numbers of tokens do
    # not have comparable throughput at all, and neither do their TPOTs.
    differing = []
    for i in sorted(shared):
        x = (real.usage.get(i) or {}).get("completion_tokens")
        y = (modelled.usage.get(i) or {}).get("completion_tokens")
        if x != y:
            differing.append((i, x, y))
    if differing:
        head = ", ".join(f"{i}: {x} against {y}" for i, x, y in differing[:5])
        bad.append(f"{len(differing)} of {len(shared)} requests produced "
                   f"different numbers of tokens on the two sides ({head}); "
                   f"throughput and TPOT are not comparable")
    return bad


# --------------------------------------------------------------------------
# metrics -- POC_STATUS.md section 0


def metrics(run: Run, indices) -> dict:
    """The fixed metric set, per request and in aggregate."""
    ttft, tpot, latency = {}, {}, {}
    arrivals, finishes, tokens = [], [], 0
    for i in indices:
        rec = run.joined[i]
        a, f, z = rec["arrive_time"], rec["first_token_time"], rec["finish_time"]
        ttft[i] = f - a
        latency[i] = z - a
        # `check_run` refuses a run without this, so it is present here.
        n = int((run.usage.get(i) or {}).get("completion_tokens") or 0)
        if n >= 2:
            tpot[i] = (z - f) / (n - 1)
        arrivals.append(a)
        finishes.append(z)
        tokens += n
    window = (max(finishes) - min(arrivals)) if arrivals else 0.0
    return {"ttft": ttft, "tpot": tpot, "latency": latency,
            "output_tokens": tokens,
            "window_s": window,
            "throughput_tok_s": (tokens / window) if window > 0 else 0.0}


#: How every quantile in this file is taken, stamped into each summary so a
#: number cannot be re-read later under a different convention. It is a plain
#: order statistic with no interpolation: the q-quantile of a sorted sample of
#: n is `s[min(n - 1, floor(q * n))]`. For n = 10 the median is the 6th value
#: and p90 the 10th, so both are *upper* picks; they are not `numpy.percentile`
#: defaults and they are not `statistics.median`, which averages the middle two
#: on an even sample. Only `mean` is an average.
QUANTILE_CONVENTION = "order statistic s[min(n-1, floor(q*n))], no interpolation"


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    s = sorted(values)
    pick = lambda q: s[min(len(s) - 1, int(q * len(s)))]  # noqa: E731
    return {"n": len(s), "median": pick(0.5), "mean": statistics.fmean(s),
            "p90": pick(0.9), "min": s[0], "max": s[-1]}


def _error(real: dict, modelled: dict) -> dict:
    """Signed error on each summary statistic, and per request."""
    out = {}
    for stat in ("median", "mean", "p90"):
        r, m = real.get(stat), modelled.get(stat)
        out[stat] = ((m - r) / r * 100.0) if r else None
    return out


def compare(real: Run, modelled: Run) -> dict:
    shared = sorted(set(real.joined) & set(modelled.joined))
    rm, mm = metrics(real, shared), metrics(modelled, shared)
    report = {"requests": len(shared),
              "real_clock": real.clock, "modelled_clock": modelled.clock,
              "metrics": {}}
    for name in ("ttft", "tpot", "latency"):
        keys = sorted(set(rm[name]) & set(mm[name]))
        r = _quantiles([rm[name][i] for i in keys])
        m = _quantiles([mm[name][i] for i in keys])
        per_request = sorted((mm[name][i] - rm[name][i]) / rm[name][i] * 100.0
                             for i in keys if rm[name][i] > 0)
        report["metrics"][name] = {
            "real": r, "modelled": m, "error_pct": _error(r, m),
            "per_request_error_pct": _quantiles(per_request),
            "real_per_request": [rm[name][i] for i in keys],
            "modelled_per_request": [mm[name][i] for i in keys],
        }
    rt, mt = rm["throughput_tok_s"], mm["throughput_tok_s"]
    report["metrics"]["throughput_tok_s"] = {
        "real": rt, "modelled": mt,
        "error_pct": ((mt - rt) / rt * 100.0) if rt else None,
        "real_window_s": rm["window_s"], "modelled_window_s": mm["window_s"],
        "output_tokens": rm["output_tokens"],
    }
    if rm["output_tokens"] != mm["output_tokens"]:
        # `check_pair` refuses this, so it is only reachable under
        # `--report-anyway`; the number still has to carry the caveat.
        report["throughput_warning"] = (
            f"the two runs produced different token counts "
            f"({rm['output_tokens']} against {mm['output_tokens']}); the "
            f"throughput comparison is not like for like")
    report["quantile_convention"] = QUANTILE_CONVENTION
    return report


# --------------------------------------------------------------------------


def _print(report: dict) -> None:
    print(f"  requests compared : {report['requests']}  "
          f"(clocks: real {report['real_clock']}, "
          f"modelled {report['modelled_clock']})")
    for name in ("ttft", "tpot", "latency"):
        block = report["metrics"][name]
        r, m, e = block["real"], block["modelled"], block["error_pct"]
        if not r.get("n"):
            print(f"\n  {name}: no requests qualify")
            continue
        print(f"\n  {name}, seconds        n={r['n']}")
        print(f"    {'real':<10} med {r['median']:9.4f}  mean {r['mean']:9.4f}"
              f"  p90 {r['p90']:9.4f}  max {r['max']:9.4f}")
        print(f"    {'modelled':<10} med {m['median']:9.4f}  mean {m['mean']:9.4f}"
              f"  p90 {m['p90']:9.4f}  max {m['max']:9.4f}")
        print(f"    {'error':<10} med {e['median']:+8.2f}%  "
              f"mean {e['mean']:+8.2f}%  p90 {e['p90']:+8.2f}%")
        pr = block["per_request_error_pct"]
        print(f"    {'per req':<10} med {pr['median']:+8.2f}%  "
              f"mean {pr['mean']:+8.2f}%  p90 {pr['p90']:+8.2f}%  "
              f"worst {max(abs(pr['min']), abs(pr['max'])):8.2f}%")
    t = report["metrics"]["throughput_tok_s"]
    print("\n  throughput, output tokens/s")
    print(f"    real {t['real']:9.3f} over {t['real_window_s']:.3f}s   "
          f"modelled {t['modelled']:9.3f} over {t['modelled_window_s']:.3f}s   "
          f"error {t['error_pct']:+.2f}%")
    if report.get("throughput_warning"):
        print(f"    WARNING: {report['throughput_warning']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True)
    ap.add_argument("--modelled", required=True)
    ap.add_argument("--expect-requests", "--repeats", type=int, default=None,
                    dest="expect_requests",
                    help="how many requests each side must declare; a run with "
                         "fewer is refused rather than reported over what "
                         "arrived. This is a count of requests, not of repeats "
                         "of the workload -- the old `--repeats` spelling is "
                         "kept only so existing scripts keep working.")
    ap.add_argument("--summary-out", help="write the report as JSON")
    ap.add_argument("--allow-unchecked-lengths", action="store_true",
                    help="accept a replay that did not pass --check-lengths. "
                         "A length-dependent client bug reshapes a length "
                         "distribution rather than shifting it, so this should "
                         "only be used when diagnosing that bug.")
    ap.add_argument("--allow-missing-provenance", action="store_true",
                    help="accept a run whose manifest does not identify the "
                         "server build, model revision and calibration. The "
                         "waiver is recorded in the summary.")
    ap.add_argument("--report-anyway", action="store_true",
                    help="print the numbers from a run that failed its checks, "
                         "for diagnosing the failure itself")
    args = ap.parse_args(argv)

    real = load_run(args.real, "real")
    modelled = load_run(args.modelled, "modelled")
    problems = {}
    for run in (real, modelled):
        bad = check_run(run, expect_requests=args.expect_requests,
                        require_length_check=not args.allow_unchecked_lengths,
                        require_provenance=not args.allow_missing_provenance)
        if bad:
            problems[run.label] = bad
    pair = check_pair(real, modelled)
    if pair:
        problems["pair"] = pair

    for label, bad in problems.items():
        print(f"  {label} is not reportable:")
        for line in bad:
            print(f"    {line}")
    if problems and not args.report_anyway:
        print("\nrefusing to report. --report-anyway prints the numbers with "
              "this notice attached, for diagnosing the failure itself.")
        return 1

    report = compare(real, modelled)
    report["problems"] = problems
    report["waivers"] = {"unchecked_lengths": args.allow_unchecked_lengths,
                         "missing_provenance": args.allow_missing_provenance}
    report["real_path"], report["modelled_path"] = args.real, args.modelled
    if problems:
        print("\n  REPORTING ANYWAY -- these numbers describe a run that did "
              "not complete as asked.\n")
    _print(report)
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        print(f"\n  summary -> {args.summary_out}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

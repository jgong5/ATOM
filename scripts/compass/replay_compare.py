"""Diff a modelled replay against the real one, per request.

`serve_compare.py` cannot be reused: it reads a directory of `engine.json` plus
`bench.json`, while `replay.py` writes one nested artifact. More importantly it
compares aggregates, and on this workload aggregates lie.

**Why latency is not the verdict.** A cc-traces run on this project once landed
at -5.0% on mean latency and was read as a good result for a day. Per request it
was +51.6% on TTFT against -30.3% on decode: two large errors in opposite
directions, summing to something that looked like agreement. That has now
happened four times. So this reports per-request TTFT and TPOT, and prints
latency below them with the reason it is not the headline.

**Why errors are paired, not pooled.** Comparing the distribution of real TTFTs
against the distribution of modelled ones answers a different question -- two
sets can match in distribution with every individual request wrong. Each
request is joined by workload index and its own error taken, so `p90` here is
the 90th percentile of *one request's* error, not the gap between two p90s.

    python scripts/compass/replay_compare.py --real real.json \\
        --modelled modelled.json --out compare.json
"""

import argparse
import json
import sys

#: Requests whose real reading is below this are excluded from *relative* error.
#: Dividing by a 2 ms TTFT turns a 1 ms absolute difference into 50%, which then
#: dominates a percentile. They stay in the absolute-error figures and are
#: counted, because dropping rows quietly is how a summary starts flattering a
#: model.
MIN_DENOMINATOR_S = 0.010


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _per_request(artifact):
    """`index -> reading`, joining the client's results to the engine's clock.

    The join is `response["id"]`, which is the same string the server keys its
    records by. It is needed because the client cannot time a simulated run --
    a socket measures how fast the simulator ran, not what it simulated -- so
    every timing here comes off the engine and the client contributes only the
    index and the token counts.
    """
    records = {r["request_id"]: r
               for r in (artifact.get("engine") or {}).get("requests", [])}
    out = {}
    for result in artifact.get("results", []):
        if not result.get("ok"):
            continue
        response = result.get("response") or {}
        record = records.get(response.get("id"))
        if record is None:
            continue
        usage = response.get("usage") or {}
        produced = int(usage.get("completion_tokens", 0) or 0)
        ttft = record.get("ttft")
        latency = record.get("latency")
        decode = None
        if ttft is not None and latency is not None and produced > 1:
            # Per *output* token after the first. A request that produced one
            # token has no inter-token interval, and charging it (latency-ttft)
            # over one token would put a near-zero in the middle of the
            # distribution.
            decode = (latency - ttft) / (produced - 1)
        out[result["index"]] = {
            "ttft_s": ttft,
            "tpot_s": decode,
            "latency_s": latency,
            "output_tokens": produced,
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "cached_tokens": record.get("num_cached_tokens"),
        }
    return out


def _workloads_agree(real, modelled):
    """Whether the two runs replayed the same thing.

    Two artifacts always diff; whether the diff means anything depends on this.
    A modelled run against a re-extracted trace, or one truncated by
    --num-requests, produces a report that looks exactly like a model error.
    """
    a, b = real.get("workload", []), modelled.get("workload", [])
    if len(a) != len(b):
        return False, f"{len(a)} requests against {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        for key in ("input_tokens", "output_tokens"):
            if x.get(key) != y.get(key):
                return False, f"request {i}: {key} {x.get(key)} against {y.get(key)}"
        if x.get("hash_ids") != y.get("hash_ids"):
            return False, f"request {i}: different hash_ids, so different sharing"
        if abs(float(x.get("arrival_s", 0)) - float(y.get("arrival_s", 0))) > 1e-6:
            return False, (f"request {i}: arrival {x.get('arrival_s')} against "
                           f"{y.get('arrival_s')}")
    return True, None


def _errors(real, modelled, field):
    """Paired signed relative error per request, and the absolute differences."""
    relative, absolute, skipped = [], [], 0
    for index, r in real.items():
        m = modelled.get(index)
        if m is None or r.get(field) is None or m.get(field) is None:
            continue
        absolute.append(m[field] - r[field])
        if r[field] < MIN_DENOMINATOR_S:
            skipped += 1
            continue
        relative.append((m[field] - r[field]) / r[field])
    return relative, absolute, skipped


def _summary(relative, absolute, skipped):
    if not relative and not absolute:
        return {"paired": 0}
    signed = sorted(relative)
    magnitude = sorted(abs(v) for v in relative)
    return {
        "paired": len(absolute),
        "relative_n": len(relative),
        "below_denominator_floor": skipped,
        # Signed, so a model that is late on everything cannot hide behind a
        # mean of magnitudes; and magnitudes, so one that is half early and
        # half late cannot hide behind a mean of signs. Both, always.
        "mean_signed": round(sum(signed) / len(signed), 6) if signed else None,
        "p50_signed": round(_percentile(signed, 0.5), 6) if signed else None,
        "p50_abs": round(_percentile(magnitude, 0.5), 6) if magnitude else None,
        "p90_abs": round(_percentile(magnitude, 0.9), 6) if magnitude else None,
        "max_abs": round(magnitude[-1], 6) if magnitude else None,
        "mean_diff_s": round(sum(absolute) / len(absolute), 6) if absolute else None,
    }


def _reuse(artifact):
    """What the prefix cache served, differenced across the run."""
    stats = artifact.get("cache_stats") or {}
    before, after = stats.get("before") or {}, stats.get("after") or {}
    if not after:
        return {"available": False}
    got = {key: int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)
           for key in ("cached_tokens", "reusable_tokens", "full_tokens")}
    got["available"] = True
    per_request = [r["cached_tokens"]
                   for r in _per_request(artifact).values()
                   if r.get("cached_tokens") is not None]
    got["per_request_cached_tokens"] = sum(per_request)
    got["requests_with_a_hit"] = sum(1 for v in per_request if v > 0)
    return got


def compare(real, modelled):
    """The whole report, including every reason it might not be one."""
    agree, why = _workloads_agree(real, modelled)
    r, m = _per_request(real), _per_request(modelled)
    paired = sorted(set(r) & set(m))

    blocking = []
    if not agree:
        blocking.append(f"the two runs replayed different workloads: {why}")
    for name, art in (("real", real), ("modelled", modelled)):
        run = art.get("run") or {}
        if run.get("failed"):
            blocking.append(f"{name} run had {run['failed']} failed requests")
        if run.get("prompt_lengths") not in (None, "passed", "not requested"):
            blocking.append(f"{name} run: prompt_lengths {run['prompt_lengths']}")
        if run.get("arrival_barrier_timed_out"):
            blocking.append(f"{name} run: the arrival barrier timed out, so its "
                            f"latencies are not the declared workload's")
        if run.get("rows_with_hash_ids") and not run.get("ignore_eos"):
            # max_tokens is a ceiling. On a trace that records what each request
            # produced, a run without this is a shorter workload than the one on
            # disk, and nothing downstream can see the difference.
            blocking.append(f"{name} run replayed a trace with recorded output "
                            f"lengths but without --ignore-eos")
    if not paired:
        blocking.append("no request could be joined between the two artifacts")

    report = {
        "requests_paired": len(paired),
        "requests_real": len(r),
        "requests_modelled": len(m),
        "ttft": _summary(*_errors(r, m, "ttft_s")),
        "tpot": _summary(*_errors(r, m, "tpot_s")),
        # Reported because it is what a reader expects to see, and listed last
        # with this note because on this workload it is the number that has
        # concealed the error four separate times.
        "latency": _summary(*_errors(r, m, "latency_s")),
        "latency_note": ("not a verdict: two large errors of opposite sign sum "
                         "to a small one. Read ttft and tpot."),
        "reuse": {"real": _reuse(real), "modelled": _reuse(modelled)},
        # Barrier state lives on the scheduler in the engine-core process and is
        # not served over HTTP, so this can only report it when a manifest
        # carries it. Said here rather than left as a silent pass.
        "arrival_barrier": ("checked" if any(
            (a.get("run") or {}).get("arrival_barrier_timed_out") is not None
            for a in (real, modelled)) else "not reported by the server"),
        "blocking": blocking,
        "verdict": "withheld" if blocking else "reportable",
    }
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", required=True, help="replay.py artifact, real engine")
    p.add_argument("--modelled", required=True, help="replay.py artifact, predicted")
    p.add_argument("--out", required=True)
    p.add_argument("--allow-blocking", action="store_true",
                   help="print the report even when a check refused it. The "
                        "numbers are written either way; this only changes the "
                        "exit code, so a script cannot pass by ignoring stdout")
    args = p.parse_args(argv)

    report = compare(_load(args.real), _load(args.modelled))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    print(f"{report['requests_paired']} requests paired -> {args.out}")
    for field in ("ttft", "tpot", "latency"):
        s = report[field]
        if not s.get("relative_n"):
            # Says which. "No readings" reads as a broken join, and a run where
            # every request was faster than the floor is a different fact from
            # one where the two sides could not be matched up at all.
            floored = s.get("below_denominator_floor") or 0
            why = (f"all {floored} below the {MIN_DENOMINATOR_S * 1000:.0f}ms "
                   f"floor" if floored else "nothing paired")
            print(f"  {field:<8} no reading: {why}")
            continue
        print(f"  {field:<8} mean {s['mean_signed']:+.1%}  p50 "
              f"{s['p50_signed']:+.1%}  p90|err| {s['p90_abs']:.1%}  "
              f"max|err| {s['max_abs']:.1%}  (n={s['relative_n']})")
    print(f"  {report['latency_note']}")
    for side in ("real", "modelled"):
        reuse = report["reuse"][side]
        if reuse.get("available"):
            print(f"  reuse {side:<8} cached {reuse['cached_tokens']} of "
                  f"{reuse['full_tokens']} tokens, "
                  f"{reuse['requests_with_a_hit']} requests hit")
    for reason in report["blocking"]:
        print(f"  BLOCKING: {reason}", file=sys.stderr)
    return 0 if not report["blocking"] or args.allow_blocking else 1


if __name__ == "__main__":
    sys.exit(main())

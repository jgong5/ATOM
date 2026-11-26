# SPDX-License-Identifier: MIT
"""One real run against its simulated runs, metric by metric, with each error.

    PYTHONPATH=$T python3 scripts/compass/one_real.py [--loose-arguments] REAL SIM [SIM...]

`python -m atom.compass.validate` grades a cell against the spread of two or
more real runs and refuses one. When the budget is one real GPU run, as for a
law fitted on that run's own record, this prints each SIM's metrics beside
REAL's and the relative error of each, and grades nothing: an error against
one run carries no spread, and a law priced on the steps it was fitted on is
an in-sample check.

Each of REAL and SIM is a run directory `client.sh` wrote an export into: a
`compass record` or `compass run` `--out`, or a `deploy.sh` OUT. The metrics
are `validate.curves.metrics`' (InferenceX's agentic ones, over the profiling
records with no error) and, over the same records, TTFT p50 and p99, TPOT p50
and p90 (aiperf's inter-token latency, per request) and E2E latency p50, in
milliseconds, and the requests counted. Every run must have been given the
same client arguments. Below the table it names REAL's stalled forwards over
its graded window (`atom.compass.validate.stalls`): seconds the real run lost
to one DP rank's slow launch, which a simulated run does not charge. Then
REAL's own 95% interval of TTFT p99, p90 TTFT and TPOT p90 over resamples of
its sends' 60 s blocks, and whether each SIM's error lies inside it
(`tails`): a tail holds a few bursts' worth of requests. Then, per
SIM, its requests paired with REAL's by conversation and turn (`paired`): a
percentile of each run is a rank statistic over its own request set, and E2E's
middle is sparse (#3720 c=8: 3.5% of requests within 5% of the median), so its
p50 moves with the requests either run finished and with each request's place
in the schedule, which the paired E2E does not. Then, per SIM, its throughput
per GPU over the requests both runs finished, and the seconds aiperf's agentic
replay skipped in each run under its global system-idle cap (`window`): a
short window's throughput is the tokens of the requests the client sent in it,
and a run whose system went idle where the other's did not jumps its whole
replay ahead. Last, per SIM, its prefix hits over the requests both ran beside
REAL's (`hits`): which DP rank a follow-up lands on, and so whether it finds
its conversation's prefix, is a draw of the schedule.
"""

import json
import random
import re
import sys
from pathlib import Path

from atom.compass.backends.json_file import read_json, read_jsonl
from atom.compass.validate import stalls
from atom.compass.validate.curves import (
    CLIENT_RUN,
    RECORDS,
    _metric,
    _positive,
    gpus,
    metrics,
    percentile,
    profiled,
)

#: The latency percentiles added to `metrics`', aiperf field and percentile, in ms.
LATENCIES = {
    "TTFT p50 ms": ("time_to_first_token", 50),
    "TTFT p99 ms": ("time_to_first_token", 99),
    "TPOT p50 ms": ("inter_token_latency", 50),
    "TPOT p90 ms": ("inter_token_latency", 90),
    "E2E p50 ms": ("request_latency", 50),
}


def run_metrics(run: Path, simulated: bool) -> dict:
    """`metrics` of `run` and `LATENCIES` over its profiled records."""
    records = profiled(read_jsonl(run / RECORDS))
    out = {"requests": len(records)}
    for name, (field, p) in LATENCIES.items():
        out[name] = percentile(_positive(_metric(r, field) for r in records), p)
    return out | metrics(run, simulated)


def arguments(run: Path) -> dict:
    return read_json(run / CLIENT_RUN)["arguments"]


def error(real, sim) -> str:
    if real is None or sim is None or real == 0:
        return "-"
    return f"{(sim - real) / real:+.1%}"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    # --loose-arguments: REAL's client is another launcher's (a recipe harness's
    # agentx run, CI's flags) over the same traces: name each argument the two
    # were given differently, then compare anyway.
    loose = "--loose-arguments" in argv
    argv = [a for a in argv if a != "--loose-arguments"]
    real, *sims = map(Path, argv)
    given = arguments(real)
    for sim in sims:
        if (theirs := arguments(sim)) != given:
            if not loose:
                print(
                    f"{sim} was given other client arguments than {real}",
                    file=sys.stderr,
                )
                return 2
            for k in sorted(given.keys() | theirs.keys()):
                if given.get(k) != theirs.get(k):
                    print(f"arguments differ: {k}: real {given.get(k)!r}, {sim.name} {theirs.get(k)!r}")
    rows = {"real": run_metrics(real, False)}
    rows |= {sim.name: run_metrics(sim, True) for sim in sims}
    names = list(rows["real"])
    print(
        f"{'metric':34}"
        + "".join(f"{k:>16}" for k in rows)
        + "".join(f"{'err ' + s.name:>18}" for s in sims)
    )
    for name in names:
        values = [rows[k][name] for k in rows]
        shown = "".join(f"{'-' if v is None else format(v, '.4g'):>16}" for v in values)
        errs = "".join(f"{error(values[0], v):>18}" for v in values[1:])
        print(f"{name:34}{shown}{errs}")
    print(stalled(real))
    print(tails(real, sims, rows))
    for sim in sims:
        print(paired(real, sim))
    for sim in sims:
        print(window(real, sim))
    for sim in sims:
        print(hits(real, sim))
    print(json.dumps(rows))
    return 0


#: Resamples of REAL's requests for the interval of its E2E p50, and their seed.
RESAMPLES, SEED = 2000, 0


def _latencies(run: Path) -> dict:
    """E2E latency of each profiled request of `run` by (conversation, turn)."""
    out = {}
    for r in profiled(read_jsonl(run / RECORDS)):
        m, e2e = r.get("metadata", {}), _metric(r, "request_latency")
        if e2e and "conversation_id" in m and "turn_index" in m:
            out[m["conversation_id"], m["turn_index"]] = e2e
    return out


def paired(real: Path, sim: Path) -> str:
    """`sim`'s E2E against `real`'s over the requests both ran: the median of
    each request's ratio and the ratio on totals, beside the 95% interval of
    `real`'s own E2E p50 over `RESAMPLES` resamples of its requests."""
    a, b = _latencies(real), _latencies(sim)
    both = sorted(set(a) & set(b))
    if not both:
        return f"{sim.name}: no request paired with the real run's"
    mine = list(a.values())
    rng, p50 = random.Random(SEED), percentile(mine, 50)
    draws = sorted(
        percentile(rng.choices(mine, k=len(mine)), 50) for _ in range(RESAMPLES)
    )
    lo, hi = (draws[int(q * (RESAMPLES - 1))] / p50 - 1 for q in (0.025, 0.975))
    ratio = percentile([b[k] / a[k] for k in both], 50)
    total = sum(b[k] for k in both) / sum(a[k] for k in both)
    return (
        f"{sim.name}: E2E paired over the {len(both)} requests both ran: median "
        f"ratio {ratio - 1:+.1%}, on totals {total - 1:+.1%}; real E2E p50's own "
        f"95% interval {lo:+.1%}..{hi:+.1%}"
    )


#: The tail percentiles `tails` gives an interval, as `run_metrics` names them,
#: and the width of the send-time blocks it resamples, in seconds.
TAILS = {
    "TTFT p99 ms": ("time_to_first_token", 99),
    "p90 TTFT": ("time_to_first_token", 90),
    "TPOT p90 ms": ("inter_token_latency", 90),
}
BLOCK_S = 60


def tails(real: Path, sims: list[Path], rows: dict) -> str:
    """The 95% interval of `real`'s own TTFT p99, p90 and TPOT p90 over
    `RESAMPLES` resamples of `BLOCK_S`-second blocks of its requests' sends,
    and which of `sims`' errors (one_real's table, `rows`) fall inside it. A
    tail is set by a few events, each holding requests sent together: a client
    burst whose follow-ups miss their prefix on the rank the router picks
    (#3720 c=64: the 41 requests sent 300 s into the window held 2.4M new
    tokens in the real run, 3.6M and 4.9M in two simulated ones), so the
    blocks resample the events, where a resample of requests would not."""
    blocks = {}
    for r in profiled(read_jsonl(real / RECORDS)):
        sent = r.get("metadata", {}).get("request_start_ns")
        if sent is None:
            return "real: no request_start_ns, no tail interval"
        blocks.setdefault(int(sent) // int(BLOCK_S * 1e9), []).append(r)
    keys, rng, out = sorted(blocks), random.Random(SEED), []
    for name, (field, p) in TAILS.items():
        values = {k: _positive(_metric(r, field) for r in blocks[k]) for k in keys}
        mine = percentile([v for k in keys for v in values[k]], p)
        draws = sorted(
            percentile(
                [v for k in rng.choices(keys, k=len(keys)) for v in values[k]], p
            )
            for _ in range(RESAMPLES)
        )
        lo, hi = (draws[int(q * (RESAMPLES - 1))] / mine - 1 for q in (0.025, 0.975))
        errs = [
            f"{s.name} {e:+.1%} {'in' if lo <= e <= hi else 'OUT'}"
            for s in sims
            if None not in (rows["real"][name], rows[s.name][name])
            for e in [rows[s.name][name] / rows["real"][name] - 1]
        ]
        out.append(f"{name} {lo:+.1%}..{hi:+.1%} ({', '.join(errs)})")
    return (
        f"real tails' own 95% interval over {len(keys)} {BLOCK_S} s send blocks: "
        + "; ".join(out)
    )


#: aiperf's log in a client export, and the line its agentic replay writes at
#: each phase's end of what its global system-idle cap skipped.
AIPERF_LOG = "artifacts/logs/aiperf.log"
IDLE_CAP = re.compile(
    r"Global system-idle cap summary: limit=\S+, jumps=(\d+), skipped=([0-9.]+)s"
)


def idle_cap(run: Path) -> tuple[int, float] | None:
    """The jumps aiperf's agentic replay made in `run` under its global
    system-idle cap and the seconds they skipped, from the last summary its log
    holds (the profiling phase's, which ends after the warmup's); None where
    the log states none."""
    try:
        found = IDLE_CAP.findall((run / AIPERF_LOG).read_text(errors="replace"))
    except OSError:
        return None
    return (int(found[-1][0]), float(found[-1][1])) if found else None


def _served(run: Path) -> dict:
    """Each profiled request of `run` by (conversation, turn): its send, its
    end at the client and its input plus output tokens."""
    out = {}
    for r in profiled(read_jsonl(run / RECORDS)):
        m = r.get("metadata", {})
        start, end = m.get("request_start_ns"), m.get("request_end_ns")
        if None in (start, end) or "conversation_id" not in m or "turn_index" not in m:
            continue
        tokens = sum(
            n
            for k in ("input_sequence_length", "output_sequence_length")
            if (n := _metric(r, k, int)) is not None
        )
        out[m["conversation_id"], m["turn_index"]] = (int(start), int(end), tokens)
    return out


def window(real: Path, sim: Path) -> str:
    """`sim`'s throughput per GPU against `real`'s over the requests both ran
    (each run's tokens of them over its span from the first one's send to the
    last one's end), beside what aiperf's global system-idle cap skipped in
    each run (`idle_cap`). The table's throughput is a window's: the tokens of
    the requests the client sent in it. Under agentic replay the client shifts
    every pending send earlier once the system has been idle for the cap, so a
    run that goes idle where the other has a request in flight sends its whole
    replay ahead (#3720 c=8: each simulated run jumped 3 times, 57 s, the real
    one 2 times, 27 s, as a lane ran 2 s ahead of the real one's; the 30 s more
    of the trace sent 4 more requests holding 1.9M input tokens: throughput
    +7.8%, and -0.2% over the 198 requests both ran)."""
    a, b = _served(real), _served(sim)
    both = set(a) & set(b)
    caps = ", ".join(
        f"{name} {'-' if c is None else f'{c[1]:.1f} s in {c[0]} jumps'}"
        for name, c in (("real", idle_cap(real)), (sim.name, idle_cap(sim)))
    )
    if not both:
        return f"{sim.name}: no request paired with the real run's; idle cap skipped {caps}"

    def per_gpu(run: Path, served: dict) -> float | None:
        span = (max(served[k][1] for k in both) - min(served[k][0] for k in both)) / 1e9
        return sum(served[k][2] for k in both) / span / gpus(run) if span > 0 else None

    return (
        f"{sim.name}: throughput per GPU over the {len(both)} requests both ran "
        f"{error(per_gpu(real, a), per_gpu(sim, b))} ({len(a) - len(both)} ran in "
        f"the real run only, {len(b) - len(both)} in {sim.name} only); the "
        f"client's global idle cap skipped {caps}"
    )


def _cached(run: Path) -> dict:
    """Each profiled request of `run` by (conversation, turn): its input tokens
    and those its engine served from the prefix cache (aiperf's
    ``usage_prompt_cache_read_tokens``), None where its record states none."""
    out = {}
    for r in profiled(read_jsonl(run / RECORDS)):
        m = r.get("metadata", {})
        isl = _metric(r, "input_sequence_length", int)
        if isl and "conversation_id" in m and "turn_index" in m:
            out[m["conversation_id"], m["turn_index"]] = (
                isl,
                _metric(r, "usage_prompt_cache_read_tokens", int),
            )
    return out


def hits(real: Path, sim: Path) -> str:
    """`sim`'s prefix hits against `real`'s over the requests both ran: each
    run's cached share of their input tokens and the change in their uncached
    ones. The DP router places a request on the rank with the fewest in
    flight, blind to where its conversation's prefix lives, so whether a
    follow-up lands on its history is a draw of the schedule; where prefill
    bounds the run, the draw moves throughput and the TTFT tail (#3720 c=256:
    one rank prefilling while the other three wait held 72% of the time; over
    26 simulated runs each point of hit rate moved throughput per GPU +1.7%
    and p90 TTFT -2.8%, and the hit rates spread 1.0 point (sd) about a mean
    1.2 points below the one real run's). A gap here beside `window`'s ~0%
    over the same requests is that draw, not the law."""
    a, b = _cached(real), _cached(sim)
    both = [
        k for k in set(a) & set(b) if a[k][1] is not None and b[k][1] is not None
    ]
    if not both:
        return f"{sim.name}: prefix hits: no request both ran states its cached tokens"
    (ia, ca), (ib, cb) = (
        (sum(x[k][0] for k in both), sum(x[k][1] for k in both)) for x in (a, b)
    )
    shift = f"{(ib - cb) / (ia - ca) - 1:+.1%}" if ia > ca else "-"
    return (
        f"{sim.name}: prefix hits over the {len(both)} requests both ran "
        f"{cb / ib:.1%} against the real run's {ca / ia:.1%} "
        f"({100 * (cb / ib - ca / ia):+.1f} pt), their uncached input tokens {shift}"
    )


def stalled(real: Path) -> str:
    """REAL's stalled forwards (`validate.stalls`), which no simulated run
    charges, or why they are not counted."""
    try:
        return "real " + stalls.text(stalls.report(real))
    except (OSError, KeyError, ValueError) as e:
        return f"real: stalled forwards not counted ({type(e).__name__}: {e})"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

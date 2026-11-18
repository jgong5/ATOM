# SPDX-License-Identifier: MIT
"""One real run against its simulated runs, metric by metric, with each error.

    PYTHONPATH=$T python3 scripts/compass/one_real.py REAL SIM [SIM...]

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
to one DP rank's slow launch, which a simulated run does not charge. Then, per
SIM, its requests paired with REAL's by conversation and turn (`paired`): a
percentile of each run is a rank statistic over its own request set, and E2E's
middle is sparse (#3720 c=8: 3.5% of requests within 5% of the median), so its
p50 moves with the requests either run finished and with each request's place
in the schedule, which the paired E2E does not.
"""

import json
import random
import sys
from pathlib import Path

from atom.compass.backends.json_file import read_json, read_jsonl
from atom.compass.validate import stalls
from atom.compass.validate.curves import (
    CLIENT_RUN,
    RECORDS,
    _metric,
    _positive,
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
    real, *sims = map(Path, argv)
    given = arguments(real)
    for sim in sims:
        if arguments(sim) != given:
            print(
                f"{sim} was given other client arguments than {real}", file=sys.stderr
            )
            return 2
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
    for sim in sims:
        print(paired(real, sim))
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


def stalled(real: Path) -> str:
    """REAL's stalled forwards (`validate.stalls`), which no simulated run
    charges, or why they are not counted."""
    try:
        return "real " + stalls.text(stalls.report(real))
    except (OSError, KeyError, ValueError) as e:
        return f"real: stalled forwards not counted ({type(e).__name__}: {e})"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

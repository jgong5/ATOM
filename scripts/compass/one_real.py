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
same client arguments.
"""

import json
import sys
from pathlib import Path

from atom.compass.backends.json_file import read_json, read_jsonl
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
    print(json.dumps(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

# SPDX-License-Identifier: MIT
"""Real and simulated curves, drawn and judged point by point as InferenceX's
agentic results compute them: `python -m atom.compass.validate --curves OUT ARM...`

An ARM directory holds one directory per point. A point holds two or more real
runs, `real*/`, and one simulated run per backend that priced it,
`step_linear/`, `step_grid/` or `op_graph/`; each run holds its client's
`artifacts/` as `client.sh` writes them and its engine record,
`steps/dp0.engine.json`. Every run of a point must have been given the same
client arguments (`artifacts/client_run.json`), whose profiling concurrency is
the point's.

A run's metrics are InferenceX's, as its agentic results code computes them at
SemiAnalysisAI/InferenceX 4b94f096 (`request_metrics.py`, and
`backends/atom.py` for the hit rates), over the records of
`artifacts/profile_export.jsonl` that hold no error and are of the profiling
phase or name none. A percentile interpolates linearly between ranks.

- throughput per GPU: the input and output tokens over the time from the first
  request's start to the last one's end, over the GPUs its engine record
  states, `tensor_parallel_size` times `data_parallel_size`;
- p90 TTFT; median TPOT, the median inter-token latency;
- p90 interactivity, one over the p90 inter-token latency; p90 E2E-normalized
  interactivity, one over the p90 of each request's latency per output token;
- GPU and CPU hit rates: the changes of `atom:prefix_cache_cached_tokens` and
  of `atom:lmcache_loaded_tokens` across the profiling window, each over that
  of `atom:prefix_cache_full_tokens`, summed over every series aiperf's
  `artifacts/server_metrics_export.json` holds. A simulated run's client
  scrapes no server metrics, so its hit rates have no value unless its
  artifacts hold that export.

A judged metric of a simulated run passes at its point when its value lies
inside the real runs' min-to-max band, or within `LIMITS` of their mean:
relative for throughput per GPU, p90 interactivity, p90 TTFT and median TPOT,
in percentage points for the hit rates. Its row prints the band beside the
distance, so a pass on a band wider than the limit shows. E2E-normalized
interactivity is drawn and reported, not judged. A point passes when every
judged metric does; it fails naming each that fails, and is otherwise refused,
naming each metric that has no value on some run, and the runs.

Charts, PNG files in OUT: one per ARM, `<ARM>.png`, and with two or more ARMs
one across them, `arms.png`. Each draws throughput per GPU against p90
interactivity and against p90 E2E-normalized interactivity, and p90 TTFT and
median TPOT against concurrency: the real curve as the real runs' mean with
bars from their min to their max, and a dashed curve per simulated backend.
"""

from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean

from atom.compass.backends.json_file import read_json, read_jsonl
from atom.compass.run import OP_GRAPH, STEP_GRID, STEP_LINEAR
from atom.compass.validate.runs import CLIENT_RUN

RECORDS = "artifacts/profile_export.jsonl"
SERVER_METRICS = "artifacts/server_metrics_export.json"
ENGINE_RECORD = "steps/dp0.engine.json"
SIMULATED = (STEP_LINEAR, STEP_GRID, OP_GRAPH)
CONCURRENCY = "phases.profiling.concurrency"
#: Each judged metric's distance from the real runs' mean: a percentage of the
#: mean, or, where True, percentage points.
LIMITS = {
    "throughput per GPU": (5, False),
    "p90 interactivity": (5, False),
    "p90 TTFT": (10, False),
    "median TPOT": (5, False),
    "GPU hit rate": (2, True),
    "CPU hit rate": (2, True),
}
REPORTED = ("p90 E2E-normalized interactivity",)
#: Each drawn quantity's axis label and the factor from its value to that axis.
AXES = {
    "concurrency": ("concurrency", 1),
    "throughput per GPU": ("throughput per GPU (tok/s/GPU)", 1),
    "p90 interactivity": ("p90 interactivity (tok/s/user)", 1),
    "p90 E2E-normalized interactivity": (
        "p90 E2E-normalized interactivity (tok/s/user)",
        1,
    ),
    "p90 TTFT": ("p90 TTFT (s)", 1),
    "median TPOT": ("median TPOT (ms)", 1000),
}
PANELS = (
    ("p90 interactivity", "throughput per GPU"),
    ("p90 E2E-normalized interactivity", "throughput per GPU"),
    ("concurrency", "p90 TTFT"),
    ("concurrency", "median TPOT"),
)
MARKERS = {"real": "o", STEP_LINEAR: "s", STEP_GRID: "^", OP_GRAPH: "D"}


def percentile(values: list[float], p: float) -> float | None:
    """The `p`th percentile, linear between the ranks either side; None of none."""
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    f = int(k)
    return v[f] if f + 1 >= len(v) else v[f] + (k - f) * (v[f + 1] - v[f])


def _number(value, kind=float):
    try:
        return kind(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _metric(record: dict, key: str, kind=float):
    m = record.get("metrics", {}).get(key)
    return _number(m.get("value") if isinstance(m, dict) else m, kind)


def _positive(values) -> list[float]:
    return [v for v in values if v is not None and math.isfinite(v) and v > 0]


def _inverse(x: float | None) -> float | None:
    return None if x is None or x <= 0 else 1 / x


def _scaled(x: float | None, scale: float) -> float | None:
    return None if x is None else x * scale


def _rate(n: float | None, d: float | None) -> float | None:
    return None if n is None or d is None or d <= 0 else n / d


def _counter(metrics: dict, stem: str) -> float | None:
    """The sum over every series of the counter `stem`'s names, each series
    its first stat of total, sum, max and avg; None where no series has one."""
    total, found = 0.0, False
    for name in (f"atom{s}{stem}{t}" for t in ("", "_total") for s in ":_"):
        entry = metrics.get(name)
        series = entry.get("series") if isinstance(entry, dict) else None
        for s in series if isinstance(series, list) else ():
            stats = s.get("stats") if isinstance(s, dict) else None
            stats = stats if isinstance(stats, dict) else {}
            values = [_number(stats.get(k)) for k in ("total", "sum", "max", "avg")]
            if v := [x for x in values if x is not None]:
                total, found = total + v[0], True
    return total if found else None


def metrics(run) -> dict:
    """InferenceX's metrics of the run directory `run`, None where it has none."""
    run = Path(run)
    rs = [
        r
        for r in read_jsonl(run / RECORDS)
        if not r.get("error")
        and r.get("metadata", {}).get("benchmark_phase") in (None, "profiling")
    ]
    ttft, itl = (
        [v / 1000 for v in _positive(_metric(r, k) for r in rs)]
        for k in ("time_to_first_token", "inter_token_latency")
    )
    per_token = []
    for r in rs:
        e2el, osl = _metric(r, "request_latency"), _metric(r, "output_sequence_length")
        if len(_positive((e2el, osl))) == 2:
            per_token.append(e2el / 1000 / osl)
    tokens = [
        sum(n for r in rs if (n := _metric(r, k, int)) is not None)
        for k in ("input_sequence_length", "output_sequence_length")
    ]
    meta = [r.get("metadata", {}) for r in rs]
    starts = [int(m["request_start_ns"]) for m in meta if m.get("request_start_ns")]
    ends = [int(m["request_end_ns"]) for m in meta if m.get("request_end_ns")]
    duration = (max(ends) - min(starts)) / 1e9 if starts and ends else 0
    engine = read_json(run / ENGINE_RECORD)
    gpus = engine["tensor_parallel_size"] * (engine.get("data_parallel_size") or 1)
    path = run / SERVER_METRICS
    server = (read_json(path).get("metrics") if path.is_file() else None) or {}
    full = _counter(server, "prefix_cache_full_tokens")
    return {
        "concurrency": read_json(run / CLIENT_RUN)["arguments"][CONCURRENCY],
        "throughput per GPU": (
            (tokens[0] + tokens[1]) / duration / gpus if duration > 0 else None
        ),
        "p90 interactivity": _inverse(percentile(itl, 90)),
        "p90 E2E-normalized interactivity": _inverse(percentile(per_token, 90)),
        "p90 TTFT": percentile(ttft, 90),
        "median TPOT": percentile(itl, 50),
        "GPU hit rate": _rate(_counter(server, "prefix_cache_cached_tokens"), full),
        "CPU hit rate": _rate(_counter(server, "lmcache_loaded_tokens"), full),
    }


def point(directory) -> dict:
    """The metrics of a point's runs: `real`, each real run's by name, and
    each simulated backend's; refused unless it holds two or more real runs
    and a simulated one, all given the same client arguments."""
    directory = Path(directory)
    real = sorted(r for r in directory.glob("real*") if r.is_dir())
    simulated = [directory / s for s in SIMULATED if (directory / s).is_dir()]
    if len(real) < 2 or not simulated:
        raise ValueError(
            f"{directory} holds {len(real)} real*/ and {[s.name for s in simulated]}; "
            "a point holds two or more real runs and a simulated one"
        )
    every = [*real, *simulated]
    if bare := [r.name for r in every if not (r / CLIENT_RUN).is_file()]:
        raise ValueError(f"{directory}'s {bare} hold no {CLIENT_RUN}")
    given = [read_json(r / CLIENT_RUN)["arguments"] for r in every]
    if odd := sorted({k for a in given for k in a if a.get(k) != given[0].get(k)}):
        raise ValueError(
            f"{directory}'s runs were given different client arguments: {odd}"
        )
    return {"real": {r.name: metrics(r) for r in real}} | {
        s.name: metrics(s) for s in simulated
    }


def points(arm) -> list[dict]:
    """The points of the ARM directory `arm`, by concurrency."""
    found = [
        point(d)
        for d in sorted(Path(arm).iterdir())
        if d.is_dir() and any(d.glob("real*"))
    ]
    at = [next(iter(p["real"].values()))["concurrency"] for p in found]
    if not found or len(set(at)) < len(at):
        raise ValueError(
            f"{arm}'s points are at concurrencies {at}; an arm holds one per concurrency"
        )
    return [p for _, p in sorted(zip(at, found), key=lambda x: x[0])]


def judge(name: str, real: dict, simulated: float | None) -> tuple[str, str]:
    """The value text and verdict of `name`'s simulated value against the real
    runs' values, `real` by run name."""
    if lacking := [
        n for n, v in (*real.items(), ("simulated", simulated)) if v is None
    ]:
        return f"no value in {lacking}", "refused" if name in LIMITS else "not judged"
    values = list(real.values())
    mean, lo, hi = fmean(values), min(values), max(values)
    if name not in LIMITS:
        return (
            f"simulated {simulated:.4g}; real mean {mean:.4g}, band {lo:.4g} to {hi:.4g}",
            "not judged",
        )
    limit, in_points = LIMITS[name]
    if in_points:
        off = abs(simulated - mean) * 100
        text = f"{off:.2f} points"
        bounds = f"simulated {simulated:.2%}; real mean {mean:.2%}, band {lo:.2%} to {hi:.2%}"
        what = f"{limit} points"
    else:
        off, text = abs(simulated / mean - 1) * 100, f"{simulated / mean - 1:+.2%}"
        bounds = (
            f"simulated {simulated:.4g}; real mean {mean:.4g}, band {lo:.4g} to {hi:.4g}"
            f" ({lo / mean - 1:+.2%} to {hi / mean - 1:+.2%})"
        )
        what = f"{limit}%"
    inside, near = lo <= simulated <= hi, off <= limit
    verdict = (
        ("pass: in band" if near else f"pass: in band, past {what} of mean")
        if inside
        else ("pass: within " + what + " of mean" if near else "fail")
    )
    return f"{bounds}; off mean {text}, limit {what}", verdict


def rows(found: list[dict]) -> list[tuple]:
    """(concurrency, backend, metric, value, verdict) per point of `found`,
    backend and metric, then the point's verdict per backend."""
    out = []
    for p in found:
        real = p["real"]
        c = next(iter(real.values()))["concurrency"]
        for backend in (s for s in SIMULATED if s in p):
            verdicts = {}
            for name in (*LIMITS, *REPORTED):
                # Shown in its axis's unit; a hit rate as a percentage.
                label, scale = AXES.get(name, (name, 1))
                theirs = {n: _scaled(m[name], scale) for n, m in real.items()}
                mine = _scaled(p[backend][name], scale)
                value, verdicts[name] = judge(name, theirs, mine)
                out.append((c, backend, label, value, verdicts[name]))
            failed = [n for n, v in verdicts.items() if v == "fail"]
            refused = [n for n, v in verdicts.items() if v == "refused"]
            verdict = (
                f"fail: {', '.join(failed)}"
                if failed
                else f"refused: {', '.join(refused)}" if refused else "pass"
            )
            out.append((c, backend, "point", f"{len(real)} real runs", verdict))
    return out


def _curve(ax, x: str, y: str, values: list[tuple], real: bool, **style) -> None:
    """Draw (concurrency, xs, ys) per point where every value is known: the
    real runs' as their mean with bars to their min and max, each point named
    by its concurrency, or one simulated run's, dashed."""
    (sx, sy), drawn = (AXES[x][1], AXES[y][1]), []
    for c, xs, ys in values:
        if None not in (*xs, *ys):
            drawn.append((c, [v * sx for v in xs], [v * sy for v in ys]))
    if not drawn:
        return
    mx, my = ([fmean(v[i]) for v in drawn] for i in (1, 2))
    if not real:
        ax.plot(mx, my, linestyle="--", **style)
        return
    err = [
        [
            [m - min(v[i]) for m, v in zip(means, drawn)],
            [max(v[i]) - m for m, v in zip(means, drawn)],
        ]
        for i, means in ((1, mx), (2, my))
    ]
    ax.errorbar(mx, my, xerr=err[0], yerr=err[1], capsize=3, **style)
    for (c, _, _), px, py in zip(drawn, mx, my):
        ax.annotate(
            f"c{c}", (px, py), xytext=(4, 4), textcoords="offset points", fontsize=7
        )


def draw(arms: dict, path) -> None:
    """The PANELS for `arms`, {name: points}, as one PNG at `path`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    for ax, (x, y) in zip(axes.flat, PANELS):
        for i, (arm, found) in enumerate(arms.items()):
            # One colour per arm across arms, per curve within one.
            for j, run in enumerate(("real", *SIMULATED)):
                if run == "real":
                    held = [list(p["real"].values()) for p in found]
                else:
                    held = [[p[run]] for p in found if run in p]
                values = [
                    (m[0]["concurrency"], [r[x] for r in m], [r[y] for r in m])
                    for m in held
                ]
                color = f"C{i if len(arms) > 1 else j}"
                style = {"marker": MARKERS[run], "label": f"{arm} {run}"}
                _curve(ax, x, y, values, run == "real", color=color, **style)
        ax.set_xlabel(AXES[x][0])
        ax.set_ylabel(AXES[y][0])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str]) -> None:
    """`OUT ARM...`: each arm's charts written to OUT, its verdicts printed."""
    if len(argv) < 2:
        raise ValueError("usage: python -m atom.compass.validate --curves OUT ARM...")
    out, *arms = map(Path, argv)
    named = {a.name: points(a) for a in arms}
    if len(named) < len(arms):
        raise ValueError(
            f"the ARMs {[str(a) for a in arms]} share a name, which names their charts"
        )
    out.mkdir(parents=True, exist_ok=True)
    for name, found in named.items():
        draw({name: found}, out / f"{name}.png")
    if len(named) > 1:
        draw(named, out / "arms.png")
    print("arm | concurrency | run | metric | value | verdict")
    for name, found in named.items():
        for row in rows(found):
            print(" | ".join(str(v) for v in (name, *row)))

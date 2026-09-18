"""Turn a sweep of closed-loop rungs into a saturation curve, real vs modelled.

Two axes, both read off the *engine's* clock and never the client's:

  tokens/s per GPU   total output tokens over the span the engine was working,
                     divided by the GPUs it was working on. The throughput the
                     hardware delivered.
  tokens/s per user  1/TPOT, median over requests. The speed one user sees.

Reading either from the client would silently measure the wrong thing on the
modelled side: a socket timing a simulated run measures how fast the simulator
ran, not what it predicted. So every number here comes from
`GET /compass/requests`, joined to the client only for token counts.

Each rung is a directory written by `replay_validate.py --clients N`, holding
`real.json` and `modelled.json`. The curve is the rungs in client order:

    python scripts/compass/saturation.py out/sweep/c1 out/sweep/c4 \
        out/sweep/c8 out/sweep/c16 --out out/sweep/saturation.json \
        --plot out/sweep/saturation.png
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _compare_module():
    """`replay_compare`'s join, rather than a second copy of it.

    The join from client result to engine record is the one place this family
    of scripts can silently disagree with itself, and a saturation curve that
    joined differently from the accuracy report would be a different
    measurement wearing the same run's name.
    """
    spec = importlib.util.spec_from_file_location(
        "replay_compare_for_saturation", _HERE / "replay_compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _busy_span(records):
    """(span, busy) seconds: wall from first arrival to last finish, and how
    much of it the engine held at least one request.

    The gap between the two is the client's round trip. It matters because the
    two sides do not pay it alike: a real clock keeps ticking while a response
    travels back and the next request comes out, and the virtual clock does
    not advance at all when the engine is empty. So the difference is reported
    rather than assumed small -- at 256k contexts it is, but that is a fact
    about this workload and not about the method.
    """
    spans = [(float(r["arrive_time"]), float(r["finish_time"]))
             for r in records
             if r.get("arrive_time") is not None
             and r.get("finish_time") is not None]
    if not spans:
        return None, None
    spans.sort()
    span = max(end for _, end in spans) - min(start for start, _ in spans)
    busy = 0.0
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if start > cur_end:
            busy += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    busy += cur_end - cur_start
    return span, busy


def rung(artifact, gpus):
    """Both axes for one side of one rung."""
    mod = _compare_module()
    readings = mod._per_request(artifact)
    records = (artifact.get("engine") or {}).get("requests", [])

    # Only the requests this run actually executed. A closed-loop rung runs a
    # subset of the trace, and dividing its tokens by the whole trace's span
    # would report a throughput no run achieved.
    ids = {(r.get("response") or {}).get("id")
           for r in artifact.get("results", []) if r.get("ok")}
    records = [r for r in records if r.get("request_id") in ids]

    span, busy = _busy_span(records)
    produced = sum(r["output_tokens"] for r in readings.values())
    arrivals = {round(float(r["arrive_time"]), 9) for r in records
                if r.get("arrive_time") is not None}
    rates = [1.0 / r["tpot_s"] for r in readings.values()
             if r.get("tpot_s") and r["tpot_s"] > 0]

    manifest = artifact.get("run") or {}
    return {
        "clients": int(manifest.get("clients") or 0),
        "closed_loop": bool(manifest.get("closed_loop")),
        "requests": len(readings),
        "output_tokens": produced,
        "engine_span_s": span,
        "engine_busy_s": busy,
        "busy_fraction": (busy / span) if span else None,
        "tokens_s_per_gpu": (produced / span / gpus) if span else None,
        "tokens_s_per_user": _median(rates),
        "tokens_s_per_user_n": len(rates),
        "distinct_arrivals": len(arrivals),
        "failed": int(manifest.get("failed") or 0),
    }


def _plot(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    styles = {"real": ("o-", "#1f77b4"), "modelled": ("s--", "#d62728")}
    for name, points in curves.items():
        style, colour = styles.get(name, ("^:", "#555555"))
        xs = [p["tokens_s_per_user"] for p in points]
        ys = [p["tokens_s_per_gpu"] for p in points]
        ax.plot(xs, ys, style, color=colour, label=name)
        # Labelled with the client count: the two axes are both *derived*, so
        # without it a reader cannot tell which end of the curve is one user
        # and which is sixteen.
        for point, x, y in zip(points, xs, ys):
            ax.annotate(f"{point['clients']}", (x, y), textcoords="offset points",
                        xytext=(5, 4), fontsize=8, color=colour)
    ax.set_xlabel("tokens/s per user  (1/TPOT, median)")
    ax.set_ylabel("tokens/s per GPU")
    ax.set_title("cc-traces closed loop: real against modelled")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", help="one rung directory per client count")
    p.add_argument("--gpus", type=int, default=1,
                   help="GPUs the engine served on; the per-GPU axis divides "
                        "by this, so a TP2 run reported as TP1 doubles it")
    p.add_argument("--out", required=True)
    p.add_argument("--plot", default=None)
    args = p.parse_args(argv)

    curves = {"real": [], "modelled": []}
    missing = []
    for directory in args.dirs:
        base = Path(directory)
        for side in ("real", "modelled"):
            path = base / f"{side}.json"
            if not path.exists():
                missing.append(str(path))
                continue
            with open(path, encoding="utf-8") as fh:
                point = rung(json.load(fh), args.gpus)
            point["dir"] = str(base)
            curves[side].append(point)

    for side in curves:
        curves[side].sort(key=lambda pt: pt["clients"])

    open_loop = [pt["dir"] for side in curves for pt in curves[side]
                 if not pt["closed_loop"]]

    print(f"{'side':<9}{'clients':>8}{'reqs':>7}{'tok/s/GPU':>12}"
          f"{'tok/s/user':>12}{'busy':>8}")
    for side in ("real", "modelled"):
        for pt in curves[side]:
            gpu = pt["tokens_s_per_gpu"]
            user = pt["tokens_s_per_user"]
            busy = pt["busy_fraction"]
            print(f"{side:<9}{pt['clients']:>8}{pt['requests']:>7}"
                  f"{(f'{gpu:.1f}' if gpu else '-'):>12}"
                  f"{(f'{user:.2f}' if user else '-'):>12}"
                  f"{(f'{busy:.3f}' if busy else '-'):>8}")

    blocking = []
    if missing:
        blocking.append(f"{len(missing)} artifact(s) absent: {missing[0]}"
                        + (" ..." if len(missing) > 1 else ""))
    if open_loop:
        # A paced rung on this plot is not a slower point on the same curve:
        # it is a different experiment, pinned to the trace's arrival rate,
        # and its throughput says nothing about saturation.
        blocking.append(f"{len(open_loop)} rung(s) were not closed-loop: "
                        f"{open_loop[0]}")
    for side in ("real", "modelled"):
        for pt in curves[side]:
            if pt["failed"]:
                blocking.append(f"{side} rung {pt['clients']} had "
                                f"{pt['failed']} failed request(s)")
            if pt["tokens_s_per_user"] is None:
                blocking.append(f"{side} rung {pt['clients']} produced no TPOT "
                                f"reading, so it has no per-user axis")
            # More requests than clients means somebody's second request was
            # sent after somebody's first came back, so they cannot all have
            # arrived at one instant. When they do, arrival was stamped on a
            # clock that never advanced -- which reads as a plausible curve:
            # TTFT absorbs the queueing each request had already passed, and
            # the span collapses onto busy time, overstating per-GPU
            # throughput. Measured at +68% before this check existed.
            if pt["requests"] > max(1, pt["clients"]) and \
                    pt["distinct_arrivals"] <= 1:
                blocking.append(
                    f"{side} rung {pt['clients']} stamped all "
                    f"{pt['requests']} requests at one arrival instant; "
                    f"arrival is being read from a clock that does not advance")
    counts = {side: [pt["clients"] for pt in curves[side]] for side in curves}
    if counts["real"] != counts["modelled"]:
        blocking.append(f"the two sides swept different client counts: "
                        f"real {counts['real']} modelled {counts['modelled']}")

    report = {"gpus": args.gpus, "curves": curves, "blocking": blocking}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    if args.plot and not blocking:
        _plot(curves, args.plot)
        print(f"\nplot -> {args.plot}")
    elif args.plot:
        # Deliberately not drawn. A curve is read at a glance and its caveats
        # are not; one drawn from a rung that failed requests would be
        # believed.
        print(f"\nplot NOT drawn; see the reasons below", file=sys.stderr)

    print(f"report -> {args.out}")
    if blocking:
        print(f"\ncurve WITHHELD, {len(blocking)} reason(s):", file=sys.stderr)
        for reason in blocking:
            print(f"  - {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

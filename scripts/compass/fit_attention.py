"""Fit prefill attention cost against the sequence-length multiset.

Every other operator in a prefill graph can be priced at whatever token count
occurs, by rewriting its shapes (`synth_shapes.py`). Attention cannot: its cost
is quadratic *within* each sequence, so two steps with the same token total and
different sequence splits do different amounts of work, and a shape rewrite
would produce a number that means nothing.

What attention does have, and the gemms do not, is smoothness -- no tile
cliffs, cost rising with sequence length in a shape the hardware does not
discontinuously change. That is the case where fitting is the right tool rather
than a way of hiding that you did not measure.

Measurement says the cost is separable per sequence: at a fixed length, cost
per sequence barely moves with how many sequences run together (-1.1% from 2 to
4 at L=2294, -0.9% from 1 to 2 at L=4694). So the model is a per-sequence cost
summed over the batch,

    seconds = sum_i f(L_i),   f(L) = a*L^2 + b*L + c

with L_i the query lengths from `cu_seqlens_q`: quadratic for the score matrix,
linear for the projections and the K/V walk, constant for setup. Note the
constant is per *sequence*, not per graph -- putting it per graph is a
materially worse fit, which is why several forms are compared below rather than
one being assumed.

Model choice is made on leave-one-out error, not in-sample error: with this few
points a richer model always wins in-sample and that says nothing about a
sequence length nobody traced.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys


def sequence_lengths(op: dict) -> list[int] | None:
    for key, value in op.get("context") or ():
        if key == "cu_seqlens_q":
            return [b - a for a, b in zip(value, value[1:])]
    return None


def solve(rows: list[list[float]], rhs: list[float]) -> list[float]:
    """Relative-error least squares.

    The costs here span 7ms to 106ms, and unweighted least squares would spend
    the parameters on the largest graph and fit the rest badly -- backwards,
    since a percentage is what the cost model is judged on at every size.
    """
    return _normal_equations([[v / y for v in r] for r, y in zip(rows, rhs)],
                             [1.0 for _ in rhs])


def _normal_equations(rows: list[list[float]],
                      rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting on the normal equations."""
    n = len(rows[0])
    a = [[sum(r[i] * r[j] for r in rows) for j in range(n)] + [
        sum(r[i] * y for r, y in zip(rows, rhs))] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-30:
            raise SystemExit(
                "design matrix is singular -- the inputs do not vary "
                "independently, so nothing can be fitted from them")
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(n):
            if r == col:
                continue
            f = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= f * a[col][c]
    return [a[i][n] / a[i][i] for i in range(n)]


def _linear(design, names):
    """A model linear in its coefficients, fitted on total seconds."""
    return {
        "names": names,
        "fit": lambda points: solve([design(p) for p in points],
                                    [p["seconds"] for p in points]),
        "predict": lambda coef, p: sum(c * x
                                       for c, x in zip(coef, design(p))),
    }


def _seq_len(p: dict) -> float:
    return p["tokens"] / p["seqs"]


def _quantised():
    """f(L) = L * (a + b * ceil(L / S)).

    Measured cost per token is not a smooth function of L: it sits on flat
    plateaus and steps between them, and the plateau index is ceil(L/S) for a
    fixed S. The score-matrix work is done in whole chunks of S keys, so the
    quadratic term is quantised rather than continuous -- which is why a power
    law leaves a periodic sawtooth no amount of refitting removes.

    Linear in (a, b) once S is fixed, so S is grid-searched and the rest
    solved.
    """
    def rows(points, S):
        return [[sum(L for L in p["lengths"]),
                 sum(L * math.ceil(L / S) for L in p["lengths"])]
                for p in points]

    def predict(coef, p):
        a, b, S = coef
        return sum(L * (a + b * math.ceil(L / S)) for L in p["lengths"])

    def fit(points):
        best = None
        for S in range(1600, 4001, 10):
            try:
                a, b = solve(rows(points, S), [p["seconds"] for p in points])
            except SystemExit:
                continue
            err = max(abs(predict((a, b, S), p) - p["seconds"]) / p["seconds"]
                      for p in points)
            if best is None or err < best[0]:
                best = (err, [a, b, S])
        if best is None:
            raise SystemExit("quantised model not identifiable")
        return best[1]

    return {"names": ("a", "b", "S"), "fit": fit, "predict": predict}


def _power():
    """f(L) = a * L^p, fitted in log space on per-sequence cost.

    Not reachable by the polynomial family: a polynomial with non-negative
    coefficients has a local exponent that rises monotonically toward 2, and
    the measured exponent does not.
    """
    return {
        "names": ("log_a", "p"),
        "fit": lambda points: _normal_equations(
            [[1.0, math.log(_seq_len(p))] for p in points],
            [math.log(p["seconds"] / p["seqs"]) for p in points]),
        "predict": lambda coef, p: p["seqs"] * math.exp(
            coef[0] + coef[1] * math.log(_seq_len(p))),
    }


# Each is sum_i f(L_i) for a different f, except "c/graph" which puts the
# constant on the step rather than the sequence -- kept to show it loses.
MODELS = {
    "L.(a + b.ceil(L/S))": _quantised(),
    "a.L^p": _power(),
    "a.L2 + b.L": _linear(lambda p: [p["sum_sq"], p["tokens"]], ("a", "b")),
    "a.L2 + b.L + c/seq": _linear(
        lambda p: [p["sum_sq"], p["tokens"], float(p["seqs"])],
        ("a", "b", "c")),
    "a.L2 + b.L + c/graph": _linear(
        lambda p: [p["sum_sq"], p["tokens"], 1.0], ("a", "b", "c")),
}


def collect(paths: list[str], prices: dict) -> list[dict]:
    from atom.compass.runtime.microbench import signature_of

    points = []
    for path in sorted(paths):
        graph = json.load(open(path))
        total = 0.0
        lengths: list[int] = []
        missing = 0
        for op in graph["ops"]:
            if "attention" not in op["name"]:
                continue
            entry = prices.get(signature_of(op))
            if entry is None:
                missing += 1
                continue
            total += entry["seconds"] * entry.get("occurrences", 1)
            if not lengths:
                lengths = sequence_lengths(op) or []
        if not lengths or total <= 0:
            print(f"  skipped {path}: nothing priced", file=sys.stderr)
            continue
        points.append({
            "path": path,
            "lengths": lengths,
            "seqs": len(lengths),
            "tokens": sum(lengths),
            "sum_sq": float(sum(L * L for L in lengths)),
            "seconds": total,
            "missing": missing,
        })
    return points


def report(points, coef, model, label: str) -> None:
    print(f"\n{label}")
    print(f"  {'seqs':>5} {'tokens':>8} {'per seq':>8} "
          f"{'priced':>10} {'fitted':>10} {'diff':>8}")
    worst = 0.0
    for p in points:
        got = model["predict"](coef, p)
        err = (got - p["seconds"]) / p["seconds"] * 100
        worst = max(worst, abs(err))
        print(f"  {p['seqs']:5d} {p['tokens']:8d} "
              f"{p['tokens'] // p['seqs']:8d} "
              f"{p['seconds'] * 1e3:9.3f}ms {got * 1e3:9.3f}ms {err:+7.2f}%")
    print(f"  worst {worst:.2f}%")


def loo(points, model) -> tuple[list[float], float, float]:
    """Refit without each point and predict it -- the only number that answers
    "what happens at a sequence length nobody traced"."""
    errs = []
    for i, held in enumerate(points):
        rest = points[:i] + points[i + 1:]
        try:
            c = model["fit"](rest)
        except SystemExit:
            return [], float("inf"), float("inf")
        errs.append(abs(model["predict"](c, held) - held["seconds"])
                    / held["seconds"] * 100)
    ordered = sorted(errs)
    return errs, ordered[len(ordered) // 2], max(ordered)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True,
                    help="glob of traced prefill graphs; comma-separated globs "
                         "are unioned")
    ap.add_argument("--prices", required=True)
    ap.add_argument("--out", help="write coefficients here")
    ap.add_argument("--min-len", type=int, default=0,
                    help="ignore graphs whose sequences are shorter than this")
    args = ap.parse_args()

    prices = json.load(open(args.prices))["prices"]
    paths = sorted({q for g in args.graphs.split(",")
                    for q in glob.glob(g.strip())})
    if not paths:
        raise SystemExit(f"--graphs {args.graphs!r} matched nothing")
    points = collect(paths, prices)
    if args.min_len:
        points = [p for p in points if p["tokens"] // p["seqs"] >= args.min_len]
    if len(points) < 4:
        raise SystemExit(f"{len(points)} usable graphs; need at least 4")

    print(f"\n{len(points)} graphs, per-sequence lengths "
          + ", ".join(str(p["tokens"] // p["seqs"]) for p in points))
    print("\nmodel selection by leave-one-out error")
    print(f"  {'model':<24} {'LOO median':>11} {'LOO worst':>10}")
    scored = []
    for name, model in MODELS.items():
        _errs, median, worst = loo(points, model)
        scored.append((median, worst, name, model))
        shown = "singular" if median == float("inf") else f"{median:10.2f}%"
        print(f"  {name:<24} {shown:>11} "
              + ("" if worst == float("inf") else f"{worst:9.2f}%"))
    scored.sort(key=lambda r: (r[0], r[1], r[2]))
    median, worst, name, model = scored[0]
    if median == float("inf"):
        raise SystemExit("no model is identifiable from these graphs")
    print(f"\nchosen: {name} (LOO median {median:.2f}%, worst {worst:.2f}%)")

    coef = model["fit"](points)
    report(points, coef, model, "in-sample fit")
    errs, _m, _w = loo(points, model)
    print("\nleave-one-out, per graph")
    for held, err in zip(points, errs):
        print(f"  {held['seqs']:2d} seqs x {held['tokens'] // held['seqs']:6d} "
              f"tokens: {err:6.2f}%")

    if args.out:
        json.dump({
            "version": 1,
            "provenance": "empirical/fitted",
            "model": name,
            "terms": list(model["names"]),
            "coefficients": dict(zip(model["names"], coef)),
            "loo_median_percent": median,
            "loo_worst_percent": worst,
            "fitted_on": [
                {k: p[k] for k in ("path", "seqs", "tokens", "seconds")}
                for p in points],
        }, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

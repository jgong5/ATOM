#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""How much of real records' steps a calibration's hull holds.

    cal_coverage.py LAW [--population RUN[:A-B] ...] RECORD...

LAW is a step linear law (`step_linear_law.json`, or its `params` as law.json),
whose target prefill vector and decode vectors name the features. A DP step is
taken as the fit and `validate.cost` take it: at its priced rank
(`step_linear_fit.priced_rank`, the rank LAW prices highest; a prefill step's
dummy forwards at their `step_linear.dummy_view`), its point that rank's features
on its vector's hull axes (`step_linear.hull_axes`), its seconds the smallest of
its ranks' `d_busy_ns`. With no `--population` a step is held when it lies in its
vector's own hull, the law's fit; with them, in the hull `step_linear_fit._hull`
gives the points of the populations' steps for that vector: each RUN a record or
a simulated run's directory (its `steps/dp*.jsonl`, a dry run's shapes), its steps
A to B only when given. A decode rung no population reached holds nothing.

Per RECORD, per vector, its steps and seconds and the share of each held; a step
the law refuses, and one only eager decodes beside a prefilling peer run (priced
0 s), are counted apart.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from atom.compass.backends import (
    CostRefused,
    StepLinearBackend,
    StepLinearLaw,
    step_labels,
)
from atom.compass.backends.hull import Hull
from atom.compass.backends.shape import BatchView, RequestShape, sum_context
from atom.compass.backends.step_linear import dummy_view
from atom.compass.backends.step_linear_fit import _hull, priced_rank


def steps(run: str):
    """(step seconds, {rank: view}) of each step of `run`, DIR or DIR:A-B, its
    prefill dummies at their dummy view."""
    path, _, span = run.partition(":")
    lo, hi = map(int, span.split("-")) if span else (0, 1 << 62)
    files = sorted(
        Path(path, "steps").glob("dp*[0-9].jsonl"), key=lambda p: int(p.stem[2:])
    )
    ranks = [[json.loads(x) for x in f.read_text().splitlines()] for f in files]
    dummies, peaks = step_labels.group_dummies(ranks), step_labels.group_peaks(ranks)
    tails, at = step_labels.group_tails(ranks), [
        {s["step"]: s for s in r} for r in ranks
    ]
    for n in sorted(set().union(*at)):
        group = {r: x[n] for r, x in enumerate(at) if n in x}
        real = {r: s for r, s in group.items() if not s["is_dummy_run"]}
        if not real or not lo <= n <= hi:
            continue
        views = {}
        for r, s in real.items():
            views[r] = BatchView(
                tuple(RequestShape(q, c, bool(d)) for q, c, d in s["rows"]),
                capture_rung=s["rung"],
                moe_rows=s.get("moe_rows"),
                moe_tokens=s.get("moe_tokens")
                or step_labels.group_tokens(list(group.values())),
                produces_output=s["produces_output"],
                draft_bs=s.get("draft_bs"),
                dp_rank=s["dp_rank"],
                group_dummies=dummies.get(n, 0),
                group_max_tokens=peaks.get(n),
                group_tail=n in tails,
                group_needs_eager=s.get("group_needs_eager", False),
            )
        peer = next(iter(views.values()))
        if peer.prefill:
            views |= {r: dummy_view(peer) for r in group if r not in real}
        busy = [s["d_busy_ns"] for s in real.values() if "d_busy_ns" in s]
        yield (min(busy) * 1e-9 if busy else 0.0), views


class Points(StepLinearBackend):
    """A step's priced rank's target vector and hull point."""

    def _priced(self, law, vector, kind, counts, what):
        if not what and kind in ("prefill", "decode") and self.got is None:
            self.got = (vector, tuple(counts[i] for i in vector.axes))
        return super()._priced(law, vector, kind, counts, what)

    def step(self, views: dict):
        """(vector or why none, point) of the step whose ranks are `views`."""
        priced = {}
        for r, view in views.items():
            self.got = None
            try:
                cost = self.price(view)
            except CostRefused:
                return "refused", None
            context = sum_context(view.requests)
            priced[r] = ((cost.target_seconds, len(view.requests), context), self.got)
        top = priced_rank({r: p[0] for r, p in priced.items()})
        return priced[top][1] or ("eager", None)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("law", type=Path)
    p.add_argument("records", nargs="+")
    p.add_argument("--population", action="append", default=[])
    a = p.parse_args(argv)
    law = json.loads(a.law.read_text())
    law = {k: v for k, v in law.get("params", law).items() if k != "draft"}
    points = Points(StepLinearLaw(law), dp=4)
    hulls = {}
    if a.population:
        held = defaultdict(list)
        for run in a.population:
            for _, views in steps(run):
                vector, x = points.step(views)
                if x is not None:
                    held[vector.name].append(x)
        for vector in [points.law.prefill, *points.law.decode.values()]:
            names = tuple(vector.features[i] for i in vector.axes)
            if len(held[vector.name]) > len(names):
                hull = _hull(np.array(held[vector.name], dtype=np.float64), names)
                hulls[vector.name] = Hull(vector.name, names, hull)
        print("population:", ", ".join(a.population))
        print("  " + "; ".join(f"{k} {len(v)}" for k, v in held.items()))
    for record in a.records:
        rows = defaultdict(lambda: np.zeros(4))  # steps, held, seconds, held seconds
        for seconds, views in steps(record):
            vector, x = points.step(views)
            key = vector if x is None else vector.name
            inside = x is not None and (
                key in hulls and hulls[key].contains(x)
                if a.population
                else vector.hull.contains(x)
            )
            rows[key] += (1, inside, seconds, seconds * inside)
        for name, kind in (("prefill all", "prefill"), ("decode all", "decode rung")):
            rows[name] = sum(
                (v for k, v in list(rows.items()) if k.startswith(kind)), np.zeros(4)
            )
        print(f"\n{record}\n  vector | steps | held | seconds | held")
        last = lambda k: int(k.split()[-1]) if k[-1].isdigit() else 999
        for k in sorted(
            rows, key=lambda k: (k.split()[0] != "prefill", "all" in k, last(k))
        ):
            n, h, sec, hs = rows[k]
            print(
                f"  {k} | {n:.0f} | {h / max(n, 1):.1%} | {sec:.1f} | {hs / max(sec, 1e-12):.1%}"
            )


if __name__ == "__main__":
    main()

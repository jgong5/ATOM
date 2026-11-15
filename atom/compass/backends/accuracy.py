# SPDX-License-Identifier: MIT
"""One run's labelled steps priced by a backend and graded: the report
`validate.cost` gives on a record, whose docstring states its rules, given the
run's samples rather than read from its directory. The step linear fit grades
each held-out run with it (`step_linear_fit.fold_figures`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

import numpy as np

from atom.compass.backends import (
    CostRefused,
    Method,
    StepGridBackend,
    StepLinearBackend,
    StepLinearLaw,
    step_linear_fit,
)
from atom.compass.backends.cost import GROUPS
from atom.compass.backends.cost import group as method_group
from atom.compass.backends.step_linear_fit import WARM_FACTOR, _band
from atom.compass.backends.step_roofline import StepFallbackBackend

WITHIN = 0.10
#: The report's groups a step enters, by its terms: a declared one only its own.
GRADED = {
    "validated": ("graded", "validated"),
    "unvalidated": ("graded", "unvalidated"),
    "declared": ("declared",),
}


def _is_grid(backend) -> bool:
    """Whether `backend` is a step grid, bare or the first one a fallback asks."""
    first = backend.first if isinstance(backend, StepFallbackBackend) else backend
    return isinstance(first, StepGridBackend)


def _leading_slow(priced: list[tuple], law: dict | None) -> set[int]:
    """Steps leading their vector's (vector, band, step, seconds, price) rows, each
    slow by the scatter `law` fitted the vector on, or by the scatter here where
    `law` records none or is None, as for an op graph."""
    out = set()
    for key in sorted({p[0] for p in priced}):
        mine = [p for p in priced if p[0] == key]
        error = np.array([p[4] / p[3] - 1.0 for p in mine])
        centre = np.median(error)
        mad = 1.4826 * np.median(np.abs(error - centre))
        vector = {}
        if law is not None:
            vector = law["prefill"] if key == "prefill" else law["decode"][key]
        if "mad" in vector.get("fit", {}):
            centre, mad = vector["fit"]["centre"], vector["fit"]["mad"]
        slow = (error < 0) & (
            np.abs(error - centre)
            > max(step_linear_fit.MAD_K * mad, step_linear_fit.MAD_FLOOR)
        )
        for p, s in zip(mine, slow):
            if not s:
                break
            out.add(p[2])
    return out


def _summary(steps: list[tuple]) -> dict:
    """(band, step, real, priced) rows of one kind as the report's figures."""
    real = np.array([s[2] for s in steps])
    priced = np.array([s[3] for s in steps])
    error = priced / real - 1.0
    largest = int(np.argmax(np.abs(priced - real)))
    rest = np.arange(len(steps)) != largest
    bands = {}
    for band in sorted({s[0] for s in steps}):
        e = error[[s[0] == band for s in steps]]
        p10, p50, p90 = np.percentile(e, (10, 50, 90))
        bands[band] = {
            "steps": len(e),
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "within_10pct": float(np.mean(np.abs(e) <= WITHIN)),
        }
    return {
        "steps": len(steps),
        "real_s": float(real.sum()),
        "priced_s": float(priced.sum()),
        "on_totals": float(priced.sum() / real.sum() - 1.0),
        "held_out_step": steps[largest][1],
        "held_out": (
            float(priced[rest].sum() / real[rest].sum() - 1.0) if rest.any() else None
        ),
        "median": float(np.median(error)),
        "bands": bands,
    }


def _rank(rows: list[tuple], buses: list[str] | None) -> dict:
    """The (real, priced) rows of the graded steps one rank was priced as."""
    real, priced = np.array(rows).T
    return {
        "buses": buses,
        "steps": len(rows),
        "on_totals": float(priced.sum() / real.sum() - 1.0),
        "median": float(np.median(priced / real - 1.0)),
    }


def run_accuracy(run, backend, steps=None, dummies=None, cards=None) -> dict:
    """`validate.cost.cost_accuracy`'s report but its ``record``, on `run`, one run's ranks'
    samples as `step_labels.label_run` labels them: `dummies` holds each rank's
    steps run as dummy forwards, read from its samples when None, and `cards`
    each rank's number, in rank order, to its cards' buses, rank i numbered i
    with none named when None."""
    cards = cards or dict.fromkeys(range(len(run)))
    number = list(cards)
    joined, cut, _ = step_linear_fit._groups(run, "base", steps=steps, dummies=dummies)
    views = [{s.step: s.view for s in rank} for rank in run]
    drafts = [{s.step: s.draft_s for s in rank} for rank in run]
    law = backend if isinstance(backend, Mapping) else None
    if law is not None:
        backend = StepLinearBackend(StepLinearLaw(law))
    grid = _is_grid(backend)
    groups = {g: {} for g in ("graded", "validated", "unvalidated", "declared")}
    declined: dict[str, dict] = {}
    fell: dict[str, dict] = {}
    warm, priced, extrapolated = [], [], 0
    for group in joined:
        key, _, seconds, step, *_ = next(iter(group.values()))
        try:
            costs = {i: backend.price(views[i][step]) for i in group}
        except CostRefused as refused:
            band = _band(key, max((r[1] for r in group.values()), key=lambda c: c[1]))
            _count(declined, band, seconds, refused.declined)
            continue
        top = step_linear_fit.priced_rank(
            {i: (c.target_seconds, group[i][1]) for i, c in costs.items()}
        )
        extrapolated += any(t.provenance.extrapolated for t in costs[top].terms)
        band, price = _band(key, group[top][1]), costs[top].target_seconds
        # Each term of a fallback's answer carries what declined the step.
        if costs[top].refusals:
            _count(fell, band, seconds, dict.fromkeys(costs[top].refusals))
        labels = [drafts[i][step] for i in group]
        draft = None
        if None not in labels:
            mine = costs[top].draft_terms
            draft = (min(labels), costs[top].draft_seconds, _group(mine))
        target = _group(costs[top].terms)
        priced.append((key, band, step, seconds, price, target, draft, costs[top], top))
    leading = set() if grid else _leading_slow(priced, law)
    parts: dict[str, list] = {}
    by_rank: dict[str, dict] = {}
    for key, band, step, seconds, price, target, draft, cost, top in priced:
        if step in leading or seconds > WARM_FACTOR * price:
            warm.append(
                {
                    "step": step,
                    "band": band,
                    "real_s": seconds,
                    "priced_s": price,
                }
            )
            continue
        kind = "prefill" if key == "prefill" else "decode"
        for group in GRADED[target]:
            groups[group].setdefault(kind, []).append((band, step, seconds, price))
        if "graded" in GRADED[target]:
            by_rank.setdefault(kind, {}).setdefault(top, []).append((seconds, price))
        parts.setdefault(kind, []).append((seconds, cost))
        if draft is not None:
            for group in GRADED[draft[2]]:
                groups[group].setdefault("draft", []).append((band, step, *draft[:2]))
    name = backend.describe() if law is None else backend.law.fitted_on
    report = {"backend": name, "cut": cut}
    for group, kinds in groups.items():
        report[group] = {
            kind: _summary(rows) for kind, rows in sorted(kinds.items(), reverse=True)
        }
    report["warm"] = warm
    report["fallback"], report["refused"] = fell, declined
    report["extrapolated"] = extrapolated
    report["ranks"] = {
        kind: {
            number[i]: _rank(rows, cards[number[i]])
            for i, rows in sorted(mine.items(), key=lambda m: number[m[0]])
        }
        for kind, mine in sorted(by_rank.items(), reverse=True)
        if len(run) > 1
    }
    report["methods"], report["terms"] = {}, {}
    for kind, priced_steps in sorted(parts.items(), reverse=True):
        real, methods, terms = np.array([s for s, _ in priced_steps]), Counter(), {}
        for i, (_, cost) in enumerate(priced_steps):
            for t in cost.terms:
                methods[
                    method_group(t.provenance.method, t.provenance.validated)
                ] += t.seconds
                name = t.name.removeprefix("op.").removesuffix(".price_list")
                terms.setdefault(name, np.zeros(len(priced_steps)))[i] += t.seconds
        report["methods"][kind] = {g: methods[g] for g in GROUPS if g in methods}
        report["terms"][kind] = {
            name: {
                "on_totals": float(s.sum() / real.sum()),
                **dict(
                    zip(("p10", "p50", "p90"), np.percentile(s / real, (10, 50, 90)))
                ),
            }
            for name, s in sorted(terms.items())
        }
    return report


def _count(into: dict, band: str, seconds: float, refusals) -> None:
    """Count a step of `seconds` in `band` of `into`, with its `refusals`."""
    r = into.setdefault(band, {"steps": 0, "real_s": 0.0, "reasons": Counter()})
    r["steps"] += 1
    r["real_s"] += seconds
    r["reasons"]["; ".join(f"{d.source}: {d.reason}" for d in refusals)] += 1


def _validated(term) -> bool:
    """Whether `term` is validated: any answer but an estimated one outside
    its hull, which the run summary counts as unvalidated."""
    p = term.provenance
    return method_group(p.method, p.validated) != "unvalidated"


def _group(terms: list) -> str:
    """The `GRADED` key of a step priced by `terms`: declared when one is,
    validated when there are some and each is, else unvalidated."""
    if any(t.provenance.method is Method.DECLARED for t in terms):
        return "declared"
    return "validated" if terms and all(map(_validated, terms)) else "unvalidated"

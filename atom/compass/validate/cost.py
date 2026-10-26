# SPDX-License-Identifier: MIT
"""Cost accuracy: a law, a step grid or an op graph, priced on a real run's own step sequence.

    python -m atom.compass.validate.cost RECORD LAW

RECORD is a real run's step record directory (`ATOM_COMPASS_PARITY_RECORD`,
`OUT_DIR/steps` of a `REAL=1 cctraces_sim.sh` cell); LAW is a law file as
`step_linear_fit` writes it. Pricing the real run's own steps separates cost error
from schedule error: both sides run the same batches. The command prices every
step with a law; `python -m atom.compass.validate` prices only those of the client
run it grades, its warmup left out, with the backend the simulated run priced with.

Each step is labelled as the fit labels it, by its device time
(`step_labels.label` under ATOM's split rule, `bound._split`, which also says
which steps carry no label and refuses a record without device windows), and priced by `StepLinearBackend`, `StepGridBackend` or `OpGraphBackend`. The
error of a step is `priced / real - 1`, its target price over its target label on
an MTP record. A step that also carries a draft label is graded again under kind
`draft`, its draft price over that label, validated only if it has draft terms and
each is. Per kind the report gives the error
on totals, on totals holding out the step that contributes most to their error,
and the median per step; per decode rung or prefill token band, the p10, p50
and p90 of the per-step error and the fraction of steps within 10%.

The graded figures hold every step whose priced rank has no declared target
term, at every backend. The same figures are given beside them over its
validated steps and over its unvalidated ones: those whose terms are not all
validated (outside the hull of the law's held-out steps, every step of a law
that records none; for the grid, outside its residual's held-out hull). A step
with a declared term is reported apart, in no graded figure; a draft is
grouped by its draft terms alike. A step that ran more
than `step_linear_fit.WARM_FACTOR` times its price is warm, a one-off first touch the
law does not model: it is named with its seconds and enters no figure. So are,
under a law or an op graph, the steps that lead a vector's priced steps, each
slower than its price by more than `step_linear_fit.MAD_K` scaled MADs of the
errors the law's vector was fitted on, as the fit names them; of that vector's
errors here if the law records no `fit.mad` or the backend is an op graph. A step
grid, bare or below a fallback, has no such rule: its bundle's points are measured
after the bench warm-up and its fit names no warm step, so only a step above that
factor is warm. A step the backend declines is counted
under `refused` with its real seconds and its reasons, the most frequent named.
A step whose priced rank a fallback answered after a source declined it
(`StepCost.refusals`), as the step roofline below a step grid answers, is
graded by its terms and also counted under `fallback`, with what declined it.
A step it extrapolated, past its support domain, is counted too. An answer of any
method but an estimated one outside its hull is validated, as the run summary
groups seconds: an op graph's analytical and measured leaves are, its
interpolated ones only inside a held-out bracket on each line they interpolate
along, and are unvalidated otherwise.

Per kind the report also gives the priced seconds of those steps by method
(`backends.cost.GROUPS`), and per cost term, an op graph's op kind its
``op.<kind>`` and ``op.<kind>.price_list`` terms together, the term's seconds
over the measured ones: on totals, and the p10, p50 and p90 per step. Summed
over the terms, they are the composed step over the measured one.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from atom.compass.backends import (
    CostRefused,
    Method,
    StepGridBackend,
    StepLinearBackend,
    StepLinearLaw,
    step_labels,
    step_linear_fit,
)
from atom.compass.backends.cost import GROUPS
from atom.compass.backends.cost import group as method_group
from atom.compass.backends.json_file import read_json, read_jsonl
from atom.compass.backends.step_linear_fit import WARM_FACTOR, _band
from atom.compass.backends.step_roofline import StepFallbackBackend
from atom.compass.drive import aiter_jit
from atom.compass.validate.bound import _split

WITHIN = 0.10
REASONS = 3  # a declined band names this many of its most frequent reasons
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


def cost_accuracy(record, backend, steps=None) -> dict:
    """The report for `backend`, a `CostBackend` or a law mapping
    `StepLinearLaw` reads, on the record directory `record`: on its steps whose
    numbers are in `steps`, every step when None. Its ``backend`` names the
    law's provenance, or the backend's `describe()` line.

    The ranks' records (`dp0.jsonl` up) are joined as `step_linear_fit` joins
    them: a step is its ranks' lines of one step index, measured as the
    smallest rank label and priced, banded and validated as the rank
    `step_linear_fit.priced_rank` picks; a step the law refuses is banded as
    its rank of most tokens. Its draft is measured and priced the same way. A
    rank that ran the step as a dummy forward is priced 0 s: the step is
    measured, priced and banded by the ranks that ran a forward. A step some
    rank lacks otherwise or cut, or whose
    ranks fall to different vectors, is counted under `cut.ranks`. Ranks whose
    records differ in length by more than the one step a stopped run leaves on
    some ranks only are refused.
    So is a record whose run.json, beside `record`, names `aiter_jit_builds`.
    """
    run = Path(record).resolve().parent / "run.json"
    if run.is_file() and (built := aiter_jit.refusal(read_json(run))):
        raise ValueError(f"{record} is not costed: {built}")
    ranks = sorted(Path(record).glob("dp*[0-9].jsonl"))
    records = [read_jsonl(r) for r in ranks]
    lines = {r.name: len(x) for r, x in zip(ranks, records)}
    if not ranks or max(lines.values()) - min(lines.values()) > 1:
        raise ValueError(
            f"{record} holds rank records of {lines} lines; one run's ranks "
            "differ by at most the step a stopped run leaves on some"
        )
    run = [step_labels.label(r, split_rule=_split)[0] for r in ranks]
    joined, cut, _ = step_linear_fit._groups(run, "base", steps=steps)
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
        key, _, seconds, step, _ = next(iter(group.values()))
        try:
            costs = {i: backend.price(views[i][step]) for i in group}
        except CostRefused as refused:
            band = _band(key, max((r[1] for r in group.values()), key=lambda c: c[1]))
            _count(declined, band, seconds, refused.declined)
            continue
        top = step_linear_fit.priced_rank(
            {i: (c.target_seconds, group[i][1]) for i, c in costs.items()}
        )
        terms = [t for c in costs.values() for t in c.terms]
        extrapolated += any(t.provenance.extrapolated for t in terms)
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
        priced.append((key, band, step, seconds, price, target, draft, costs[top]))
    leading = set() if grid else _leading_slow(priced, law)
    parts: dict[str, list] = {}
    for key, band, step, seconds, price, target, draft, cost in priced:
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
        parts.setdefault(kind, []).append((seconds, cost))
        if draft is not None:
            for group in GRADED[draft[2]]:
                groups[group].setdefault("draft", []).append((band, step, *draft[:2]))
    name = backend.describe() if law is None else backend.law.fitted_on
    report = {"record": str(record), "backend": name, "cut": cut}
    for group, kinds in groups.items():
        report[group] = {
            kind: _summary(rows) for kind, rows in sorted(kinds.items(), reverse=True)
        }
    report["warm"] = warm
    report["fallback"], report["refused"] = fell, declined
    report["extrapolated"] = extrapolated
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


def declined_text(r: dict) -> str:
    """A declined band's steps, real seconds and most frequent reasons."""
    reasons = r["reasons"].most_common()
    named = json.dumps(dict(reasons[:REASONS]))
    more = f" and {len(reasons) - REASONS} more" if len(reasons) > REASONS else ""
    return f"{r['steps']} steps, real {r['real_s']:.2f} s; reasons {named}{more}"


def _text(report: dict) -> str:
    out = [f"record {report['record']}", f"law {report['backend']}"]
    out.append(f"not labelled: {json.dumps(report['cut'])}")
    for group, title in (
        ("graded", "graded: every step with no declared term"),
        ("validated", "of which validated: inside the held-out hull"),
        ("unvalidated", "of which unvalidated"),
        ("declared", "declared, apart"),
    ):
        out.append(f"\n{title}")
        for kind, s in report[group].items():
            held = "-" if s["held_out"] is None else f"{s['held_out']:+.2%}"
            out += [
                (
                    f"{kind}  {s['steps']} steps  real {s['real_s']:.2f} s"
                    f"  priced {s['priced_s']:.2f} s"
                ),
                f"  on totals              {s['on_totals']:+.2%}",
                f"  holding out step {s['held_out_step']:<6}{held}",
                f"  median per step        {s['median']:+.2%}",
                "  band | steps | p10 | p50 | p90 | within 10%",
            ]
            for band, b in s["bands"].items():
                out.append(
                    f"  {band} | {b['steps']} | {b['p10']:+.1%} | {b['p50']:+.1%}"
                    f" | {b['p90']:+.1%} | {b['within_10pct']:.1%}"
                )
    for w in report["warm"]:
        out.append(
            f"\nwarm step {w['step']} ({w['band']}): real {w['real_s']:.2f} s,"
            f" priced {w['priced_s']:.2f} s"
        )
    for band, r in report["refused"].items():
        out.append(f"\nrefused {band}: {declined_text(r)}")
    return "\n".join(out)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.validate.cost",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("record", help="a real run's step record directory")
    parser.add_argument("law", help="a law file, as step_linear_fit writes it")
    args = parser.parse_args(argv)
    print(_text(cost_accuracy(args.record, json.loads(Path(args.law).read_text()))))


if __name__ == "__main__":
    main()

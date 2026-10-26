# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The step grid's bundle, from its grid's measured points and its residual's targets.

`bundle(definition, planned, points, targets, seed)` takes the grid definition,
`step_grid_plan.plan`'s output for it, each planned line's measurement in the
plan's line order (a mapping holding `seconds`, `statistic` and `repeats`, the
kept repeats), and the residual's targets, `step_labels.Sample`s whose
provenance names their `source`. Each measurement becomes the grid point on its
line's node, with its statistic, the count of its kept repeats and the KV
blocks its rows shared where its line states them (`shared`): prefill's
`[B, T, c]` in the grid of chunks that produce output or of those that do not,
decode's `[rung, B, c]`. The grid keeps the plan's nodes, graph mode and timing
scope, holds seconds, and states `VALUE`, what its points hold.

A target measured from a ragged view (`source` `ragged`, `measure steps
--ragged`) fits alpha (`step_grid.py`), never the residual. Per decode rung,
each `ALPHA_FORMS` form is fitted on the rung's ragged views, by least squares
on lookup over measured time less 1, `HELD_OUT` of them held out by `seed`;
a view whose rows read one context, or whose mean or longest row the grid
does not read, takes no part. The form with the least median absolute error
on its held-out views is the rung's. A rung keeps it only if two calibration
cells (target runs) have `CELL_STEPS` decode steps at the rung that the grid
reads with and without it, a cell with fewer not counted, and at none of them
does the residual, fitted without that cell, price the cell's steps at the rung
further from their times at p50 with alpha than without; rungs that fail drop
out and the rest are judged again until none fails. The grid holds the alpha
of each rung kept.

Every other target with a device label that the grid answers, past the KV
pool included, becomes a triple of its view, the lookup and its label, and
`step_grid_residual.fit` fits the residual on the triples. A target the grid
refuses is named in the report with the grid's reason; one with no device
label is counted by why it was cut. A fit with no target the grid answers
refuses.

The bundle is the member `run/step_grid.py` loads: the grid, the residual, the
plan's bounds, the definition, and the sources its triples came from. The
report is the residual's by kind, each kind's selected member also broken down
by source: the triples its MAD pass kept and dropped and those held out, the
median and scaled MAD of its relative residuals over those kept and dropped,
as the member's are, and the median absolute one over those held out, a band
it holds no factor for counting as a factor of 0, as the fit counts it. It
counts the triples whose lookup is extrapolated past the KV pool, and gives per rung each alpha form's terms and errors, the views the
grid does not read, how many it reads at their longest row past the KV pool,
the cells' p50 without and with alpha, and whether the rung keeps it.
"""

import copy
import statistics
from collections import Counter
from fractions import Fraction

import numpy as np

from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.step_grid import ALPHA, StepGrid, length_statistics
from atom.compass.backends.step_grid_residual import (
    HELD_OUT,
    RECIPE,
    Residual,
    fit,
    kind,
)
from atom.compass.backends.step_linear_fit import ROWMOD_CELL_STEPS as CELL_STEPS

#: What a bundle's point holds: `measure steps`' bench window, the device time
#: from the forward's entry to its exit with work queued ahead.
VALUE = "bench window"
#: The forms alpha may take, each by how many of `step_grid.ALPHA`'s terms it fits.
ALPHA_FORMS = {"constant": 1, "ratio": 2, "ratio and spread": 3}


def bundle(definition, planned, points, targets, seed: int) -> tuple[dict, dict]:
    """The bundle and its report; or a ValueError."""
    keys = ("graph_mode", "scope", "prefill", "decode")
    grid = copy.deepcopy({k: planned[k] for k in keys}) | {"unit": "s", "value": VALUE}
    for line, measurement in zip(planned["lines"], points, strict=True):
        (n_q, context, decode), b = line["rows"][0], len(line["rows"])
        point = {"statistic": measurement["statistic"]}
        point["repeats"] = len(measurement["repeats"])
        point["time"] = measurement["seconds"]
        if "shared" in line:
            point["shared"] = line["shared"]
        if decode:
            at, into = [line["rung"], b, context - 1], grid["decode"]["points"]
        else:
            at = [b, b * n_q, context - n_q]
            into = grid["prefill"]["output" if line["produces_output"] else "no_output"]
        into.append({"at": at, **point})
    views = [s for s in targets if s.provenance["source"] == "ragged"]
    targets = [s for s in targets if s.provenance["source"] != "ragged"]
    terms, alphas = _alpha(StepGrid(grid), views, seed)
    labelled = [s for s in targets if s.device_s is not None]
    kept = _kept(grid, terms, labelled, seed, alphas)
    grid = _alpha_grid(grid, {r: terms[r] for r in kept})
    lookup = StepGrid(grid)
    triples, sources, unpriced, past = [], [], [], 0
    cut = Counter(s.cut for s in targets if s.device_s is None)
    for s in labelled:
        try:
            _, _, value, _, _, beyond = lookup.value(s.view)
        except CostRefused as refused:
            reason = "; ".join(r.reason for r in refused.declined)
            run = s.provenance["run"]
            unpriced.append(
                {"step": f"{run} dp{s.rank} step {s.step}", "reason": reason}
            )
            continue
        past += bool(beyond)
        triples.append((s.view, float(value * lookup.unit), s.device_s))
        sources.append(s.provenance["source"])
    if not triples:
        raise ValueError(
            f"none of the {len(targets)} targets is a step the grid answers, "
            "with a device label and inside a complete cell"
        )
    residual, kinds = fit(triples, seed)
    fitted = Residual(residual).kinds
    for k, report in kinds.items():
        held, dropped, by = set(report["held_out"]), set(report["dropped"]), {}
        for i, (view, looked_up, measured) in enumerate(triples):
            if kind(view) != k:
                continue
            x = tuple(float(v) for v in RECIPE(k, view.requests))
            try:
                factor = sum(w * m(view, x) for _, w, m in fitted[k].members)
            except LookupError:
                factor = 0.0
            part = 2 if i in held else 1 if i in dropped else 0
            errors = by.setdefault(sources[i], ([], [], []))
            errors[part].append(factor * looked_up / measured - 1.0)
        report["sources"] = {source: _spread(*e) for source, e in sorted(by.items())}
    out = {"grid": grid, "residual": residual, "bounds": planned["bounds"]}
    out |= {"definition": definition, "sources": sorted(set(sources))}
    report = {"kinds": kinds, "unpriced": unpriced, "cut": dict(cut)}
    return out, report | {"past_pool": past, "alpha": alphas}


def _alpha(grid: StepGrid, views, seed: int) -> tuple[dict, dict]:
    """Each decode rung's alpha terms, the form with the least held-out error
    fitted on its ragged `views`; and the report by rung."""
    rows, report = {}, {}
    for s in views:
        rung = s.view.capture_rung
        got = report.setdefault(str(rung), {"views": 0, "unread": [], "past_pool": 0})
        got["views"] += 1
        longest = max(r.context_tokens - r.query_tokens for r in s.view.requests)
        try:
            cells, at = grid.locate(s.view)
            mean = grid.read(cells, s.view, at)[0]
            top, _, _, beyond = grid.read(cells, s.view, (at[0], Fraction(longest)))
        except CostRefused as refused:
            got["unread"].append("; ".join(r.reason for r in refused.declined))
            continue
        got["past_pool"] += bool(beyond)
        ratio, spread = length_statistics(s.view.requests)
        if ratio:
            g, t = float(mean * grid.unit), float(top * grid.unit)
            rows.setdefault(rung, []).append((g, t, s.device_s, ratio, spread))
    terms = {}
    for rung, got in rows.items():
        g, top, t, ratio, spread = map(np.array, zip(*got))
        x = np.column_stack([np.ones(len(t)), ratio, spread])
        w, e = (top - g) / t, g / t - 1
        held = np.random.default_rng(seed).permutation(len(t))[: int(len(t) * HELD_OUT)]
        dev = np.setdiff1d(np.arange(len(t)), held)
        forms = {}
        for form, n in ALPHA_FORMS.items():
            if len(dev) <= n or not len(held):
                continue
            z = np.zeros(3)
            z[:n] = np.linalg.lstsq(x[dev, :n] * w[dev, None], -e[dev], rcond=None)[0]
            error = np.abs(e + w * np.clip(x @ z, 0.0, 1.0))
            forms[form] = {
                "terms": z.tolist(),
                "fitted_error": float(np.median(error[dev])),
                "held_out_error": float(np.median(error[held])),
            }
        if forms:
            best = min(forms, key=lambda f: forms[f]["held_out_error"])
            terms[rung] = tuple(forms[best]["terms"])
            report[str(rung)] |= {"forms": forms, "form": best}
    return terms, report


def _alpha_grid(grid: dict, terms: dict) -> dict:
    """`grid` holding the alpha `terms` gives per rung."""
    alpha = {str(r): dict(zip(ALPHA, t)) for r, t in terms.items()}
    return grid | {"decode": grid["decode"] | {"alpha": alpha}}


def _kept(grid: dict, terms: dict, targets, seed: int, report: dict) -> list:
    """The rungs of `terms` that keep their alpha, as the module says, the
    cells' evidence written into `report`."""
    decode = [s for s in targets if not s.view.prefill]
    steps = Counter((s.provenance["run"], s.view.capture_rung) for s in decode)
    cells = sorted({c for (c, r), n in steps.items() if r in terms and n >= CELL_STEPS})
    base, kept = _p50s(grid, {}, decode, cells, terms, seed), sorted(terms)
    while kept:
        tried = _p50s(grid, {r: terms[r] for r in kept}, decode, cells, terms, seed)
        now = []
        for rung in kept:
            pairs = {
                c: (base[c, rung], tried[c, rung])
                for c in cells
                if (c, rung) in base and (c, rung) in tried
            }
            report[str(rung)]["cells"] = pairs
            if len(pairs) >= 2 and all(abs(w) <= abs(o) for o, w in pairs.values()):
                now.append(rung)
        if now == kept:
            break
        kept = now
    for rung in terms:
        report[str(rung)]["kept"] = rung in kept
    return kept


def _p50s(grid: dict, terms: dict, decode, cells, rungs, seed: int) -> dict:
    """Per cell and rung of `rungs` where the cell has `CELL_STEPS` steps the
    grid reads, the p50 of priced over measured less 1 of those `decode`
    targets, the residual fitted on the other cells' with `terms`' alpha."""
    lookup, priced = StepGrid(_alpha_grid(grid, terms)), []
    for s in decode:
        try:
            priced.append((s, float(lookup.value(s.view)[2] * lookup.unit)))
        except CostRefused:
            continue
    out = {}
    for cell in cells:
        others = [
            (s.view, g, s.device_s) for s, g in priced if s.provenance["run"] != cell
        ]
        try:
            fitted = Residual(fit(others, seed)[0]).kinds["decode"]
        except ValueError:
            continue
        errors = {}
        for s, g in priced:
            if s.provenance["run"] != cell or s.view.capture_rung not in rungs:
                continue
            x = tuple(float(v) for v in RECIPE("decode", s.view.requests))
            try:
                factor = sum(w * m(s.view, x) for _, w, m in fitted.members)
            except LookupError:
                continue
            errors.setdefault(s.view.capture_rung, []).append(factor * g / s.device_s)
        for rung, e in errors.items():
            if len(e) >= CELL_STEPS:
                out[cell, rung] = float(np.median(e)) - 1.0
    return out


def _spread(kept: list, dropped: list, held: list) -> dict:
    fitted = kept + dropped
    centre = statistics.median(fitted) if fitted else None
    return {
        "kept": len(kept),
        "dropped": len(dropped),
        "held_out": len(held),
        "centre": centre,
        "mad": (
            1.4826 * statistics.median(abs(e - centre) for e in fitted)
            if fitted
            else None
        ),
        "held_out_error": statistics.median(map(abs, held)) if held else None,
    }

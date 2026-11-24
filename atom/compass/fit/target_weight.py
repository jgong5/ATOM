# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""A target-weighted held-out error for choosing a step law's form (#3720).

Leave-one-segment-out on a calibration record grades a form on the record's
own mix of steps. The graded workloads (the targets) run another mix: cell A's
design records hold prefills in stretches at the corners of the reachable
range, its agentic targets mostly isolated 8-32k chunks beside decoding peers.
Two forms that tie on the design's own mix can then differ by points on the
targets (LLM-Booster#752, `research/doe_regression.md` sections 5.1-5.2).
Under such a covariate shift the importance-weighted held-out error is
unbiased where the plain one is not (Sugiyama, Krauledat and Mueller 2007):
each held-out design step's error is weighted by `w(x) = q_target(x) /
q_design(x)`, how much commoner a step like it is in the targets than in the
design.

`target_weighted_held_out` is the criterion: it estimates `w` over a
`DESCRIPTOR` row of each step by a logistic classifier between the design's
steps and the targets' (`_density_ratio`), each target step weighed by its
mass (its seconds), and returns the weighted mean magnitude of the held-out
errors, their weighted error on totals and the Kish effective sample size of
the weights. A small effective size says the design holds few steps like the
targets', which no weighting repairs. The targets' steps need no device: a
simulated run of each target writes the step records `record_descriptors`
reads, and so does a real one.

The criterion is opt-in: nothing in the fit calls it; a form comparison does,
after the identifiability gate (VIF) has filtered the forms.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from atom.compass.backends.shape import (
    BatchView,
    RequestShape,
    sum_context,
    sum_tokens,
)
from atom.compass.backends.step_labels import (
    group_dummies,
    group_peaks,
    group_prefills,
    prefill_duties,
)
from atom.compass.backends.step_linear import token_bucket

#: A prefill group step's descriptor, read from its busiest rank's view
#: (`descriptor`): its tokens, its group's MoE gather rows and the peers'
#: share of them, its cached history, its sequences, the group's dummy
#: forwards, whether it runs eagerly past the largest token-graph bucket, and
#: one less its prefill duty (a prefill right after decode steps is cold).
DESCRIPTOR = (
    "log_tokens",
    "log_rows",
    "peer_share",
    "log_history",
    "log_seqs",
    "dummies",
    "eager",
    "cold",
)
#: The L2 penalty of the classifier, on its standardized coefficients, per
#: unit of total sample weight.
RIDGE = 1e-3
#: The largest weight kept, as a multiple of the mean: past it a weight is
#: clipped, bounding the variance a few design steps alone can carry.
CLIP = 50.0


def descriptor(views, buckets: tuple[int, ...] = ()) -> tuple[float, ...]:
    """`DESCRIPTOR` of a prefill group step from `views`, its ranks' views
    that prefill (a dummy forward's none): the busiest one's, the one with
    the most tokens. `buckets` are the prefill token-graph buckets."""
    view = max(views, key=lambda v: sum_tokens(v.prefill))
    own = sum_tokens(view.prefill)
    rows = own if view.moe_rows is None else view.moe_rows
    history = sum_context(view.prefill) - own
    eager = 1.0
    if buckets and (bucket := token_bucket(view, tuple(buckets))) is not None:
        eager = float(bucket == 0)
    duty = view.prefill_duty
    return (
        float(np.log1p(own)),
        float(np.log1p(rows)),
        (rows - own) / rows if rows else 0.0,
        float(np.log1p(max(history, 0))),
        float(np.log1p(len(view.prefill))),
        float(view.group_dummies),
        eager,
        0.0 if duty is None else 1.0 - min(max(duty, 0.0), 1.0),
    )


def record_descriptors(
    steps_dir, buckets: tuple[int, ...] = ()
) -> tuple[np.ndarray, np.ndarray]:
    """Each prefill group step of the step records in `steps_dir` (one
    run's `dp<r>.jsonl`, real or simulated): its `descriptor` row and its
    mass, the seconds it ran: a simulated run's largest `device_s` over its
    ranks (`dp<r>.cost.jsonl`), a real one's largest `d_busy_ns`. A
    simulated record's duty is read from its cost lines' clock, each forward
    from `t_s` to `t_s` plus `group_s`, as the runner reads it. A step some
    rank mixes prefill and decode at, or whose mass no rank states, enters
    none."""
    paths = sorted(
        Path(steps_dir).glob("dp*[0-9].jsonl"), key=lambda p: int(p.stem[2:])
    )
    ranks = [[json.loads(x) for x in p.read_text().splitlines()] for p in paths]
    seconds = [{} for _ in ranks]
    for path, lines, mine in zip(paths, ranks, seconds):
        cost = path.with_suffix(".cost.jsonl")
        if "t_enter_ns" in (lines[0] if lines else {}) or not cost.exists():
            for s in lines:
                if "d_busy_ns" in s:
                    mine[s["step"]] = s["d_busy_ns"] * 1e-9
            continue
        stamps = {}
        for x in cost.read_text().splitlines():
            c = json.loads(x)
            if "step" in c and c.get("t_s") is not None:
                stamps[c["step"]] = (c["t_s"], c["t_s"] + c["group_s"])
                mine[c["step"]] = c["device_s"]
        for s in lines:
            if s["step"] in stamps:
                enter, exit_ = stamps[s["step"]]
                s["t_enter_ns"], s["t_exit_ns"] = enter * 1e9, exit_ * 1e9
    dummies, peaks = group_dummies(ranks), group_peaks(ranks)
    prefills = group_prefills(ranks)
    duties = [
        dict(zip((s["step"] for s in x), prefill_duties(x, prefills))) for x in ranks
    ]
    steps: dict[int, list] = {}
    for r, lines in enumerate(ranks):
        for s in lines:
            if s["is_dummy_run"] or not s["rows"]:
                continue
            dp = s.get("moe_rows") is not None
            view = BatchView(
                tuple(RequestShape(q, c, bool(d)) for q, c, d in s["rows"]),
                capture_rung=s.get("rung"),
                moe_rows=s.get("moe_rows"),
                moe_tokens=s.get("moe_tokens"),
                produces_output=s.get("produces_output"),
                dp_rank=s.get("dp_rank") if dp else None,
                group_dummies=dummies.get(s["step"], 0) if dp else 0,
                group_max_tokens=peaks.get(s["step"]) if dp else None,
                group_needs_eager=s.get("group_needs_eager", False),
                prefill_duty=duties[r].get(s["step"]),
            )
            steps.setdefault(s["step"], []).append((r, view))
    rows, mass = [], []
    for step, held in sorted(steps.items()):
        views = [v for _, v in held]
        if not any(v.prefill for v in views) or any(
            v.prefill and v.decode for v in views
        ):
            continue
        got = [seconds[r].get(step) for r, _ in held]
        if any(g is None for g in got):
            continue
        rows.append(descriptor([v for v in views if v.prefill], buckets))
        mass.append(max(got))
    return np.array(rows, dtype=np.float64).reshape(-1, len(DESCRIPTOR)), np.array(mass)


def _expand(x: np.ndarray, centre: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """`x` standardized, with each column's square and an intercept: the
    classifier's log-odds is quadratic in the descriptor, a density ratio of
    two Gaussians' shape."""
    z = (x - centre) / scale
    return np.column_stack([np.ones(len(z)), z, z * z])


def _density_ratio(
    design: np.ndarray, target: np.ndarray, mass: np.ndarray | None = None
) -> np.ndarray:
    """`q_target / q_design` at each `design` row, from a logistic classifier
    between the design's rows and the `target` rows, each target row weighed
    by its `mass` (1 each when None), the two sets at equal total weight, so
    that the ratio is the classifier's odds. An L2 penalty of `RIDGE` holds
    the solve where the sets separate. The weights are scaled to mean 1 and
    clipped at `CLIP`."""
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mass = np.ones(len(target)) if mass is None else np.asarray(mass, dtype=np.float64)
    both = np.vstack([design, target])
    centre = both.mean(axis=0)
    scale = both.std(axis=0)
    scale[scale == 0] = 1.0
    x = _expand(both, centre, scale)
    z = np.r_[np.zeros(len(design)), np.ones(len(target))]
    w = np.r_[np.full(len(design), 0.5 / len(design)), 0.5 * mass / mass.sum()]
    # Constant columns, both sets at one value, carry no information.
    live = np.r_[True, np.ptp(x[:, 1:], axis=0) > 0]
    x = x[:, live]
    beta = np.zeros(x.shape[1])
    penalty = np.full(x.shape[1], RIDGE)
    penalty[0] = 0.0
    for _ in range(100):
        eta = np.clip(x @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = x.T @ (w * (p - z)) + penalty * beta
        hess = (x * (w * p * (1 - p))[:, None]).T @ x + np.diag(penalty)
        step = np.linalg.solve(hess + 1e-12 * np.eye(len(beta)), grad)
        beta -= step
        if np.abs(step).max() < 1e-9:
            break
    ratio = np.exp(np.clip(x[: len(design)] @ beta, -30, 30))
    ratio /= ratio.mean()
    return np.minimum(ratio, CLIP)


def target_weighted_held_out(
    design,
    errors,
    target,
    mass=None,
    labels=None,
) -> dict:
    """The target-weighted held-out error of a form: `errors`, each design
    step's held-out relative error (its price under the law fitted without
    its segment, over its label, less 1), weighted by `q_target / q_design`
    at its `design` row (`DESCRIPTOR`), the density ratio between the
    `target` rows, each of `mass` (its seconds; 1 each when None), and the
    design's (`_density_ratio`).

    Returns `error`, the weighted mean of the errors' magnitudes, the
    criterion (lower is better); `bias`, the weighted error on totals,
    `sum w label error / sum w label` with `labels` the design steps' seconds
    (each 1 when None); `unweighted`, the plain mean magnitude, as plain
    leave-one-segment-out grades a form; `ess`, the Kish effective sample
    size of the weights, `(sum w)^2 / sum w^2`, against `steps`; and
    `weights`. A step with no held-out error (NaN) enters no figure.
    """
    design = np.asarray(design, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if len(design) != len(errors):
        raise ValueError(f"{len(design)} design rows for {len(errors)} errors")
    target = np.asarray(target, dtype=np.float64)
    if not len(target):
        raise ValueError("no target step to weight the design's steps by")
    if design.shape[1] != target.shape[1]:
        raise ValueError(
            f"design rows of {design.shape[1]} features, target rows of "
            f"{target.shape[1]}"
        )
    held = ~np.isnan(errors)
    w = _density_ratio(design, target, mass)
    labels = np.ones(len(errors)) if labels is None else np.asarray(labels, float)
    wh, eh, lh = w[held], errors[held], labels[held]
    return {
        "error": float((wh * np.abs(eh)).sum() / wh.sum()),
        "bias": float((wh * lh * eh).sum() / (wh * lh).sum()),
        "unweighted": float(np.abs(eh).mean()),
        "ess": float(wh.sum() ** 2 / (wh * wh).sum()),
        "steps": int(held.sum()),
        "weights": w,
    }

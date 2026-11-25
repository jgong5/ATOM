# SPDX-License-Identifier: MIT
"""The target-weighted held-out error a law's form is chosen by (#3720,
LLM-Booster#752): each held-out design step's error weighted by how much
commoner a step like it is in the targets than in the design, estimated by a
classifier between the two sets' descriptor rows. No device."""

import json

import numpy as np
import pytest

from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.fit.target_weight import (
    DESCRIPTOR,
    _density_ratio,
    descriptor,
    record_descriptors,
    target_weighted_held_out,
)

BUCKETS = (256, 512, 1024)


def test_the_weights_follow_the_targets_density_over_the_design_s():
    rng = np.random.default_rng(0)
    design = rng.uniform(0, 1, (4000, 1))
    target = rng.normal(0.8, 0.05, (2000, 1))
    w = _density_ratio(design, target)
    near = np.abs(design[:, 0] - 0.8) < 0.03
    far = np.abs(design[:, 0] - 0.2) < 0.03
    assert w[near].mean() > 20 * w[far].mean()
    assert w.mean() == pytest.approx(1.0, rel=0.1)  # clipped after the scaling
    # Both sets alike: every weight near 1.
    same = _density_ratio(design, rng.uniform(0, 1, (4000, 1)))
    assert np.percentile(same, 95) < 1.3 and np.percentile(same, 5) > 0.7


def test_a_target_step_s_mass_moves_the_weights_toward_it():
    rng = np.random.default_rng(1)
    design = rng.uniform(0, 1, (3000, 1))
    target = np.r_[rng.normal(0.2, 0.05, (500, 1)), rng.normal(0.8, 0.05, (500, 1))]
    heavy_high = np.r_[np.ones(500), np.full(500, 9.0)]
    w = _density_ratio(design, target, heavy_high)
    low = np.abs(design[:, 0] - 0.2) < 0.03
    high = np.abs(design[:, 0] - 0.8) < 0.03
    assert w[high].mean() > 4 * w[low].mean()


def test_the_criterion_picks_the_form_right_where_the_targets_are():
    """Form A is exact near the targets and off elsewhere; form B is a little
    off everywhere. Plain held-out error picks B, the weighted one A."""
    rng = np.random.default_rng(2)
    design = rng.uniform(0, 1, (5000, 1))
    target = rng.normal(0.85, 0.04, (3000, 1))
    near = np.abs(design[:, 0] - 0.85) < 0.12
    noise = rng.normal(0, 0.002, len(design))
    a = np.where(near, 0.0, 0.06) + noise
    b = np.full(len(design), 0.02) + noise
    got = {
        f: target_weighted_held_out(design, e, target) for f, e in (("a", a), ("b", b))
    }
    assert got["b"]["unweighted"] < got["a"]["unweighted"]
    assert got["a"]["error"] < got["b"]["error"]
    assert got["a"]["ess"] < len(design) / 2  # the design's few target-like steps
    assert got["a"]["steps"] == len(design)


def test_the_bias_is_the_weighted_error_on_totals_and_nan_steps_enter_none():
    design = np.array([[0.0], [1.0], [2.0], [3.0]])
    errors = np.array([0.1, -0.1, np.nan, 0.1])
    labels = np.array([1.0, 3.0, 5.0, 1.0])
    got = target_weighted_held_out(design, errors, design, labels=labels)
    w = got["weights"][[0, 1, 3]]
    e, lab = errors[[0, 1, 3]], labels[[0, 1, 3]]
    assert got["bias"] == pytest.approx((w * lab * e).sum() / (w * lab).sum())
    assert got["steps"] == 3
    with pytest.raises(ValueError, match="design rows"):
        target_weighted_held_out(design, errors[:2], design)
    with pytest.raises(ValueError, match="no target step"):
        target_weighted_held_out(design, errors, np.zeros((0, 1)))


def _prefill(tokens, history=0, peak=None, moe=None, dummies=0, duty=None):
    rows = (RequestShape(tokens, tokens + history, False),)
    return BatchView(
        rows,
        moe_rows=moe,
        moe_tokens=moe,
        group_max_tokens=peak,
        group_dummies=dummies,
        prefill_duty=duty,
    )


def test_a_group_step_is_described_by_its_busiest_rank():
    small = _prefill(300, peak=5000, moe=5303, dummies=0, duty=0.2)
    busy = _prefill(5000, history=1000, peak=5000, moe=5303, duty=0.25)
    got = dict(zip(DESCRIPTOR, descriptor([small, busy], BUCKETS)))
    assert got["log_tokens"] == pytest.approx(np.log1p(5000))
    assert got["log_rows"] == pytest.approx(np.log1p(5303))
    assert got["peer_share"] == pytest.approx(303 / 5303)
    assert got["log_history"] == pytest.approx(np.log1p(1000))
    assert got["eager"] == 1.0 and got["cold"] == pytest.approx(0.75)
    graphed = dict(
        zip(DESCRIPTOR, descriptor([_prefill(200, peak=300, moe=500)], BUCKETS))
    )
    assert graphed["eager"] == 0.0 and graphed["cold"] == 0.0


def _line(step, rank, rows, moe, dummy=False, **extra):
    return {
        "step": step,
        "dp_rank": rank,
        "is_dummy_run": dummy,
        "produces_output": True,
        "rows": rows,
        "rung": None,
        "moe_rows": moe,
        "moe_tokens": moe,
        **extra,
    }


def test_a_simulated_record_s_prefill_steps_are_read_with_their_cost_seconds(tmp_path):
    """Two ranks: a decode step, then a prefill beside a dummy, then a
    prefill on both. The duty is read from the cost lines' clock."""
    steps = [
        ([[1, 900, True]], [[1, 400, True]], 2, (0.0, 0.01), (0.010, 0.011)),
        ([[3000, 3000, False]], [[1, 1, False]], 3001, (0.01, 0.2), (0.200, 0.001)),
        ([[600, 600, False]], [[700, 700, False]], 1300, (0.21, 0.05), (0.04, 0.05)),
    ]
    for r in range(2):
        lines, cost = [], [{"version": 1}]
        for i, (a, b, moe, (t, g), seconds) in enumerate(steps):
            rows = (a, b)[r]
            dummy = i == 1 and r == 1
            lines.append(_line(i, r, rows, moe, dummy))
            cost.append({"step": i, "t_s": t, "group_s": g, "device_s": seconds[r]})
        (tmp_path / f"dp{r}.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in lines)
        )
        (tmp_path / f"dp{r}.cost.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in cost)
        )
    rows, mass = record_descriptors(tmp_path, BUCKETS)
    assert rows.shape == (2, len(DESCRIPTOR))
    assert mass.tolist() == [0.2, 0.05]
    first, second = (dict(zip(DESCRIPTOR, r)) for r in rows)
    assert first["log_tokens"] == pytest.approx(np.log1p(3000))
    assert first["dummies"] == 1.0 and first["eager"] == 1.0
    assert first["cold"] == 1.0  # after a decode step alone
    assert second["log_tokens"] == pytest.approx(np.log1p(700))
    assert second["eager"] == 0.0 and 0.0 < second["cold"] < 1.0

# SPDX-License-Identifier: MIT
"""A prefill step's `cold_peer_rows`: its peers' MoE rows, by how far it lies
outside a stretch of prefill steps (#3747).

The duty is `shape.PrefillDuty` over a rank's forwards: from a real record's
`t_enter_ns` / `t_exit_ns` and its group's prefill steps (`step_labels`), and
on a simulated run from the engine's clock (`_group_step_seconds`). The law
prices the column (`StepLinearBackend`) and the fit, asked, fits it.
"""

import json
import math
import re

import numpy as np
import pytest
from test_backend_step_linear_fit import _engine_file, traced
from test_dp_step_max import Recording, prefill, runner

from atom.compass.backends import (
    BatchView,
    CostRefused,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.shape import DUTY_TAU, PrefillDuty
from atom.compass.backends.step_labels import (
    group_prefills,
    label_run,
    prefill_duties,
)
from atom.compass.backends.step_linear import cold_peer_rows, dummy_view
from atom.compass.backends.step_linear_fit import fit, label_records

TAU = DUTY_TAU


def test_the_duty_is_the_exponentially_weighted_share_of_time_in_prefill():
    duty = PrefillDuty()
    assert duty.at(5.0) == 0.0
    duty.add(0.0, 10 * TAU, True)
    assert duty.at(10 * TAU) == pytest.approx(1 - math.exp(-10))
    # Idle time and a forward of no prefill step decay it alike.
    idle = PrefillDuty()
    idle.add(0.0, TAU, True)
    busy = PrefillDuty()
    busy.add(0.0, TAU, True)
    busy.add(TAU, 2 * TAU, False)
    assert idle.at(2 * TAU) == pytest.approx(busy.at(2 * TAU))
    assert idle.at(2 * TAU) == pytest.approx((1 - math.exp(-1)) * math.exp(-1))
    # A forward entered before the last exit starts there; none reads back.
    busy.add(1.5 * TAU, 2.5 * TAU, True)
    assert busy.time == 2.5 * TAU
    assert busy.at(TAU) == busy.at(2.5 * TAU)
    with pytest.raises(ValueError, match="before its entry"):
        busy.add(4 * TAU, 3 * TAU, True)


def _line(step, enter, exit_, rows, dummy=False):
    return {
        "step": step,
        "is_dummy_run": dummy,
        "rows": rows,
        "t_enter_ns": int(enter * 1e9),
        "t_exit_ns": int(exit_ * 1e9),
    }


def test_a_rank_s_duty_counts_its_forwards_of_its_group_s_prefill_steps():
    # Rank 1 runs a dummy at step 0, where rank 0 prefills; both decode at 1.
    rank0 = [
        _line(0, 0.0, 2.0, [[100, 100, False]]),
        _line(1, 2.0, 2.5, [[1, 101, True]]),
        _line(2, 6.0, 8.0, [[50, 150, False]]),
    ]
    rank1 = [
        _line(0, 0.0, 2.0, [[1, 1, False]], dummy=True),
        _line(1, 2.0, 2.5, [[1, 9, True]]),
        _line(2, 6.0, 8.0, [[1, 1, False]], dummy=True),
    ]
    prefills = group_prefills([rank0, rank1])
    assert prefills == {0, 2}
    expected = PrefillDuty()
    expected.add(0.0, 2.0, True)
    expected.add(2.0, 2.5, False)
    for lines in (rank0, rank1):
        got = prefill_duties(lines, prefills)
        assert got == pytest.approx([0.0, 1 - math.exp(-2.0 / TAU), expected.at(6.0)])
    # A line without its stamps states no duty for any line of the rank.
    del rank0[1]["t_exit_ns"]
    assert prefill_duties(rank0, prefills) == [None] * 3


#: The known per-rank prefill law, and its cold_peer_rows coefficient.
LAW = {
    "step": 0.02,
    "tokens": 6e-5,
    "query_square": 2e-10,
    "query_cached": 3e-10,
    "moe_rows": 2e-5,
    "cold_peer_rows": 1.5e-5,
}
DP = 4


def _views(rng, ranks=DP):
    """A DP prefill step's views, rank 0 the busiest, each without its duty."""
    per_rank = []
    for r in range(ranks):
        lo, hi = (6000, 8192) if r == 0 else (16, 4096)
        rows = []
        for _ in range(int(rng.integers(1, 4))):
            q = int(rng.integers(lo, hi))
            rows.append(RequestShape(q, q + int(rng.integers(0, 30000)), False))
        per_rank.append(tuple(rows))
    moe = sum(r.query_tokens for rows in per_rank for r in rows)
    return [BatchView(rows, moe_rows=moe, produces_output=True) for rows in per_rank]


def _price(view):
    q = np.array([r.query_tokens for r in view.prefill])
    n = np.array([r.context_tokens for r in view.prefill])
    x = (
        1,
        q.sum(),
        (q * q).sum(),
        (q * (n - q)).sum(),
        view.moe_rows,
        cold_peer_rows(view),
    )
    return float(np.dot(list(LAW.values()), x))


def _cold_record(tmp_path, seed=0, n=360):
    """A DP4 record of prefill steps in stretches of 1 to 30 split by idle
    time, each labelled with the group's step under `LAW`: the max over its
    ranks' prices, each rank's at its duty. Returns the records and each
    step's views, duty stated."""
    import dataclasses

    rng = np.random.default_rng(seed)
    duty, t, steps = PrefillDuty(), 0.0, []
    left = 0
    for i in range(n):
        if left == 0:
            t += float(rng.uniform(0.0, 3 * TAU))
            left = int(rng.integers(1, 31))
        left -= 1
        at = duty.at(t)
        views = [dataclasses.replace(v, prefill_duty=at) for v in _views(rng)]
        seconds = max(map(_price, views))
        steps.append((views, t, seconds))
        duty.add(t, t + seconds, True)
        t += seconds
    paths = []
    for r in range(DP):
        lines = []
        for i, (views, enter, seconds) in enumerate(steps):
            rows = [
                [x.query_tokens, x.context_tokens, x.decode] for x in views[r].requests
            ]
            line = {
                "step": i,
                "dp_rank": r,
                "is_dummy_run": False,
                "produces_output": True,
                "batch": [
                    [r * 100_000 + i * 10 + j, q, c] for j, (q, c, _) in enumerate(rows)
                ],
                "named": [],
                "rows": rows,
                "rung": None,
                "moe_rows": views[r].moe_rows,
                "t_enter_ns": round(enter * 1e9),
                "t_launch_ns": [round(enter * 1e9)] * 2,
                "t_recv_ns": [round(enter * 1e9)] * 2,
                "d_busy_ns": round(seconds * 1e9),
                "t_exit_ns": round((enter + seconds) * 1e9),
            }
            lines.append(traced(line))
        path = tmp_path / f"dp{r}.jsonl"
        path.write_text("".join(json.dumps(x) + "\n" for x in lines))
        _engine_file(path)
        paths.append(path)
    return paths, steps


def test_a_record_s_views_state_the_duty_its_stamps_give(tmp_path):
    paths, steps = _cold_record(tmp_path, n=60)
    for rank, (samples, _) in enumerate(label_run(paths)):
        got = [s.view.prefill_duty for s in samples]
        assert got == pytest.approx(
            [views[rank].prefill_duty for views, *_ in steps], abs=1e-6
        )
    duties = [v[0].prefill_duty for v, *_ in steps]
    assert min(duties) < 0.05 and max(duties) > 0.9


def test_the_fit_asked_for_cold_peer_rows_recovers_its_coefficient(tmp_path):
    paths, _ = _cold_record(tmp_path)
    samples = label_records(paths, DP)[0]
    law = fit(samples, {"form": "base", "dp": DP, "cold": True})[0]

    vector = law["prefill"]
    assert list(vector["coefficients"]) == list(LAW)
    assert vector["coefficients"] == pytest.approx(LAW, rel=1e-4)
    assert vector["duty_tau"] == DUTY_TAU
    # Unasked the vector has no such column; at one rank it refuses.
    plain = fit(samples, {"form": "base", "dp": DP})[0]["prefill"]
    assert "cold_peer_rows" not in plain["coefficients"]
    with pytest.raises(ValueError, match="these records run one rank"):
        fit(samples, {"form": "base", "dp": 1, "cold": True})


def _law(duty_tau=DUTY_TAU):
    hull = {"facets": [[-1, 0, 0, 0, 0, 0]]}
    prefill = {"coefficients": dict(LAW), "hull": hull, "duty_tau": duty_tau}
    eager = {"coefficients": {"step": 0.0, "context": 0.0, "graph_padding": 0.0}}
    eager["hull"] = {"equal": {"graph_padding": 0}, "facets": [[-1, 0]]}
    return {"provenance": "a known law", "prefill": prefill, "decode": {"eager": eager}}


def test_the_law_prices_a_ranks_peer_rows_by_one_less_its_duty():
    backend = StepLinearBackend(StepLinearLaw(_law()), dp=DP)
    rows = (RequestShape(300, 1300, False),)
    view = BatchView(rows, moe_rows=1000, dp_rank=0, prefill_duty=0.25)

    terms = {t.name: t.seconds for t in backend.price(view).terms}
    assert terms["prefill.cold_peer_rows"] == pytest.approx(700 * 0.75 * 1.5e-5)
    # A peer's dummy forward pays for every row but its own one.
    dummy = cold_peer_rows(dummy_view(view))
    assert dummy == pytest.approx(999 * 0.75)
    # One rank has no peers, and a duty past 1 reads as 1.
    assert cold_peer_rows(BatchView(rows)) == 0.0
    assert cold_peer_rows(BatchView(rows, moe_rows=1000, prefill_duty=1.2)) == 0.0
    with pytest.raises(CostRefused, match="no prefill duty"):
        backend.price(BatchView(rows, moe_rows=1000, dp_rank=0))


def test_a_law_whose_duty_runs_at_another_time_constant_is_refused():
    with pytest.raises(
        ValueError, match=re.escape(f"a view states its duty at {DUTY_TAU} s")
    ):
        StepLinearLaw(_law(duty_tau=2 * DUTY_TAU))


class _Clock:
    """A host whose clock a test sets, charging each forward its price."""

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def charge(self, batch, device_s, draft_s, refuse=True):
        return 2.0


def test_the_runner_states_its_duty_from_the_engine_s_clock():
    backend = Recording()
    r = runner(backend)
    # Without a host the view states no duty.
    r.forward(prefill(40))
    assert backend.views[-1].prefill_duty is None

    r = runner(backend)
    r.compass_host = clock = _Clock()
    r.forward(prefill(40))  # 0 s to 2 s, a prefill
    clock.t = 2.0
    r.forward(prefill(40))  # 2 s to 4 s
    clock.t = 4.0 + TAU  # idle for TAU
    r.dummy_execution()  # a decode, 2 s long
    clock.t = 6.0 + TAU
    r.forward(prefill(40))
    duty = [v.prefill_duty for v in backend.views[-4:]]
    after_two = 1 - math.exp(-4.0 / TAU)
    assert duty == pytest.approx(
        [
            0.0,
            1 - math.exp(-2.0 / TAU),
            after_two * math.exp(-1),
            after_two * math.exp(-1 - 2.0 / TAU),
        ]
    )

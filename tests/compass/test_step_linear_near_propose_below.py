# SPDX-License-Identifier: MIT
"""A draft law whose `propose` pass runs slow close behind its host only on
an output prefill under some token count bounds its `near_propose` there, and
prices a longer one's pass at its far price whatever its lag."""

import json

import pytest
from test_backend_step_linear_fit import _fit, _record, _steps
from test_step_linear_draft import PROPOSE, SHARE
from test_step_linear_near_propose import LATE, WAIT, _output_prefills, _priced

from atom.compass.backends import CostRefused

#: The prefill tokens from which a pass its host is late to runs at its far price.
BELOW = 12288


def _bounded(path, steps: int, below: int = BELOW, every: int = 3):
    """`_output_prefills`, but a pass whose host is back `LATE` runs `WAIT`
    longer only on a prefill of fewer than `below` tokens."""
    plan, i = [], 0
    for rows, rung, s in _steps(3, n=steps):
        late = slow = propose = 0
        if not rows[0][2]:
            late, i = i % every == 0, i + 1
            slow = late and sum(q for q, _, _ in rows) < below
            propose = PROPOSE[0] + PROPOSE[1] * len(rows) + WAIT * slow
        plan.append((rows, rung, s, propose, late))
    _record(path, [(r, g, s * (1 + SHARE) + p) for r, g, s, p, _ in plan])
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for line, (_, _, s, propose, late) in zip(lines, plan):
        line["d_target_ns"] = round(s * 1e9)
        if propose:
            line["d_propose_ns"] = round(propose * 1e9)
            line["t_recv_ns"][1] += round(LATE * 1e9) * late
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return path


@pytest.fixture(scope="module")
def law(tmp_path_factory):
    return _fit([_bounded(tmp_path_factory.mktemp("below") / "dp0.jsonl", 1600)])


def test_a_near_pass_that_runs_slow_only_under_a_token_count_is_bounded_there(law):
    near = law["draft"]["near_propose"]
    assert near["below"] == BELOW
    assert (near["coefficients"]["step"], near["coefficients"]["seqs"]) == (
        pytest.approx((PROPOSE[0] + WAIT, PROPOSE[1]), rel=1e-4)
    )
    split = near["fit"]["split"]
    assert split["error"] < 1e-9 < split["error_unbounded"] < split["error_without"]


def test_a_bounded_near_propose_prices_a_longer_prefill_at_its_far_price(law):
    assert _priced(law, -LATE, BELOW - 1) == pytest.approx(
        sum(PROPOSE) + WAIT, rel=1e-4
    )
    assert _priced(law, -LATE, BELOW) == pytest.approx(sum(PROPOSE), rel=1e-4)
    assert _priced(law, 0.0, BELOW - 1) == pytest.approx(sum(PROPOSE), rel=1e-4)
    # Its lag does not move its price, so a view stating none is priced.
    assert _priced(law, None, BELOW) == pytest.approx(sum(PROPOSE), rel=1e-4)
    with pytest.raises(CostRefused, match="how far behind its host"):
        _priced(law, None, BELOW - 1)


def test_a_near_pass_that_runs_slow_at_any_length_is_not_bounded(tmp_path):
    near = _fit([_output_prefills(tmp_path / "dp0.jsonl", 1600)])["draft"][
        "near_propose"
    ]
    assert "below" not in near and "error_unbounded" not in near["fit"]["split"]

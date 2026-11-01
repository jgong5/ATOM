# SPDX-License-Identifier: MIT
"""A draft law prices the `propose` pass of an output prefill whose device
reaches it close behind its host apart, when the records' propose labels split
at that lag, and a simulated runner reads the lag off its host clock."""

import json

import numpy as np
import pytest
from test_backend_step_linear_fit import _fit, _record, _steps
from test_dp_step_max import prefill, runner
from test_step_linear_draft import PROPOSE, SHARE

from atom.compass.backends import (
    BatchView,
    CostRefused,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.runner.overrides import HostDeviceRecurrence

#: What a pass the device reaches close behind its host adds, in seconds, and
#: how long after its device starts the step that host returns from its wait.
WAIT, LATE = 1e-3, 10.0


def _output_prefills(path, steps: int, every: int = 3):
    """`_mtp_record` of `steps` of `_steps(3)`, whose prefills all produce
    output and each run `propose` after their draft at its `PROPOSE` price;
    unless `every` is 0, every `every`-th runs it `WAIT` longer, its host back
    from the wait `LATE` after its device start."""
    plan, i = [], 0
    for rows, rung, s in _steps(3, n=steps):
        near = propose = 0
        if not rows[0][2]:
            near, i = every and i % every == 0, i + 1
            propose = PROPOSE[0] + PROPOSE[1] * len(rows) + WAIT * near
        plan.append((rows, rung, s, propose, near))
    _record(path, [(r, g, s * (1 + SHARE) + p) for r, g, s, p, _ in plan])
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for line, (_, _, s, propose, near) in zip(lines, plan):
        line["d_target_ns"] = round(s * 1e9)
        if propose:
            line["d_propose_ns"] = round(propose * 1e9)
            line["t_recv_ns"][1] += round(LATE * 1e9) * near
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return path


@pytest.fixture(scope="module")
def law(tmp_path_factory):
    return _fit([_output_prefills(tmp_path_factory.mktemp("near") / "dp0.jsonl", 1600)])


def _proposed(cost) -> float:
    return sum(
        t.seconds for t in cost.draft_terms if t.name.startswith("draft.propose.")
    )


def _priced(law, start_lag_s) -> float:
    """The seconds `law`'s draft prices a one-request output prefill's
    `propose` pass at, its device starting it `start_lag_s` behind its host."""
    row = RequestShape(512, 612, False)
    view = BatchView((row,), produces_output=True, start_lag_s=start_lag_s)
    return _proposed(StepLinearBackend(StepLinearLaw(law)).price(view))


def test_a_pass_the_device_reaches_close_behind_its_host_is_fitted_and_priced_apart(
    law,
):
    near = law["draft"]["near_propose"]
    assert (near["coefficients"]["step"], near["coefficients"]["seqs"]) == (
        pytest.approx((PROPOSE[0] + WAIT, PROPOSE[1]), rel=1e-4)
    )
    assert near["fit"]["split"]["error"] < 1e-9 < near["fit"]["split"]["error_without"]
    assert _priced(law, -LATE) == pytest.approx(sum(PROPOSE) + WAIT, rel=1e-4)
    assert _priced(law, 0.0) == pytest.approx(sum(PROPOSE), rel=1e-4)
    with pytest.raises(CostRefused, match="how far behind its host"):
        _priced(law, None)


@pytest.mark.parametrize("steps, every", [(300, 3), (1600, 160)])
def test_a_split_of_too_few_passes_or_too_few_a_side_is_not_adopted(
    tmp_path, steps, every
):
    """150 output prefills, under `PROPOSE_SPLIT_STEPS`, where scatter that
    follows no lag clears the gain; or a side of 5, under `PROPOSE_SIDE_STEPS`."""
    path = _output_prefills(tmp_path / "dp0.jsonl", steps, every)
    assert "near_propose" not in _fit([path])["draft"]


def test_a_propose_pass_whose_scatter_follows_no_lag_is_not_split(tmp_path):
    """Labels scattered 5% about one price: a split cuts the error only by the
    two coefficients it adds."""
    path = _output_prefills(tmp_path / "dp0.jsonl", 1600, every=0)
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    rng = np.random.default_rng(0)
    for line in (x for x in lines if "d_propose_ns" in x):
        scattered = round(line["d_propose_ns"] * rng.uniform(0.95, 1.05))
        line["d_busy_ns"] += scattered - line["d_propose_ns"]
        line["d_propose_ns"] = scattered
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert "near_propose" not in _fit([path])["draft"]


def test_a_near_propose_the_law_cannot_read_whole_is_refused(law):
    near = law["draft"]["near_propose"]
    for draft, match in (
        ({"near_propose": {k: v for k, v in near.items() if k != "lag"}}, "no lag"),
        ({"propose": None}, "no propose vector"),
        ({"short_propose": near}, "token count.*refit"),
    ):
        with pytest.raises(ValueError, match=match):
            StepLinearLaw({**law, "draft": {**law["draft"], **draft}})


class _Kept(StepLinearBackend):
    """Keeps the last step's price."""

    def price(self, batch_view):
        self.cost = super().price(batch_view)
        return self.cost


@pytest.mark.parametrize("launch, wait", [(0.0, 0.0), (LATE, WAIT)])
def test_a_simulated_runner_prices_the_pass_from_its_host_clock_s_lag(
    law, launch, wait
):
    """A host whose launch outlasts the step's target queues the pass after
    the device reached it."""
    backend = _Kept(StepLinearLaw(law))
    r = runner(backend)
    laws = {"prefill": {"W_launch": [launch, 0.0, 0.0]}}
    r.compass_host = HostDeviceRecurrence(laws, ("E", "W_pre", "W_post"), lambda: 0.0)
    r.forward(prefill(8))
    assert r.refused_commands() == ()
    assert _proposed(backend.cost) == pytest.approx(sum(PROPOSE) + wait, rel=1e-4)

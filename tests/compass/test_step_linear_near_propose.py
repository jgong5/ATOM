# SPDX-License-Identifier: MIT
"""A draft law that holds `near_propose`, as one fitted before the fit's lag
split went may, prices the `propose` pass of an output prefill whose device
reaches it close behind its host apart, and a simulated runner reads the lag
off its host clock. Propose labels fit one `propose` vector whatever their lag."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_backend_step_linear_fit import _fit, _record, _steps, traced
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
#: The lag a law's `near_propose` splits at, in seconds: the one the fit's lag
#: split found on these records before it went.
LAG = 2.0**-20


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
    path.write_text("".join(json.dumps(traced(x)) + "\n" for x in lines))
    return path


def with_near_propose(law, **extra) -> dict:
    """`law` with a `near_propose` beside its draft's `propose`, `WAIT` more a
    pass, below `LAG` and with `extra`'s entries, as a law fitted before the
    fit's lag split went holds one."""
    coefficients = {"step": PROPOSE[0] + WAIT, "seqs": PROPOSE[1]}
    split = {**law["draft"]["propose"], "coefficients": coefficients, "lag": LAG}
    return {**law, "draft": {**law["draft"], "near_propose": split | extra}}


@pytest.fixture(scope="module")
def law(tmp_path_factory):
    record = _output_prefills(tmp_path_factory.mktemp("near") / "dp0.jsonl", 1600, 0)
    return with_near_propose(_fit([record]))


def _proposed(cost) -> float:
    return sum(
        t.seconds for t in cost.draft_terms if t.name.startswith("draft.propose.")
    )


def _priced(law, start_lag_s, tokens=512) -> float:
    """The seconds `law`'s draft prices a one-request output prefill's
    `propose` pass at, its device starting it `start_lag_s` behind its host."""
    row = RequestShape(tokens, tokens + 100, False)
    view = BatchView((row,), produces_output=True, start_lag_s=start_lag_s)
    return _proposed(StepLinearBackend(StepLinearLaw(law)).price(view))


def test_propose_labels_that_follow_a_lag_fit_one_propose_vector(tmp_path):
    """Every third pass runs `WAIT` longer, its host back from the wait `LATE`
    after its device start: one `propose` vector prices them all."""
    draft = _fit([_output_prefills(tmp_path / "dp0.jsonl", 1600)])["draft"]
    assert "propose" in draft and "near_propose" not in draft


def test_a_pass_the_device_reaches_close_behind_its_host_is_priced_apart(law):
    assert _priced(law, -LATE) == pytest.approx(sum(PROPOSE) + WAIT, rel=1e-4)
    assert _priced(law, 0.0) == pytest.approx(sum(PROPOSE), rel=1e-4)
    with pytest.raises(CostRefused, match="how far behind its host"):
        _priced(law, None)


def test_a_near_propose_the_law_cannot_read_whole_is_refused(law):
    near = law["draft"]["near_propose"]
    for draft, match in (
        ({"near_propose": {k: v for k, v in near.items() if k != "lag"}}, "no lag"),
        ({"propose": None}, "no propose vector"),
        ({"short_propose": {**near, "below": 1024}}, "near_propose.*short_propose"),
    ):
        with pytest.raises(ValueError, match=match):
            StepLinearLaw({**law, "draft": {**law["draft"], **draft}})


def test_a_law_split_at_a_token_count_prices_the_pass_by_it(law):
    """A law fitted before the lag split, as the checked-in MI455 ones were."""
    draft = {k: v for k, v in law["draft"].items() if k != "near_propose"}
    short = {**law["draft"]["near_propose"], "below": 1024}
    old = {**law, "draft": {**draft, "short_propose": short}}
    assert _priced(old, None, 1023) == pytest.approx(sum(PROPOSE) + WAIT, rel=1e-4)
    assert _priced(old, None, 1024) == pytest.approx(sum(PROPOSE), rel=1e-4)
    del short["below"]
    with pytest.raises(ValueError, match="short_propose.*no below"):
        StepLinearLaw(old)


def test_every_checked_in_law_loads():
    laws = sorted((Path(__file__).parents[2] / "scripts/compass/laws").glob("*.json"))
    assert laws
    for path in laws:
        StepLinearLaw(json.loads(path.read_text()))


def test_off_mtp_the_start_lag_waits_for_the_last_output_s_target():
    """The host returns from the wait once the previous output forward's target
    ends, after a launch that took no time."""
    batch = SimpleNamespace(
        total_seqs_num_prefill=1,
        total_tokens_num=8,
        total_seqs_num=1,
        produces_output=lambda: True,
    )
    host = HostDeviceRecurrence({}, (), lambda: 0.0)
    assert host.start_lag(batch) == 0.0
    host.charge(batch, 1.0, 0.0)
    assert host.start_lag(batch) == 0.0


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

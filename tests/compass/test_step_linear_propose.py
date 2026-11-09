# SPDX-License-Identifier: MIT
"""Propose passes of two lengths fit one `propose` vector, and a law split at
a token count, as the checked-in MI455 laws are, prices a short output
prefill's `propose` pass by its `short_propose` vector."""

import json
from pathlib import Path

import pytest
from test_backend_step_linear_fit import _fit, _record, _steps, traced
from test_step_linear_draft import PROPOSE, SHARE

from atom.compass.backends import (
    BatchView,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)

#: What a longer `propose` pass adds, in seconds.
EXTRA = 1e-3


def _output_prefills(path, steps: int, every: int = 3):
    """`_mtp_record` of `steps` of `_steps(3)`, whose prefills all produce
    output and each run `propose` after their draft at its `PROPOSE` price;
    unless `every` is 0, every `every`-th runs it `EXTRA` longer."""
    plan, i = [], 0
    for rows, rung, s in _steps(3, n=steps):
        propose = 0
        if not rows[0][2]:
            longer, i = every and i % every == 0, i + 1
            propose = PROPOSE[0] + PROPOSE[1] * len(rows) + EXTRA * longer
        plan.append((rows, rung, s, propose))
    _record(path, [(r, g, s * (1 + SHARE) + p) for r, g, s, p in plan])
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for line, (_, _, s, propose) in zip(lines, plan):
        line["d_target_ns"] = round(s * 1e9)
        if propose:
            line["d_propose_ns"] = round(propose * 1e9)
    path.write_text("".join(json.dumps(traced(x)) + "\n" for x in lines))
    return path


@pytest.fixture(scope="module")
def law(tmp_path_factory):
    record = _output_prefills(tmp_path_factory.mktemp("law") / "dp0.jsonl", 1600, 0)
    return _fit([record])


def _proposed(cost) -> float:
    return sum(
        t.seconds for t in cost.draft_terms if t.name.startswith("draft.propose.")
    )


def _priced(law, tokens=512) -> float:
    """The seconds `law`'s draft prices a one-request output prefill's
    `propose` pass at."""
    row = RequestShape(tokens, tokens + 100, False)
    view = BatchView((row,), produces_output=True)
    return _proposed(StepLinearBackend(StepLinearLaw(law)).price(view))


def test_propose_passes_of_two_lengths_fit_one_propose_vector(tmp_path):
    """Every third pass runs `EXTRA` longer: one `propose` vector prices them
    all."""
    draft = _fit([_output_prefills(tmp_path / "dp0.jsonl", 1600)])["draft"]
    assert "propose" in draft and "near_propose" not in draft


def test_a_law_split_at_a_token_count_prices_the_pass_by_it(law):
    """As the checked-in MI455 laws are split."""
    coefficients = {"step": PROPOSE[0] + EXTRA, "seqs": PROPOSE[1]}
    short = {**law["draft"]["propose"], "coefficients": coefficients, "below": 1024}
    old = {**law, "draft": {**law["draft"], "short_propose": short}}
    assert _priced(old, 1023) == pytest.approx(sum(PROPOSE) + EXTRA, rel=1e-4)
    assert _priced(old, 1024) == pytest.approx(sum(PROPOSE), rel=1e-4)
    del short["below"]
    with pytest.raises(ValueError, match="short_propose.*no below"):
        StepLinearLaw(old)


def test_every_checked_in_law_loads():
    laws = sorted((Path(__file__).parents[2] / "scripts/compass/laws").glob("*.json"))
    assert laws
    for path in laws:
        StepLinearLaw(json.loads(path.read_text()))

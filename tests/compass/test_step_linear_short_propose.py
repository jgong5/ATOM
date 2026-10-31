# SPDX-License-Identifier: MIT
"""A draft law prices a short output prefill's `propose` pass apart, when the
records' propose labels split at a token count."""

import json

import numpy as np
import pytest
from test_backend_step_linear_fit import _fit
from test_step_linear_draft import PROPOSE, _proposing

from atom.compass.backends import (
    BatchView,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)

#: What an output prefill of fewer than `SHORT` tokens adds to its `propose`
#: pass, in seconds.
WAIT, SHORT = 1e-3, 4096


def _rewrite(path, change):
    """`change(line)` on each output prefill line of the record at `path`."""
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for line in lines:
        if not line["rows"][0][2] and line["produces_output"]:
            change(line)
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return path


def _proposed(law, tokens: int, output: bool = True) -> float:
    """The seconds `law`'s draft prices the `propose` pass of a one-request
    prefill at."""
    row = RequestShape(tokens, tokens + 100, False)
    cost = StepLinearBackend(StepLinearLaw(law)).price(
        BatchView((row,), produces_output=output)
    )
    return sum(
        t.seconds for t in cost.draft_terms if t.name.startswith("draft.propose.")
    )


def test_a_short_output_prefill_s_propose_pass_is_fitted_and_priced_apart(tmp_path):
    def wait(line):
        if sum(r[0] for r in line["rows"]) < SHORT:
            line["d_propose_ns"] += round(WAIT * 1e9)
            line["d_busy_ns"] += round(WAIT * 1e9)

    path, _ = _proposing(tmp_path / "dp0.jsonl")
    law = _fit([_rewrite(path, wait)])

    assert _proposed(law, SHORT - 1) == pytest.approx(sum(PROPOSE) + WAIT, rel=1e-4)
    assert _proposed(law, SHORT) == pytest.approx(sum(PROPOSE), rel=1e-4)
    assert _proposed(law, SHORT - 1, output=False) == 0
    short = law["draft"]["short_propose"]
    assert short["below"] == SHORT
    assert (short["coefficients"]["step"], short["coefficients"]["seqs"]) == (
        pytest.approx((PROPOSE[0] + WAIT, PROPOSE[1]), rel=1e-4)
    )
    split = short["fit"]["split"]
    assert split["error"] < 1e-9 < split["error_without"]


def test_a_propose_pass_whose_scatter_follows_no_token_count_is_not_split(tmp_path):
    """Labels scattered 5% about one price: a split cuts the error only by the
    two coefficients it adds."""
    rng = np.random.default_rng(0)

    def scatter(line):
        scattered = round(line["d_propose_ns"] * rng.uniform(0.95, 1.05))
        line["d_busy_ns"] += scattered - line["d_propose_ns"]
        line["d_propose_ns"] = scattered

    path, _ = _proposing(tmp_path / "dp0.jsonl")
    law = _fit([_rewrite(path, scatter)])
    assert "short_propose" not in law["draft"]
    assert _proposed(law, SHORT - 1) == pytest.approx(sum(PROPOSE), rel=0.05)

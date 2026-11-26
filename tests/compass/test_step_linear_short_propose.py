# SPDX-License-Identifier: MIT
"""Under COMPASS_FIT_SHORT_PROPOSE_BELOW the fit splits an output prefill's
propose pass at a prefill token count: the shorter ones fit the draft law's
`short_propose` vector, with the count as its `below`, the rest `propose`.

A drafter whose propose pass runs over the chunk's tokens (Qwen3.8-Flash-Next's
MTP: 3.8 ms p50 under 2,048 tokens, 20.4 ms at 16k) otherwise fits one constant
near the long passes' and drops the short ones as outliers."""

import json

import pytest
from test_backend_step_linear_fit import _record, _steps, traced
from test_step_linear_draft import PROPOSE, SHARE

from atom.compass.backends import (
    BatchView,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_linear_fit import (
    SHORT_PROPOSE_ENV,
    env_options,
    fit,
    label_records,
)

#: The prefill token count the records' propose pass changes length at.
BELOW = 4096
#: What a pass over at least `BELOW` prefill tokens adds, in seconds.
LONG = 15e-3


def _propose(rows) -> float:
    """A prefill step's propose pass: `PROPOSE`, `LONG` more from `BELOW`
    prefill tokens on."""
    tokens = sum(q for q, _, decode in rows if not decode)
    return PROPOSE[0] + PROPOSE[1] * len(rows) + LONG * (tokens >= BELOW)


def _output_prefills(path, steps: int = 1600):
    """An MTP record of `steps` of `_steps(3)`, whose prefills all produce
    output and run `_propose` after their draft."""
    plan = [
        (rows, rung, s, 0 if rows[0][2] else _propose(rows))
        for rows, rung, s in _steps(3, n=steps)
    ]
    _record(path, [(r, g, s * (1 + SHARE) + p) for r, g, s, p in plan])
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for line, (_, _, s, propose) in zip(lines, plan):
        line["d_target_ns"] = round(s * 1e9)
        if propose:
            line["d_propose_ns"] = round(propose * 1e9)
    path.write_text("".join(json.dumps(traced(x)) + "\n" for x in lines))
    return path


@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    record = _output_prefills(tmp_path_factory.mktemp("short") / "dp0.jsonl")
    return label_records([record], 1)[0]


def _fit(samples, **options):
    return fit(samples, {"form": "base", "dp": 1, **options})[0]


def _priced(law, tokens) -> float:
    """The seconds `law`'s draft prices a one-request output prefill's propose
    pass of `tokens` prefill tokens at."""
    view = BatchView((RequestShape(tokens, tokens + 100, False),), produces_output=True)
    cost = StepLinearBackend(StepLinearLaw(law)).price(view)
    return sum(
        t.seconds for t in cost.draft_terms if t.name.startswith("draft.propose.")
    )


def test_the_switch_takes_a_token_count():
    assert env_options({SHORT_PROPOSE_ENV: "8192"}) == {"short_propose_below": 8192}
    assert env_options({SHORT_PROPOSE_ENV: "0"}) == {}
    assert env_options({}) == {}
    with pytest.raises(ValueError, match=SHORT_PROPOSE_ENV):
        env_options({SHORT_PROPOSE_ENV: "8k"})


def test_off_the_draft_law_fits_one_propose_vector_as_before(samples):
    off = _fit(samples)
    assert json.dumps(off, sort_keys=True) == json.dumps(
        _fit(samples, **env_options({})), sort_keys=True
    )
    assert "short_propose" not in off["draft"]


def test_on_the_short_passes_fit_short_propose_under_its_below(samples):
    law = _fit(samples, short_propose_below=BELOW)
    draft = law["draft"]
    assert draft["short_propose"]["below"] == BELOW
    assert draft["short_propose"]["coefficients"] == pytest.approx(
        {"step": PROPOSE[0], "seqs": PROPOSE[1]}, rel=1e-4
    )
    assert draft["propose"]["coefficients"] == pytest.approx(
        {"step": PROPOSE[0] + LONG, "seqs": PROPOSE[1]}, rel=1e-4
    )
    assert _priced(law, BELOW - 1) == pytest.approx(sum(PROPOSE), rel=1e-4)
    assert _priced(law, BELOW) == pytest.approx(sum(PROPOSE) + LONG, rel=1e-4)


def test_a_count_no_pass_reaches_fits_propose_alone(samples):
    draft = _fit(samples, short_propose_below=1_000_000)["draft"]
    assert "short_propose" not in draft
    assert draft["propose"]["fit"]["samples"] > 0


def test_a_split_is_refused_above_one_rank(samples):
    with pytest.raises(ValueError, match=SHORT_PROPOSE_ENV):
        fit(samples, {"form": "base", "dp": 2, "short_propose_below": BELOW})

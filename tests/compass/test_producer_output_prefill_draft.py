# SPDX-License-Identifier: MIT
"""A P/D producer's output prefill under a draft law with `output_prefill_kv` and no `propose`: its drafter runs only
the draft-KV pass, which the law prices among the step's target terms (step_linear's module text), so the step has no
`draft_terms`. The speculative runner must charge it, not refuse it as a law that "prices no draft" and charge 0 s: that
dropped the target's prefill too, 18.7% of a 1P1D producer's device time (cell B2 r1_c128, #3720).
"""

import pytest
from test_backend_step_linear_fit import _steps
from test_dp_step_max import prefill, runner
from test_speculative_step import _speculative
from test_step_linear_draft import _fit, _mtp_record, _output

from atom.compass.backends import StepLinearBackend, StepLinearLaw
from atom.compass.validate.cost import cost_accuracy


def _producer_law(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    law = _fit([_mtp_record(tmp_path / "dp0.jsonl", _steps(3))])
    law["draft"]["output_prefill_kv"] = True
    for k in ("propose", "short_propose", "forward"):
        law["draft"].pop(k, None)
    return law


def test_an_output_prefill_whose_draft_is_its_kv_pass_is_charged_not_refused(tmp_path):
    law = _producer_law(tmp_path)
    priced = StepLinearBackend(StepLinearLaw(law)).price(_output([(8, 8, False)]))
    assert priced.draft_terms == ()
    assert any(t.name.startswith("draft.") for t in priced.terms)
    tagged = _speculative(runner(StepLinearBackend(StepLinearLaw(law))))
    got = tagged.forward(prefill(8)).predicted_s
    assert tagged.refused_commands() == ()
    assert got == pytest.approx(priced.seconds)
    assert got > 0


def test_a_law_with_no_draft_at_all_is_still_refused(tmp_path):
    law = _producer_law(tmp_path)
    del law["draft"]
    plain = _speculative(runner(StepLinearBackend(StepLinearLaw(law))))
    assert plain.forward(prefill(8)).predicted_s == 0.0
    assert plain.refused_commands() == ("cost:StepLinearBackend: prices no draft",)


def test_a_draft_the_law_prices_among_the_target_is_graded_with_it_not_apart(tmp_path):
    """The figures grade a step's draft only where the law prices one apart: an
    output prefill's draft-KV pass is priced among its target's terms, and its
    draft label holds only the rest of the drafter's time, which no vector
    prices (MiniMax-M3.1's PP4 prefill, stage 3: a 22 us residual graded against
    0 s, a median -100%)."""
    record = _mtp_record(tmp_path / "dp0.jsonl", _steps(3))
    law = _producer_law(tmp_path / "fit")
    backend = StepLinearBackend(StepLinearLaw(law))
    report = cost_accuracy(record.parent, law)

    draft, decode = report["graded"]["draft"], report["graded"]["decode"]
    # Each decode prices its draft apart; no output prefill here does.
    assert backend.price(_output([(8, 8, False)])).draft_terms == ()
    assert draft["steps"] == decode["steps"] > 0
    assert draft["median"] > -0.5

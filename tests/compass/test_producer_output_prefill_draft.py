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


def _producer_law(tmp_path):
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

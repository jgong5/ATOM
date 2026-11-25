# SPDX-License-Identifier: MIT
"""A step linear law's vector that states `declared`, fitted on another subject,
prices its terms `declared` with that note and never validated, where any other
vector's are `estimated`. The crs005 prefill law prices a prefill-side rung-2
decode by the crs005 decode law's rung-2 vector so, where it refused the rung.
"""

from test_backend_step_linear import law, view
from test_crs005_dpa8_laws import backend, decode_step

from atom.compass.backends import Method, StepLinearBackend, StepLinearLaw

ORIGIN = "fitted under another engine config"

#: The prefill side's rung-2 decode in #975's clean warm pass (c=256, step 526)
#: as its costliest rank ran it: the group's 9 MoE tokens, that rank's one row.
RUNG_2 = decode_step(2, 9, (186401,))


def test_a_declared_vector_prices_declared_terms_with_its_note_never_validated():
    spec = law()
    spec["decode"]["2"]["declared"] = ORIGIN
    step = view((1, 400, True), rung=2)

    cost = StepLinearBackend(StepLinearLaw(spec)).price(step)

    assert {t.provenance.method for t in cost.terms} == {Method.DECLARED}
    assert {t.provenance.note for t in cost.terms} == {ORIGIN}
    assert not any(t.provenance.validated for t in cost.terms)
    assert cost.seconds == StepLinearBackend(StepLinearLaw(law())).price(step).seconds


def test_the_crs005_prefill_law_prices_rung_2_by_the_decode_law_s_vector():
    prefill, decode = backend("crs005", "prefill"), backend("crs005", "decode")

    cost, fitted = prefill.price(RUNG_2), decode.price(RUNG_2)

    assert {t.provenance.method for t in cost.terms} == {Method.DECLARED}
    assert {t.provenance.method for t in fitted.terms} == {Method.ESTIMATED}
    assert cost.seconds == fitted.seconds > 0
